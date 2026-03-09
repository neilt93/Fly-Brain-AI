"""
Sanity checks for the brain-body bridge (v2).

Run these before and after connecting the real brain model.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.interfaces import BodyObservation, BrainInput, BrainOutput, LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter


def _make_obs(**overrides):
    """Create a BodyObservation with defaults."""
    defaults = dict(
        joint_angles=np.random.randn(42).astype(np.float32),
        joint_velocities=np.random.randn(42).astype(np.float32),
        contact_forces=np.random.rand(6).astype(np.float32),
        body_velocity=np.random.randn(3).astype(np.float32),
        body_orientation=np.random.randn(3).astype(np.float32),
    )
    defaults.update(overrides)
    return BodyObservation(**defaults)


def test_encoder_output_range():
    """Encoder output rates should be in [0, max_rate_hz]."""
    ids = np.arange(20, dtype=np.int64)
    enc = SensoryEncoder(ids, max_rate_hz=100.0)

    bi = enc.encode(_make_obs())
    assert bi.firing_rates_hz.min() >= 0, f"Negative rate: {bi.firing_rates_hz.min()}"
    assert bi.firing_rates_hz.max() <= 100.0, f"Rate exceeds max: {bi.firing_rates_hz.max()}"
    assert len(bi.neuron_ids) == 20
    print("  encoder_output_range: PASS")


def test_channel_encoder():
    """Channel-aware encoder maps different obs fields to different neuron subsets."""
    ids = np.arange(20, dtype=np.int64)
    channel_map = {
        "gustatory": list(range(0, 5)),
        "proprioceptive": list(range(5, 12)),
        "mechanosensory": list(range(12, 17)),
        "vestibular": list(range(17, 20)),
    }
    enc = SensoryEncoder(ids, channel_map=channel_map, max_rate_hz=100.0, baseline_rate_hz=10.0)

    # High contact forces should boost gustatory + mechanosensory channels
    obs_high_contact = _make_obs(contact_forces=np.ones(6, dtype=np.float32))
    obs_low_contact = _make_obs(contact_forces=np.zeros(6, dtype=np.float32))

    bi_high = enc.encode(obs_high_contact)
    bi_low = enc.encode(obs_low_contact)

    # Gustatory neurons (0-4) should be higher with high contact
    gust_high = bi_high.firing_rates_hz[:5].mean()
    gust_low = bi_low.firing_rates_hz[:5].mean()
    assert gust_high > gust_low, f"Gustatory not contact-responsive: {gust_high} vs {gust_low}"

    # Mechanosensory neurons (12-16) should be higher with high contact
    mech_high = bi_high.firing_rates_hz[12:17].mean()
    mech_low = bi_low.firing_rates_hz[12:17].mean()
    assert mech_high > mech_low, f"Mechanosensory not contact-responsive: {mech_high} vs {mech_low}"

    # All rates in valid range
    assert bi_high.firing_rates_hz.min() >= 0
    assert bi_high.firing_rates_hz.max() <= 100.0

    print("  channel_encoder: PASS")


def test_channel_independence():
    """Changing one channel's input shouldn't affect unrelated channels."""
    ids = np.arange(20, dtype=np.int64)
    channel_map = {
        "gustatory": list(range(0, 5)),
        "proprioceptive": list(range(5, 12)),
        "mechanosensory": list(range(12, 17)),
        "vestibular": list(range(17, 20)),
    }
    enc = SensoryEncoder(ids, channel_map=channel_map, max_rate_hz=100.0, baseline_rate_hz=10.0)

    # Two obs: same everything except joint angles
    base_obs = _make_obs()
    diff_obs = BodyObservation(
        joint_angles=base_obs.joint_angles * 5,  # very different
        joint_velocities=base_obs.joint_velocities,
        contact_forces=base_obs.contact_forces,
        body_velocity=base_obs.body_velocity,
        body_orientation=base_obs.body_orientation,
    )

    bi_base = enc.encode(base_obs)
    bi_diff = enc.encode(diff_obs)

    # Proprioceptive neurons (5-11) should change
    prop_delta = np.abs(bi_diff.firing_rates_hz[5:12] - bi_base.firing_rates_hz[5:12]).mean()
    # Vestibular neurons (17-19) should NOT change (same body vel/ori)
    vest_delta = np.abs(bi_diff.firing_rates_hz[17:20] - bi_base.firing_rates_hz[17:20]).mean()

    assert prop_delta > vest_delta, (
        f"Channel independence violated: prop_delta={prop_delta:.4f} vest_delta={vest_delta:.4f}"
    )
    print(f"  channel_independence: PASS (prop_delta={prop_delta:.4f} vest_delta={vest_delta:.4f})")


def test_monotonic_forward():
    """Higher forward readout rates → higher forward_drive."""
    ids = np.arange(25, dtype=np.int64)
    decoder = DescendingDecoder(
        forward_ids=ids[:5], turn_left_ids=ids[5:10], turn_right_ids=ids[10:15],
        rhythm_ids=ids[15:20], stance_ids=ids[20:25],
    )

    low_rates = np.zeros(25, dtype=np.float32)
    high_rates = np.zeros(25, dtype=np.float32)
    high_rates[:5] = 80.0  # forward group

    low_cmd = decoder.decode(BrainOutput(ids, low_rates))
    high_cmd = decoder.decode(BrainOutput(ids, high_rates))

    assert high_cmd.forward_drive > low_cmd.forward_drive, (
        f"Not monotonic: low={low_cmd.forward_drive}, high={high_cmd.forward_drive}"
    )
    print("  monotonic_forward: PASS")


def test_turn_asymmetry():
    """Left readout > right readout → positive turn_drive."""
    ids = np.arange(25, dtype=np.int64)
    decoder = DescendingDecoder(
        forward_ids=ids[:5], turn_left_ids=ids[5:10], turn_right_ids=ids[10:15],
        rhythm_ids=ids[15:20], stance_ids=ids[20:25],
    )

    rates = np.zeros(25, dtype=np.float32)
    rates[5:10] = 60.0   # turn_left group active
    rates[10:15] = 0.0    # turn_right group silent

    cmd = decoder.decode(BrainOutput(ids, rates))
    assert cmd.turn_drive > 0, f"Expected positive turn_drive, got {cmd.turn_drive}"
    print("  turn_asymmetry: PASS")


def test_locomotion_bridge_action_shape():
    """LocomotionBridge produces valid FlyGym action dict."""
    bridge = LocomotionBridge()
    bridge.warmup(100)

    cmd = LocomotionCommand(forward_drive=0.8, turn_drive=0.1, step_frequency=1.2, stance_gain=1.0)
    action = bridge.step(cmd)

    assert "joints" in action, "Missing 'joints' key"
    assert "adhesion" in action, "Missing 'adhesion' key"
    assert action["joints"].shape == (42,), f"Wrong joints shape: {action['joints'].shape}"
    assert action["adhesion"].shape == (6,), f"Wrong adhesion shape: {action['adhesion'].shape}"
    print("  locomotion_bridge_action_shape: PASS")


def test_flygym_adapter_extraction():
    """FlyGymAdapter extracts correct shapes from obs dict."""
    adapter = FlyGymAdapter()
    obs = {
        "joints": np.random.randn(3, 42),
        "fly": np.random.randn(4, 3),
        "contact_forces": np.random.randn(30, 3),
        "end_effectors": np.random.randn(6, 3),
    }
    body_obs = adapter.extract_body_observation(obs)

    assert body_obs.joint_angles.shape == (42,)
    assert body_obs.joint_velocities.shape == (42,)
    assert body_obs.contact_forces.shape == (6,)
    assert body_obs.body_velocity.shape == (3,)
    assert body_obs.body_orientation.shape == (3,)
    print("  flygym_adapter_extraction: PASS")


def test_fake_brain_loop():
    """Full loop with FakeBrainRunner + channel encoder produces varying commands."""
    from bridge.brain_runner import FakeBrainRunner

    sensory_ids = np.arange(20, dtype=np.int64)
    readout_ids = np.arange(100, 125, dtype=np.int64)
    channel_map = {
        "gustatory": list(range(0, 5)),
        "proprioceptive": list(range(5, 12)),
        "mechanosensory": list(range(12, 17)),
        "vestibular": list(range(17, 20)),
    }

    brain = FakeBrainRunner(readout_neuron_ids=readout_ids)
    encoder = SensoryEncoder(sensory_ids, channel_map=channel_map)
    decoder = DescendingDecoder(
        forward_ids=readout_ids[:5], turn_left_ids=readout_ids[5:10],
        turn_right_ids=readout_ids[10:15], rhythm_ids=readout_ids[15:20],
        stance_ids=readout_ids[20:25],
    )

    commands = []
    for _ in range(10):
        bi = encoder.encode(_make_obs())
        bo = brain.step(bi)
        cmd = decoder.decode(bo)
        commands.append(cmd.forward_drive)

    variance = np.var(commands)
    assert variance > 1e-6, f"Commands not varying: variance={variance}"
    print(f"  fake_brain_loop: PASS (drive variance={variance:.6f})")


def main():
    print("Running sanity checks (v2)...")
    test_encoder_output_range()
    test_channel_encoder()
    test_channel_independence()
    test_monotonic_forward()
    test_turn_asymmetry()
    test_locomotion_bridge_action_shape()
    test_flygym_adapter_extraction()
    test_fake_brain_loop()
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
