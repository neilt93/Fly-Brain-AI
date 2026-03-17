"""
Export closed-loop brain-body simulation data for Unity visualization.

Runs the full bridge (sensory encoder → 139k LIF brain → descending decoder → CPG body)
and exports timeseries_plastic.json + connectome_activity.json to FlyBrainViz/Assets/Resources/.

Usage:
    cd plastic-fly
    python export_unity_demo.py                        # default 10k steps
    python export_unity_demo.py --body-steps 20000     # longer demo
    python export_unity_demo.py --fake-brain            # test without Brian2
"""

import sys
import json
import time
import argparse
import os
import tempfile
import numpy as np
from pathlib import Path


def _write_json_atomic(path, obj):
    """Write JSON to *path* atomically via a temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge.config import BridgeConfig
from bridge.interfaces import LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter

UNITY_RESOURCES = Path(__file__).resolve().parent.parent / "FlyBrainViz" / "Assets" / "Resources"
LOG_INTERVAL = 20  # log every 20 sim steps → dt = 20 * 1e-4 = 0.002s (smooth animation)


def run_and_export(body_steps=10000, warmup_steps=3000, use_fake_brain=False, seed=42):
    import flygym

    cfg = BridgeConfig()

    # --- Load populations ---
    if not cfg.sensory_ids_path.exists():
        print("Run: python scripts/select_populations.py")
        return
    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)
    print(f"Populations: {len(sensory_ids)} sensory, {len(readout_ids)} readout")

    # --- Components ---
    encoder = SensoryEncoder.from_channel_map(
        sensory_ids, cfg.channel_map_path,
        max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    print(f"Building brain ({brain_label})...")
    brain = create_brain_runner(
        sensory_ids=sensory_ids, readout_ids=readout_ids,
        use_fake=use_fake_brain, warmup_ms=cfg.brain_warmup_ms,
    )

    # --- FlyGym ---
    print("Initializing FlyGym...")
    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    arena = flygym.arena.FlatTerrain()
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=arena, timestep=1e-4)
    obs, info = sim.reset()

    # Get joint names from fly
    joint_names = [str(n) for n in fly_obj.actuated_joints]
    print(f"  {len(joint_names)} actuated joints")

    # --- Warmup --- start with tripod phase pattern, ramp magnitude from 0
    print(f"Warming up ({warmup_steps} steps)...")
    tripod_phases = np.array([0, np.pi, 0, np.pi, 0, np.pi])  # alternating tripod
    locomotion.cpg.reset(init_phases=tripod_phases, init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        obs, _, terminated, truncated, info = sim.step(action)
        if terminated or truncated:
            print("  Ended during warmup!")
            sim.close()
            return

    # --- Main loop: collect data ---
    bspb = cfg.body_steps_per_brain
    print(f"\nRunning {body_steps} body steps, brain every {bspb} steps ({brain_label})")

    positions = []
    contacts_log = []
    contact_forces_log = []
    end_effectors_log = []
    joint_angles_log = []
    tripod_scores = []
    brain_commands = []

    current_cmd = LocomotionCommand(forward_drive=1.0)
    brain_steps = 0
    t0 = time.time()

    for step in range(body_steps):
        # Brain step
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            current_cmd = decoder.decode(brain_output)
            brain_steps += 1

            if brain_steps % 10 == 1:
                active = int(np.sum(brain_output.firing_rates_hz > 0))
                print(f"  brain #{brain_steps:3d}: fwd={current_cmd.forward_drive:+.3f} "
                      f"turn={current_cmd.turn_drive:+.3f} active={active}/{len(readout_ids)}")

        # Body step
        action = locomotion.step(current_cmd)
        try:
            obs, _, terminated, truncated, info = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

        # Log every LOG_INTERVAL steps
        if step % LOG_INTERVAL == 0:
            fly_state = np.asarray(obs["fly"])         # (4, 3)
            joints = np.asarray(obs["joints"])          # (3, 42)
            cf_raw = np.asarray(obs["contact_forces"])  # (30, 3)
            ee = np.asarray(obs["end_effectors"])       # (6, 3)

            # Position (MuJoCo coords — Unity converts)
            positions.append(fly_state[0].tolist())

            # Joint angles (42 DOFs)
            joint_angles_log.append(joints[0].tolist())

            # Contact forces per leg: max magnitude across 5 sensors
            cf_mags = np.linalg.norm(cf_raw, axis=1)  # (30,)
            per_leg = [float(cf_mags[i*5:(i+1)*5].max()) for i in range(6)]
            contact_forces_log.append(per_leg)

            # Binary contacts (force > threshold)
            binary_contacts = [1.0 if f > 0.1 else 0.0 for f in per_leg]
            contacts_log.append(binary_contacts)

            # End effectors
            end_effectors_log.append(ee.tolist())

            # Tripod score: legs 0,2,4 vs 1,3,5 alternating contact
            left_set = [binary_contacts[0], binary_contacts[2], binary_contacts[4]]
            right_set = [binary_contacts[1], binary_contacts[3], binary_contacts[5]]
            ls, rs = sum(left_set), sum(right_set)
            if ls + rs > 0:
                tripod = 1.0 - abs(ls - rs) / (ls + rs)
            else:
                tripod = 0.0
            tripod_scores.append(tripod)

            brain_commands.append({
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
                "step_frequency": current_cmd.step_frequency,
                "stance_gain": current_cmd.stance_gain,
            })

    sim.close()
    elapsed = time.time() - t0
    n_frames = len(positions)
    print(f"\nDone: {step+1} body steps, {brain_steps} brain steps, {n_frames} frames in {elapsed:.1f}s")

    # --- Fix height only (no smoothing) ---
    pos_arr = np.array(positions)

    # Height fix: subtract initial Z so fly sits on the ground.
    # Ground is at Unity Y=-0.8, fly legs are ~0.5 units (globalScale=0.5).
    # So torso should be at Unity Y = -0.8 + 0.5 = -0.3
    initial_z = pos_arr[0, 2]
    target_unity_y = -0.3
    pos_arr[:, 2] = pos_arr[:, 2] - initial_z + target_unity_y

    positions = pos_arr.tolist()
    print(f"  Height offset: Z-{initial_z:.3f} -> Unity Y~{target_unity_y} (raw positions, no smoothing)")

    # --- Export timeseries_plastic.json ---
    dt = LOG_INTERVAL * 1e-4  # seconds per frame

    ts = {
        "controller": "brain_bridge",
        "dt": dt,
        "n_frames": n_frames,
        "positions": positions,
        "contacts": contacts_log,
        "contact_forces": contact_forces_log,
        "end_effectors": end_effectors_log,
        "joint_angles": joint_angles_log,
        "joint_names": joint_names,
        "tripod_score": tripod_scores,
        "weight_drifts": [cmd["forward_drive"] for cmd in brain_commands],
        "perturbation_idx": 0,
    }

    out_path = UNITY_RESOURCES / "timeseries_plastic.json"
    _write_json_atomic(out_path, ts)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nExported {out_path} ({n_frames} frames, {size_mb:.1f} MB)")

    # Also write a minimal timeseries_fixed.json (baseline straight walk, no brain)
    # so comparison mode works too
    ts_fixed = dict(ts)
    ts_fixed["controller"] = "fixed_baseline"
    ts_fixed["weight_drifts"] = [0.0] * n_frames
    fixed_path = UNITY_RESOURCES / "timeseries_fixed.json"
    _write_json_atomic(fixed_path, ts_fixed)
    print(f"Exported {fixed_path} (clone for comparison mode)")

    return n_frames


def export_connectome(n_walking_frames):
    """Run the connectome brain viz export to match frame count."""
    print(f"\n{'='*60}")
    print("EXPORTING CONNECTOME BRAIN ACTIVITY")
    print(f"{'='*60}")

    # Import and run the existing export script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_connectome_viz",
        str(Path(__file__).resolve().parent / "export_connectome_viz.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Run brain sim
    spike_trains, i2flyid, flyid2i, exc_indices, df_comp = mod.run_brain_simulation(freq_hz=100)

    # Select neurons
    selected = mod.select_viz_neurons(spike_trains, exc_indices, n_neurons=250)

    # Classify
    neuron_types, neuron_names = mod.classify_neurons_fast(selected, exc_indices, spike_trains)

    # Connections
    connections = mod.extract_subgraph(selected, max_connections=300)

    # Layout + rates
    positions = mod.generate_3d_layout(len(selected), neuron_types)
    binned = mod.bin_spikes_to_rates(spike_trains, selected)
    firing_rates = mod.tile_to_frames(binned, n_walking_frames)

    # Export
    mod.export_json(
        UNITY_RESOURCES / "connectome_activity.json",
        positions, connections, neuron_types, neuron_names,
        firing_rates, n_total_neurons=len(df_comp)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export brain-body demo for Unity")
    parser.add_argument("--body-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-connectome", action="store_true",
                        help="Skip connectome export (body timeseries only)")
    args = parser.parse_args()

    print("=" * 60)
    print("UNITY DEMO EXPORT: Brain-Body Closed Loop")
    print("=" * 60)

    n_frames = run_and_export(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        seed=args.seed,
    )

    if n_frames and not args.skip_connectome:
        export_connectome(n_frames)

    print("\n" + "=" * 60)
    print("DONE. Open Unity and press Play.")
    print("=" * 60)
