"""
Chemotaxis experiment: can the connectome-driven fly navigate toward an odor source?

Odor signals injected into olfactory receptor neurons propagate through the
connectome's antennal lobe and mushroom body circuits, producing turning
behavior without any engineered navigation logic.

Setup:
  - OdorArena with single odor source at (x, y) offset from start
  - Fly starts at origin facing +X
  - ORNs (100 neurons: 50 left, 50 right) receive bilateral odor signal
  - Optional: R7/R8 photoreceptors (100 neurons) receive visual input

Usage:
    python experiments/chemotaxis.py --fake-brain         # fast test (no brain)
    python experiments/chemotaxis.py                      # real brain, odor only
    python experiments/chemotaxis.py --enable-vision      # odor + vision
    python experiments/chemotaxis.py --odor-y 10          # odor source 10mm left
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter


def _write_json_atomic(path: Path, payload: dict):
    """Write JSON atomically so checkpoints stay readable if the run is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _record_frame(obs, positions, odor_readings):
    pos = np.array(obs["fly"][0])
    positions.append(pos.tolist())
    if "odor_intensity" in obs:
        odor_readings.append(obs["odor_intensity"].tolist())


def _save_checkpoint(output_path, config_payload, steps_completed, body_steps,
                     positions, odor_readings, episode_log, status):
    _write_json_atomic(output_path / "chemotaxis_checkpoint.json", {
        "status": status,
        "steps_completed": steps_completed,
        "body_steps": body_steps,
        "config": config_payload,
        "positions": positions,
        "odor_readings": odor_readings,
        "episode_log": episode_log,
    })


def run_chemotaxis(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    enable_vision: bool = False,
    odor_source: tuple = (15.0, 5.0, 0.0),  # (x, y, z) in mm — ahead and to the left
    odor_intensity: float = 1.0,
    seed: int = 42,
    output_dir: str = "logs/chemotaxis",
):
    import flygym

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Use v3 sensory populations (with olfactory + visual) ---
    sensory_ids_path = cfg.data_dir / "sensory_ids_v3.npy"
    channel_map_path = cfg.data_dir / "channel_map_v3.json"

    if not sensory_ids_path.exists():
        print("Run: python -c 'see setup in chemotaxis.py docstring'")
        print("Need data/sensory_ids_v3.npy and data/channel_map_v3.json")
        return None

    sensory_ids = np.load(sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)
    print("Loaded populations: %d sensory (v3), %d readout" % (len(sensory_ids), len(readout_ids)))

    # --- Components ---
    encoder = SensoryEncoder.from_channel_map(
        sensory_ids, channel_map_path,
        max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    print("Building brain (%s)..." % brain_label)
    brain = create_brain_runner(
        sensory_ids=sensory_ids,
        readout_ids=readout_ids,
        use_fake=use_fake_brain,
        warmup_ms=cfg.brain_warmup_ms,
    )

    # --- FlyGym with OdorArena ---
    print("Initializing FlyGym with OdorArena...")
    print("  Odor source at (%.1f, %.1f, %.1f) mm, intensity=%.1f" % (
        odor_source[0], odor_source[1], odor_source[2], odor_intensity))

    # OdorArena: place odor source
    odor_source_pos = np.array([[odor_source[0], odor_source[1], odor_source[2]]])
    peak_intensity = np.array([[odor_intensity]])  # (n_sources, n_dims)

    arena = flygym.arena.OdorArena(
        odor_source=odor_source_pos,
        peak_odor_intensity=peak_intensity,
        diffuse_func=lambda x: x**-2,  # inverse-square falloff
    )

    fly_obj = flygym.Fly(
        enable_adhesion=True,
        init_pose="stretch",
        control="position",
        enable_olfaction=True,
        enable_vision=enable_vision,
    )
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=arena, timestep=1e-4)
    obs, info = sim.reset()

    # Check what observations we get
    has_odor = "odor_intensity" in obs
    has_vision = "vision" in obs
    print("  Olfaction: %s (shape: %s)" % (has_odor, obs.get("odor_intensity", np.array([])).shape if has_odor else "N/A"))
    print("  Vision: %s (shape: %s)" % (has_vision, obs.get("vision", np.array([])).shape if has_vision else "N/A"))

    # --- Warmup ---
    print("Warming up (%d steps)..." % warmup_steps)
    locomotion.warmup(0)
    locomotion.cpg.reset(init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
                         init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, terminated, truncated, info = sim.step(action)
            if terminated or truncated:
                sim.close()
                return None
        except Exception as e:
            print("  Warmup error: %s" % e)
            sim.close()
            return None

    # --- Main loop ---
    bspb = cfg.body_steps_per_brain
    print("\nRunning chemotaxis: %d body steps, brain every %d steps (%s)" % (
        body_steps, bspb, brain_label))

    positions = []
    odor_readings = []
    episode_log = []
    brain_steps = 0
    current_cmd = LocomotionCommand(forward_drive=1.0)
    step = 0
    config_payload = {
        "body_steps": body_steps,
        "odor_source": list(odor_source),
        "odor_intensity": odor_intensity,
        "enable_vision": enable_vision,
        "use_fake_brain": use_fake_brain,
        "n_sensory": len(sensory_ids),
        "n_readout": len(readout_ids),
    }

    t_start = time.time()

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            current_cmd = decoder.decode(brain_output)
            brain_steps += 1

            # Log odor signal
            if body_obs.odor_intensity is not None:
                odor = body_obs.odor_intensity
                if odor.ndim == 2 and odor.shape[1] >= 4:
                    l_odor = float((odor[0, 0] + odor[0, 2]) * 0.5)
                    r_odor = float((odor[0, 1] + odor[0, 3]) * 0.5)
                else:
                    l_odor = r_odor = 0.0
            else:
                l_odor = r_odor = 0.0

            mean_rate = float(np.mean(brain_output.firing_rates_hz))
            active = int(np.sum(brain_output.firing_rates_hz > 0))

            if brain_steps % 5 == 1:
                print("  brain #%3d: fwd=%+.3f turn=%+.3f | odor L=%.3f R=%.3f | rate=%.0fHz active=%d/%d" % (
                    brain_steps, current_cmd.forward_drive, current_cmd.turn_drive,
                    l_odor, r_odor, mean_rate, active, len(readout_ids)))

            episode_log.append({
                "body_step": step,
                "brain_step": brain_steps,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
                "odor_left": l_odor,
                "odor_right": r_odor,
                "readout_mean_hz": mean_rate,
                "readout_active": active,
            })

        action = locomotion.step(current_cmd)
        try:
            obs, _, terminated, truncated, info = sim.step(action)
        except Exception as e:
            print("  Physics error at step %d: %s" % (step, e))
            break

        if step % 50 == 0:
            _record_frame(obs, positions, odor_readings)
            _save_checkpoint(output_path, config_payload, step, body_steps,
                             positions, odor_readings, episode_log, "running")

        if terminated or truncated:
            print("  Episode ended at step %d" % step)
            break

    sim.close()
    elapsed = time.time() - t_start

    # --- Results ---
    print("\n" + "=" * 60)
    print("CHEMOTAXIS RESULTS")
    print("=" * 60)
    print("Steps: %d/%d in %.1fs, %d brain steps" % (step + 1, body_steps, elapsed, brain_steps))

    if len(positions) >= 2:
        start = np.array(positions[0])
        end = np.array(positions[-1])
        dist = np.linalg.norm(end - start)
        print("Start: (%.2f, %.2f, %.2f)" % tuple(start))
        print("End:   (%.2f, %.2f, %.2f)" % tuple(end))
        print("Distance traveled: %.2f mm" % dist)

        # Heading toward odor source
        to_odor = np.array(odor_source[:2]) - start[:2]
        to_end = end[:2] - start[:2]
        if np.linalg.norm(to_odor) > 0 and np.linalg.norm(to_end) > 0:
            cos_angle = np.dot(to_odor, to_end) / (np.linalg.norm(to_odor) * np.linalg.norm(to_end))
            angle_to_odor = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            print("Angle between trajectory and odor: %.1f deg" % angle_to_odor)

            # Distance to odor source
            dist_to_odor_start = np.linalg.norm(np.array(odor_source[:2]) - start[:2])
            dist_to_odor_end = np.linalg.norm(np.array(odor_source[:2]) - end[:2])
            print("Distance to odor: %.2f -> %.2f mm (%s)" % (
                dist_to_odor_start, dist_to_odor_end,
                "APPROACHING" if dist_to_odor_end < dist_to_odor_start else "DIVERGING"))

    # Odor gradient analysis
    if episode_log:
        l_odors = [e["odor_left"] for e in episode_log]
        r_odors = [e["odor_right"] for e in episode_log]
        turns = [e["turn_drive"] for e in episode_log]
        print("\nOdor signal: L=%.4f+/-%.4f  R=%.4f+/-%.4f" % (
            np.mean(l_odors), np.std(l_odors), np.mean(r_odors), np.std(r_odors)))
        print("Mean turn drive: %+.4f" % np.mean(turns))

        # Correlation: does left>right odor correlate with turn direction?
        odor_diff = np.array(l_odors) - np.array(r_odors)
        if np.std(odor_diff) > 1e-6 and np.std(turns) > 1e-6:
            corr = np.corrcoef(odor_diff, turns)[0, 1]
            print("Odor L-R vs turn_drive correlation: %.3f" % corr)
        else:
            print("Odor L-R vs turn_drive: insufficient variance")

    # --- Validation ---
    print("\nVALIDATION:")
    if len(positions) >= 2:
        # Test 1: Did the fly move?
        moved = dist > 1.0
        print("  [%s] Movement: %.2f mm (>1mm)" % ("PASS" if moved else "FAIL", dist))

        # Test 2: Did it approach the odor?
        if 'dist_to_odor_end' in dir():
            approached = dist_to_odor_end < dist_to_odor_start
            print("  [%s] Approached odor: %.2f -> %.2f mm" % (
                "PASS" if approached else "FAIL", dist_to_odor_start, dist_to_odor_end))

        # Test 3: Was there odor asymmetry?
        if episode_log:
            had_asymmetry = np.std(odor_diff) > 1e-4
            print("  [%s] Odor asymmetry detected: std=%.6f" % (
                "PASS" if had_asymmetry else "FAIL", np.std(odor_diff)))

    # --- Save ---
    results = {
        "config": config_payload,
        "positions": positions,
        "odor_readings": odor_readings,
        "episode_log": episode_log,
    }
    _write_json_atomic(output_path / "chemotaxis_results.json", results)
    print("\nSaved to %s/chemotaxis_results.json" % output_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chemotaxis experiment")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--enable-vision", action="store_true")
    parser.add_argument("--odor-x", type=float, default=15.0)
    parser.add_argument("--odor-y", type=float, default=5.0)
    parser.add_argument("--odor-intensity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="logs/chemotaxis")
    args = parser.parse_args()

    run_chemotaxis(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        enable_vision=args.enable_vision,
        odor_source=(args.odor_x, args.odor_y, 0.0),
        odor_intensity=args.odor_intensity,
        seed=args.seed,
        output_dir=args.output_dir,
    )
