"""
Closed-loop brain-body walking experiment.

body state → sensory encoder → brain model → descending decoder
→ locomotion bridge (VNC) → FlyGym body

Usage:
    # First: generate neuron population files
    python scripts/select_populations.py

    # Then: run with fake brain (test loop without Brian2)
    python experiments/closed_loop_walk.py --fake-brain

    # Then: run with real brain
    python experiments/closed_loop_walk.py

    # Longer run
    python experiments/closed_loop_walk.py --body-steps 10000
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


def run_closed_loop(
    body_steps: int = 2000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    output_dir: str = "logs/closed_loop",
    seed: int = 42,
):
    import flygym

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load neuron populations ---
    if not cfg.sensory_ids_path.exists():
        print("Neuron population files not found. Run: python scripts/select_populations.py")
        return None

    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)
    print(f"Loaded populations: {len(sensory_ids)} sensory, {len(readout_ids)} readout")

    # --- Initialize components ---
    if cfg.channel_map_path.exists():
        encoder = SensoryEncoder.from_channel_map(
            sensory_ids, cfg.channel_map_path,
            max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
        )
        print(f"  Using channel-aware encoder (v2)")
    else:
        encoder = SensoryEncoder(sensory_ids, max_rate_hz=cfg.max_rate_hz)
        print(f"  Using flat encoder (v1 fallback)")
    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    print(f"\nInitializing brain ({brain_label})...")
    brain = create_brain_runner(
        sensory_ids=sensory_ids,
        readout_ids=readout_ids,
        use_fake=use_fake_brain,
        warmup_ms=cfg.brain_warmup_ms,
    )

    # --- Initialize FlyGym ---
    print("Initializing FlyGym...")
    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    arena = flygym.arena.FlatTerrain()
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=arena, timestep=1e-4)
    obs, info = sim.reset()

    # Warmup CPG
    print(f"Warming up locomotion bridge ({warmup_steps} steps)...")
    locomotion.warmup(0)
    locomotion.cpg.reset(init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
                         init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, terminated, truncated, info = sim.step(action)
            if terminated or truncated:
                print("  Episode ended during warmup!")
                sim.close()
                return None
        except Exception as e:
            print(f"  Physics error during warmup: {e}")
            sim.close()
            return None

    # --- Main closed loop ---
    bspb = cfg.body_steps_per_brain
    print(f"\nRunning closed loop: {body_steps} body steps, "
          f"brain every {bspb} steps ({cfg.brain_dt_ms}ms), {brain_label}")
    print()

    episode_log = []
    positions = []
    brain_steps = 0
    current_cmd = LocomotionCommand(forward_drive=1.0)

    t_start = time.time()
    t_brain = 0.0

    for step in range(body_steps):
        # --- Brain step ---
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)

            tb0 = time.time()
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            t_brain += time.time() - tb0

            current_cmd = decoder.decode(brain_output)
            brain_steps += 1

            mean_rate = float(np.mean(brain_output.firing_rates_hz))
            active = int(np.sum(brain_output.firing_rates_hz > 0))

            if brain_steps % 5 == 1:
                print(f"  brain #{brain_steps:3d}: "
                      f"fwd={current_cmd.forward_drive:+.3f} "
                      f"turn={current_cmd.turn_drive:+.3f} "
                      f"freq={current_cmd.step_frequency:.2f} "
                      f"stance={current_cmd.stance_gain:.2f} "
                      f"| rate={mean_rate:.0f}Hz active={active}/{len(readout_ids)}")

            episode_log.append({
                "body_step": step,
                "brain_step": brain_steps,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
                "step_frequency": current_cmd.step_frequency,
                "stance_gain": current_cmd.stance_gain,
                "readout_mean_hz": mean_rate,
                "readout_active": active,
            })

        # --- Body step ---
        action = locomotion.step(current_cmd)
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break

        if step % 50 == 0:
            positions.append(np.array(obs["fly"][0]).tolist())

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    sim.close()
    elapsed = time.time() - t_start

    # --- Results ---
    print(f"\nDone: {step+1} body steps, {brain_steps} brain steps in {elapsed:.1f}s")
    print(f"  Brain time: {t_brain:.1f}s ({t_brain/max(elapsed,0.01)*100:.0f}%)")

    if len(positions) >= 2:
        start, end = np.array(positions[0]), np.array(positions[-1])
        dist = np.linalg.norm(end - start)
        print(f"  Distance: {dist:.2f}mm | Final pos: ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})")

    # --- Validation ---
    print(f"\n{'='*50}")
    print("VALIDATION")
    print(f"{'='*50}")

    if len(episode_log) >= 2:
        drives = [e["forward_drive"] for e in episode_log]
        var = np.var(drives)
        print(f"1. Mechanical closure: drive variance = {var:.6f} "
              f"{'PASS' if var > 1e-6 else 'MARGINAL'}")

        if len(positions) >= 2:
            dx = np.array(positions[-1])[0] - np.array(positions[0])[0]
            mfwd = np.mean(drives)
            print(f"2. Monotonicity: mean_fwd={mfwd:.3f} dx={dx:.2f}mm "
                  f"{'PASS' if dx > 0 and mfwd > 0 else 'CHECK'}")

        print(f"3. Stability: {step+1}/{body_steps} steps "
              f"{'PASS' if step+1 >= body_steps * 0.9 else 'PARTIAL'}")

    # --- Save ---
    results = {
        "config": {
            "body_steps": body_steps, "brain_dt_ms": cfg.brain_dt_ms,
            "body_steps_per_brain": bspb, "use_fake_brain": use_fake_brain,
            "n_sensory": len(sensory_ids), "n_readout": len(readout_ids),
        },
        "summary": {
            "steps_completed": step + 1, "brain_steps": brain_steps,
            "elapsed_s": elapsed, "brain_time_s": t_brain,
        },
        "positions": positions,
        "episode_log": episode_log,
    }
    with open(output_path / "closed_loop_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}/closed_loop_results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop brain-body walking")
    parser.add_argument("--body-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true", help="Use fake brain (no Brian2)")
    parser.add_argument("--output-dir", default="logs/closed_loop")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_closed_loop(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        output_dir=args.output_dir,
        seed=args.seed,
    )
