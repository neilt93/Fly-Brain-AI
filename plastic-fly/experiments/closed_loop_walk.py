"""
Closed-loop brain-body walking experiment.

body state → sensory encoder → brain model → descending decoder
→ motor layer → FlyGym body

Motor layer modes:
  1. CPG (default): DescendingDecoder → VNC-lite → LocomotionBridge (PreprogrammedSteps)
  2. VNC connectome: DescendingDecoder → VNCBridge (MANC Brian2 LIF → MN decoder)
     This replaces the CPG with a real connectome-constrained VNC.

Usage:
    # First: generate neuron population files
    python scripts/select_populations.py

    # CPG mode (default): fake brain
    python experiments/closed_loop_walk.py --fake-brain

    # CPG mode: real brain
    python experiments/closed_loop_walk.py

    # VNC mode: fake brain + fake VNC (test loop)
    python experiments/closed_loop_walk.py --fake-brain --use-vnc-fake

    # VNC mode: real brain + fake VNC (test MN decoder)
    python experiments/closed_loop_walk.py --use-vnc-fake

    # VNC mode: real brain + real VNC (full connectome)
    python experiments/closed_loop_walk.py --use-vnc

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
from bridge.flygym_adapter import FlyGymAdapter


def run_closed_loop(
    body_steps: int = 2000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    output_dir: str = "logs/closed_loop",
    seed: int = 42,
    use_vnc_lite: bool = True,
    motor_mode: str = "cpg",  # "cpg", "vnc", "vnc-fake"
    vnc_shuffle_seed: int | None = None,
    ablate_groups: list[str] | None = None,
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
    adapter = FlyGymAdapter()

    # --- Motor layer setup ---
    use_vnc = motor_mode in ("vnc", "vnc-fake")
    vnc_bridge = None
    locomotion = None
    vnc_lite = None

    if use_vnc:
        from bridge.vnc_bridge import VNCBridge
        use_fake_vnc = (motor_mode == "vnc-fake")
        vnc_bridge = VNCBridge(use_fake_vnc=use_fake_vnc, shuffle_seed=vnc_shuffle_seed)
        motor_label = f"VNC ({'fake' if use_fake_vnc else 'MANC Brian2'})"
        if vnc_shuffle_seed is not None:
            motor_label += f" SHUFFLED(seed={vnc_shuffle_seed})"
        if ablate_groups:
            motor_label += f" ABLATE({','.join(ablate_groups)})"
        print(f"  Using VNC connectome bridge ({motor_label})")
    else:
        from bridge.locomotion_bridge import LocomotionBridge
        from bridge.vnc_lite import VNCLite
        locomotion = LocomotionBridge(seed=seed)
        vnc_lite = VNCLite() if use_vnc_lite else None
        motor_label = "CPG" + (" + VNC-lite" if vnc_lite else "")
        if vnc_lite:
            print(f"  Using VNC-lite motor layer")

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

    # VNC body stepping: step VNC at body frequency for smooth oscillation.
    vnc_body_dt_s = 1e-4  # 0.1ms per body step (matches FlyGym timestep)

    # --- Warmup ---
    if use_vnc:
        # VNC mode: initialize MN decoder with FlyGym init pose, then ramp.
        # The exponential smoothing transitions from init to VNC-driven angles.
        init_joints = np.array(obs["joints"][0], dtype=np.float64)
        vnc_bridge.reset(init_angles=init_joints)
        print(f"Warming up VNC + physics ({warmup_steps} steps, ramp from init pose)...")
        for i in range(warmup_steps):
            # Ramp forward drive from 0 to 20 during warmup
            ramp = min(1.0, i / max(warmup_steps * 0.5, 1.0))
            neutral_rates = {"forward": 20.0 * ramp, "turn_left": 0.0,
                             "turn_right": 0.0, "rhythm": 10.0 * ramp,
                             "stance": 10.0 * ramp}
            action = vnc_bridge.step(neutral_rates, dt_s=vnc_body_dt_s)
            try:
                obs, _, terminated, truncated, info = sim.step(action)
                if terminated or truncated:
                    print("  Episode ended during warmup!")
                    sim.close()
                    return None
            except Exception as e:
                print(f"  Physics error during warmup step {i}: {e}")
                sim.close()
                return None
    else:
        # CPG mode: standard CPG warmup
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
          f"brain every {bspb} steps ({cfg.brain_dt_ms}ms), {brain_label}, {motor_label}")
    print()

    episode_log = []
    positions = []
    joint_angle_frames = []
    contact_frames = []
    contact_binary_frames = []
    end_effector_frames = []
    brain_steps = 0
    current_cmd = LocomotionCommand(forward_drive=1.0)
    current_group_rates = {"forward": 20.0, "turn_left": 0.0, "turn_right": 0.0,
                           "rhythm": 10.0, "stance": 10.0}
    log_interval = 50  # record every 50 steps for Unity

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

            group_rates = decoder.get_group_rates(brain_output)

            # Apply ablation: zero out specified DN group rates
            if ablate_groups:
                for g in ablate_groups:
                    if g in group_rates:
                        group_rates[g] = 0.0

            if use_vnc:
                # VNC mode: step VNC at brain frequency, cache for body steps
                current_group_rates = group_rates
                vnc_bridge.step_brain(group_rates, sim_ms=cfg.brain_dt_ms)
                # Create a LocomotionCommand for logging (approximate from group rates)
                current_cmd = LocomotionCommand(
                    forward_drive=float(np.tanh(group_rates["forward"] / cfg.rate_scale)),
                    turn_drive=float(np.tanh((group_rates["turn_left"] - group_rates["turn_right"]) / cfg.rate_scale)),
                    step_frequency=1.0,
                    stance_gain=1.0,
                )
            elif vnc_lite:
                current_cmd = vnc_lite.step(group_rates, dt_s=cfg.brain_dt_ms / 1000.0, body_obs=body_obs)
            else:
                current_cmd = decoder.decode(brain_output)

            brain_steps += 1

            mean_rate = float(np.mean(brain_output.firing_rates_hz))
            active = int(np.sum(brain_output.firing_rates_hz > 0))

            if brain_steps % 5 == 1:
                if use_vnc:
                    print(f"  brain #{brain_steps:3d}: "
                          f"fwd_rate={group_rates['forward']:.0f}Hz "
                          f"turn_L={group_rates['turn_left']:.0f}Hz "
                          f"turn_R={group_rates['turn_right']:.0f}Hz "
                          f"| rate={mean_rate:.0f}Hz active={active}/{len(readout_ids)}")
                else:
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
                "motor_mode": motor_mode,
            })

        # --- Body step ---
        if use_vnc:
            # VNC mode: step VNC at body frequency for smooth oscillation
            action = vnc_bridge.step(current_group_rates, dt_s=vnc_body_dt_s)
        else:
            # CPG mode: step the CPG every body step
            action = locomotion.step(current_cmd)

        try:
            obs, reward, terminated, truncated, info = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break

        if step % log_interval == 0:
            positions.append(np.array(obs["fly"][0]).tolist())
            joint_angle_frames.append(np.array(obs["joints"][0]).tolist())
            # Reduce 30x3 contact forces to 6 per-leg magnitudes
            raw_cf = np.array(obs["contact_forces"])  # (30, 3)
            per_leg = []
            per_leg_binary = []
            for leg_i in range(6):
                leg_forces = raw_cf[leg_i*5:(leg_i+1)*5]  # 5 contact points per leg
                mag = float(np.linalg.norm(leg_forces))
                per_leg.append(mag)
                per_leg_binary.append(1.0 if mag > 0.1 else 0.0)
            contact_frames.append(per_leg)
            contact_binary_frames.append(per_leg_binary)
            end_effector_frames.append(np.array(obs["end_effectors"]).tolist())

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    sim.close()
    elapsed = time.time() - t_start

    # --- Results ---
    print(f"\nDone: {step+1} body steps, {brain_steps} brain steps in {elapsed:.1f}s")
    print(f"  Brain time: {t_brain:.1f}s ({t_brain/max(elapsed,0.01)*100:.0f}%)")
    print(f"  Motor mode: {motor_label}")

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
            "motor_mode": motor_mode,
            "vnc_shuffle_seed": vnc_shuffle_seed,
            "ablate_groups": ablate_groups,
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

    # --- Export Unity timeseries ---
    if joint_angle_frames:
        n_frames = len(joint_angle_frames)
        dt = log_interval * 1e-4  # timestep * log_interval

        joint_names = list(fly_obj.actuated_joints) if hasattr(fly_obj, 'actuated_joints') else []

        controller_name = f"vnc_connectome" if use_vnc else "brain_driven"
        unity_ts = {
            "controller": controller_name,
            "dt": dt,
            "n_frames": n_frames,
            "positions": positions[:n_frames],
            "contacts": contact_binary_frames[:n_frames],
            "contact_forces": contact_frames[:n_frames],
            "end_effectors": end_effector_frames[:n_frames],
            "joint_angles": joint_angle_frames[:n_frames],
            "joint_names": joint_names,
            "tripod_score": [0.0] * n_frames,
            "weight_drifts": [],
            "perturbation_idx": 0,
        }

        # Save to logs
        ts_name = f"timeseries_{controller_name}.json"
        unity_file = output_path / ts_name
        with open(unity_file, "w") as f:
            json.dump(unity_ts, f)
        print(f"Unity timeseries: {unity_file} ({n_frames} frames)")

        # Copy to Unity Resources
        unity_res = Path(__file__).resolve().parent.parent.parent / "FlyBrainViz" / "Assets" / "Resources"
        if unity_res.exists():
            import shutil
            dst = unity_res / "timeseries_plastic.json"
            shutil.copy2(unity_file, dst)
            print(f"Copied to {dst}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop brain-body walking")
    parser.add_argument("--body-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true", help="Use fake brain (no Brian2)")
    parser.add_argument("--output-dir", default="logs/closed_loop")
    parser.add_argument("--seed", type=int, default=42)

    # Motor layer mode
    motor_group = parser.add_mutually_exclusive_group()
    motor_group.add_argument("--use-vnc", action="store_true",
                             help="Use real MANC VNC connectome (replaces CPG)")
    motor_group.add_argument("--use-vnc-fake", action="store_true",
                             help="Use fake VNC (oscillatory MN patterns, no Brian2 VNC)")
    parser.add_argument("--no-vnc-lite", action="store_true",
                        help="Disable VNC-lite in CPG mode (use raw decoder)")
    parser.add_argument("--vnc-shuffle", type=int, default=None, metavar="SEED",
                        help="Shuffle VNC connectivity (random seed)")
    parser.add_argument("--ablate", nargs="+", default=None, metavar="GROUP",
                        help="Zero out DN group rates (e.g. --ablate forward)")

    args = parser.parse_args()

    if args.use_vnc:
        motor_mode = "vnc"
    elif args.use_vnc_fake:
        motor_mode = "vnc-fake"
    else:
        motor_mode = "cpg"

    run_closed_loop(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        output_dir=args.output_dir,
        seed=args.seed,
        use_vnc_lite=not args.no_vnc_lite,
        motor_mode=motor_mode,
        vnc_shuffle_seed=args.vnc_shuffle,
        ablate_groups=args.ablate,
    )
