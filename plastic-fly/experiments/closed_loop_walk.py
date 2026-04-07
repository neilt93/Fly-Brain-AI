"""
Closed-loop brain-body walking experiment.

body state → sensory encoder → brain model → descending decoder
→ motor layer → FlyGym body

Motor layer modes:
  1. CPG (default): DescendingDecoder → VNC-lite → LocomotionBridge (PreprogrammedSteps)
  2. VNC connectome: DescendingDecoder → VNCBridge (MANC Brian2 LIF → MN decoder)
     This replaces the CPG with a real connectome-constrained VNC.
  3. VNC firing rate: DescendingDecoder → FiringRateVNCBridge (Pugliese rate model)
     Rhythm emerges from network dynamics (no external CPG/sine).

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

    # VNC firing rate mode: real brain + Pugliese rate VNC (emergent rhythm)
    python experiments/closed_loop_walk.py --use-vnc-rate
    python experiments/closed_loop_walk.py --use-vnc-rate --fake-brain

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


def _write_json_atomic(path: Path, payload: dict):
    """Write JSON atomically so checkpoints stay readable if the run is interrupted."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    try:
        for _attempt in range(5):
            try:
                tmp_path.replace(path)
                break
            except PermissionError:
                if _attempt < 4:
                    import time; time.sleep(0.05 * (_attempt + 1))
                else:
                    import shutil; shutil.copy2(str(tmp_path), str(path))
                    tmp_path.unlink(missing_ok=True)
    except PermissionError:
        # Windows: target file may be locked; fall back to remove-then-rename
        try:
            os.remove(str(path))
        except FileNotFoundError:
            pass
        os.rename(str(tmp_path), str(path))


def _record_frame(
    obs: dict,
    positions: list,
    joint_angle_frames: list,
    contact_frames: list,
    contact_binary_frames: list,
    end_effector_frames: list,
):
    positions.append(np.array(obs.get("fly", np.zeros((1, 3))))[0].tolist())
    joint_angle_frames.append(np.array(obs.get("joints", np.zeros((1, 42))))[0].tolist())

    raw_cf = np.array(obs.get("contact_forces", np.zeros((30, 3))), dtype=np.float32)
    per_leg = []
    per_leg_binary = []
    for leg_i in range(6):
        leg_forces = raw_cf[leg_i * 5:(leg_i + 1) * 5]
        mag = float(np.linalg.norm(leg_forces))
        per_leg.append(mag)
        per_leg_binary.append(1.0 if mag > 0.1 else 0.0)
    contact_frames.append(per_leg)
    contact_binary_frames.append(per_leg_binary)

    end_effector_frames.append(np.array(obs.get("end_effectors", np.zeros((6, 3)))).tolist())


def _save_checkpoint(
    output_path: Path,
    config_payload: dict,
    steps_completed: int,
    brain_steps: int,
    elapsed_s: float,
    brain_time_s: float,
    positions: list,
    episode_log: list,
    joint_angle_frames: list,
    contact_frames: list,
    contact_binary_frames: list,
    end_effector_frames: list,
    joint_names: list[str],
    status: str,
):
    checkpoint = {
        "config": config_payload,
        "summary": {
            "steps_completed": steps_completed,
            "brain_steps": brain_steps,
            "elapsed_s": elapsed_s,
            "brain_time_s": brain_time_s,
            "status": status,
        },
        "positions": positions,
        "episode_log": episode_log,
        "joint_angles": joint_angle_frames,
        "joint_names": joint_names,
        "contacts": contact_binary_frames,
        "contact_forces": contact_frames,
        "end_effectors": end_effector_frames,
    }
    _write_json_atomic(output_path / "closed_loop_checkpoint.json", checkpoint)


def run_closed_loop(
    body_steps: int = 2000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    output_dir: str = "logs/closed_loop",
    seed: int = 42,
    use_vnc_lite: bool = True,
    motor_mode: str = "cpg",  # "cpg", "vnc", "vnc-fake", "vnc-rate"
    vnc_shuffle_seed: int | None = None,
    ablate_groups: list[str] | None = None,
    use_cpg: bool = False,
    connectome: str = "flywire",
    vnc_connectome: str | None = None,
):
    import flygym

    cfg = BridgeConfig(connectome=connectome)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load neuron populations ---
    if not cfg.sensory_ids_path.exists():
        print("Neuron population files not found. Run: python scripts/select_populations.py")
        return None

    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)
    print(f"Loaded populations: {len(sensory_ids)} sensory, {len(readout_ids)} readout")

    config_payload = {
        "body_steps": body_steps,
        "brain_dt_ms": cfg.brain_dt_ms,
        "body_steps_per_brain": cfg.body_steps_per_brain,
        "use_fake_brain": use_fake_brain,
        "motor_mode": motor_mode,
        "vnc_shuffle_seed": vnc_shuffle_seed,
        "ablate_groups": ablate_groups,
        "n_sensory": len(sensory_ids),
        "n_readout": len(readout_ids),
    }

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
    use_vnc_rate = motor_mode == "vnc-rate"
    vnc_bridge = None
    vnc_rate_bridge = None
    locomotion = None
    vnc_lite = None

    if use_vnc_rate:
        from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge
        _vnc_src = vnc_connectome or ("banc" if connectome == "banc" else "manc")
        if _vnc_src == "banc":
            vnc_rate_bridge = FiringRateVNCBridge.from_banc()
            motor_label = "VNC Firing Rate (BANC female, emergent rhythm)"
        else:
            vnc_rate_bridge = FiringRateVNCBridge()
            motor_label = "VNC Firing Rate (MANC male, emergent rhythm)"
        if ablate_groups:
            motor_label += f" ABLATE({','.join(ablate_groups)})"
        print(f"  Using firing rate VNC bridge ({motor_label})")
    elif use_vnc:
        from bridge.vnc_bridge import VNCBridge
        use_fake_vnc = (motor_mode == "vnc-fake")
        vnc_bridge = VNCBridge(use_fake_vnc=use_fake_vnc, shuffle_seed=vnc_shuffle_seed,
                               use_cpg=use_cpg)
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
        connectome=connectome,
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
    if use_vnc_rate:
        # VNC firing rate mode: initialize MN decoder, warmup VNC network,
        # then ramp physics.
        init_joints = np.array(obs["joints"][0], dtype=np.float64)
        vnc_rate_bridge.reset(init_angles=init_joints)
        vnc_rate_bridge.warmup(warmup_ms=200.0)
        print(f"Warming up VNC-rate + physics ({warmup_steps} steps, ramp from init pose)...")
        for i in range(warmup_steps):
            ramp = min(1.0, i / max(warmup_steps * 0.5, 1.0))
            neutral_rates = {"forward": 20.0 * ramp, "turn_left": 0.0,
                             "turn_right": 0.0, "rhythm": 10.0 * ramp,
                             "stance": 10.0 * ramp}
            action = vnc_rate_bridge.step(neutral_rates, dt_s=vnc_body_dt_s)
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
    elif use_vnc:
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
    steps_completed = 0
    last_completed_step = -1
    last_logged_step = -1
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

            if use_vnc_rate:
                # VNC firing rate mode: cache group rates (VNC steps at body freq)
                current_group_rates = group_rates
                current_cmd = LocomotionCommand(
                    forward_drive=float(np.tanh(group_rates["forward"] / cfg.rate_scale)),
                    turn_drive=float(np.tanh((group_rates["turn_left"] - group_rates["turn_right"]) / cfg.rate_scale)),
                    step_frequency=1.0,
                    stance_gain=1.0,
                )
            elif use_vnc:
                # VNC mode: step VNC at brain frequency, cache for body steps
                current_group_rates = group_rates
                vnc_bridge.step_brain(group_rates, sim_ms=cfg.brain_dt_ms,
                                     body_obs=body_obs)
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
                if use_vnc_rate or use_vnc:
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
        if use_vnc_rate:
            # VNC firing rate mode: VNC steps at body frequency (emergent rhythm)
            action = vnc_rate_bridge.step(current_group_rates, dt_s=vnc_body_dt_s)
        elif use_vnc:
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
        steps_completed = step + 1
        last_completed_step = step

        if step % log_interval == 0:
            _record_frame(
                obs,
                positions,
                joint_angle_frames,
                contact_frames,
                contact_binary_frames,
                end_effector_frames,
            )
            last_logged_step = step
            _save_checkpoint(
                output_path=output_path,
                config_payload=config_payload,
                steps_completed=step + 1,
                brain_steps=brain_steps,
                elapsed_s=time.time() - t_start,
                brain_time_s=t_brain,
                positions=positions,
                episode_log=episode_log,
                joint_angle_frames=joint_angle_frames,
                contact_frames=contact_frames,
                contact_binary_frames=contact_binary_frames,
                end_effector_frames=end_effector_frames,
                joint_names=list(fly_obj.actuated_joints) if hasattr(fly_obj, "actuated_joints") else [],
                status="running",
            )

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    sim.close()
    elapsed = time.time() - t_start
    if steps_completed > 0 and last_logged_step != last_completed_step:
        _record_frame(
            obs,
            positions,
            joint_angle_frames,
            contact_frames,
            contact_binary_frames,
            end_effector_frames,
        )

    # --- Results ---
    print(f"\nDone: {steps_completed} body steps, {brain_steps} brain steps in {elapsed:.1f}s")
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

        print(f"3. Stability: {steps_completed}/{body_steps} steps "
              f"{'PASS' if steps_completed >= body_steps * 0.9 else 'PARTIAL'}")

    # --- Save ---
    final_status = "completed" if steps_completed >= body_steps else "partial"
    results = {
        "config": config_payload,
        "summary": {
            "steps_completed": steps_completed, "brain_steps": brain_steps,
            "elapsed_s": elapsed, "brain_time_s": t_brain,
            "status": final_status,
        },
        "positions": positions,
        "episode_log": episode_log,
    }
    _write_json_atomic(output_path / "closed_loop_results.json", results)
    _save_checkpoint(
        output_path=output_path,
        config_payload=config_payload,
        steps_completed=steps_completed,
        brain_steps=brain_steps,
        elapsed_s=elapsed,
        brain_time_s=t_brain,
        positions=positions,
        episode_log=episode_log,
        joint_angle_frames=joint_angle_frames,
        contact_frames=contact_frames,
        contact_binary_frames=contact_binary_frames,
        end_effector_frames=end_effector_frames,
        joint_names=list(fly_obj.actuated_joints) if hasattr(fly_obj, "actuated_joints") else [],
        status=final_status,
    )
    print(f"\nSaved to {output_path}/closed_loop_results.json")

    # --- Export Unity timeseries ---
    if joint_angle_frames:
        n_frames = len(joint_angle_frames)
        dt = log_interval * 1e-4  # timestep * log_interval

        joint_names = list(fly_obj.actuated_joints) if hasattr(fly_obj, 'actuated_joints') else []

        if use_vnc_rate:
            controller_name = "vnc_rate"
        elif use_vnc:
            controller_name = "vnc_connectome"
        else:
            controller_name = "brain_driven"
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
        _write_json_atomic(unity_file, unity_ts)
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
    motor_group.add_argument("--use-vnc-rate", action="store_true",
                             help="Use Pugliese firing rate VNC (emergent rhythm, no external CPG)")
    parser.add_argument("--no-vnc-lite", action="store_true",
                        help="Disable VNC-lite in CPG mode (use raw decoder)")
    parser.add_argument("--vnc-shuffle", type=int, default=None, metavar="SEED",
                        help="Shuffle VNC connectivity (random seed)")
    parser.add_argument("--ablate", nargs="+", default=None, metavar="GROUP",
                        help="Zero out DN group rates (e.g. --ablate forward)")
    parser.add_argument("--cpg", action="store_true",
                        help="Use Pugliese CPG instead of sine rhythm (VNC modes only)")
    parser.add_argument("--connectome", choices=["flywire", "banc"], default="flywire",
                        help="Connectome dataset for brain (default: flywire)")
    parser.add_argument("--vnc-connectome", choices=["manc", "banc"], default=None,
                        help="VNC connectome for --use-vnc-rate (default: matches --connectome)")

    args = parser.parse_args()

    if args.use_vnc:
        motor_mode = "vnc"
    elif args.use_vnc_fake:
        motor_mode = "vnc-fake"
    elif args.use_vnc_rate:
        motor_mode = "vnc-rate"
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
        use_cpg=args.cpg,
        connectome=args.connectome,
        vnc_connectome=args.vnc_connectome,
    )
