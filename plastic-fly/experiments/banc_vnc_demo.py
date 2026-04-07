#!/usr/bin/env python
"""
BANC VNC comprehensive demo: exercises all capabilities in one run.

Phases:
  1. Forward walking (5k steps)
  2. Left turn (3k steps)
  3. Right turn (3k steps)
  4. Looming escape left (2k steps)
  5. Forward recovery (2k steps)

Exports Unity-compatible timeseries and prints summary stats.

Usage:
    python experiments/banc_vnc_demo.py
    python experiments/banc_vnc_demo.py --body-steps 20000
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    for attempt in range(5):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(0.05 * (attempt + 1))
            else:
                import shutil
                shutil.copy2(str(tmp), str(path))
                tmp.unlink(missing_ok=True)


def run_demo(body_steps: int = 15000, output_dir: str = "logs/banc_vnc_demo"):
    import flygym
    from bridge.banc_loader import load_banc_vnc
    from bridge.vnc_firing_rate import FiringRateVNCConfig
    from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BANC VNC Comprehensive Demo")
    print("=" * 60)

    # Load BANC
    print("\nLoading BANC connectome...")
    t0 = time.time()
    data = load_banc_vnc(
        exc_mult=1.0, inh_mult=1.0, inh_scale=1.0,
        normalize_weights=False, verbose=False,
    )
    print(f"  {data.n_neurons} neurons, {data.n_synapses:,} synapses ({time.time()-t0:.1f}s)")

    # Build bridge
    cfg = FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
        use_adaptation=False, normalize_weights=False,
        use_delay=True, delay_inh_ms=3.0, param_cv=0.05, seed=42,
    )
    bridge = FiringRateVNCBridge.from_banc(banc_data=data, cfg=cfg, fallback_blend=0.3)
    bridge.warmup(warmup_ms=200.0)

    # FlyGym
    fly = flygym.Fly(enable_adhesion=True, draw_adhesion=False)
    sim = flygym.SingleFlySimulation(fly=fly, timestep=1e-4)
    obs, _ = sim.reset()

    # Phase schedule
    phase_frac = body_steps / 15000.0  # scale phases with total steps
    phases = [
        ("forward",       int(5000 * phase_frac),
         {"forward": 15.0, "turn_left": 0.0, "turn_right": 0.0,
          "rhythm": 10.0, "stance": 5.0, "escape_left": 0.0, "escape_right": 0.0}),
        ("turn_left",     int(3000 * phase_frac),
         {"forward": 15.0, "turn_left": 20.0, "turn_right": 0.0,
          "rhythm": 10.0, "stance": 5.0, "escape_left": 0.0, "escape_right": 0.0}),
        ("turn_right",    int(3000 * phase_frac),
         {"forward": 15.0, "turn_left": 0.0, "turn_right": 20.0,
          "rhythm": 10.0, "stance": 5.0, "escape_left": 0.0, "escape_right": 0.0}),
        ("escape_left",   int(2000 * phase_frac),
         {"forward": 15.0, "turn_left": 0.0, "turn_right": 0.0,
          "rhythm": 10.0, "stance": 5.0, "escape_left": 40.0, "escape_right": 0.0}),
        ("recovery",      int(2000 * phase_frac),
         {"forward": 15.0, "turn_left": 0.0, "turn_right": 0.0,
          "rhythm": 10.0, "stance": 5.0, "escape_left": 0.0, "escape_right": 0.0}),
    ]

    # Run
    print(f"\nRunning {body_steps} body steps across {len(phases)} phases...")
    positions = []
    joint_log = []
    phase_log = []
    vnc_activity_log = []
    leg_rate_log = []
    log_every = max(1, body_steps // 200)  # ~200 frames for Unity
    step_count = 0
    init_pos = obs["fly"][0, :3].copy()

    t_start = time.time()
    for phase_name, phase_steps, group_rates in phases:
        phase_start_pos = obs["fly"][0, :2].copy()
        print(f"  Phase: {phase_name} ({phase_steps} steps)...", end="", flush=True)

        for s in range(phase_steps):
            action = bridge.step(group_rates, dt_s=1e-4)
            try:
                obs, _, _, _, _ = sim.step({
                    "joints": action["joints"],
                    "adhesion": action.get("adhesion", np.ones(6)),
                })
            except (RuntimeError, ValueError):
                print(f" CRASH at step {s}")
                break

            if step_count % log_every == 0:
                positions.append(obs["fly"][0, :3].tolist())
                joint_log.append(action["joints"].tolist())
                phase_log.append(phase_name)
                # Log VNC neural activity (top 50 most active neurons)
                all_rates = bridge.vnc.get_all_rates()
                top_idx = np.argsort(all_rates)[-50:]
                vnc_activity_log.append({
                    "mn_mean": float(bridge.vnc.get_mn_rates().mean()),
                    "mn_max": float(bridge.vnc.get_mn_rates().max()),
                    "dn_mean": float(bridge.vnc.get_dn_rates().mean()),
                    "network_mean": float(all_rates.mean()),
                    "n_active": int((all_rates > 1.0).sum()),
                    "top50_rates": all_rates[top_idx].tolist(),
                    "top50_indices": top_idx.tolist(),
                })
                # Per-leg flex/ext rates
                leg_rates = {}
                for li, ln in enumerate(["LF", "LM", "LH", "RF", "RM", "RH"]):
                    f, e = bridge.vnc.get_flexor_extensor_rates(li)
                    leg_rates[ln] = {"flex": round(f, 1), "ext": round(e, 1)}
                leg_rate_log.append(leg_rates)
            step_count += 1

        phase_disp = obs["fly"][0, :2] - phase_start_pos
        phase_dist = float(np.linalg.norm(phase_disp))
        phase_heading = float(np.degrees(np.arctan2(phase_disp[1], phase_disp[0])))
        print(f" dist={phase_dist:.2f}mm heading={phase_heading:+.0f}deg")

    wall_time = time.time() - t_start
    total_disp = obs["fly"][0, :2] - init_pos[:2]
    total_dist = float(np.linalg.norm(total_disp))
    final_heading = float(np.degrees(np.arctan2(total_disp[1], total_disp[0])))

    print(f"\nDone in {wall_time:.1f}s ({step_count/wall_time:.0f} steps/s)")
    print(f"Total: {total_dist:.2f}mm, heading={final_heading:+.0f}deg")

    # Export Unity timeseries
    joint_names = [f"joint_{i}" for i in range(42)]
    try:
        joint_names = [f"{leg}_{dof}" for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
                       for dof in ["Coxa", "Coxa_roll", "Coxa_yaw", "Femur",
                                   "Femur_roll", "Tibia", "Tarsus1"]]
    except Exception:
        pass

    timeseries = {
        "controller": "banc_vnc_demo",
        "dt": log_every * 1e-4,
        "n_frames": len(positions),
        "positions": positions,
        "joint_angles": joint_log,
        "joint_names": joint_names,
        "phases": phase_log,
        "vnc_activity": vnc_activity_log,
        "leg_rates": leg_rate_log,
        "vnc_info": {
            "n_neurons": data.n_neurons,
            "n_dn": data.n_dn,
            "n_mn": data.n_mn,
            "n_premotor": data.n_premotor,
            "n_synapses": data.n_synapses,
            "source": "BANC (female Drosophila, Harvard Dataverse)",
        },
    }
    ts_path = output_path / "timeseries_banc_demo.json"
    _write_json_atomic(ts_path, timeseries)
    print(f"\nUnity timeseries: {ts_path} ({len(positions)} frames)")

    # Copy to Unity Resources
    unity_res = Path(__file__).resolve().parent.parent.parent / "FlyBrainViz" / "Assets" / "Resources"
    if unity_res.exists():
        dst = unity_res / "timeseries_plastic.json"
        _write_json_atomic(dst, timeseries)
        print(f"Copied to {dst}")

    # Gait analysis
    gait = bridge.compute_gait_score()
    print(f"Tripod gait score: {gait['tripod_score']:.3f} "
          f"({gait['n_active_legs']}/6 active legs)")

    # Save summary
    summary = {
        "body_steps": step_count,
        "wall_time_s": wall_time,
        "total_distance_mm": total_dist,
        "final_heading_deg": final_heading,
        "phases": [p[0] for p in phases],
        "n_frames": len(positions),
        "model": "BANC firing-rate VNC (Pugliese ODE)",
        "neurons": data.n_neurons,
        "synapses": data.n_synapses,
        "tripod_score": gait["tripod_score"],
        "n_active_legs": gait["n_active_legs"],
    }
    _write_json_atomic(output_path / "banc_demo_summary.json", summary)
    print(f"Summary: {output_path / 'banc_demo_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BANC VNC comprehensive demo")
    parser.add_argument("--body-steps", type=int, default=15000)
    parser.add_argument("--output-dir", default="logs/banc_vnc_demo")
    args = parser.parse_args()
    run_demo(body_steps=args.body_steps, output_dir=args.output_dir)
