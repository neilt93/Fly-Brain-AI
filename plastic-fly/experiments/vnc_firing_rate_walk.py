"""
Standalone test: firing rate VNC drives FlyGym body via connectome-emergent rhythm.

Builds a FiringRateVNCRunner with the best config, stimulates DNg100 (forward
walking command), and steps both VNC + FlyGym body for 5000 steps. Logs
trajectory, MN traces, and anti-phase quality per leg.

Two conditions:
  1. Intact: DNg100 stimulated at 60 Hz + DN baseline at 25 Hz
  2. Forward-ablated: DNg100 silenced, baseline only (control)

Outputs:
  - logs/vnc_rate_walk/results.json  (positions, distances, MN stats)
  - logs/vnc_rate_walk/trajectory.png  (XY trajectory + heading)
  - logs/vnc_rate_walk/mn_traces.png  (per-leg flex/ext traces)

Usage:
    python experiments/vnc_firing_rate_walk.py
    python experiments/vnc_firing_rate_walk.py --body-steps 10000
    python experiments/vnc_firing_rate_walk.py --no-fallback   # pure network rhythm
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.vnc_firing_rate import FiringRateVNCConfig, LEG_ORDER
from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    for _attempt in range(5):
        try:
            tmp_path.replace(path)
            break
        except PermissionError:
            if _attempt < 4:
                time.sleep(0.05 * (_attempt + 1))
            else:
                import shutil
                shutil.copy2(str(tmp_path), str(path))
                tmp_path.unlink(missing_ok=True)


def run_condition(
    name: str,
    body_steps: int,
    warmup_steps: int,
    dng100_rate: float,
    dn_baseline: float,
    silence_dng100: bool,
    fallback_blend: float,
    output_dir: Path,
):
    """Run one condition (intact or ablated) and return results."""
    import flygym

    print(f"\n{'='*60}")
    print(f"CONDITION: {name}")
    print(f"  DNg100 rate: {dng100_rate} Hz, baseline: {dn_baseline} Hz")
    print(f"  Silence DNg100: {silence_dng100}")
    print(f"  Fallback blend: {fallback_blend}")
    print(f"{'='*60}\n")

    # Build bridge
    cfg = FiringRateVNCConfig(
        segments=("T1", "T2", "T3"),
        min_mn_synapses=3,
    )
    bridge = FiringRateVNCBridge(
        cfg=cfg,
        dn_baseline_hz=dn_baseline,
        substeps_per_body=5,
        fallback_blend=fallback_blend,
        fallback_freq_hz=10.0,
        mn_rate_scale=35.0,
        mn_alpha=0.3,
    )

    # Initialize FlyGym
    print("Initializing FlyGym...")
    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    arena = flygym.arena.FlatTerrain()
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=arena, timestep=1e-4)
    obs, info = sim.reset()

    # Initialize decoder with current pose
    init_joints = np.array(obs["joints"][0], dtype=np.float64)
    bridge.reset(init_angles=init_joints)

    # Warmup the VNC network (settle transients)
    bridge.warmup(warmup_ms=200.0)

    # If ablated, silence DNg100 after warmup
    if silence_dng100:
        bridge.vnc.silence_dn_type("DNg100")
        print("  DNg100 SILENCED")

    # Group rates that map to DN stimulation
    # DNg100 is the primary forward command neuron
    group_rates = {
        "forward": dng100_rate if not silence_dng100 else 0.0,
        "turn_left": 0.0,
        "turn_right": 0.0,
        "rhythm": 10.0,
        "stance": 5.0,
    }

    # Body step frequency
    dt_s = 1e-4  # 0.1ms = FlyGym timestep

    # Data storage
    positions = []
    flex_traces = {leg: [] for leg in LEG_ORDER}
    ext_traces = {leg: [] for leg in LEG_ORDER}
    mean_mn_rates = []

    # Record interval (every 10 steps = every 1ms)
    record_every = 10

    # Warmup physics (ramp from init pose)
    print(f"Warming up physics ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        ramp = min(1.0, i / max(warmup_steps * 0.5, 1.0))
        ramp_rates = {
            "forward": group_rates["forward"] * ramp,
            "turn_left": 0.0,
            "turn_right": 0.0,
            "rhythm": group_rates["rhythm"] * ramp,
            "stance": group_rates["stance"] * ramp,
        }
        action = bridge.step(ramp_rates, dt_s=dt_s)
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

    # Main loop
    print(f"\nRunning {body_steps} body steps...")
    t_start = time.time()
    steps_completed = 0

    for step in range(body_steps):
        action = bridge.step(group_rates, dt_s=dt_s)

        try:
            obs, _, terminated, truncated, info = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break

        steps_completed = step + 1

        # Record data
        if step % record_every == 0:
            pos = np.array(obs.get("fly", np.zeros((1, 3))))[0]
            positions.append(pos.tolist())

            for leg_idx, leg_name in enumerate(LEG_ORDER):
                flex_rate, ext_rate = bridge.vnc.get_flexor_extensor_rates(leg_idx)
                flex_traces[leg_name].append(flex_rate)
                ext_traces[leg_name].append(ext_rate)

            mean_mn_rates.append(float(bridge.vnc.get_mn_rates().mean()))

        # Log every 500ms (5000 steps)
        if step % 5000 == 0 and step > 0:
            elapsed = time.time() - t_start
            rate = step / max(elapsed, 0.01)
            pos = np.array(obs.get("fly", np.zeros((1, 3))))[0]
            dist = np.linalg.norm(pos - np.array(positions[0]))
            print(f"  step {step:6d}: pos=({pos[0]:.2f}, {pos[1]:.2f}) "
                  f"dist={dist:.2f}mm ({rate:.0f} steps/s)")

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    sim.close()
    elapsed = time.time() - t_start
    print(f"Done: {steps_completed} steps in {elapsed:.1f}s "
          f"({steps_completed/max(elapsed,0.01):.0f} steps/s)")

    # Compute results
    if len(positions) < 2:
        print("  WARNING: too few position samples")
        return None

    start = np.array(positions[0])
    end = np.array(positions[-1])
    diff = end - start
    dist = float(np.linalg.norm(diff))
    dx = float(diff[0])  # forward displacement
    heading = float(np.degrees(np.arctan2(diff[1], diff[0])))
    fwd_eff = dx / max(dist, 0.001)

    print(f"\n  {name} RESULTS:")
    print(f"    Distance: {dist:.3f}mm")
    print(f"    Forward dx: {dx:.3f}mm")
    print(f"    Heading: {heading:.1f} deg")
    print(f"    Forward efficiency: {fwd_eff:.2f}")
    print(f"    Mean MN rate: {np.mean(mean_mn_rates):.1f} Hz")
    print(f"    Anti-phase quality: {bridge._antiphase_quality}")
    print(f"    VNC time: {bridge.vnc.current_time_ms:.0f}ms")

    # Save checkpoint
    results = {
        "name": name,
        "body_steps": body_steps,
        "steps_completed": steps_completed,
        "elapsed_s": elapsed,
        "distance_mm": dist,
        "dx_mm": dx,
        "heading_deg": heading,
        "forward_efficiency": fwd_eff,
        "mean_mn_rate_hz": float(np.mean(mean_mn_rates)),
        "antiphase_quality": bridge._antiphase_quality.tolist(),
        "positions": positions,
        "config": {
            "dng100_rate": dng100_rate,
            "dn_baseline": dn_baseline,
            "silence_dng100": silence_dng100,
            "fallback_blend": fallback_blend,
        },
    }

    cond_dir = output_dir / name
    cond_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cond_dir / "results.json", results)

    # Add trace data for plotting (not in main JSON to keep it small)
    results["_flex_traces"] = flex_traces
    results["_ext_traces"] = ext_traces
    results["_mean_mn_rates"] = mean_mn_rates

    return results


def plot_results(results_list: list, output_dir: Path):
    """Generate trajectory and MN trace figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # --- Trajectory plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for res in results_list:
        if res is None:
            continue
        pos = np.array(res["positions"])
        label = res["name"]
        color = "tab:blue" if "intact" in label.lower() else "tab:red"
        axes[0].plot(pos[:, 0], pos[:, 1], label=label, color=color, linewidth=1.5)
        axes[0].plot(pos[0, 0], pos[0, 1], "o", color=color, markersize=8)
        axes[0].plot(pos[-1, 0], pos[-1, 1], "s", color=color, markersize=8)

    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[0].set_title("Walking Trajectory")
    axes[0].legend()
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # Distance over time
    for res in results_list:
        if res is None:
            continue
        pos = np.array(res["positions"])
        start = pos[0]
        dists = np.linalg.norm(pos - start, axis=1)
        t_ms = np.arange(len(dists)) * 1.0  # 1ms per record (record_every=10)
        label = res["name"]
        color = "tab:blue" if "intact" in label.lower() else "tab:red"
        axes[1].plot(t_ms, dists, label=label, color=color, linewidth=1.5)

    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Distance from start (mm)")
    axes[1].set_title("Distance Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot: {output_dir / 'trajectory.png'}")

    # --- MN traces plot (intact only) ---
    intact_res = next((r for r in results_list if r and "intact" in r["name"].lower()), None)
    if intact_res is None:
        return

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    flex_traces = intact_res["_flex_traces"]
    ext_traces = intact_res["_ext_traces"]
    n_samples = len(flex_traces[LEG_ORDER[0]])
    t_ms = np.arange(n_samples) * 1.0  # 1ms per sample

    for leg_idx, leg_name in enumerate(LEG_ORDER):
        ax = axes[leg_idx]
        flex = np.array(flex_traces[leg_name])
        ext = np.array(ext_traces[leg_name])

        ax.plot(t_ms, flex, color="tab:red", linewidth=0.8, alpha=0.8, label="Flexor")
        ax.plot(t_ms, ext, color="tab:blue", linewidth=0.8, alpha=0.8, label="Extensor")

        quality = intact_res["antiphase_quality"][leg_idx]
        ax.set_ylabel(f"{leg_name}\n(q={quality:.2f})")
        if leg_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (ms)")
    axes[0].set_title("Motor Neuron Rates: Flexor (red) vs Extensor (blue)")

    plt.tight_layout()
    fig.savefig(output_dir / "mn_traces.png", dpi=150)
    plt.close(fig)
    print(f"Saved MN traces: {output_dir / 'mn_traces.png'}")

    # --- Anti-phase quality bar chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    qualities = intact_res["antiphase_quality"]
    colors = ["tab:green" if q > 0.2 else "tab:orange" if q > 0.1 else "tab:red"
              for q in qualities]
    ax.bar(LEG_ORDER, qualities, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Anti-phase Quality")
    ax.set_title("Per-Leg Flexor/Extensor Alternation Quality")
    ax.axhline(y=0.2, color="gray", linestyle="--", linewidth=0.5, label="Threshold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "antiphase_quality.png", dpi=150)
    plt.close(fig)
    print(f"Saved anti-phase quality: {output_dir / 'antiphase_quality.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Firing rate VNC walking test: connectome-emergent rhythm drives FlyGym"
    )
    parser.add_argument("--body-steps", type=int, default=5000,
                        help="Body steps per condition (default: 5000)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Physics warmup steps (default: 500)")
    parser.add_argument("--dng100-rate", type=float, default=60.0,
                        help="DNg100 stimulation rate in Hz (default: 60)")
    parser.add_argument("--dn-baseline", type=float, default=25.0,
                        help="DN baseline rate in Hz (default: 25)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable fallback rhythm (pure network rhythm only)")
    parser.add_argument("--fallback-blend", type=float, default=0.3,
                        help="Fallback rhythm blend weight (default: 0.3)")
    parser.add_argument("--output-dir", default="logs/vnc_rate_walk",
                        help="Output directory")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip the ablation condition (faster)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback_blend = 0.0 if args.no_fallback else args.fallback_blend

    results_list = []

    # Condition 1: Intact (DNg100 stimulated)
    r = run_condition(
        name="intact",
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        dng100_rate=args.dng100_rate,
        dn_baseline=args.dn_baseline,
        silence_dng100=False,
        fallback_blend=fallback_blend,
        output_dir=output_dir,
    )
    results_list.append(r)

    # Condition 2: Forward ablated (DNg100 silenced)
    if not args.skip_ablation:
        r = run_condition(
            name="forward_ablated",
            body_steps=args.body_steps,
            warmup_steps=args.warmup_steps,
            dng100_rate=args.dng100_rate,
            dn_baseline=args.dn_baseline,
            silence_dng100=True,
            fallback_blend=fallback_blend,
            output_dir=output_dir,
        )
        results_list.append(r)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for res in results_list:
        if res is None:
            continue
        print(f"  {res['name']:20s}: dist={res['distance_mm']:.3f}mm  "
              f"dx={res['dx_mm']:.3f}mm  heading={res['heading_deg']:.1f}deg  "
              f"fwd_eff={res['forward_efficiency']:.2f}")

    if len(results_list) >= 2 and all(r is not None for r in results_list):
        intact = results_list[0]
        ablated = results_list[1]
        if intact["distance_mm"] > 0.01:
            reduction = 1.0 - ablated["distance_mm"] / intact["distance_mm"]
            print(f"\n  Forward ablation effect: {reduction*100:.1f}% distance reduction")
            if reduction > 0.3:
                print("  PASS: significant causal effect of DNg100 on locomotion")
            else:
                print("  NOTE: modest causal effect (network may have redundant forward pathways)")

    # Generate plots
    print()
    plot_results(results_list, output_dir)

    # Save combined summary
    summary = {
        "conditions": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results_list if r is not None
        ],
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
