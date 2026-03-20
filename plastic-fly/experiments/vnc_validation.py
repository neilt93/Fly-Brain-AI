"""
VNC connectome validation: multi-condition ablation + shuffle battery.

Runs the full pipeline (real brain + real VNC) across multiple conditions
and seeds, producing a summary table with confidence intervals.

Conditions:
  1. Intact (baseline)
  2. Forward DN ablation
  3. Turn-left DN ablation
  4. Turn-right DN ablation
  5. Shuffled VNC connectivity

Usage:
    python experiments/vnc_validation.py                    # quick (5k steps, 1 seed)
    python experiments/vnc_validation.py --seeds 3          # 3 seeds per condition
    python experiments/vnc_validation.py --body-steps 20000 # longer runs
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
                import time; time.sleep(0.05 * (_attempt + 1))
            else:
                import shutil; shutil.copy2(str(tmp_path), str(path))
                tmp_path.unlink(missing_ok=True)


def run_condition(name, body_steps, seed, ablate_groups=None, vnc_shuffle_seed=None,
                  use_fake_brain=False):
    """Run one condition and return results dict."""
    from experiments.closed_loop_walk import run_closed_loop

    tag = f"{name}_s{seed}"
    output_dir = f"logs/vnc_validation/{tag}"

    results = run_closed_loop(
        body_steps=body_steps,
        warmup_steps=500,
        use_fake_brain=use_fake_brain,
        output_dir=output_dir,
        seed=seed,
        use_vnc_lite=True,
        motor_mode="vnc",
        vnc_shuffle_seed=vnc_shuffle_seed,
        ablate_groups=ablate_groups,
    )

    if results is None or len(results.get("positions", [])) < 2:
        return {"name": name, "seed": seed, "dist": 0.0, "dx": 0.0, "dy": 0.0,
                "heading": 0.0, "fwd_eff": 0.0, "steps": 0}

    pos = results["positions"]
    start = np.array(pos[0])
    end = np.array(pos[-1])
    diff = end - start
    dist = float(np.linalg.norm(diff))
    dx, dy = float(diff[0]), float(diff[1])
    heading = float(np.degrees(np.arctan2(dy, dx)))
    fwd_eff = dx / max(dist, 0.001)
    n_steps = results["summary"]["steps_completed"]

    return {
        "name": name, "seed": seed,
        "dist": dist, "dx": dx, "dy": dy,
        "heading": heading, "fwd_eff": fwd_eff,
        "steps": n_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="VNC validation battery")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds per condition")
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    seed_list = [args.base_seed + i * 37 for i in range(args.seeds)]

    conditions = [
        ("intact",      {"ablate_groups": None, "vnc_shuffle_seed": None}),
        ("fwd_ablated", {"ablate_groups": ["forward"], "vnc_shuffle_seed": None}),
        ("turnL_ablated", {"ablate_groups": ["turn_left"], "vnc_shuffle_seed": None}),
        ("turnR_ablated", {"ablate_groups": ["turn_right"], "vnc_shuffle_seed": None}),
        ("shuffled",    {"ablate_groups": None, "vnc_shuffle_seed": None}),  # shuffle seed set per run
    ]

    all_results = []
    t0 = time.time()

    for cond_name, cond_kwargs in conditions:
        for seed in seed_list:
            print(f"\n{'='*60}")
            print(f"  {cond_name.upper()} | seed={seed} | {args.body_steps} steps")
            print(f"{'='*60}")

            kwargs = dict(cond_kwargs)
            if cond_name == "shuffled":
                kwargs["vnc_shuffle_seed"] = seed  # use seed as shuffle seed

            result = run_condition(
                name=cond_name,
                body_steps=args.body_steps,
                seed=seed,
                use_fake_brain=args.fake_brain,
                **kwargs,
            )
            all_results.append(result)
            print(f"  -> dist={result['dist']:.2f}mm dx={result['dx']:+.2f}mm "
                  f"heading={result['heading']:+.1f}° fwd_eff={result['fwd_eff']:.0%}")

    elapsed = time.time() - t0

    # --- Summary table ---
    print(f"\n\n{'='*70}")
    print(f"VNC VALIDATION SUMMARY ({args.body_steps} steps, {args.seeds} seed(s))")
    print(f"{'='*70}")
    print(f"{'Condition':<18} {'n':>2} {'Dist (mm)':>12} {'dx (mm)':>12} "
          f"{'heading':>10} {'fwd_eff':>8}")
    print("-" * 70)

    for cond_name, _ in conditions:
        cond_results = [r for r in all_results if r["name"] == cond_name]
        n = len(cond_results)
        dists = [r["dist"] for r in cond_results]
        dxs = [r["dx"] for r in cond_results]
        headings = [r["heading"] for r in cond_results]
        fwd_effs = [r["fwd_eff"] for r in cond_results]

        if n == 1:
            dist_str = f"{dists[0]:.2f}"
            dx_str = f"{dxs[0]:+.2f}"
            head_str = f"{headings[0]:+.1f}°"
            eff_str = f"{fwd_effs[0]:.0%}"
        else:
            dist_str = f"{np.mean(dists):.2f}±{np.std(dists):.2f}"
            dx_str = f"{np.mean(dxs):+.2f}±{np.std(dxs):.2f}"
            head_str = f"{np.mean(headings):+.1f}±{np.std(headings):.0f}°"
            eff_str = f"{np.mean(fwd_effs):.0%}±{np.std(fwd_effs):.0%}"

        print(f"{cond_name:<18} {n:>2} {dist_str:>12} {dx_str:>12} "
              f"{head_str:>10} {eff_str:>8}")

    # Ablation effect sizes
    intact_dists = [r["dist"] for r in all_results if r["name"] == "intact"]
    intact_mean = np.mean(intact_dists) if intact_dists else 1.0
    print(f"\n  Ablation effects (vs intact mean {intact_mean:.2f}mm):")
    for cond_name in ["fwd_ablated", "turnL_ablated", "turnR_ablated", "shuffled"]:
        cond_dists = [r["dist"] for r in all_results if r["name"] == cond_name]
        if cond_dists:
            cond_mean = np.mean(cond_dists)
            pct = (cond_mean - intact_mean) / max(intact_mean, 0.001) * 100
            print(f"    {cond_name}: {cond_mean:.2f}mm ({pct:+.0f}%)")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    out_path = Path("logs/vnc_validation/summary.json")
    _write_json_atomic(out_path, {"results": all_results, "args": vars(args)})
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
