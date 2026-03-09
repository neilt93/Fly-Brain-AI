"""
Multi-seed robustness study for the brain-body bridge.

Runs ablation conditions across multiple seeds and computes
confidence intervals on key causal metrics.

Usage:
    python experiments/robustness_study.py --fake-brain --seeds 5   # fast test
    python experiments/robustness_study.py --seeds 10               # real brain
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.ablation_study import run_single_condition, ABLATION_CONDITIONS
from bridge.config import BridgeConfig


# Focused conditions for robustness (skip redundant ones)
ROBUSTNESS_CONDITIONS = [
    "baseline",
    "ablate_forward",
    "ablate_turn_left",
    "ablate_turn_right",
    "ablate_rhythm",
    "ablate_stance",
]


def ci95(values):
    """Compute mean and 95% confidence interval."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return float(np.mean(arr)), 0.0
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean, 1.96 * se


def run_robustness_study(
    n_seeds: int = 10,
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    output_dir: str = "logs/robustness",
):
    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(cfg.decoder_groups_path) as f:
        decoder_groups = json.load(f)

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    seeds = list(range(42, 42 + n_seeds))

    print("=" * 70)
    print("ROBUSTNESS STUDY (%s, %d seeds, %d body steps)" % (brain_label, n_seeds, body_steps))
    print("Seeds: %s" % seeds)
    print("=" * 70)

    # Collect results: {condition: {metric: [values_per_seed]}}
    all_results = {cond: [] for cond in ROBUSTNESS_CONDITIONS}

    t_total = time.time()
    for si, seed in enumerate(seeds):
        print("\n--- Seed %d (%d/%d) ---" % (seed, si + 1, n_seeds))
        for cond_name in ROBUSTNESS_CONDITIONS:
            cond = ABLATION_CONDITIONS[cond_name]
            t0 = time.time()
            r = run_single_condition(
                condition_name=cond_name,
                condition=cond,
                body_steps=body_steps,
                warmup_steps=warmup_steps,
                use_fake_brain=use_fake_brain,
                seed=seed,
                decoder_groups=decoder_groups,
            )
            elapsed = time.time() - t0
            if "error" in r:
                print("  %s: ERROR %s" % (cond_name, r["error"]))
                continue
            all_results[cond_name].append(r)
            b = r["behavior"]
            print("  %s: fwd=%.2fmm path=%.2fmm sym=%.2f freq=%.0fHz (%.1fs)"
                  % (cond_name, b["forward_distance"], b["total_path_length"],
                     b["contact_symmetry"], b["step_frequency_hz"], elapsed))

    elapsed_total = time.time() - t_total
    print("\n\nTotal time: %.1fs" % elapsed_total)

    # === Extract key metrics per condition per seed ===
    def get_metrics(results_list):
        return {
            "forward_distance": [r["behavior"]["forward_distance"] for r in results_list],
            "total_path_length": [r["behavior"]["total_path_length"] for r in results_list],
            "final_heading": [r["behavior"]["final_heading"] for r in results_list],
            "contact_symmetry": [r["behavior"]["contact_symmetry"] for r in results_list],
            "step_frequency_hz": [r["behavior"]["step_frequency_hz"] for r in results_list],
            "fall_count": [r["behavior"]["fall_count"] for r in results_list],
            "contact_duty_cycle": [r["behavior"]["contact_duty_cycle"] for r in results_list],
            "mean_forward_drive": [np.mean([e["forward_drive"] for e in r["episode_log"]])
                                   for r in results_list],
            "mean_turn_drive": [np.mean([e["turn_drive"] for e in r["episode_log"]])
                                for r in results_list],
            "mean_step_frequency": [np.mean([e["step_frequency"] for e in r["episode_log"]])
                                    for r in results_list],
            "mean_stance_gain": [np.mean([e["stance_gain"] for e in r["episode_log"]])
                                 for r in results_list],
        }

    metrics = {cond: get_metrics(all_results[cond]) for cond in ROBUSTNESS_CONDITIONS
               if all_results[cond]}

    if "baseline" not in metrics:
        print("No baseline results!")
        return

    # === Summary table ===
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (mean +/- 95%% CI, n=%d seeds)" % n_seeds)
    print("=" * 70)

    key_metrics = [
        ("forward_distance", "FwdDist(mm)"),
        ("total_path_length", "PathLen(mm)"),
        ("mean_forward_drive", "FwdDrive"),
        ("mean_turn_drive", "TurnDrive"),
        ("mean_step_frequency", "StepFreq"),
        ("mean_stance_gain", "StanceGain"),
        ("contact_symmetry", "CntSym"),
    ]

    header = "%-20s" + " %16s" * len(key_metrics)
    print(header % tuple(["Condition"] + [m[1] for m in key_metrics]))
    print("-" * (20 + 17 * len(key_metrics)))

    for cond in ROBUSTNESS_CONDITIONS:
        if cond not in metrics:
            continue
        row = [cond]
        for mk, _ in key_metrics:
            vals = metrics[cond][mk]
            m, ci = ci95(vals)
            row.append("%+.3f +/- %.3f" % (m, ci))
        print("%-20s" % row[0] + " ".join("%16s" % v for v in row[1:]))

    # === Causal effect sizes with CIs ===
    print("\n" + "=" * 70)
    print("CAUSAL EFFECT SIZES (ablation - baseline)")
    print("=" * 70)

    effects = {}

    def effect_test(name, condition, metric, expect_negative=True):
        if condition not in metrics:
            print("  [SKIP] %s" % name)
            return
        base_vals = np.array(metrics["baseline"][metric])
        abl_vals = np.array(metrics[condition][metric])
        n = min(len(base_vals), len(abl_vals))
        diffs = abl_vals[:n] - base_vals[:n]
        m, ci = ci95(diffs)
        effects[name] = {"mean": m, "ci95": ci, "n": n}

        if expect_negative:
            sig = m + ci < 0  # entire CI below zero
        else:
            sig = m - ci > 0  # entire CI above zero

        label = "SIG" if sig else "n.s."
        print("  [%s] %s: %.3f +/- %.3f (n=%d)" % (label, name, m, ci, n))

    effect_test("fwd_ablation_distance",   "ablate_forward", "forward_distance", expect_negative=True)
    effect_test("fwd_ablation_path",       "ablate_forward", "total_path_length", expect_negative=True)
    effect_test("fwd_ablation_drive",      "ablate_forward", "mean_forward_drive", expect_negative=True)
    effect_test("rhythm_ablation_freq",    "ablate_rhythm",  "mean_step_frequency", expect_negative=True)
    effect_test("stance_ablation_gain",    "ablate_stance",  "mean_stance_gain", expect_negative=True)

    # Turn contrast: ablate_left - ablate_right turn_drive (should be negative)
    if "ablate_turn_left" in metrics and "ablate_turn_right" in metrics:
        left_vals = np.array(metrics["ablate_turn_left"]["mean_turn_drive"])
        right_vals = np.array(metrics["ablate_turn_right"]["mean_turn_drive"])
        n = min(len(left_vals), len(right_vals))
        diffs = left_vals[:n] - right_vals[:n]
        m, ci = ci95(diffs)
        effects["turn_contrast"] = {"mean": m, "ci95": ci, "n": n}
        sig = m + ci < 0  # left should be more negative than right
        print("  [%s] turn_contrast (L-R): %.3f +/- %.3f (n=%d)" % (
            "SIG" if sig else "n.s.", m, ci, n))

    # === Count significant effects ===
    # Significant if the CI doesn't cross zero (or CI is zero meaning no variance)
    n_sig = sum(1 for e in effects.values()
                if abs(e["mean"]) > e["ci95"])
    print("\nSignificant effects: %d/%d" % (n_sig, len(effects)))

    # === Save ===
    save_data = {
        "config": {
            "n_seeds": n_seeds, "seeds": seeds, "body_steps": body_steps,
            "use_fake_brain": use_fake_brain,
        },
        "metrics": {
            cond: {mk: vals for mk, vals in m.items()}
            for cond, m in metrics.items()
        },
        "effects": effects,
        "elapsed_s": elapsed_total,
    }
    with open(output_path / "robustness_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nSaved to %s/robustness_results.json" % output_path)

    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed robustness study")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--output-dir", default="logs/robustness")
    args = parser.parse_args()

    run_robustness_study(
        n_seeds=args.seeds,
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        output_dir=args.output_dir,
    )
