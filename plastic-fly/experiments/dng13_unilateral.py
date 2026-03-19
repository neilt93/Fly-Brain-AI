"""
DNg13 unilateral silencing: explain WHY bilateral silencing INCREASES |turn|.

Hypothesis: removing both L and R creates asymmetric network effects (not
perfectly balanced due to stochastic brain dynamics), amplifying turning.

Conditions (4):
  1. baseline — no intervention
  2. silence_both — zero both L and R DNg13 neurons
  3. silence_L — zero left DNg13 only
  4. silence_R — zero right DNg13 only

Expected:
  - silence_L: turn_drive shifts positive (turns right, losing left-turning DN)
  - silence_R: turn_drive shifts negative (turns left, losing right-turning DN)
  - silence_both: |turn_drive| increases (residual network asymmetry amplified)

Usage:
    cd plastic-fly
    python experiments/dng13_unilateral.py                  # real brain
    python experiments/dng13_unilateral.py --fake-brain      # fast test
    python experiments/dng13_unilateral.py --seeds 5         # more seeds
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.dn_phenotype_prediction import (
    run_trial, apply_neuron_intervention, resolve_dn_type_ids, _write_json_atomic,
)

BASE = Path(__file__).resolve().parent.parent.parent


def run_dng13_unilateral(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    n_seeds: int = 3,
    readout_version: int = 5,
):
    print("=" * 80)
    print("DNg13 UNILATERAL SILENCING EXPERIMENT")
    print(f"  body_steps={body_steps}, seeds={n_seeds}, "
          f"brain={'FAKE' if use_fake_brain else 'Brian2 LIF'}")
    print("=" * 80)

    ann = pd.read_csv(BASE / "brain-model" / "flywire_annotations_matched.csv", low_memory=False)

    # Resolve DNg13 L and R neurons
    all_ids = resolve_dn_type_ids("DNg13", ann)
    left_ids = []
    right_ids = []
    for rid in all_ids:
        row = ann[ann["root_id"] == rid]
        side = str(row["side"].values[0]) if len(row) > 0 else "?"
        if side.startswith("l") or side == "L":
            left_ids.append(rid)
        elif side.startswith("r") or side == "R":
            right_ids.append(rid)
        else:
            # Unknown side — include in both to be safe
            left_ids.append(rid)
            right_ids.append(rid)

    print(f"\n  DNg13 neurons: {len(all_ids)} total")
    print(f"    Left:  {left_ids}")
    print(f"    Right: {right_ids}")

    output_dir = Path("logs/dng13_unilateral")
    seeds = list(range(42, 42 + n_seeds))

    conditions = {
        "baseline":     [],
        "silence_both": all_ids,
        "silence_L":    left_ids,
        "silence_R":    right_ids,
    }

    all_results = {cond: [] for cond in conditions}

    for cond_name, target_ids in conditions.items():
        mode = "baseline" if cond_name == "baseline" else "silence"
        for seed in seeds:
            label = f"DNg13_{cond_name}_s{seed}"
            print(f"  {label}...", end=" ", flush=True)

            r = run_trial(
                target_ids=target_ids,
                mode=mode,
                body_steps=body_steps,
                warmup_steps=warmup_steps,
                use_fake_brain=use_fake_brain,
                seed=seed,
                readout_version=readout_version,
            )

            if "error" in r:
                print(f"ERROR: {r['error']}")
                continue

            b = r["behavior"]
            print(f"fwd={b['forward_distance']:+.2f}mm "
                  f"heading={np.degrees(b['final_heading']):+.1f}deg "
                  f"turn={r['mean_turn_drive']:+.3f} "
                  f"|turn|={r['mean_abs_turn_drive']:.3f} "
                  f"({r['elapsed_s']:.1f}s)")

            _write_json_atomic(output_dir / "checkpoints" / f"{label}.json", r)
            all_results[cond_name].append(r)

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Condition':<15s} {'fwd_dist':>10s} {'heading':>10s} "
          f"{'turn_drive':>12s} {'|turn|':>10s}")
    print("  " + "-" * 60)

    means = {}
    for cond_name in conditions:
        trials = all_results[cond_name]
        if not trials:
            print(f"  {cond_name:<15s} {'NO DATA':>10s}")
            continue

        dists = [t["behavior"]["forward_distance"] for t in trials]
        headings = [np.degrees(t["behavior"]["final_heading"]) for t in trials]
        turns = [t["mean_turn_drive"] for t in trials]
        abs_turns = [t["mean_abs_turn_drive"] for t in trials]

        m = {
            "fwd_dist": np.mean(dists),
            "heading": np.mean(headings),
            "turn_drive": np.mean(turns),
            "abs_turn": np.mean(abs_turns),
        }
        means[cond_name] = m

        print(f"  {cond_name:<15s} {m['fwd_dist']:>+9.2f}mm {m['heading']:>+9.1f}deg "
              f"{m['turn_drive']:>+11.3f} {m['abs_turn']:>9.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Causal tests
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("CAUSAL TESTS")
    print(f"{'='*80}")

    tests_passed = 0
    tests_total = 0

    def log_test(name, passed, desc):
        nonlocal tests_passed, tests_total
        tests_total += 1
        tests_passed += int(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {desc}")

    bl = means.get("baseline")
    sb = means.get("silence_both")
    sl = means.get("silence_L")
    sr = means.get("silence_R")

    if bl and sl:
        # Test 1: L-silencing shifts turn_drive positive (right turn)
        shift_L = sl["turn_drive"] - bl["turn_drive"]
        log_test("L_silencing_rightward",
                 shift_L > 0,
                 f"L-silencing turn_drive shift: {shift_L:+.4f} (expect positive)")

    if bl and sr:
        # Test 2: R-silencing shifts turn_drive negative (left turn)
        shift_R = sr["turn_drive"] - bl["turn_drive"]
        log_test("R_silencing_leftward",
                 shift_R < 0,
                 f"R-silencing turn_drive shift: {shift_R:+.4f} (expect negative)")

    if bl and sl and sr:
        # Test 3: L and R produce opposite-sign shifts
        shift_L = sl["turn_drive"] - bl["turn_drive"]
        shift_R = sr["turn_drive"] - bl["turn_drive"]
        log_test("opposite_shifts",
                 shift_L * shift_R < 0,
                 f"L shift={shift_L:+.4f}, R shift={shift_R:+.4f} (should be opposite)")

    if bl and sb:
        # Test 4: Bilateral silencing increases |turn|
        abs_turn_increase = sb["abs_turn"] - bl["abs_turn"]
        log_test("bilateral_increases_abs_turn",
                 abs_turn_increase > 0,
                 f"|turn| change: {abs_turn_increase:+.4f} (expect positive)")

    print(f"\n  TOTAL: {tests_passed}/{tests_total} tests passed")

    # ═══════════════════════════════════════════════════════════════════════
    # Interpretation
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("MECHANISTIC INTERPRETATION")
    print(f"{'='*80}")

    if sl and sr and bl and sb:
        shift_L = sl['turn_drive'] - bl['turn_drive']
        shift_R = sr['turn_drive'] - bl['turn_drive']
        abs_change = sb['abs_turn'] - bl['abs_turn']
        print(f"\n  DNg13 is a bilateral turning DN pair (turn_left/turn_right groups).")
        print(f"\n  Unilateral silencing:")
        print(f"    - L-silencing: turn shift {shift_L:+.4f}")
        print(f"    - R-silencing: turn shift {shift_R:+.4f}")
        if shift_L * shift_R < 0:
            print(f"    => Confirms DNg13 L/R have OPPOSING steering effects.")
        print(f"\n  Bilateral silencing:")
        print(f"    - |turn| change: {abs_change:+.4f}")
        if abs_change > 0:
            print(f"    => PARADOXICAL INCREASE: removing both steering neurons amplifies turning")
            print(f"       because residual network asymmetry (stochastic Brian2 dynamics) is")
            print(f"       no longer damped by the balanced DNg13 pair.")
        else:
            print(f"    => Expected decrease in turning magnitude")

    # Save
    save_payload = {
        "means": means,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "dng13_ids": {
            "all": [int(x) for x in all_ids],
            "left": [int(x) for x in left_ids],
            "right": [int(x) for x in right_ids],
        },
        "params": {
            "body_steps": body_steps,
            "n_seeds": n_seeds,
            "readout_version": readout_version,
            "use_fake_brain": use_fake_brain,
        },
    }
    _write_json_atomic(output_dir / "results.json", save_payload)
    print(f"Saved to {output_dir}/results.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNg13 unilateral silencing")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--readout-version", type=int, default=5, choices=[1, 4, 5])
    args = parser.parse_args()

    run_dng13_unilateral(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        n_seeds=args.seeds,
        readout_version=args.readout_version,
    )
