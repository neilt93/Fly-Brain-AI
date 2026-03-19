"""
Reproduce known lesion/activation phenotypes from published optogenetic data.

Validates our model against published results, documents which phenotypes it
reproduces and which it honestly cannot (due to model limitations).

Targets:
  - DNa01 boost (Yang 2024): should increase forward locomotion. LIKELY PASS.
  - DNa02 silence: should reduce steering. Partially testable.
  - DNp01/Giant Fiber boost: should produce rapid acceleration. Partially testable.
  - MDN/DNp50: backward walking. NOT testable (forward_drive clamped to 0.1).

Each target: silence + boost (3x), 5000 steps, multiple seeds.

Usage:
    cd plastic-fly
    python experiments/published_phenotype_validation.py --fake-brain
    python experiments/published_phenotype_validation.py --seeds 5
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the per-neuron intervention from dn_phenotype_prediction
from experiments.dn_phenotype_prediction import (
    run_trial, resolve_dn_type_ids, find_group_assignment, _write_json_atomic,
)
from bridge.config import BridgeConfig

BASE = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# Published phenotype database
# ═══════════════════════════════════════════════════════════════════════════

PUBLISHED_PHENOTYPES = {
    "DNa01": {
        "source": "Yang 2024 / Cande 2018",
        "activation_phenotype": "Locomotor increase + ipsilateral steering",
        "silencing_phenotype": "Reduced locomotion and turning",
        "dn_aliases": ["DNae001", "DNa01"],
        "testable": True,
        "test_type": "both",  # "silence", "boost", or "both"
        "expected_boost": {
            "forward_distance_change": "increase",
            "description": "Boosting should increase forward locomotion",
        },
        "expected_silence": {
            "forward_distance_change": "decrease_or_unchanged",
            "turn_change": "decrease",
            "description": "Silencing should reduce steering drive",
        },
        "known_limitations": [],
    },
    "DNa02": {
        "source": "Cande 2018",
        "activation_phenotype": "Locomotor increase + steering",
        "silencing_phenotype": "Reduced steering",
        "dn_aliases": ["DNa02"],
        "testable": True,
        "test_type": "both",
        "expected_boost": {
            "forward_distance_change": "increase_or_unchanged",
            "description": "Boosting should increase locomotion or steering",
        },
        "expected_silence": {
            "turn_change": "decrease",
            "description": "Silencing should reduce steering",
        },
        "known_limitations": [
            "DNa02 is also linked to backward walking initiation, "
            "which our model cannot reproduce (forward_drive clamped to 0.1)."
        ],
    },
    "DNp01": {
        "source": "von Reyn 2014 / Cande 2018 (Giant Fiber)",
        "activation_phenotype": "Escape jump / rapid leg extension",
        "silencing_phenotype": "Reduced escape response",
        "dn_aliases": ["DNp01", "GF"],
        "testable": True,
        "test_type": "boost",
        "expected_boost": {
            "forward_distance_change": "increase",
            "description": "Giant Fiber activation should produce rapid forward motion "
                          "(jump mechanics not in FlyGym, but fast leg extension should push forward)",
        },
        "expected_silence": None,
        "known_limitations": [
            "FlyGym lacks jump mechanics — we can only test forward thrust, not actual escape jump.",
            "Giant fiber activates leg extensors for take-off; our model maps this through turning group.",
        ],
    },
    "DNg13": {
        "source": "Cande 2018",
        "activation_phenotype": "Steering / turning",
        "silencing_phenotype": "Reduced turning",
        "dn_aliases": ["DNg13"],
        "testable": True,
        "test_type": "both",
        "expected_boost": {
            "turn_change": "increase",
            "description": "Boosting should increase turning drive",
        },
        "expected_silence": {
            "turn_change": "decrease",
            "description": "Silencing should reduce turning",
        },
        "known_limitations": [],
    },
}

# Not testable — document as honest limitations
UNTESTABLE_PHENOTYPES = {
    "MDN/DNp50": {
        "source": "Bidaye 2014 / Cande 2018",
        "phenotype": "Backward walking",
        "reason": "MDN (DNp50) not in our readout pool. Also, forward_drive is clamped "
                  "to min 0.1 — our model cannot produce backward locomotion.",
    },
    "DNp09": {
        "source": "Cande 2018",
        "phenotype": "Running then freezing",
        "reason": "Freezing requires bistable dynamics not captured by our tonic decoder.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_published_validation(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    n_seeds: int = 3,
    readout_version: int = 5,
):
    print("=" * 80)
    print("PUBLISHED PHENOTYPE VALIDATION")
    print(f"  body_steps={body_steps}, seeds={n_seeds}, "
          f"brain={'FAKE' if use_fake_brain else 'Brian2 LIF'}")
    print("=" * 80)

    ann = pd.read_csv(BASE / "brain-model" / "flywire_annotations_matched.csv", low_memory=False)
    cfg = BridgeConfig()

    if readout_version == 5:
        dec_path = cfg.data_dir / "decoder_groups_v5_steering.json"
    elif readout_version == 4:
        dec_path = cfg.data_dir / "decoder_groups_v4_looming.json"
    else:
        dec_path = cfg.decoder_groups_path
    with open(dec_path) as f:
        decoder_groups = json.load(f)

    output_dir = Path("logs/published_phenotype_validation")
    seeds = list(range(42, 42 + n_seeds))

    # Resolve DN types
    print("\n--- DN TYPE RESOLUTION ---")
    dn_info = {}
    for dn_type, meta in PUBLISHED_PHENOTYPES.items():
        all_ids = []
        for alias in meta["dn_aliases"]:
            all_ids.extend(resolve_dn_type_ids(alias, ann))
        all_ids = list(set(all_ids))

        assignments = find_group_assignment(all_ids, decoder_groups)
        all_groups = set()
        for g in assignments.values():
            all_groups.update(g)

        in_readout = sum(1 for g in assignments.values() if g)

        dn_info[dn_type] = {
            "root_ids": all_ids,
            "groups": list(all_groups),
            "n_in_readout": in_readout,
        }
        print(f"  {dn_type}: {len(all_ids)} neurons, groups={list(all_groups)}, "
              f"in_readout={in_readout}/{len(all_ids)}")

    # Run experiments
    all_results = {}
    verdicts = {}

    for dn_type, meta in PUBLISHED_PHENOTYPES.items():
        info = dn_info[dn_type]
        root_ids = info["root_ids"]

        if not root_ids or info["n_in_readout"] == 0:
            print(f"\n  {dn_type}: SKIPPED (not in readout pool)")
            verdicts[dn_type] = "UNTESTABLE"
            continue

        print(f"\n{'='*60}")
        print(f"  {dn_type} ({meta['source']})")
        print(f"  Published: {meta['activation_phenotype']}")
        print(f"  Groups in our model: {info['groups']}")
        if meta["known_limitations"]:
            for lim in meta["known_limitations"]:
                print(f"  LIMITATION: {lim}")
        print(f"{'='*60}")

        conditions = ["baseline"]
        if meta["test_type"] in ("silence", "both"):
            conditions.append("silence")
        if meta["test_type"] in ("boost", "both"):
            conditions.append("boost")

        dn_results = {cond: [] for cond in conditions}

        for seed in seeds:
            for cond in conditions:
                label = f"{dn_type}_{cond}_s{seed}"
                print(f"    {label}...", end=" ", flush=True)

                r = run_trial(
                    target_ids=root_ids,
                    mode=cond,
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
                      f"|turn|={r['mean_abs_turn_drive']:.3f} "
                      f"heading={np.degrees(b['final_heading']):+.1f}deg "
                      f"({r['elapsed_s']:.1f}s)")

                dn_results[cond].append(r)
                _write_json_atomic(output_dir / "checkpoints" / f"{label}.json", r)

        all_results[dn_type] = dn_results

        # --- Evaluate ---
        baseline_trials = dn_results.get("baseline", [])
        if not baseline_trials:
            verdicts[dn_type] = "NO_DATA"
            continue

        bl_dist = np.mean([t["behavior"]["forward_distance"] for t in baseline_trials])
        bl_abs_turn = np.mean([t["mean_abs_turn_drive"] for t in baseline_trials])

        tests_passed = 0
        tests_total = 0

        # Boost tests
        if "boost" in dn_results and dn_results["boost"]:
            bo_dist = np.mean([t["behavior"]["forward_distance"] for t in dn_results["boost"]])
            bo_abs_turn = np.mean([t["mean_abs_turn_drive"] for t in dn_results["boost"]])

            expected = meta.get("expected_boost", {})
            if "forward_distance_change" in expected:
                tests_total += 1
                fwd_exp = expected["forward_distance_change"]
                if fwd_exp == "increase" and bo_dist > bl_dist:
                    tests_passed += 1
                    print(f"    [PASS] Boost fwd: {bl_dist:.2f} -> {bo_dist:.2f}mm (increase)")
                elif fwd_exp == "increase_or_unchanged" and bo_dist >= bl_dist * 0.9:
                    tests_passed += 1
                    print(f"    [PASS] Boost fwd: {bl_dist:.2f} -> {bo_dist:.2f}mm (increase/unchanged)")
                else:
                    print(f"    [FAIL] Boost fwd: {bl_dist:.2f} -> {bo_dist:.2f}mm "
                          f"(expected {fwd_exp})")

            if "turn_change" in expected:
                tests_total += 1
                turn_exp = expected["turn_change"]
                if turn_exp == "increase" and bo_abs_turn > bl_abs_turn:
                    tests_passed += 1
                    print(f"    [PASS] Boost |turn|: {bl_abs_turn:.4f} -> {bo_abs_turn:.4f} (increase)")
                else:
                    print(f"    [FAIL] Boost |turn|: {bl_abs_turn:.4f} -> {bo_abs_turn:.4f} "
                          f"(expected {turn_exp})")

        # Silence tests
        if "silence" in dn_results and dn_results["silence"]:
            sl_dist = np.mean([t["behavior"]["forward_distance"] for t in dn_results["silence"]])
            sl_abs_turn = np.mean([t["mean_abs_turn_drive"] for t in dn_results["silence"]])

            expected = meta.get("expected_silence", {})
            if expected and "forward_distance_change" in expected:
                tests_total += 1
                fwd_exp = expected["forward_distance_change"]
                if fwd_exp == "decrease" and sl_dist < bl_dist:
                    tests_passed += 1
                    print(f"    [PASS] Silence fwd: {bl_dist:.2f} -> {sl_dist:.2f}mm (decrease)")
                elif fwd_exp == "decrease_or_unchanged" and sl_dist <= bl_dist * 1.1:
                    tests_passed += 1
                    print(f"    [PASS] Silence fwd: {bl_dist:.2f} -> {sl_dist:.2f}mm (decrease/unchanged)")
                else:
                    print(f"    [FAIL] Silence fwd: {bl_dist:.2f} -> {sl_dist:.2f}mm "
                          f"(expected {fwd_exp})")

            if expected and "turn_change" in expected:
                tests_total += 1
                turn_exp = expected["turn_change"]
                if turn_exp == "decrease" and sl_abs_turn < bl_abs_turn:
                    tests_passed += 1
                    print(f"    [PASS] Silence |turn|: {bl_abs_turn:.4f} -> {sl_abs_turn:.4f} (decrease)")
                else:
                    print(f"    [FAIL] Silence |turn|: {bl_abs_turn:.4f} -> {sl_abs_turn:.4f} "
                          f"(expected {turn_exp})")

        if tests_total > 0:
            verdicts[dn_type] = f"{tests_passed}/{tests_total}"
        else:
            verdicts[dn_type] = "NO_TESTS"

    # ═══════════════════════════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'DN Type':<10s} {'Source':<25s} {'Verdict':<12s} {'Groups':<25s}")
    print("  " + "-" * 72)

    for dn_type in list(PUBLISHED_PHENOTYPES.keys()):
        meta = PUBLISHED_PHENOTYPES[dn_type]
        v = verdicts.get(dn_type, "?")
        groups = dn_info.get(dn_type, {}).get("groups", [])
        print(f"  {dn_type:<10s} {meta['source']:<25s} {v:<12s} {str(groups):<25s}")

    print(f"\n  HONEST LIMITATIONS (documented, not claimed):")
    for dn_type, info in UNTESTABLE_PHENOTYPES.items():
        print(f"    {dn_type}: {info['phenotype']}")
        print(f"      Reason: {info['reason']}")

    # Count pass/fail
    total_testable = 0
    total_passed = 0
    for v in verdicts.values():
        if "/" in str(v):
            p, t = v.split("/")
            total_passed += int(p)
            total_testable += int(t)

    print(f"\n  TOTAL: {total_passed}/{total_testable} tests passed")
    print(f"  Untestable phenotypes documented: {len(UNTESTABLE_PHENOTYPES)}")

    # Save
    save_payload = {
        "verdicts": verdicts,
        "total_passed": total_passed,
        "total_testable": total_testable,
        "untestable": {k: v for k, v in UNTESTABLE_PHENOTYPES.items()},
        "params": {
            "body_steps": body_steps,
            "n_seeds": n_seeds,
            "readout_version": readout_version,
            "use_fake_brain": use_fake_brain,
        },
    }
    _write_json_atomic(output_dir / "results.json", save_payload)
    print(f"\nSaved to {output_dir}/results.json")

    return verdicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Published phenotype validation")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--readout-version", type=int, default=5, choices=[1, 4, 5])
    args = parser.parse_args()

    run_published_validation(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        n_seeds=args.seeds,
        readout_version=args.readout_version,
    )
