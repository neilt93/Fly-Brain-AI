"""
Test: Compare decoder accuracy between v3 (without DNa01/DNg13) and
v5 (with DNa01/DNg13 steering DNs from Yang et al. 2024).

Uses synthetic brain outputs to measure:
1. Decoder group sizes
2. Turn group balance (L vs R symmetry)
3. Steering sensitivity when DNa01/DNg13 fire
4. Group assignment correctness
5. No cross-contamination
6. Isolated steering DN activation
7. Turn drive symmetry
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.descending_decoder import DescendingDecoder
from bridge.interfaces import BrainOutput

data_dir = Path(__file__).resolve().parent.parent / "data"

# DNa01 and DNg13 IDs
dna01_left = [720575940618167579, 720575940627787609]
dna01_right = [720575940644438551, 720575940626328707]
dng13_left = [720575940616471052]
dng13_right = [720575940606112940]
new_ids = set(dna01_left + dna01_right + dng13_left + dng13_right)


def main():
    with open(data_dir / "decoder_groups_v3.json") as f:
        v3_groups = json.load(f)
    with open(data_dir / "decoder_groups_v5_steering.json") as f:
        v5_groups = json.load(f)

    print("=" * 65)
    print("DECODER GROUPS COMPARISON: v3 vs v5 (with DNa01 + DNg13)")
    print("=" * 65)

    print()
    print("TEST 1: Group sizes")
    print("-" * 45)
    for group in ["forward_ids", "turn_left_ids", "turn_right_ids", "rhythm_ids", "stance_ids"]:
        n3 = len(set(v3_groups[group]))
        n5 = len(set(v5_groups[group]))
        delta = n5 - n3
        marker = " <-- CHANGED" if delta != 0 else ""
        print("  %-20s: v3=%3d  v5=%3d  delta=%+d%s" % (group, n3, n5, delta, marker))

    print()
    print("TEST 2: Turn group balance (L vs R)")
    print("-" * 45)
    for label, groups in [("v3", v3_groups), ("v5", v5_groups)]:
        nl = len(set(groups["turn_left_ids"]))
        nr = len(set(groups["turn_right_ids"]))
        ratio = nl / max(nr, 1)
        print("  %s: left=%d, right=%d, L/R ratio=%.3f" % (label, nl, nr, ratio))

    print()
    print("TEST 3: Steering sensitivity (synthetic brain output)")
    print("-" * 45)

    dec_v3 = DescendingDecoder.from_json(data_dir / "decoder_groups_v3.json")
    dec_v5 = DescendingDecoder.from_json(data_dir / "decoder_groups_v5_steering.json")

    all_v3_ids = set()
    all_v5_ids = set()
    for g in v3_groups.values():
        all_v3_ids.update(g)
    for g in v5_groups.values():
        all_v5_ids.update(g)
    all_ids = sorted(all_v3_ids | all_v5_ids)
    id_to_idx = {nid: i for i, nid in enumerate(all_ids)}

    rates_baseline = np.full(len(all_ids), 20.0, dtype=np.float32)
    bo_baseline = BrainOutput(
        neuron_ids=np.array(all_ids, dtype=np.int64),
        firing_rates_hz=rates_baseline,
    )
    cmd_v3_base = dec_v3.decode(bo_baseline)
    cmd_v5_base = dec_v5.decode(bo_baseline)
    print("  Baseline (all 20Hz):")
    print("    v3: fwd=%.4f turn=%.4f freq=%.4f" % (
        cmd_v3_base.forward_drive, cmd_v3_base.turn_drive, cmd_v3_base.step_frequency))
    print("    v5: fwd=%.4f turn=%.4f freq=%.4f" % (
        cmd_v5_base.forward_drive, cmd_v5_base.turn_drive, cmd_v5_base.step_frequency))

    rates_left_steer = np.full(len(all_ids), 20.0, dtype=np.float32)
    for fid in dna01_left + dng13_left:
        if fid in id_to_idx:
            rates_left_steer[id_to_idx[fid]] = 80.0

    bo_left = BrainOutput(
        neuron_ids=np.array(all_ids, dtype=np.int64),
        firing_rates_hz=rates_left_steer,
    )
    cmd_v3_left = dec_v3.decode(bo_left)
    cmd_v5_left = dec_v5.decode(bo_left)
    print()
    print("  Left steering (DNa01_L + DNg13_L at 80Hz):")
    print("    v3: fwd=%.4f turn=%.6f" % (cmd_v3_left.forward_drive, cmd_v3_left.turn_drive))
    print("    v5: fwd=%.4f turn=%.6f" % (cmd_v5_left.forward_drive, cmd_v5_left.turn_drive))
    delta_turn_v3 = cmd_v3_left.turn_drive - cmd_v3_base.turn_drive
    delta_turn_v5 = cmd_v5_left.turn_drive - cmd_v5_base.turn_drive
    print("    turn delta: v3=%.6f  v5=%.6f" % (delta_turn_v3, delta_turn_v5))

    rates_right_steer = np.full(len(all_ids), 20.0, dtype=np.float32)
    for fid in dna01_right + dng13_right:
        if fid in id_to_idx:
            rates_right_steer[id_to_idx[fid]] = 80.0

    bo_right = BrainOutput(
        neuron_ids=np.array(all_ids, dtype=np.int64),
        firing_rates_hz=rates_right_steer,
    )
    cmd_v3_right = dec_v3.decode(bo_right)
    cmd_v5_right = dec_v5.decode(bo_right)
    print()
    print("  Right steering (DNa01_R + DNg13_R at 80Hz):")
    print("    v3: fwd=%.4f turn=%.6f" % (cmd_v3_right.forward_drive, cmd_v3_right.turn_drive))
    print("    v5: fwd=%.4f turn=%.6f" % (cmd_v5_right.forward_drive, cmd_v5_right.turn_drive))
    delta_turn_v3r = cmd_v3_right.turn_drive - cmd_v3_base.turn_drive
    delta_turn_v5r = cmd_v5_right.turn_drive - cmd_v5_base.turn_drive
    print("    turn delta: v3=%.6f  v5=%.6f" % (delta_turn_v3r, delta_turn_v5r))

    print()
    print("TEST 4: Verify DNa01/DNg13 group assignments")
    print("-" * 45)
    checks = [
        ("DNa01_L1", 720575940618167579, "turn_left_ids"),
        ("DNa01_L2", 720575940627787609, "turn_left_ids"),
        ("DNa01_R1", 720575940644438551, "turn_right_ids"),
        ("DNa01_R2", 720575940626328707, "turn_right_ids"),
        ("DNg13_L",  720575940616471052, "turn_left_ids"),
        ("DNg13_R",  720575940606112940, "turn_right_ids"),
    ]
    test4_pass = True
    for name, fid, expected_group in checks:
        in_v3 = fid in set(v3_groups.get(expected_group, []))
        in_v5 = fid in set(v5_groups.get(expected_group, []))
        status = "PASS" if (not in_v3 and in_v5) else "FAIL"
        if status == "FAIL":
            test4_pass = False
        print("  %s (%d): v3=%s -> v5=%s  [%s]" % (name, fid, in_v3, in_v5, status))

    print()
    print("TEST 5: No cross-contamination (new IDs only in turn groups)")
    print("-" * 45)
    test5_pass = True
    for group in ["forward_ids", "rhythm_ids", "stance_ids"]:
        v5_set = set(v5_groups[group])
        found = new_ids & v5_set
        if found:
            print("  FAIL: %d steering DNs found in %s!" % (len(found), group))
            test5_pass = False
        else:
            print("  %s: clean (no steering DN contamination) [PASS]" % group)

    print()
    print("TEST 6: Isolated steering DN test (only new neurons fire)")
    print("-" * 45)

    rates_pure_left = np.zeros(len(all_ids), dtype=np.float32)
    for fid in dna01_left + dng13_left:
        if fid in id_to_idx:
            rates_pure_left[id_to_idx[fid]] = 100.0

    bo_pure_left = BrainOutput(
        neuron_ids=np.array(all_ids, dtype=np.int64),
        firing_rates_hz=rates_pure_left,
    )
    cmd_v3_pure = dec_v3.decode(bo_pure_left)
    cmd_v5_pure = dec_v5.decode(bo_pure_left)
    print("  Pure left steering (new neurons only at 100Hz):")
    print("    v3 turn_drive = %.6f (should be ~0, neurons not in v3)" % cmd_v3_pure.turn_drive)
    print("    v5 turn_drive = %.6f (should be positive/nonzero)" % cmd_v5_pure.turn_drive)
    test6a_pass = abs(cmd_v3_pure.turn_drive) < 0.001 and abs(cmd_v5_pure.turn_drive) > 0.001
    print("    [%s]" % ("PASS" if test6a_pass else "FAIL"))

    rates_pure_right = np.zeros(len(all_ids), dtype=np.float32)
    for fid in dna01_right + dng13_right:
        if fid in id_to_idx:
            rates_pure_right[id_to_idx[fid]] = 100.0

    bo_pure_right = BrainOutput(
        neuron_ids=np.array(all_ids, dtype=np.int64),
        firing_rates_hz=rates_pure_right,
    )
    cmd_v3_pure_r = dec_v3.decode(bo_pure_right)
    cmd_v5_pure_r = dec_v5.decode(bo_pure_right)
    print("  Pure right steering (new neurons only at 100Hz):")
    print("    v3 turn_drive = %.6f (should be ~0)" % cmd_v3_pure_r.turn_drive)
    print("    v5 turn_drive = %.6f (should be negative/nonzero)" % cmd_v5_pure_r.turn_drive)
    test6b_pass = abs(cmd_v3_pure_r.turn_drive) < 0.001 and abs(cmd_v5_pure_r.turn_drive) > 0.001
    print("    [%s]" % ("PASS" if test6b_pass else "FAIL"))

    print()
    print("TEST 7: Turn drive symmetry (L activation = -R activation)")
    print("-" * 45)
    asym = abs(cmd_v5_pure.turn_drive + cmd_v5_pure_r.turn_drive)
    print("  |turn_L + turn_R| = %.6f (ideal=0)" % asym)
    test7_pass = asym < 0.02
    print("  [%s]" % ("PASS" if test7_pass else ("MARGINAL" if asym < 0.05 else "FAIL")))

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    tests_passed = sum([test4_pass, test5_pass, test6a_pass, test6b_pass, test7_pass])
    total_tests = 5
    print("  Tests passed: %d/%d" % (tests_passed, total_tests))
    print()
    print("  v5 adds 6 neurons (3L + 3R) to turn groups:")
    print("    DNa01: 4 neurons (2L in turn_left, 2R in turn_right)")
    print("    DNg13: 2 neurons (1L in turn_left, 1R in turn_right)")
    print()
    if tests_passed == total_tests:
        print("  ALL TESTS PASS - v5 decoder groups ready for integration")
    else:
        print("  %d test(s) failed - investigate before integration" % (total_tests - tests_passed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test steering DN decoder groups")
    parser.add_argument("--connectome", choices=["flywire", "banc"], default="flywire",
                        help="Connectome dataset")
    args = parser.parse_args()
    main()
