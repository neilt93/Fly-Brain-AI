"""
Blind test: Cande 2018 optogenetic phenotypes vs our decoder group assignments.

Cande et al. 2018 (eLife) activated individual DN types and recorded behavioral
phenotypes. We have decoder groups (forward, turn_left, turn_right, rhythm,
stance) that assign DNs to motor functions. This test checks whether our
group assignments predict the correct behavioral phenotypes.

Three parts:
  A) Annotation mapping across ALL decoder versions (v1..v4): which Cande DNs
     are in each readout pool, and which group are they assigned to?
  B) Boost experiment (v1 decoder, fake brain): boost each decoder group
     individually, record all 4 command values, compare baseline vs boosted.
     This gives our model's behavioral predictions per group.
  C) Combined comparison: do our group assignments match Cande phenotypes?

Usage:
    cd plastic-fly
    python experiments/cande2018_blind_test.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

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

from bridge.config import BridgeConfig
from bridge.interfaces import LocomotionCommand, BrainOutput
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter


# ═══════════════════════════════════════════════════════════════════════════
# Cande 2018 phenotype database (from literature)
# ═══════════════════════════════════════════════════════════════════════════

CANDE_PHENOTYPES = {
    # DN type: (behavioral phenotype, expected decoder group(s))
    "DNa01": ("locomotor increase / steering", ["forward", "turn"]),
    "DNa02": ("locomotor increase / steering", ["forward", "turn"]),
    "DNp26": ("locomotor increase", ["forward"]),
    "DNg25": ("transient fast running", ["forward", "rhythm"]),
    "DNp09": ("running then freezing", ["forward", "rhythm"]),
    "DNg07": ("head grooming", ["stance"]),       # non-locomotor
    "DNg08": ("head grooming", ["stance"]),        # non-locomotor
    "DNg10": ("anterior reaching / steering", ["turn"]),
    "DNg13": ("steering", ["turn"]),
    "DNp29": ("abdomen grooming", ["stance"]),     # non-locomotor
    "DNd02": ("slow movement", ["forward", "rhythm"]),
    "DNp02": ("slow movement", ["forward", "rhythm"]),
    "DNp01": ("escape jump (giant fiber)", ["forward", "rhythm"]),
    "DNb01": ("anterior leg twitch then freeze", ["stance"]),
    "DNb05": ("steering (Yang 2024)", ["turn"]),
    "DNb06": ("contraversive steering (Yang 2024)", ["turn"]),
}

# FlyWire cell_type aliases for each Cande DN name
DN_ALIASES = {
    "DNa01": ["DNae001", "DNa01"],
    "DNa02": ["DNa02"],
    "DNp26": ["DNp26"],
    "DNg25": ["DNg25"],
    "DNp09": ["DNp09", "DNp71"],
    "DNg07": ["DNg07"],
    "DNg08": ["DNg08_a", "DNg08_b"],
    "DNg10": ["DNg10"],
    "DNg13": ["DNg13"],
    "DNp29": ["DNp29"],
    "DNd02": ["DNd02"],
    "DNp02": ["DNp02"],
    "DNp01": ["DNp01", "GF"],
    "DNb01": ["DNb01"],
    "DNb05": ["DNb05"],
    "DNb06": ["DNb06"],
}

DECODER_VERSIONS = {
    "v1 (47 neurons)": "decoder_groups.json",
    "v2 (204 neurons)": "decoder_groups_v2.json",
    "v3 (324 neurons)": "decoder_groups_v3.json",
    "v4_looming (350 neurons)": "decoder_groups_v4_looming.json",
}


def load_annotations():
    """Load FlyWire annotations and build lookup tables."""
    cfg = BridgeConfig()
    ann_path = cfg.brain_repo_root / "flywire_annotations_matched.csv"
    ann = pd.read_csv(ann_path, low_memory=False)

    type_to_ids = defaultdict(list)
    for _, row in ann[["root_id", "cell_type"]].dropna(subset=["cell_type"]).iterrows():
        type_to_ids[str(row["cell_type"])].append(int(row["root_id"]))

    hemi_to_ids = defaultdict(list)
    for _, row in ann[["root_id", "hemibrain_type"]].dropna(subset=["hemibrain_type"]).iterrows():
        hemi_to_ids[str(row["hemibrain_type"])].append(int(row["root_id"]))

    id_to_type = {}
    for _, row in ann[["root_id", "cell_type"]].dropna(subset=["cell_type"]).iterrows():
        id_to_type[int(row["root_id"])] = str(row["cell_type"])

    id_to_side = {}
    for _, row in ann[["root_id", "side"]].dropna(subset=["side"]).iterrows():
        id_to_side[int(row["root_id"])] = str(row["side"])

    return type_to_ids, hemi_to_ids, id_to_type, id_to_side


def resolve_dn_type(dn_name, type_to_ids, hemi_to_ids):
    """Resolve a Cande DN name to FlyWire root IDs."""
    ids = set()
    for alias in DN_ALIASES.get(dn_name, [dn_name]):
        ids.update(type_to_ids.get(alias, []))
        ids.update(hemi_to_ids.get(alias, []))
    return list(ids)


GROUP_MATCH = {
    "forward": {"forward"},
    "turn": {"turn_left", "turn_right"},
    "rhythm": {"rhythm"},
    "stance": {"stance"},
}


def check_match(our_groups, expected_groups):
    """Check if our group assignment overlaps with any expected group."""
    expected_expanded = set()
    for eg in expected_groups:
        expected_expanded.update(GROUP_MATCH.get(eg, set()))
    return bool(our_groups & expected_expanded)


# ═══════════════════════════════════════════════════════════════════════════
# PART A: Annotation mapping across all decoder versions
# ═══════════════════════════════════════════════════════════════════════════

def part_a_annotation_mapping():
    """Map Cande 2018 DN types to decoder groups across all versions."""
    print("\n" + "=" * 80)
    print("PART A: CANDE 2018 DN TYPE -> DECODER GROUP MAPPING (ALL VERSIONS)")
    print("=" * 80)

    type_to_ids, hemi_to_ids, id_to_type, id_to_side = load_annotations()
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Resolve all DN IDs once
    dn_flyids = {}
    for dn_name in CANDE_PHENOTYPES:
        dn_flyids[dn_name] = resolve_dn_type(dn_name, type_to_ids, hemi_to_ids)

    # Test each version
    best_version = None
    best_testable = 0
    best_results = None

    for version_name, filename in DECODER_VERSIONS.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\n  {version_name}: FILE NOT FOUND ({filename})")
            continue

        with open(filepath) as f:
            dg = json.load(f)

        all_readout = set()
        for gids in dg.values():
            all_readout.update(int(x) for x in gids)

        id_to_groups = defaultdict(set)
        for gname, gids in dg.items():
            clean = gname.replace("_ids", "")
            for nid in gids:
                id_to_groups[int(nid)].add(clean)

        print(f"\n  --- {version_name} ({filename}, {len(all_readout)} unique neurons) ---")
        print(f"  {'DN Type':<10} {'Phenotype':<32} {'FlyWire':<7} {'In Pool':<8} "
              f"{'Our Group(s)':<25} {'Expected':<18} {'Match'}")
        print("  " + "-" * 110)

        results = []
        for dn_name, (phenotype, expected_groups) in CANDE_PHENOTYPES.items():
            flyids = dn_flyids[dn_name]
            in_pool = [nid for nid in flyids if nid in all_readout]
            our_groups = set()
            for nid in in_pool:
                our_groups.update(id_to_groups.get(nid, set()))

            match = None
            if in_pool:
                match = check_match(our_groups, expected_groups)

            n_ids = len(flyids)
            pool_str = f"{len(in_pool)}/{n_ids}" if flyids else "0/0"
            groups_str = ",".join(sorted(our_groups)) if our_groups else "-"
            expected_str = ",".join(expected_groups)
            match_str = "YES" if match is True else ("NO" if match is False else "-")

            print(f"  {dn_name:<10} {phenotype:<32} {n_ids:<7} {pool_str:<8} "
                  f"{groups_str:<25} {expected_str:<18} {match_str}")

            results.append({
                "dn_type": dn_name, "phenotype": phenotype,
                "n_flyids": n_ids, "n_in_pool": len(in_pool),
                "in_pool": len(in_pool) > 0, "our_groups": list(our_groups),
                "expected": expected_groups, "match": match,
            })

        testable = [r for r in results if r["match"] is not None]
        correct = [r for r in testable if r["match"]]
        print(f"\n  Score: {len(correct)}/{len(testable)} correct "
              f"({len(testable)} testable, {len(results) - len(testable)} untestable)")

        if len(testable) > best_testable:
            best_testable = len(testable)
            best_version = version_name
            best_results = results

    print(f"\n  Best version for testing: {best_version} ({best_testable} testable)")
    return best_results, best_version


# ═══════════════════════════════════════════════════════════════════════════
# PART B: Boost experiments (using v1 decoder, the active bridge decoder)
# ═══════════════════════════════════════════════════════════════════════════

def run_boost_trial(
    boost_group: str | None,
    body_steps: int = 3000,
    warmup_steps: int = 500,
    boost_factor: float = 3.0,
    seed: int = 42,
):
    """Run closed-loop trial with one decoder group boosted (fake brain)."""
    import flygym

    cfg = BridgeConfig()
    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)

    encoder = SensoryEncoder.from_channel_map(
        sensory_ids, cfg.channel_map_path,
        max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    with open(cfg.decoder_groups_path) as f:
        decoder_groups = json.load(f)

    brain = create_brain_runner(
        sensory_ids=sensory_ids, readout_ids=readout_ids, use_fake=True,
    )

    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=flygym.arena.FlatTerrain(), timestep=1e-4)
    obs, _ = sim.reset()

    locomotion.warmup(0)
    locomotion.cpg.reset(init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
                         init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        obs, _, term, trunc, _ = sim.step(action)
        if term or trunc:
            sim.close()
            return None

    bspb = cfg.body_steps_per_brain
    current_cmd = LocomotionCommand(forward_drive=1.0)
    positions = []
    commands = []

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)

            if boost_group is not None:
                rates = brain_output.firing_rates_hz.copy()
                id_to_idx = {int(nid): i for i, nid in enumerate(brain_output.neuron_ids)}
                for nid in decoder_groups.get(boost_group, []):
                    if int(nid) in id_to_idx:
                        rates[id_to_idx[int(nid)]] *= boost_factor
                brain_output = BrainOutput(
                    neuron_ids=brain_output.neuron_ids, firing_rates_hz=rates,
                )

            current_cmd = decoder.decode(brain_output)
            commands.append({
                "step": step,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
                "step_frequency": current_cmd.step_frequency,
                "stance_gain": current_cmd.stance_gain,
            })

        action = locomotion.step(current_cmd)
        try:
            obs, _, term, trunc, _ = sim.step(action)
        except (RuntimeError, ValueError):  # MuJoCo physics instability
            break
        if step % 50 == 0:
            positions.append(np.array(obs["fly"][0]).tolist())
        if term or trunc:
            break

    sim.close()
    if len(positions) < 2:
        return None

    positions = np.array(positions)
    start, end = positions[0], positions[-1]
    forward_distance = float(np.linalg.norm(end - start))

    headings = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        headings.append(float(np.arctan2(dy, dx)))
    cumulative_turn = float(np.sum(np.abs(np.diff(headings)))) if len(headings) > 1 else 0.0

    fwd_drives = [c["forward_drive"] for c in commands]
    turn_drives = [c["turn_drive"] for c in commands]
    freq_drives = [c["step_frequency"] for c in commands]
    stance_drives = [c["stance_gain"] for c in commands]

    return {
        "forward_distance": forward_distance,
        "cumulative_turn": cumulative_turn,
        "mean_forward_drive": float(np.mean(fwd_drives)),
        "mean_turn_drive": float(np.mean(turn_drives)),
        "mean_abs_turn_drive": float(np.mean(np.abs(turn_drives))),
        "mean_step_freq": float(np.mean(freq_drives)),
        "mean_stance_gain": float(np.mean(stance_drives)),
        "n_commands": len(commands),
    }


def part_b_boost_experiments():
    """Boost each decoder group and record behavioral changes."""
    print("\n" + "=" * 80)
    print("PART B: DECODER GROUP BOOST EXPERIMENTS (fake brain, 3x boost)")
    print("=" * 80)

    groups_to_test = [
        None, "forward_ids", "turn_left_ids", "turn_right_ids",
        "rhythm_ids", "stance_ids",
    ]

    results = {}
    for group in groups_to_test:
        label = group.replace("_ids", "") if group else "baseline"
        print(f"  Running {label}...", end=" ", flush=True)
        t0 = time.time()
        r = run_boost_trial(boost_group=group, body_steps=3000, boost_factor=3.0)
        elapsed = time.time() - t0
        results[label] = r
        if r:
            print(f"dist={r['forward_distance']:.1f}mm "
                  f"fwd={r['mean_forward_drive']:.3f} "
                  f"turn={r['mean_turn_drive']:+.3f} "
                  f"freq={r['mean_step_freq']:.2f} "
                  f"stance={r['mean_stance_gain']:.2f} "
                  f"({elapsed:.1f}s)")
        else:
            print(f"FAILED ({elapsed:.1f}s)")

    bl = results.get("baseline")
    if not bl:
        print("  ERROR: baseline failed")
        return results

    # Delta table
    print(f"\n  {'Group':<15} {'d(Fwd)':<12} {'d(Turn)':<12} {'d(|Turn|)':<12} "
          f"{'d(Freq)':<12} {'d(Stance)':<12} {'d(Dist mm)':<12}")
    print("  " + "-" * 87)

    print(f"  {'baseline':<15} {bl['mean_forward_drive']:<12.3f} "
          f"{bl['mean_turn_drive']:<+12.3f} "
          f"{bl['mean_abs_turn_drive']:<12.3f} "
          f"{bl['mean_step_freq']:<12.2f} "
          f"{bl['mean_stance_gain']:<12.2f} "
          f"{bl['forward_distance']:<12.1f}")

    for group in groups_to_test[1:]:
        label = group.replace("_ids", "")
        r = results.get(label)
        if not r:
            print(f"  {label:<15} FAILED")
            continue
        d = {k: r[k] - bl[k] for k in [
            "mean_forward_drive", "mean_turn_drive", "mean_abs_turn_drive",
            "mean_step_freq", "mean_stance_gain", "forward_distance",
        ]}
        print(f"  {label:<15} {d['mean_forward_drive']:<+12.3f} "
              f"{d['mean_turn_drive']:<+12.3f} "
              f"{d['mean_abs_turn_drive']:<+12.3f} "
              f"{d['mean_step_freq']:<+12.2f} "
              f"{d['mean_stance_gain']:<+12.2f} "
              f"{d['forward_distance']:<+12.1f}")

    # Build per-group behavioral prediction
    print(f"\n  MODEL BEHAVIORAL PREDICTIONS (from boost experiment):")
    group_predictions = {}
    for group in ["forward", "turn_left", "turn_right", "rhythm", "stance"]:
        r = results.get(group)
        if not r:
            group_predictions[group] = "no data"
            continue
        d_fwd = r["mean_forward_drive"] - bl["mean_forward_drive"]
        d_turn = r["mean_turn_drive"] - bl["mean_turn_drive"]
        d_abs_turn = r["mean_abs_turn_drive"] - bl["mean_abs_turn_drive"]
        d_freq = r["mean_step_freq"] - bl["mean_step_freq"]
        d_stance = r["mean_stance_gain"] - bl["mean_stance_gain"]

        parts = []
        if abs(d_fwd) > 0.01:
            parts.append(f"fwd {'UP' if d_fwd > 0 else 'DOWN'} ({d_fwd:+.3f})")
        if abs(d_abs_turn) > 0.01:
            direction = "L" if d_turn > 0 else "R"
            parts.append(f"turn {direction} ({d_abs_turn:+.3f})")
        if abs(d_freq) > 0.05:
            parts.append(f"freq {'UP' if d_freq > 0 else 'DOWN'} ({d_freq:+.2f})")
        if abs(d_stance) > 0.02:
            parts.append(f"stance {'UP' if d_stance > 0 else 'DOWN'} ({d_stance:+.2f})")

        pred = "; ".join(parts) if parts else "no significant change"
        group_predictions[group] = pred
        print(f"    {group:<15} -> {pred}")

    return results, group_predictions


# ═══════════════════════════════════════════════════════════════════════════
# PART C: Final comparison table
# ═══════════════════════════════════════════════════════════════════════════

def part_c_comparison(annotation_results, boost_results, group_predictions, best_version):
    """Print final blind test comparison table."""
    print("\n" + "=" * 80)
    print("PART C: BLIND TEST COMPARISON TABLE")
    print(f"  Annotation source: {best_version}")
    print(f"  Boost source: v1 decoder (active bridge, 47 neurons)")
    print("=" * 80)

    bl = boost_results.get("baseline")
    if not bl:
        print("  ERROR: no baseline")
        return

    # Build group boost phenotype summaries
    boost_pheno = {}
    for group in ["forward", "turn_left", "turn_right", "rhythm", "stance"]:
        r = boost_results.get(group)
        if not r:
            boost_pheno[group] = "?"
            continue
        d_fwd = r["mean_forward_drive"] - bl["mean_forward_drive"]
        d_abs_turn = r["mean_abs_turn_drive"] - bl["mean_abs_turn_drive"]
        d_freq = r["mean_step_freq"] - bl["mean_step_freq"]
        d_stance = r["mean_stance_gain"] - bl["mean_stance_gain"]
        parts = []
        if abs(d_fwd) > 0.01: parts.append("locomotion" if d_fwd > 0 else "slow")
        if abs(d_abs_turn) > 0.01: parts.append("steering")
        if abs(d_freq) > 0.05: parts.append("fast" if d_freq > 0 else "slow")
        if abs(d_stance) > 0.02: parts.append("stance change")
        boost_pheno[group] = ", ".join(parts) if parts else "neutral"

    # Header
    print(f"\n  {'DN Type':<8} {'Cande 2018 Phenotype':<30} {'#IDs':<5} {'Pool':<5} "
          f"{'Our Group(s)':<22} {'Boost Phenotype':<25} {'Expected':<16} {'MATCH'}")
    print("  " + "-" * 130)

    verdicts = []
    for result in annotation_results:
        dn = result["dn_type"]
        phenotype = result["phenotype"]
        n_ids = result["n_flyids"]
        n_in = result["n_in_pool"]
        in_pool = result["in_pool"]
        our_groups = result["our_groups"]
        match = result["match"]

        groups_str = ",".join(sorted(our_groups)) if our_groups else "-"
        expected_str = ",".join(result["expected"])
        pool_str = f"{n_in}/{n_ids}" if n_ids > 0 else "0/0"

        if not in_pool:
            verdict = "UNTESTABLE"
            bp = "-"
        else:
            # Get boost phenotypes for all groups this DN is in
            bp_set = set()
            for g in our_groups:
                bp_set.add(boost_pheno.get(g, "?"))
            bp = " | ".join(sorted(bp_set))
            if len(bp) > 25:
                bp = bp[:22] + "..."
            verdict = "PASS" if match else "FAIL"

        print(f"  {dn:<8} {phenotype:<30} {n_ids:<5} {pool_str:<5} "
              f"{groups_str:<22} {bp:<25} {expected_str:<16} {verdict}")
        verdicts.append(verdict)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SCORE
    # ═══════════════════════════════════════════════════════════════════════
    testable = [v for v in verdicts if v in ("PASS", "FAIL")]
    passes = [v for v in testable if v == "PASS"]
    untestable = verdicts.count("UNTESTABLE")

    print(f"\n" + "=" * 80)
    print(f"  FINAL BLIND TEST SCORE")
    print(f"  " + "-" * 40)
    print(f"  Total Cande DN types:         {len(CANDE_PHENOTYPES)}")
    print(f"  Identified in FlyWire:        {sum(1 for r in annotation_results if r['n_flyids'] > 0)}")
    print(f"  Present in readout pool:      {sum(1 for r in annotation_results if r['in_pool'])}")
    print(f"  Testable:                     {len(testable)}")
    print(f"  PASS (correct assignment):    {len(passes)}")
    print(f"  FAIL (wrong assignment):      {len(testable) - len(passes)}")
    if testable:
        print(f"  ACCURACY:                     {len(passes)}/{len(testable)} "
              f"= {100*len(passes)/len(testable):.0f}%")
    print(f"  Untestable (not in pool):     {untestable}")
    print(f"=" * 80)

    # Detailed failure analysis
    failures = [(annotation_results[i], verdicts[i])
                for i in range(len(verdicts)) if verdicts[i] == "FAIL"]
    if failures:
        print(f"\n  FAILURE ANALYSIS:")
        for result, _ in failures:
            dn = result["dn_type"]
            our = ",".join(sorted(result["our_groups"]))
            expected = ",".join(result["expected"])
            print(f"    {dn}: assigned to [{our}], expected [{expected}]")
            print(f"      Cande phenotype: {result['phenotype']}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Cande 2018 Blind Test: Decoder Group vs Optogenetic Phenotypes")
    print("=" * 80)
    t0 = time.time()

    # Part A: annotation mapping across all versions
    annotation_results, best_version = part_a_annotation_mapping()

    # Part B: boost experiments with v1 (active bridge decoder)
    boost_results, group_predictions = part_b_boost_experiments()

    # Part C: combined comparison
    part_c_comparison(annotation_results, boost_results, group_predictions, best_version)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save
    output_dir = Path("logs/cande2018_blind_test")
    summary = {
        "annotation_results": annotation_results,
        "best_version": best_version,
        "boost_metrics": {k: v for k, v in boost_results.items() if v is not None},
        "group_predictions": group_predictions,
        "elapsed_s": elapsed,
    }
    _write_json_atomic(output_dir / "results.json", summary)
    print(f"Saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
