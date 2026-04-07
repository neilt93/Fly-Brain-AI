"""
Systematic bottleneck DN discovery across all 6 sensory modalities.

Generalizes the DNb05 finding: for each modality, find every DN that serves
as a sole or dominant gateway. A "bottleneck" DN is one reached by the target
modality but <=1 other modality. Ablating it should collapse target throughput
(>30%) with high specificity (>5x over other modalities).

Usage:
    cd plastic-fly
    python experiments/systematic_bottleneck.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig

BASE = Path(__file__).resolve().parent.parent.parent  # "Connectome Fly Brain"


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
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


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (reuses bottleneck_causal.py pattern)
# ═══════════════════════════════════════════════════════════════════════════

def load_connectome():
    """Load connectivity, annotations, and readout population."""
    cfg = BridgeConfig()
    conn = pd.read_parquet(cfg.connectivity_path)
    ann = pd.read_csv(BASE / "brain-model" / "flywire_annotations_matched.csv", low_memory=False)
    comp = pd.read_csv(cfg.completeness_path)
    root_ids = comp["Unnamed: 0"].values
    valid_set = set(root_ids.tolist())

    with open(cfg.data_dir / "channel_map_v4_looming.json") as f:
        channel_map = json.load(f)
    with open(cfg.data_dir / "decoder_groups_v4_looming.json") as f:
        decoder_groups = json.load(f)

    readout_ids = np.load(cfg.data_dir / "readout_ids_v4_looming.npy")
    small_mask = readout_ids < 1_000_000
    readout_rootids = set()
    readout_rootids.update(readout_ids[~small_mask].tolist())
    readout_rootids.update(root_ids[readout_ids[small_mask]].tolist())

    return conn, ann, valid_set, channel_map, decoder_groups, readout_rootids


def get_modality_neurons(ann, valid_set, channel_map, nine_modality=False):
    """Build modality neuron sets.

    Args:
        nine_modality: If True, treat somatosensory subchannels as first-class
            modalities alongside the 5 other major modalities (9 total).
            This tests exclusivity against ALL categories, not just the 6 major ones.
    """
    somato_channels = ["proprioceptive", "mechanosensory", "vestibular", "gustatory"]
    visual_channels = ["visual_left", "visual_right", "lplc2_left", "lplc2_right"]
    olfactory_channels = ["olfactory_left", "olfactory_right"]

    def flatten(channels):
        s = set()
        for ch in channels:
            s.update(channel_map[ch])
        return s

    if nine_modality:
        # 9-modality mode: subchannels are first-class
        modalities = {}
        for ch in somato_channels:
            modalities[ch] = set(channel_map[ch])
        modalities["visual"] = flatten(visual_channels)
        modalities["olfactory"] = flatten(olfactory_channels)
    else:
        modalities = {
            "somatosensory": flatten(somato_channels),
            "visual": flatten(visual_channels),
            "olfactory": flatten(olfactory_channels),
        }

    # Somatosensory subchannels (for gustatory-stance analysis in 6-modality mode)
    subchannels = {}
    for ch in somato_channels:
        subchannels[ch] = set(channel_map[ch])

    aud_mask = (ann["cell_class"] == "mechanosensory") & (ann["cell_sub_class"] == "auditory")
    modalities["auditory"] = set(ann[aud_mask]["root_id"].values) & valid_set

    thermo_mask = ann["cell_class"] == "thermosensory"
    modalities["thermosensory"] = set(ann[thermo_mask]["root_id"].values) & valid_set

    hygro_mask = ann["cell_class"] == "hygrosensory"
    modalities["hygrosensory"] = set(ann[hygro_mask]["root_id"].values) & valid_set

    return modalities, subchannels


def compute_modality_dn_throughput(conn, modalities, readout_rootids, ablated_dns=None):
    """Compute 1-hop synaptic throughput from each modality to readout DNs."""
    active_readout = readout_rootids - (ablated_dns or set())
    dn_edges = conn[conn["Postsynaptic_ID"].isin(active_readout)]

    results = {}
    for mod_name, mod_neurons in modalities.items():
        edges = dn_edges[dn_edges["Presynaptic_ID"].isin(mod_neurons)]
        dns_reached = set(edges["Postsynaptic_ID"].unique())
        total_syn = int(edges["Connectivity"].sum()) if len(edges) > 0 else 0
        results[mod_name] = {
            "dns_reached": len(dns_reached),
            "dns_set": dns_reached,
            "total_edges": len(edges),
            "total_syn": total_syn,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_bottleneck_candidates(conn, modalities, readout_rootids):
    """For each modality, find DNs reached by it but <=1 other modality."""
    # First: compute which DNs each modality reaches at 1-hop
    dn_edges = conn[conn["Postsynaptic_ID"].isin(readout_rootids)]

    modality_dns = {}
    for mod_name, mod_neurons in modalities.items():
        edges = dn_edges[dn_edges["Presynaptic_ID"].isin(mod_neurons)]
        modality_dns[mod_name] = set(edges["Postsynaptic_ID"].unique())

    # For each modality, find candidate bottleneck DNs
    candidates = {}
    mod_names = list(modalities.keys())

    for target_mod in mod_names:
        target_dns = modality_dns[target_mod]
        cands = []

        for dn_id in target_dns:
            # Count how many OTHER modalities reach this DN
            other_count = 0
            other_mods = []
            for other_mod in mod_names:
                if other_mod == target_mod:
                    continue
                if dn_id in modality_dns[other_mod]:
                    other_count += 1
                    other_mods.append(other_mod)

            # Bottleneck candidate: reached by target + at most 1 other
            if other_count <= 1:
                cands.append({
                    "dn_id": dn_id,
                    "other_modalities": other_mods,
                    "other_count": other_count,
                    "exclusive": other_count == 0,
                })

        candidates[target_mod] = cands

    return candidates, modality_dns


def run_bottleneck_ablation(conn, modalities, readout_rootids, dn_id, ann):
    """Ablate a single DN and measure per-modality throughput collapse."""
    ablated = {dn_id}

    baseline = compute_modality_dn_throughput(conn, modalities, readout_rootids)
    ablated_result = compute_modality_dn_throughput(conn, modalities, readout_rootids, ablated)

    impacts = {}
    for mod_name in modalities:
        b = baseline[mod_name]
        a = ablated_result[mod_name]
        dn_loss = b["dns_reached"] - a["dns_reached"]
        syn_loss = b["total_syn"] - a["total_syn"]
        dn_pct = (dn_loss / b["dns_reached"] * 100) if b["dns_reached"] > 0 else 0
        syn_pct = (syn_loss / b["total_syn"] * 100) if b["total_syn"] > 0 else 0
        impacts[mod_name] = {
            "dn_loss": dn_loss,
            "syn_loss": syn_loss,
            "dn_pct": dn_pct,
            "syn_pct": syn_pct,
        }

    # Get DN annotation
    ann_row = ann[ann["root_id"] == dn_id]
    cell_type = str(ann_row["cell_type"].values[0]) if len(ann_row) > 0 else "unknown"
    side = str(ann_row["side"].values[0]) if len(ann_row) > 0 else "?"

    return {
        "dn_id": int(dn_id),
        "cell_type": cell_type,
        "side": side,
        "impacts": impacts,
    }


def evaluate_bottleneck(result, target_modality):
    """Check if a DN passes bottleneck criteria for a target modality.

    Two tiers:
      STRONG: >30% collapse AND >5x specificity (or zero other-modality impact)
      WEAK:   >15% collapse AND >5x specificity (high specificity, moderate collapse)
    """
    impacts = result["impacts"]
    target_impact = impacts[target_modality]

    # Use max of DN% and syn% collapse
    target_collapse = max(target_impact["dn_pct"], target_impact["syn_pct"])

    max_other = 0
    for mod_name, imp in impacts.items():
        if mod_name == target_modality:
            continue
        max_other = max(max_other, imp["dn_pct"], imp["syn_pct"])

    specificity = target_collapse / max_other if max_other > 0 else float("inf")

    strong = target_collapse > 30 and (specificity > 5 or max_other == 0)
    weak = target_collapse > 15 and (specificity > 5 or max_other == 0)

    return {
        "target_collapse_pct": target_collapse,
        "max_other_collapse_pct": max_other,
        "specificity_ratio": specificity,
        "passes_strong": strong,
        "passes_weak": weak,
        "tier": "STRONG" if strong else ("WEAK" if weak else "none"),
    }


def find_cumulative_bottlenecks(conn, modalities, readout_rootids, candidates, ann, top_k=5):
    """For each modality, ablate top-K exclusive candidates together and measure cumulative collapse.

    This finds minimum ablation sets that collectively constitute a bottleneck.
    """
    results = {}
    for mod_name, cands in candidates.items():
        if not cands:
            results[mod_name] = None
            continue

        # Sort by exclusivity (exclusive first), then by synapse count
        exclusive = [c for c in cands if c["exclusive"]]
        if not exclusive:
            results[mod_name] = None
            continue

        # Cumulatively ablate exclusive DNs, tracking collapse
        baseline = compute_modality_dn_throughput(conn, modalities, readout_rootids)
        ablated_set = set()
        cumulative = []

        for i, c in enumerate(exclusive[:top_k]):
            ablated_set.add(c["dn_id"])
            abl_result = compute_modality_dn_throughput(conn, modalities, readout_rootids, ablated_set)

            b_target = baseline[mod_name]
            a_target = abl_result[mod_name]
            target_dn_pct = ((b_target["dns_reached"] - a_target["dns_reached"]) /
                             b_target["dns_reached"] * 100) if b_target["dns_reached"] > 0 else 0
            target_syn_pct = ((b_target["total_syn"] - a_target["total_syn"]) /
                              b_target["total_syn"] * 100) if b_target["total_syn"] > 0 else 0

            max_other_pct = 0
            for other_mod in modalities:
                if other_mod == mod_name:
                    continue
                b_o = baseline[other_mod]
                a_o = abl_result[other_mod]
                o_dn = ((b_o["dns_reached"] - a_o["dns_reached"]) /
                        b_o["dns_reached"] * 100) if b_o["dns_reached"] > 0 else 0
                o_syn = ((b_o["total_syn"] - a_o["total_syn"]) /
                         b_o["total_syn"] * 100) if b_o["total_syn"] > 0 else 0
                max_other_pct = max(max_other_pct, o_dn, o_syn)

            ann_row = ann[ann["root_id"] == c["dn_id"]]
            ct = str(ann_row["cell_type"].values[0]) if len(ann_row) > 0 else "?"

            cumulative.append({
                "n_ablated": i + 1,
                "last_added": ct,
                "target_dn_pct": target_dn_pct,
                "target_syn_pct": target_syn_pct,
                "max_other_pct": max_other_pct,
            })

        results[mod_name] = cumulative

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_systematic_bottleneck(nine_modality=False):
    mode_str = "9-MODALITY" if nine_modality else "6-MODALITY"
    print("=" * 80)
    print(f"SYSTEMATIC BOTTLENECK DN DISCOVERY ({mode_str})")
    print("=" * 80)

    conn, ann, valid_set, channel_map, decoder_groups, readout_rootids = load_connectome()
    modalities, subchannels = get_modality_neurons(
        ann, valid_set, channel_map, nine_modality=nine_modality,
    )

    # Build group map for characterizing DNs
    group_map = {}
    for k, v in decoder_groups.items():
        name = k.replace("_ids", "")
        for gid in v:
            group_map[int(gid)] = name

    print(f"\nModality populations:")
    for name, neurons in modalities.items():
        print(f"  {name}: {len(neurons)} neurons")

    # Step 1: Find bottleneck candidates per modality
    print(f"\n{'='*80}")
    print("STEP 1: CANDIDATE IDENTIFICATION")
    print(f"{'='*80}")

    candidates, modality_dns = find_bottleneck_candidates(conn, modalities, readout_rootids)

    for mod_name in sorted(candidates.keys()):
        cands = candidates[mod_name]
        total_dns = len(modality_dns[mod_name])
        exclusive = sum(1 for c in cands if c["exclusive"])
        print(f"\n  {mod_name}: {total_dns} DNs reached at 1-hop, "
              f"{len(cands)} bottleneck candidates ({exclusive} exclusive)")
        for c in sorted(cands, key=lambda x: x["other_count"]):
            dn_id = c["dn_id"]
            ann_row = ann[ann["root_id"] == dn_id]
            ct = str(ann_row["cell_type"].values[0]) if len(ann_row) > 0 else "?"
            grp = group_map.get(dn_id, "?")
            excl_str = "EXCLUSIVE" if c["exclusive"] else f"+{','.join(c['other_modalities'])}"
            print(f"    {dn_id} ({ct}, group={grp}) [{excl_str}]")

    # Step 2: Ablate each candidate and measure specificity
    print(f"\n{'='*80}")
    print("STEP 2: CAUSAL ABLATION PER CANDIDATE")
    print(f"{'='*80}")

    all_results = {}
    confirmed_bottlenecks = []

    for mod_name in sorted(candidates.keys()):
        cands = candidates[mod_name]
        if not cands:
            print(f"\n  {mod_name}: no candidates to test")
            continue

        print(f"\n  --- {mod_name.upper()} ({len(cands)} candidates) ---")
        mod_results = []

        for c in cands:
            dn_id = c["dn_id"]
            result = run_bottleneck_ablation(conn, modalities, readout_rootids, dn_id, ann)
            evaluation = evaluate_bottleneck(result, mod_name)
            result["evaluation"] = evaluation

            tier = evaluation["tier"]
            spec = evaluation["specificity_ratio"]
            spec_str = f"{spec:.1f}x" if spec != float("inf") else "inf"

            if evaluation["target_collapse_pct"] > 5 or tier != "none":
                print(f"    {result['cell_type']:>12s} ({result['side']}): "
                      f"target={evaluation['target_collapse_pct']:.1f}% "
                      f"max_other={evaluation['max_other_collapse_pct']:.1f}% "
                      f"spec={spec_str} [{tier}]")

            if tier != "none":
                confirmed_bottlenecks.append({
                    "modality": mod_name,
                    "dn_id": int(dn_id),
                    "cell_type": result["cell_type"],
                    "side": result["side"],
                    "target_collapse": evaluation["target_collapse_pct"],
                    "specificity": evaluation["specificity_ratio"],
                    "decoder_group": group_map.get(dn_id, "unknown"),
                    "tier": tier,
                })

            mod_results.append(result)

        all_results[mod_name] = mod_results

    # Step 3: Somatosensory subchannels (only in 6-modality mode; 9-modality includes them already)
    if not nine_modality:
        print(f"\n{'='*80}")
        print("STEP 3: SOMATOSENSORY SUBCHANNEL BOTTLENECKS")
        print(f"{'='*80}")

        sub_modalities = {}
        for ch_name, ch_neurons in subchannels.items():
            sub_modalities[ch_name] = ch_neurons

        sub_candidates, sub_dns = find_bottleneck_candidates(conn, sub_modalities, readout_rootids)

        for ch_name in sorted(sub_candidates.keys()):
            cands = sub_candidates[ch_name]
            total_dns = len(sub_dns[ch_name])
            exclusive = sum(1 for c in cands if c["exclusive"])
            print(f"\n  {ch_name}: {total_dns} DNs, {len(cands)} candidates ({exclusive} exclusive)")

            for c in sorted(cands, key=lambda x: x["other_count"])[:10]:
                dn_id = c["dn_id"]
                result = run_bottleneck_ablation(conn, sub_modalities, readout_rootids, dn_id, ann)
                evaluation = evaluate_bottleneck(result, ch_name)

                tier = evaluation["tier"]
                if evaluation["target_collapse_pct"] > 5 or tier != "none":
                    spec = evaluation["specificity_ratio"]
                    spec_str = f"{spec:.1f}x" if spec != float("inf") else "inf"
                    print(f"    {result['cell_type']:>12s}: target={evaluation['target_collapse_pct']:.1f}% "
                          f"spec={spec_str} [{tier}]")

                    if tier != "none":
                        confirmed_bottlenecks.append({
                            "modality": f"somatosensory/{ch_name}",
                            "dn_id": int(dn_id),
                            "cell_type": result["cell_type"],
                            "side": result["side"],
                            "target_collapse": evaluation["target_collapse_pct"],
                            "specificity": evaluation["specificity_ratio"],
                            "decoder_group": group_map.get(dn_id, "unknown"),
                            "tier": tier,
                        })
    else:
        print(f"\n  (Skipping Step 3 -- subchannels are first-class in 9-modality mode)")

    # Step 4: Cumulative ablation (top-K exclusive DNs per modality)
    print(f"\n{'='*80}")
    print("STEP 4: CUMULATIVE EXCLUSIVE DN ABLATION")
    print(f"{'='*80}")

    cumulative = find_cumulative_bottlenecks(
        conn, modalities, readout_rootids, candidates, ann, top_k=5,
    )
    for mod_name, cum_data in cumulative.items():
        if cum_data is None:
            print(f"\n  {mod_name}: no exclusive candidates")
            continue
        print(f"\n  {mod_name}:")
        print(f"    {'#Ablated':>8s} {'Last Added':>14s} {'Target DN%':>10s} "
              f"{'Target Syn%':>11s} {'Max Other%':>10s}")
        for entry in cum_data:
            print(f"    {entry['n_ablated']:>8d} {entry['last_added']:>14s} "
                  f"{entry['target_dn_pct']:>9.1f}% {entry['target_syn_pct']:>10.1f}% "
                  f"{entry['max_other_pct']:>9.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("SUMMARY: CONFIRMED BOTTLENECK DNs")
    print(f"{'='*80}")

    if not confirmed_bottlenecks:
        print("  No single-DN bottlenecks passed criteria.")
        print("  This indicates DISTRIBUTED processing -- most modalities use")
        print("  redundant DN pools (unlike the thermo/hygro exception).")
    else:
        print(f"\n  {'Tier':<7s} {'Modality':<25s} {'Cell Type':<14s} {'Side':<6s} "
              f"{'Collapse':<10s} {'Specificity':<12s} {'Group':<12s}")
        print("  " + "-" * 87)
        for b in sorted(confirmed_bottlenecks, key=lambda x: (-1 if x.get("tier") == "STRONG" else 0, -x["target_collapse"])):
            spec_str = f"{b['specificity']:.1f}x" if b["specificity"] != float("inf") else "inf"
            print(f"  {b.get('tier','?'):<7s} {b['modality']:<25s} {b['cell_type']:<14s} "
                  f"{b['side']:<6s} {b['target_collapse']:<9.1f}% {spec_str:<12s} "
                  f"{b['decoder_group']:<12s}")

    strong = [b for b in confirmed_bottlenecks if b.get("tier") == "STRONG"]
    weak = [b for b in confirmed_bottlenecks if b.get("tier") == "WEAK"]
    novel_count = len([b for b in confirmed_bottlenecks
                       if b["cell_type"] != "DNb05"])
    print(f"\n  STRONG bottlenecks (>30% collapse, >5x specificity): {len(strong)}")
    print(f"  WEAK bottlenecks (>15% collapse, >5x specificity):   {len(weak)}")
    print(f"  Novel (beyond DNb05): {novel_count}")
    print(f"  SUCCESS: {'PASS' if novel_count >= 2 else 'PARTIAL' if novel_count >= 1 else 'INFORMATIVE NULL'}")

    # Save results
    output_dir = Path("logs/systematic_bottleneck")
    output_payload = {
        "confirmed_bottlenecks": confirmed_bottlenecks,
        "novel_count": novel_count,
        "strong_count": len(strong),
        "weak_count": len(weak),
        "per_modality_candidates": {
            mod: len(cands) for mod, cands in candidates.items()
        },
        "per_modality_dns_reached": {
            mod: len(dns) for mod, dns in modality_dns.items()
        },
        "cumulative_ablation": {
            mod: data for mod, data in cumulative.items() if data is not None
        },
    }
    _write_json_atomic(output_dir / "results.json", output_payload)
    print(f"\nSaved to {output_dir}/results.json")

    return confirmed_bottlenecks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Systematic bottleneck DN discovery")
    parser.add_argument("--nine-modality", action="store_true",
                        help="Use 9-modality mode (subchannels as first-class)")
    args = parser.parse_args()
    run_systematic_bottleneck(nine_modality=args.nine_modality)
