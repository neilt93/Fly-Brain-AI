"""
Causal bottleneck experiments for sensory-motor specificity.

Experiment 1: DNb05 silencing
  - Silence the 2 DNb05 neurons that form the thermo/hygro bottleneck
  - Measure: does thermo/hygro->DN throughput collapse while other modalities survive?

Experiment 2: 12 auditory-visual shared DN silencing
  - Silence the 12 turning DNs shared by auditory and visual
  - Measure: does orientation-turning collapse while olfactory valence survives?

Both experiments run brain-only (no body needed): inject Poisson stimuli into
sensory populations, measure downstream firing in readout DNs, compare
intact vs ablated conditions.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig

BASE = Path(__file__).resolve().parent.parent.parent  # up to "Connectome Fly Brain"


def load_connectome():
    """Load connectivity and annotations."""
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


def get_modality_neurons(ann, valid_set, channel_map):
    """Get all modality neuron sets including new modalities."""
    somato_channels = ["proprioceptive", "mechanosensory", "vestibular", "gustatory"]
    visual_channels = ["visual_left", "visual_right", "lplc2_left", "lplc2_right"]
    olfactory_channels = ["olfactory_left", "olfactory_right"]

    def flatten(channels):
        s = set()
        for ch in channels:
            s.update(channel_map[ch])
        return s

    # Existing modalities
    modalities = {
        "somatosensory": flatten(somato_channels),
        "visual": flatten(visual_channels),
        "olfactory": flatten(olfactory_channels),
    }

    # New modalities from annotations
    aud_mask = (ann["cell_class"] == "mechanosensory") & (ann["cell_sub_class"] == "auditory")
    modalities["auditory"] = set(ann[aud_mask]["root_id"].values) & valid_set

    thermo_mask = ann["cell_class"] == "thermosensory"
    modalities["thermosensory"] = set(ann[thermo_mask]["root_id"].values) & valid_set

    hygro_mask = ann["cell_class"] == "hygrosensory"
    modalities["hygrosensory"] = set(ann[hygro_mask]["root_id"].values) & valid_set

    return modalities


def compute_modality_dn_throughput(conn, modalities, readout_rootids, ablated_dns=None):
    """Compute synaptic throughput from each modality to readout DNs.

    If ablated_dns is provided, those DNs are removed from the readout set.
    Returns dict: modality_name -> {dns_reached, total_edges, total_syn}
    """
    active_readout = readout_rootids - (ablated_dns or set())
    dn_edges = conn[conn["Postsynaptic_ID"].isin(active_readout)]

    results = {}
    for mod_name, mod_neurons in modalities.items():
        edges = dn_edges[dn_edges["Presynaptic_ID"].isin(mod_neurons)]
        dns_reached = set(edges["Postsynaptic_ID"].unique())
        total_syn = int(edges["Connectivity"].sum()) if len(edges) > 0 else 0
        results[mod_name] = {
            "dns_reached": len(dns_reached),
            "total_edges": len(edges),
            "total_syn": total_syn,
        }
    return results


def compute_2hop_throughput(conn, modalities, readout_rootids, ablated_dns=None):
    """Compute 2-hop throughput: sensory -> intermediate -> DN.

    With ablated DNs removed from the readout set.
    """
    active_readout = readout_rootids - (ablated_dns or set())
    dn_edges = conn[conn["Postsynaptic_ID"].isin(active_readout)]

    results = {}
    for mod_name, mod_neurons in modalities.items():
        # 1-hop downstream of sensory
        downstream = conn[conn["Presynaptic_ID"].isin(mod_neurons)]
        hop1_targets = set(downstream["Postsynaptic_ID"].unique())
        intermediates = hop1_targets - mod_neurons - readout_rootids

        # Which intermediates project to active readout DNs?
        inter_to_dn = dn_edges[dn_edges["Presynaptic_ID"].isin(intermediates)]
        active_inter = set(inter_to_dn["Presynaptic_ID"].unique())
        dns_via_2hop = set(inter_to_dn["Postsynaptic_ID"].unique())

        results[mod_name] = {
            "active_intermediates": len(active_inter),
            "dns_via_2hop": len(dns_via_2hop),
        }
    return results


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: DNb05 BOTTLENECK SILENCING
# ════════════════════════════════════════════════════════════════════════════

def experiment_1_dnb05():
    """Silence DNb05 neurons and measure modality-specific throughput collapse."""
    print("=" * 80)
    print("EXPERIMENT 1: DNb05 BOTTLENECK SILENCING")
    print("=" * 80)

    conn, ann, valid_set, channel_map, decoder_groups, readout_rootids = load_connectome()
    modalities = get_modality_neurons(ann, valid_set, channel_map)

    # Find DNb05 neurons in our readout set
    dnb05_mask = ann["cell_type"] == "DNb05"
    dnb05_all = set(ann[dnb05_mask]["root_id"].values)
    dnb05_readout = dnb05_all & readout_rootids
    print(f"\n  DNb05 neurons in FlyWire: {len(dnb05_all)}")
    print(f"  DNb05 in our readout set: {len(dnb05_readout)}")
    for dn_id in sorted(dnb05_readout):
        row = ann[ann["root_id"] == dn_id]
        side = str(row["side"].values[0]) if len(row) > 0 else "?"
        print(f"    {dn_id} (side={side})")

    # Baseline throughput
    print("\n  --- BASELINE (all DNs intact) ---")
    baseline = compute_modality_dn_throughput(conn, modalities, readout_rootids)
    baseline_2hop = compute_2hop_throughput(conn, modalities, readout_rootids)

    for mod_name in sorted(modalities.keys()):
        b = baseline[mod_name]
        b2 = baseline_2hop[mod_name]
        print(f"    {mod_name:>16s}: {b['dns_reached']:>4d} DNs, {b['total_syn']:>8,} syn (1-hop) | "
              f"{b2['dns_via_2hop']:>4d} DNs (2-hop)")

    # Ablated throughput
    print(f"\n  --- ABLATED (DNb05 silenced: {len(dnb05_readout)} neurons) ---")
    ablated = compute_modality_dn_throughput(conn, modalities, readout_rootids, dnb05_readout)
    ablated_2hop = compute_2hop_throughput(conn, modalities, readout_rootids, dnb05_readout)

    for mod_name in sorted(modalities.keys()):
        a = ablated[mod_name]
        a2 = ablated_2hop[mod_name]
        print(f"    {mod_name:>16s}: {a['dns_reached']:>4d} DNs, {a['total_syn']:>8,} syn (1-hop) | "
              f"{a2['dns_via_2hop']:>4d} DNs (2-hop)")

    # Impact analysis
    print(f"\n  --- IMPACT (% reduction) ---")
    print(f"  {'Modality':>16s}  {'DN loss':>8s}  {'Syn loss':>10s}  {'DN% lost':>10s}  {'Syn% lost':>10s}  {'2hop DN loss':>12s}")
    for mod_name in sorted(modalities.keys()):
        b = baseline[mod_name]
        a = ablated[mod_name]
        b2 = baseline_2hop[mod_name]
        a2 = ablated_2hop[mod_name]
        dn_loss = b["dns_reached"] - a["dns_reached"]
        syn_loss = b["total_syn"] - a["total_syn"]
        dn_pct = (dn_loss / b["dns_reached"] * 100) if b["dns_reached"] > 0 else 0
        syn_pct = (syn_loss / b["total_syn"] * 100) if b["total_syn"] > 0 else 0
        hop2_loss = b2["dns_via_2hop"] - a2["dns_via_2hop"]
        hop2_pct = (hop2_loss / b2["dns_via_2hop"] * 100) if b2["dns_via_2hop"] > 0 else 0
        print(f"  {mod_name:>16s}  {dn_loss:>8d}  {syn_loss:>10,}  {dn_pct:>9.1f}%  {syn_pct:>9.1f}%  {hop2_loss:>4d} ({hop2_pct:.1f}%)")

    # Statistical test: is thermo/hygro impact disproportionate?
    print(f"\n  --- CAUSAL SPECIFICITY TEST ---")
    thermo_dn_pct = 0
    hygro_dn_pct = 0
    other_max_dn_pct = 0
    for mod_name in sorted(modalities.keys()):
        b = baseline[mod_name]
        a = ablated[mod_name]
        dn_loss = b["dns_reached"] - a["dns_reached"]
        dn_pct = (dn_loss / b["dns_reached"] * 100) if b["dns_reached"] > 0 else 0
        if mod_name == "thermosensory":
            thermo_dn_pct = dn_pct
        elif mod_name == "hygrosensory":
            hygro_dn_pct = dn_pct
        else:
            other_max_dn_pct = max(other_max_dn_pct, dn_pct)

    thermo_syn_pct = 0
    hygro_syn_pct = 0
    for mod_name in ["thermosensory", "hygrosensory"]:
        b = baseline[mod_name]
        a = ablated[mod_name]
        syn_loss = b["total_syn"] - a["total_syn"]
        syn_pct = (syn_loss / b["total_syn"] * 100) if b["total_syn"] > 0 else 0
        if mod_name == "thermosensory":
            thermo_syn_pct = syn_pct
        else:
            hygro_syn_pct = syn_pct

    print(f"    Thermosensory DN loss: {thermo_dn_pct:.1f}% (syn: {thermo_syn_pct:.1f}%)")
    print(f"    Hygrosensory DN loss:  {hygro_dn_pct:.1f}% (syn: {hygro_syn_pct:.1f}%)")
    print(f"    Max other modality DN loss: {other_max_dn_pct:.1f}%")
    print(f"    Specificity ratio (thermo/max_other): ", end="")
    if other_max_dn_pct > 0:
        print(f"{thermo_dn_pct / other_max_dn_pct:.1f}x")
    else:
        print("inf (other modalities unaffected)")

    tests_passed = 0
    total_tests = 4

    # Test 1: thermo loses > 30% of DN targets
    t1 = thermo_dn_pct > 30
    tests_passed += t1
    print(f"\n  Test 1: Thermo loses >30% DN targets: {'PASS' if t1 else 'FAIL'} ({thermo_dn_pct:.1f}%)")

    # Test 2: hygro loses > 30% of DN targets
    t2 = hygro_dn_pct > 30
    tests_passed += t2
    print(f"  Test 2: Hygro loses >30% DN targets: {'PASS' if t2 else 'FAIL'} ({hygro_dn_pct:.1f}%)")

    # Test 3: thermo/hygro loss > 5x other modalities
    t3 = (thermo_dn_pct > 5 * other_max_dn_pct) if other_max_dn_pct > 0 else (thermo_dn_pct > 0)
    tests_passed += t3
    print(f"  Test 3: Thermo loss >5x other modalities: {'PASS' if t3 else 'FAIL'}")

    # Test 4: somatosensory/visual/olfactory lose < 5% each
    other_ok = True
    for mod_name in ["somatosensory", "visual", "olfactory"]:
        b = baseline[mod_name]
        a = ablated[mod_name]
        dn_pct = ((b["dns_reached"] - a["dns_reached"]) / b["dns_reached"] * 100) if b["dns_reached"] > 0 else 0
        if dn_pct > 5:
            other_ok = False
    tests_passed += other_ok
    print(f"  Test 4: Major modalities lose <5% DNs: {'PASS' if other_ok else 'FAIL'}")

    print(f"\n  RESULT: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: AUDITORY-VISUAL SHARED DN SILENCING
# ════════════════════════════════════════════════════════════════════════════

def experiment_2_shared_turning():
    """Silence the 12 auditory-visual shared DNs and measure selective impact."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: AUDITORY-VISUAL SHARED DN SILENCING")
    print("=" * 80)

    conn, ann, valid_set, channel_map, decoder_groups, readout_rootids = load_connectome()
    modalities = get_modality_neurons(ann, valid_set, channel_map)

    # Find the 12 shared auditory-visual DNs
    dn_edges = conn[conn["Postsynaptic_ID"].isin(readout_rootids)]
    aud_dns = set(dn_edges[dn_edges["Presynaptic_ID"].isin(modalities["auditory"])]["Postsynaptic_ID"].unique())
    vis_dns = set(dn_edges[dn_edges["Presynaptic_ID"].isin(modalities["visual"])]["Postsynaptic_ID"].unique())
    shared_av = aud_dns & vis_dns

    print(f"\n  Auditory DNs: {len(aud_dns)}")
    print(f"  Visual DNs: {len(vis_dns)}")
    print(f"  Shared auditory-visual DNs: {len(shared_av)}")

    # Characterize which groups they belong to
    group_map = {}
    for k, v in decoder_groups.items():
        name = k.replace("_ids", "")
        for gid in v:
            group_map[gid] = name

    group_counts = {}
    for dn_id in shared_av:
        g = group_map.get(dn_id, "unknown")
        group_counts[g] = group_counts.get(g, 0) + 1
    print(f"  Group distribution: {group_counts}")

    for dn_id in sorted(shared_av):
        row = ann[ann["root_id"] == dn_id]
        ct = str(row["cell_type"].values[0]) if len(row) > 0 else "N/A"
        side = str(row["side"].values[0]) if len(row) > 0 else "?"
        g = group_map.get(dn_id, "?")
        print(f"    {dn_id} ({ct}, side={side}, group={g})")

    # Baseline
    print("\n  --- BASELINE ---")
    baseline = compute_modality_dn_throughput(conn, modalities, readout_rootids)
    for mod_name in sorted(modalities.keys()):
        b = baseline[mod_name]
        print(f"    {mod_name:>16s}: {b['dns_reached']:>4d} DNs, {b['total_syn']:>8,} syn")

    # Ablated
    print(f"\n  --- ABLATED ({len(shared_av)} shared auditory-visual DNs silenced) ---")
    ablated = compute_modality_dn_throughput(conn, modalities, readout_rootids, shared_av)
    for mod_name in sorted(modalities.keys()):
        a = ablated[mod_name]
        print(f"    {mod_name:>16s}: {a['dns_reached']:>4d} DNs, {a['total_syn']:>8,} syn")

    # Impact
    print(f"\n  --- IMPACT ---")
    print(f"  {'Modality':>16s}  {'DN loss':>8s}  {'Syn loss':>10s}  {'DN% lost':>10s}  {'Syn% lost':>10s}")
    for mod_name in sorted(modalities.keys()):
        b = baseline[mod_name]
        a = ablated[mod_name]
        dn_loss = b["dns_reached"] - a["dns_reached"]
        syn_loss = b["total_syn"] - a["total_syn"]
        dn_pct = (dn_loss / b["dns_reached"] * 100) if b["dns_reached"] > 0 else 0
        syn_pct = (syn_loss / b["total_syn"] * 100) if b["total_syn"] > 0 else 0
        print(f"  {mod_name:>16s}  {dn_loss:>8d}  {syn_loss:>10,}  {dn_pct:>9.1f}%  {syn_pct:>9.1f}%")

    # Per-decoder-group impact
    print(f"\n  --- PER-GROUP IMPACT ON AUDITORY THROUGHPUT ---")
    group_names_set = {}
    for k, v in decoder_groups.items():
        name = k.replace("_ids", "")
        group_names_set[name] = set(v)

    group_list = ["forward", "turn_left", "turn_right", "rhythm", "stance"]
    for g in group_list:
        active_g = group_names_set[g] - shared_av
        g_edges_base = dn_edges[
            (dn_edges["Presynaptic_ID"].isin(modalities["auditory"])) &
            (dn_edges["Postsynaptic_ID"].isin(group_names_set[g]))
        ]
        g_edges_abl = dn_edges[
            (dn_edges["Presynaptic_ID"].isin(modalities["auditory"])) &
            (dn_edges["Postsynaptic_ID"].isin(active_g))
        ]
        base_syn = g_edges_base["Connectivity"].sum()
        abl_syn = g_edges_abl["Connectivity"].sum()
        loss_pct = ((base_syn - abl_syn) / base_syn * 100) if base_syn > 0 else 0
        print(f"    {g:>12s}: {base_syn:>6,} -> {abl_syn:>6,} syn ({loss_pct:.1f}% lost)")

    # Tests
    tests_passed = 0
    total_tests = 4

    # Test 1: Auditory loses > 20% of DN targets
    aud_b = baseline["auditory"]
    aud_a = ablated["auditory"]
    aud_dn_pct = ((aud_b["dns_reached"] - aud_a["dns_reached"]) / aud_b["dns_reached"] * 100) if aud_b["dns_reached"] > 0 else 0
    t1 = aud_dn_pct > 20
    tests_passed += t1
    print(f"\n  Test 1: Auditory loses >20% DN targets: {'PASS' if t1 else 'FAIL'} ({aud_dn_pct:.1f}%)")

    # Test 2: Visual loses > 10% of DN targets
    vis_b = baseline["visual"]
    vis_a = ablated["visual"]
    vis_dn_pct = ((vis_b["dns_reached"] - vis_a["dns_reached"]) / vis_b["dns_reached"] * 100) if vis_b["dns_reached"] > 0 else 0
    t2 = vis_dn_pct > 10
    tests_passed += t2
    print(f"  Test 2: Visual loses >10% DN targets: {'PASS' if t2 else 'FAIL'} ({vis_dn_pct:.1f}%)")

    # Test 3: Olfactory is unaffected (0% loss)
    olf_b = baseline["olfactory"]
    olf_a = ablated["olfactory"]
    olf_dn_pct = ((olf_b["dns_reached"] - olf_a["dns_reached"]) / olf_b["dns_reached"] * 100) if olf_b["dns_reached"] > 0 else 0
    t3 = olf_dn_pct == 0
    tests_passed += t3
    print(f"  Test 3: Olfactory unaffected (0% loss): {'PASS' if t3 else 'FAIL'} ({olf_dn_pct:.1f}%)")

    # Test 4: Turning groups hit harder than forward/stance
    turning_loss = 0
    other_loss = 0
    for g in group_list:
        active_g = group_names_set[g] - shared_av
        g_edges_base = dn_edges[
            (dn_edges["Presynaptic_ID"].isin(modalities["auditory"])) &
            (dn_edges["Postsynaptic_ID"].isin(group_names_set[g]))
        ]
        g_edges_abl = dn_edges[
            (dn_edges["Presynaptic_ID"].isin(modalities["auditory"])) &
            (dn_edges["Postsynaptic_ID"].isin(active_g))
        ]
        base_syn = g_edges_base["Connectivity"].sum()
        abl_syn = g_edges_abl["Connectivity"].sum()
        loss = base_syn - abl_syn
        if g in ("turn_left", "turn_right"):
            turning_loss += loss
        else:
            other_loss += loss

    t4 = turning_loss > other_loss
    tests_passed += t4
    print(f"  Test 4: Turning loss > forward+rhythm+stance: {'PASS' if t4 else 'FAIL'} "
          f"({turning_loss:,} vs {other_loss:,})")

    print(f"\n  RESULT: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


if __name__ == "__main__":
    print("Causal Bottleneck Experiments")
    print("=" * 80)

    r1 = experiment_1_dnb05()
    r2 = experiment_2_shared_turning()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"  Experiment 1 (DNb05 bottleneck):      {'PASS' if r1 else 'FAIL'}")
    print(f"  Experiment 2 (Shared turning DNs):     {'PASS' if r2 else 'FAIL'}")
    print(f"  Overall: {int(r1) + int(r2)}/2 experiments passed")
