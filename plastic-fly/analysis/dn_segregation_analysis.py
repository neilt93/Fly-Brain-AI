"""
Deep analysis of descending neuron (DN) segregation across sensory modalities
in the Drosophila FlyWire connectome.

Analyses:
  A. Somatosensory subchannel breakdown at 1-hop
  B. The visual-somatosensory shared DNs
  C. Synapse weight analysis per modality->DN
  D. Neuropil distribution of modality->DN synapses
  E. Interneuron layer characterization (2-hop)
  F. Information flow asymmetry per decoder group
"""

import json
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

# ── Paths ───────────────────────────────────────────────────────────────────
BASE = "C:/Users/neilt/Connectome Fly Brain"
CHANNEL_MAP = f"{BASE}/plastic-fly/data/channel_map_v4_looming.json"
READOUT_IDS = f"{BASE}/plastic-fly/data/readout_ids_v4_looming.npy"
DECODER_GROUPS = f"{BASE}/plastic-fly/data/decoder_groups_v4_looming.json"
CONNECTIVITY = f"{BASE}/brain-model/Connectivity_783.parquet"
COMPLETENESS = f"{BASE}/brain-model/Completeness_783.csv"
ANNOTATIONS = f"{BASE}/brain-model/flywire_annotations_matched.csv"

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
with open(CHANNEL_MAP) as f:
    channel_map = json.load(f)
with open(DECODER_GROUPS) as f:
    decoder_groups = json.load(f)

# Map readout IDs (mix of indices and root_ids) to all root_ids
comp = pd.read_csv(COMPLETENESS)
root_ids_array = comp["Unnamed: 0"].values  # index -> root_id

raw_ids = np.load(READOUT_IDS)
small_mask = raw_ids < 1_000_000
readout_rootids = set()
readout_rootids.update(raw_ids[~small_mask].tolist())
readout_rootids.update(root_ids_array[raw_ids[small_mask]].tolist())
print(f"  Readout DNs: {len(readout_rootids)}")

# Load connectivity ONCE
print("Loading connectivity (15M rows)...")
conn = pd.read_parquet(CONNECTIVITY)
print(f"  Connectivity: {conn.shape[0]:,} edges")

# Load annotations
ann = pd.read_csv(ANNOTATIONS, low_memory=False)
ann_map = dict(zip(ann["root_id"], ann.index))
print(f"  Annotations: {ann.shape[0]:,} neurons")

# ── Define modality groups ──────────────────────────────────────────────────
somatosensory_channels = ["proprioceptive", "mechanosensory", "vestibular", "gustatory"]
visual_channels = ["visual_left", "visual_right", "lplc2_left", "lplc2_right"]
olfactory_channels = ["olfactory_left", "olfactory_right"]

# Flatten into modality sets
def get_modality_neurons(channels):
    s = set()
    for ch in channels:
        s.update(channel_map[ch])
    return s

somato_neurons = get_modality_neurons(somatosensory_channels)
visual_neurons = get_modality_neurons(visual_channels)
olfactory_neurons = get_modality_neurons(olfactory_channels)

modalities = {
    "somatosensory": somato_neurons,
    "visual": visual_neurons,
    "olfactory": olfactory_neurons,
}

print(f"\nModality sizes:")
print(f"  Somatosensory: {len(somato_neurons)} neurons")
for ch in somatosensory_channels:
    print(f"    {ch}: {len(channel_map[ch])}")
print(f"  Visual: {len(visual_neurons)} neurons")
for ch in visual_channels:
    print(f"    {ch}: {len(channel_map[ch])}")
print(f"  Olfactory: {len(olfactory_neurons)} neurons")
for ch in olfactory_channels:
    print(f"    {ch}: {len(channel_map[ch])}")

# ── Precompute 1-hop edges from each modality to readout DNs ───────────────
print("\nComputing 1-hop connections (sensory -> DN)...")

# Filter connectivity to only edges where post is a readout DN
dn_edges = conn[conn["Postsynaptic_ID"].isin(readout_rootids)].copy()
print(f"  Edges terminating on readout DNs: {dn_edges.shape[0]:,}")

# For each modality, filter edges where pre is in that modality
modality_dn_edges = {}
for mod_name, mod_neurons in modalities.items():
    edges = dn_edges[dn_edges["Presynaptic_ID"].isin(mod_neurons)]
    modality_dn_edges[mod_name] = edges
    dns_reached = edges["Postsynaptic_ID"].nunique()
    print(f"  {mod_name} -> DN: {edges.shape[0]:,} edges, {dns_reached} unique DNs reached")

# Per-subchannel edges
subchannel_dn_edges = {}
for ch in somatosensory_channels + visual_channels + olfactory_channels:
    ch_neurons = set(channel_map[ch])
    edges = dn_edges[dn_edges["Presynaptic_ID"].isin(ch_neurons)]
    subchannel_dn_edges[ch] = edges

# ── Decoder group mapping ──────────────────────────────────────────────────
# Rename keys: forward_ids -> forward, etc.
group_names = {}
for k, v in decoder_groups.items():
    name = k.replace("_ids", "")
    group_names[name] = set(v)

# Create DN -> group mapping
dn_to_group = {}
for gname, gids in group_names.items():
    for gid in gids:
        dn_to_group[gid] = gname

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS A: Somatosensory subchannel breakdown at 1-hop
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS A: Somatosensory subchannel breakdown at 1-hop")
print("=" * 80)

somato_subchannel_dns = {}
for ch in somatosensory_channels:
    dns = set(subchannel_dn_edges[ch]["Postsynaptic_ID"].unique())
    somato_subchannel_dns[ch] = dns
    print(f"\n  {ch}: reaches {len(dns)} DNs via {subchannel_dn_edges[ch].shape[0]} edges")

print("\n  Pairwise Jaccard overlaps WITHIN somatosensory subchannels:")
print(f"  {'':25s} ", end="")
for ch2 in somatosensory_channels:
    print(f"{ch2:>16s}", end="")
print()

for ch1 in somatosensory_channels:
    print(f"  {ch1:25s} ", end="")
    for ch2 in somatosensory_channels:
        s1, s2 = somato_subchannel_dns[ch1], somato_subchannel_dns[ch2]
        if len(s1 | s2) == 0:
            j = 0.0
        else:
            j = len(s1 & s2) / len(s1 | s2)
        print(f"{j:16.3f}", end="")
    print()

# Intersection sizes
print("\n  Pairwise intersection sizes:")
for ch1, ch2 in combinations(somatosensory_channels, 2):
    s1, s2 = somato_subchannel_dns[ch1], somato_subchannel_dns[ch2]
    inter = s1 & s2
    print(f"    {ch1} & {ch2}: {len(inter)} shared DNs (|union|={len(s1|s2)})")

# Exclusive DNs per subchannel
print("\n  Exclusive DNs (reached by ONLY this subchannel, not others in somato):")
for ch in somatosensory_channels:
    others = set()
    for ch2 in somatosensory_channels:
        if ch2 != ch:
            others |= somato_subchannel_dns[ch2]
    excl = somato_subchannel_dns[ch] - others
    print(f"    {ch}: {len(excl)} exclusive DNs")

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS B: Visual-somatosensory shared DNs
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS B: Visual-somatosensory shared DNs at 1-hop")
print("=" * 80)

vis_dns = set(modality_dn_edges["visual"]["Postsynaptic_ID"].unique())
som_dns = set(modality_dn_edges["somatosensory"]["Postsynaptic_ID"].unique())
olf_dns = set(modality_dn_edges["olfactory"]["Postsynaptic_ID"].unique())

shared_vs = vis_dns & som_dns
print(f"\n  Visual DNs: {len(vis_dns)}")
print(f"  Somatosensory DNs: {len(som_dns)}")
print(f"  Olfactory DNs: {len(olf_dns)}")
print(f"  Visual & Somatosensory shared: {len(shared_vs)}")
print(f"  Visual & Olfactory shared: {len(vis_dns & olf_dns)}")
print(f"  Somatosensory & Olfactory shared: {len(som_dns & olf_dns)}")
print(f"  All three shared: {len(vis_dns & som_dns & olf_dns)}")

print(f"\n  Detailed breakdown of the {len(shared_vs)} visual-somatosensory shared DNs:")
print(f"  {'DN root_id':>22s}  {'Decoder group':>14s}  {'Somato subch':>30s}  {'Visual subch':>30s}  {'cell_type':>15s}  {'flow':>10s}")

for dn_id in sorted(shared_vs):
    # Which somatosensory subchannels reach this DN?
    som_sub = []
    for ch in somatosensory_channels:
        if dn_id in somato_subchannel_dns.get(ch, set()):
            som_sub.append(ch)

    # Which visual subchannels reach this DN?
    vis_sub = []
    for ch in visual_channels:
        ch_dns = set(subchannel_dn_edges[ch]["Postsynaptic_ID"].unique())
        if dn_id in ch_dns:
            vis_sub.append(ch)

    # Decoder group
    dg = dn_to_group.get(dn_id, "???")

    # Annotation
    ann_row = ann[ann["root_id"] == dn_id]
    cell_type = ann_row["cell_type"].values[0] if len(ann_row) > 0 else "N/A"
    flow_val = ann_row["flow"].values[0] if len(ann_row) > 0 else "N/A"

    print(f"  {dn_id:>22d}  {dg:>14s}  {','.join(som_sub):>30s}  {','.join(vis_sub):>30s}  {str(cell_type):>15s}  {str(flow_val):>10s}")

# Summary: which subchannels contribute most
print("\n  Subchannel contribution to shared DNs:")
for ch in somatosensory_channels:
    count = sum(1 for dn_id in shared_vs if dn_id in somato_subchannel_dns.get(ch, set()))
    print(f"    {ch}: {count}/{len(shared_vs)}")
for ch in visual_channels:
    ch_dns = set(subchannel_dn_edges[ch]["Postsynaptic_ID"].unique())
    count = sum(1 for dn_id in shared_vs if dn_id in ch_dns)
    print(f"    {ch}: {count}/{len(shared_vs)}")

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS C: Synapse weight analysis
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS C: Synapse weight analysis (modality -> DN connections)")
print("=" * 80)

print(f"\n  {'Modality':>20s}  {'Edges':>8s}  {'Mean syn':>10s}  {'Median':>8s}  {'Max':>6s}  {'Std':>8s}  {'Total syn':>12s}")
for mod_name, edges in modality_dn_edges.items():
    syn = edges["Connectivity"]
    print(f"  {mod_name:>20s}  {len(syn):>8,}  {syn.mean():>10.2f}  {syn.median():>8.1f}  {syn.max():>6d}  {syn.std():>8.2f}  {syn.sum():>12,}")

print(f"\n  Per subchannel:")
print(f"  {'Subchannel':>20s}  {'Edges':>8s}  {'Mean syn':>10s}  {'Median':>8s}  {'Max':>6s}  {'Std':>8s}  {'Total syn':>12s}")
for ch in somatosensory_channels + visual_channels + olfactory_channels:
    edges = subchannel_dn_edges[ch]
    if len(edges) == 0:
        print(f"  {ch:>20s}  {'0':>8s}  {'N/A':>10s}  {'N/A':>8s}  {'N/A':>6s}  {'N/A':>8s}  {'0':>12s}")
        continue
    syn = edges["Connectivity"]
    print(f"  {ch:>20s}  {len(syn):>8,}  {syn.mean():>10.2f}  {syn.median():>8.1f}  {syn.max():>6d}  {syn.std():>8.2f}  {syn.sum():>12,}")

# Distribution of excitatory vs inhibitory
print(f"\n  Excitatory fraction per modality (Excitatory column):")
for mod_name, edges in modality_dn_edges.items():
    exc_frac = (edges["Excitatory"] == 1).mean()
    print(f"    {mod_name}: {exc_frac:.3f} excitatory ({(edges['Excitatory']==1).sum()}/{len(edges)})")

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS D: Neuropil distribution
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS D: Neuropil distribution (using neuron hemilineage as proxy)")
print("=" * 80)
print("  (No per-synapse neuropil in connectivity; using presynaptic neuron's")
print("   ito_lee_hemilineage from annotations as anatomical region proxy.)")

# Map presynaptic neuron -> hemilineage
ann_hemilineage = dict(zip(ann["root_id"], ann["ito_lee_hemilineage"]))
ann_superclass = dict(zip(ann["root_id"], ann["super_class"]))

for mod_name, edges in modality_dn_edges.items():
    edges_copy = edges.copy()
    edges_copy["hemilineage"] = edges_copy["Presynaptic_ID"].map(ann_hemilineage)
    hl_counts = edges_copy.groupby("hemilineage")["Connectivity"].sum().sort_values(ascending=False)
    print(f"\n  {mod_name} -> DN synapses by presynaptic hemilineage (top 10):")
    for hl, cnt in hl_counts.head(10).items():
        pct = cnt / hl_counts.sum() * 100
        print(f"    {str(hl):35s}  {cnt:>8,} syn  ({pct:5.1f}%)")
    # How many have no hemilineage
    na_count = edges_copy["hemilineage"].isna().sum()
    print(f"    (no hemilineage annotation: {na_count}/{len(edges_copy)} edges)")

# Also break down by presynaptic neuron's super_class
print("\n  Presynaptic neuron super_class distribution per modality:")
for mod_name, edges in modality_dn_edges.items():
    edges_copy = edges.copy()
    edges_copy["super_class"] = edges_copy["Presynaptic_ID"].map(ann_superclass)
    sc_counts = edges_copy.groupby("super_class")["Connectivity"].sum().sort_values(ascending=False)
    print(f"\n  {mod_name}:")
    for sc, cnt in sc_counts.items():
        pct = cnt / sc_counts.sum() * 100
        print(f"    {str(sc):25s}  {cnt:>8,} syn  ({pct:5.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS E: Interneuron layer characterization (2-hop)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS E: Interneuron layer characterization (2-hop)")
print("=" * 80)

# 2-hop: sensory -> intermediate -> DN
# For each modality, find all neurons directly downstream of sensory (1-hop from sensory)
# Then among those, which ones project to readout DNs?

print("\nFinding intermediate neurons (sensory -> interneuron -> DN)...")
intermediate_sets = {}
mod_names_list = ["somatosensory", "visual", "olfactory"]
for mod_name in mod_names_list:
    mod_neurons = modalities[mod_name]
    # Step 1: find all direct downstream of sensory neurons
    sensory_downstream = conn[conn["Presynaptic_ID"].isin(mod_neurons)]
    hop1_targets = set(sensory_downstream["Postsynaptic_ID"].unique())

    # Exclude sensory neurons themselves and readout DNs from intermediates
    intermediates = hop1_targets - mod_neurons - readout_rootids

    # Step 2: which of these intermediates project to readout DNs?
    inter_to_dn = dn_edges[dn_edges["Presynaptic_ID"].isin(intermediates)]
    active_intermediates = set(inter_to_dn["Presynaptic_ID"].unique())
    dns_via_2hop = set(inter_to_dn["Postsynaptic_ID"].unique())

    print(f"\n  {mod_name}:")
    print(f"    Total 1-hop downstream targets: {len(hop1_targets):,}")
    print(f"    Intermediates (excl sensory & DN): {len(intermediates):,}")
    print(f"    Active intermediates (also project to DN): {len(active_intermediates):,}")
    print(f"    DNs reached via 2-hop: {len(dns_via_2hop)}")
    intermediate_sets[mod_name] = active_intermediates

# Cross-modality intermediate sharing
print("\n  Cross-modality intermediate sharing:")
mod_names = ["somatosensory", "visual", "olfactory"]
for m1, m2 in combinations(mod_names, 2):
    i1 = intermediate_sets[m1]
    i2 = intermediate_sets[m2]
    shared = i1 & i2
    union = i1 | i2
    jaccard = len(shared) / len(union) if union else 0
    print(f"    {m1} & {m2}: {len(shared)} shared intermediates "
          f"(Jaccard={jaccard:.3f}, |union|={len(union):,})")

all_inter = set()
for mn in mod_names:
    all_inter |= intermediate_sets[mn]
shared_all = intermediate_sets["somatosensory"] & intermediate_sets["visual"] & intermediate_sets["olfactory"]
print(f"    All three share: {len(shared_all)} intermediates")
print(f"    Total unique intermediates across all modalities: {len(all_inter):,}")

# Characterize shared intermediates
if shared_all:
    shared_flow = [ann_superclass.get(n, "unknown") for n in shared_all]
    from collections import Counter
    flow_counts = Counter(shared_flow)
    print(f"\n    Super_class of all-three-shared intermediates:")
    for cls, cnt in flow_counts.most_common(10):
        print(f"      {str(cls):25s}  {cnt}")

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS F: Information flow asymmetry
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS F: Information flow asymmetry per decoder group")
print("=" * 80)

print(f"\n  Total syn_count from each modality into each decoder group,")
print(f"  normalized by group size (= per-neuron drive).")

# Build a table: modality x group
group_list = ["forward", "turn_left", "turn_right", "rhythm", "stance"]
print(f"\n  {'':>20s}", end="")
for g in group_list:
    print(f"  {g:>12s}", end="")
print(f"  {'TOTAL':>12s}")
print(f"  {'':>20s}", end="")
for g in group_list:
    print(f"  {'(n=' + str(len(group_names[g])) + ')':>12s}", end="")
print()

# Per modality
for mod_name in ["somatosensory", "visual", "olfactory"]:
    edges = modality_dn_edges[mod_name]
    print(f"\n  {mod_name} (total syn):")
    print(f"  {'':>20s}", end="")
    total_syn = 0
    for g in group_list:
        g_edges = edges[edges["Postsynaptic_ID"].isin(group_names[g])]
        s = g_edges["Connectivity"].sum()
        total_syn += s
        print(f"  {s:>12,}", end="")
    print(f"  {total_syn:>12,}")

    print(f"  {'per-neuron drive':>20s}", end="")
    for g in group_list:
        g_edges = edges[edges["Postsynaptic_ID"].isin(group_names[g])]
        s = g_edges["Connectivity"].sum()
        per_neuron = s / len(group_names[g]) if group_names[g] else 0
        print(f"  {per_neuron:>12.1f}", end="")
    print()

    # Also: fraction of group neurons reached
    print(f"  {'coverage (%)':>20s}", end="")
    for g in group_list:
        g_edges = edges[edges["Postsynaptic_ID"].isin(group_names[g])]
        reached = g_edges["Postsynaptic_ID"].nunique()
        coverage = reached / len(group_names[g]) * 100 if group_names[g] else 0
        print(f"  {coverage:>11.1f}%", end="")
    print()

# Which modality has strongest per-neuron drive per group?
print(f"\n  Strongest per-neuron drive per group:")
for g in group_list:
    best_mod = None
    best_drive = 0
    for mod_name in ["somatosensory", "visual", "olfactory"]:
        edges = modality_dn_edges[mod_name]
        g_edges = edges[edges["Postsynaptic_ID"].isin(group_names[g])]
        drive = g_edges["Connectivity"].sum() / len(group_names[g]) if group_names[g] else 0
        if drive > best_drive:
            best_drive = drive
            best_mod = mod_name
    print(f"    {g:>12s}: {best_mod} ({best_drive:.1f} syn/neuron)")

# Additional: per-subchannel drive into each group
print(f"\n  Per-subchannel per-neuron drive (top 3 per group):")
all_subchannels = somatosensory_channels + visual_channels + olfactory_channels
for g in group_list:
    drives = []
    for ch in all_subchannels:
        edges = subchannel_dn_edges[ch]
        g_edges = edges[edges["Postsynaptic_ID"].isin(group_names[g])]
        drive = g_edges["Connectivity"].sum() / len(group_names[g]) if group_names[g] else 0
        drives.append((ch, drive))
    drives.sort(key=lambda x: -x[1])
    top3 = drives[:3]
    top_str = ", ".join(f"{ch}={d:.1f}" for ch, d in top3)
    print(f"    {g:>12s}: {top_str}")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Modality-exclusive DNs
vis_only = vis_dns - som_dns - olf_dns
som_only = som_dns - vis_dns - olf_dns
olf_only = olf_dns - vis_dns - som_dns
tri_modal = vis_dns & som_dns & olf_dns

print(f"\n  DN segregation at 1-hop:")
print(f"    Visual-only DNs:         {len(vis_only)}")
print(f"    Somatosensory-only DNs:  {len(som_only)}")
print(f"    Olfactory-only DNs:      {len(olf_only)}")
print(f"    Visual+Somato shared:    {len(shared_vs)}")
print(f"    Visual+Olfactory shared: {len(vis_dns & olf_dns - som_dns)}")
print(f"    Somato+Olfactory shared: {len(som_dns & olf_dns - vis_dns)}")
print(f"    Tri-modal (all three):   {len(tri_modal)}")
print(f"    No modality input:       {len(readout_rootids - vis_dns - som_dns - olf_dns)}")
print(f"    Total readout DNs:       {len(readout_rootids)}")

segregation_label = "highly segregated" if len(shared_vs) < 0.1 * len(readout_rootids) else "moderately overlapping"
print(f"\n  Key finding: Sensory modalities are {segregation_label} at the DN level.")
convergence_pct = len(vis_dns & som_dns) / len(vis_dns | som_dns) * 100 if vis_dns | som_dns else 0
print(f"  Visual-somatosensory Jaccard: {len(vis_dns & som_dns)}/{len(vis_dns | som_dns)} = {convergence_pct:.1f}%")

print("\nDone.")
