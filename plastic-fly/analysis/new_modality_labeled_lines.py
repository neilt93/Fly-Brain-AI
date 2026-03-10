"""
Auditory, thermosensory, and hygrosensory labeled line analysis.

Tests whether these additional sensory modalities maintain segregated
pathways through the connectome to descending neurons (DNs), extending
the existing somatosensory/visual/olfactory segregation analysis.

Analyses:
  1. Population census from FlyWire annotations
  2. 1-hop connectivity: new modality neurons -> readout DNs
  3. Cross-modality Jaccard overlap (all 6 modalities)
  4. 2-hop interneuron pool analysis
  5. Per-decoder-group drive from new modalities
  6. Shuffled control for new modalities
"""

import json
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parent.parent.parent)  # up to "Connectome Fly Brain"
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

comp = pd.read_csv(COMPLETENESS)
root_ids_array = comp["Unnamed: 0"].values
valid_set = set(root_ids_array)

raw_ids = np.load(READOUT_IDS)
small_mask = raw_ids < 1_000_000
readout_rootids = set()
readout_rootids.update(raw_ids[~small_mask].tolist())
readout_rootids.update(root_ids_array[raw_ids[small_mask]].tolist())
print(f"  Readout DNs: {len(readout_rootids)}")

print("Loading connectivity (15M rows)...")
conn = pd.read_parquet(CONNECTIVITY)
print(f"  Connectivity: {conn.shape[0]:,} edges")

print("Loading annotations...")
ann = pd.read_csv(ANNOTATIONS, low_memory=False)
print(f"  Annotations: {ann.shape[0]:,} neurons")

# ── Extract new modality neuron populations ─────────────────────────────────
print("\n" + "=" * 80)
print("1. NEW MODALITY POPULATION CENSUS")
print("=" * 80)

# Auditory (JO neurons)
aud_mask = (ann["cell_class"] == "mechanosensory") & (ann["cell_sub_class"] == "auditory")
auditory_all = set(ann[aud_mask]["root_id"].values)
auditory_valid = auditory_all & valid_set
print(f"\n  Auditory (Johnston's Organ):")
print(f"    Total in FlyWire: {len(auditory_all)}")
print(f"    Valid in connectome: {len(auditory_valid)}")

# Split bilateral
aud_ann = ann[aud_mask & ann["root_id"].isin(valid_set)]
aud_left = set(aud_ann[aud_ann["side"] == "left"]["root_id"].values)
aud_right = set(aud_ann[aud_ann["side"] == "right"]["root_id"].values)
aud_center = auditory_valid - aud_left - aud_right
print(f"    Left: {len(aud_left)}, Right: {len(aud_right)}, Center/unknown: {len(aud_center)}")

# Thermosensory
thermo_mask = ann["cell_class"] == "thermosensory"
thermo_all = set(ann[thermo_mask]["root_id"].values)
thermo_valid = thermo_all & valid_set
print(f"\n  Thermosensory:")
print(f"    Total in FlyWire: {len(thermo_all)}")
print(f"    Valid in connectome: {len(thermo_valid)}")
thermo_ann = ann[thermo_mask & ann["root_id"].isin(valid_set)]
for sub in thermo_ann["cell_sub_class"].unique():
    n = (thermo_ann["cell_sub_class"] == sub).sum()
    types = thermo_ann[thermo_ann["cell_sub_class"] == sub]["cell_type"].value_counts()
    print(f"    {sub}: {n} neurons --{dict(types)}")

# Hygrosensory
hygro_mask = ann["cell_class"] == "hygrosensory"
hygro_all = set(ann[hygro_mask]["root_id"].values)
hygro_valid = hygro_all & valid_set
print(f"\n  Hygrosensory:")
print(f"    Total in FlyWire: {len(hygro_all)}")
print(f"    Valid in connectome: {len(hygro_valid)}")
hygro_ann = ann[hygro_mask & ann["root_id"].isin(valid_set)]
for sub in hygro_ann["cell_sub_class"].unique():
    n = (hygro_ann["cell_sub_class"] == sub).sum()
    types = hygro_ann[hygro_ann["cell_sub_class"] == sub]["cell_type"].value_counts()
    print(f"    {sub}: {n} neurons --{dict(types)}")

# ── Build full modality dict (existing + new) ──────────────────────────────
somatosensory_channels = ["proprioceptive", "mechanosensory", "vestibular", "gustatory"]
visual_channels = ["visual_left", "visual_right", "lplc2_left", "lplc2_right"]
olfactory_channels = ["olfactory_left", "olfactory_right"]

def get_modality_neurons(channels):
    s = set()
    for ch in channels:
        s.update(channel_map[ch])
    return s

modalities = {
    "somatosensory": get_modality_neurons(somatosensory_channels),
    "visual": get_modality_neurons(visual_channels),
    "olfactory": get_modality_neurons(olfactory_channels),
    "auditory": auditory_valid,
    "thermosensory": thermo_valid,
    "hygrosensory": hygro_valid,
}

print(f"\n  All modality sizes:")
for name, neurons in modalities.items():
    print(f"    {name}: {len(neurons)} neurons")

# ==============================================================================
# 2. 1-HOP CONNECTIVITY: new modalities -> readout DNs
# ==============================================================================
print("\n" + "=" * 80)
print("2. 1-HOP CONNECTIVITY TO READOUT DNs")
print("=" * 80)

dn_edges = conn[conn["Postsynaptic_ID"].isin(readout_rootids)].copy()
print(f"  Total edges terminating on readout DNs: {dn_edges.shape[0]:,}")

modality_dn_edges = {}
modality_dns = {}
for mod_name, mod_neurons in modalities.items():
    edges = dn_edges[dn_edges["Presynaptic_ID"].isin(mod_neurons)]
    modality_dn_edges[mod_name] = edges
    dns_reached = set(edges["Postsynaptic_ID"].unique())
    modality_dns[mod_name] = dns_reached
    total_syn = edges["Connectivity"].sum() if len(edges) > 0 else 0
    print(f"  {mod_name:>16s} -> DN: {edges.shape[0]:>6,} edges, "
          f"{len(dns_reached):>4d} DNs reached, {total_syn:>8,} total synapses")

# ==============================================================================
# 3. CROSS-MODALITY JACCARD OVERLAP (all 6 modalities)
# ==============================================================================
print("\n" + "=" * 80)
print("3. CROSS-MODALITY JACCARD OVERLAP AT 1-HOP")
print("=" * 80)

mod_names = list(modalities.keys())

# Full matrix
print(f"\n  {'':>16s}", end="")
for m in mod_names:
    print(f"  {m[:10]:>10s}", end="")
print()

for m1 in mod_names:
    print(f"  {m1:>16s}", end="")
    for m2 in mod_names:
        d1, d2 = modality_dns[m1], modality_dns[m2]
        if len(d1 | d2) == 0:
            j = 0.0
        else:
            j = len(d1 & d2) / len(d1 | d2)
        print(f"  {j:>10.3f}", end="")
    print()

# Pairwise details
print(f"\n  Pairwise overlap details:")
print(f"  {'Pair':>35s}  {'Shared':>7s}  {'Union':>7s}  {'Jaccard':>8s}")
for m1, m2 in combinations(mod_names, 2):
    d1, d2 = modality_dns[m1], modality_dns[m2]
    shared = d1 & d2
    union = d1 | d2
    j = len(shared) / len(union) if union else 0
    print(f"  {m1+' & '+m2:>35s}  {len(shared):>7d}  {len(union):>7d}  {j:>8.3f}")

# ==============================================================================
# 4. 2-HOP INTERNEURON POOL ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("4. 2-HOP INTERNEURON POOL ANALYSIS")
print("=" * 80)

intermediate_sets = {}
for mod_name in mod_names:
    mod_neurons = modalities[mod_name]
    if len(mod_neurons) == 0:
        intermediate_sets[mod_name] = set()
        continue

    # Step 1: all direct downstream of sensory
    downstream = conn[conn["Presynaptic_ID"].isin(mod_neurons)]
    hop1_targets = set(downstream["Postsynaptic_ID"].unique())

    # Exclude self and readout DNs
    intermediates = hop1_targets - mod_neurons - readout_rootids

    # Step 2: which project to readout DNs?
    inter_to_dn = dn_edges[dn_edges["Presynaptic_ID"].isin(intermediates)]
    active_intermediates = set(inter_to_dn["Presynaptic_ID"].unique())
    dns_via_2hop = set(inter_to_dn["Postsynaptic_ID"].unique())

    intermediate_sets[mod_name] = active_intermediates
    print(f"\n  {mod_name:>16s}:")
    print(f"    1-hop downstream: {len(hop1_targets):,}")
    print(f"    Active intermediates (-> DN): {len(active_intermediates):,}")
    print(f"    DNs reachable via 2-hop: {len(dns_via_2hop)}")

print(f"\n  Cross-modality intermediate pool overlap:")
print(f"  {'Pair':>35s}  {'Shared':>7s}  {'Union':>7s}  {'Jaccard':>8s}")
for m1, m2 in combinations(mod_names, 2):
    i1, i2 = intermediate_sets[m1], intermediate_sets[m2]
    shared = i1 & i2
    union = i1 | i2
    j = len(shared) / len(union) if union else 0
    print(f"  {m1+' & '+m2:>35s}  {len(shared):>7d}  {len(union):>7d}  {j:>8.3f}")

# ==============================================================================
# 5. PER-DECODER-GROUP DRIVE FROM NEW MODALITIES
# ==============================================================================
print("\n" + "=" * 80)
print("5. PER-DECODER-GROUP DRIVE (syn/neuron) FROM EACH MODALITY")
print("=" * 80)

group_names = {}
for k, v in decoder_groups.items():
    name = k.replace("_ids", "")
    group_names[name] = set(v)

group_list = ["forward", "turn_left", "turn_right", "rhythm", "stance"]

print(f"\n  {'Modality':>16s}", end="")
for g in group_list:
    print(f"  {g:>12s}", end="")
print(f"  {'TOTAL':>12s}")

for mod_name in mod_names:
    edges = modality_dn_edges[mod_name]
    print(f"  {mod_name:>16s}", end="")
    total = 0
    for g in group_list:
        g_edges = edges[edges["Postsynaptic_ID"].isin(group_names[g])]
        drive = g_edges["Connectivity"].sum() / len(group_names[g]) if group_names[g] else 0
        total += g_edges["Connectivity"].sum()
        print(f"  {drive:>12.1f}", end="")
    print(f"  {total:>12,}")

# Which modality dominates each group?
print(f"\n  Dominant modality per group (by syn/neuron):")
for g in group_list:
    drives = []
    for mod_name in mod_names:
        edges = modality_dn_edges[mod_name]
        g_edges = edges[edges["Postsynaptic_ID"].isin(group_names[g])]
        drive = g_edges["Connectivity"].sum() / len(group_names[g]) if group_names[g] else 0
        drives.append((mod_name, drive))
    drives.sort(key=lambda x: -x[1])
    top3 = ", ".join(f"{m}={d:.1f}" for m, d in drives[:3] if d > 0)
    print(f"    {g:>12s}: {top3}")

# ==============================================================================
# 6. SHUFFLED CONTROL: Is segregation specific to real wiring?
# ==============================================================================
print("\n" + "=" * 80)
print("6. SHUFFLED CONNECTOME CONTROL (5 permutations)")
print("=" * 80)

n_perms = 5
np.random.seed(42)

# Focus on new modalities + key comparisons
focus_pairs = [
    ("auditory", "visual"),
    ("auditory", "somatosensory"),
    ("auditory", "olfactory"),
    ("thermosensory", "somatosensory"),
    ("thermosensory", "visual"),
    ("hygrosensory", "somatosensory"),
    ("hygrosensory", "thermosensory"),
]

# Real Jaccard
real_jaccards = {}
for m1, m2 in focus_pairs:
    d1, d2 = modality_dns[m1], modality_dns[m2]
    union = d1 | d2
    j = len(d1 & d2) / len(union) if union else 0
    real_jaccards[(m1, m2)] = j

# Shuffled: permute postsynaptic targets
post_ids = dn_edges["Postsynaptic_ID"].values.copy()
shuffled_jaccards = {pair: [] for pair in focus_pairs}

for perm in range(n_perms):
    shuffled_post = post_ids.copy()
    np.random.shuffle(shuffled_post)
    dn_edges_shuffled = dn_edges.copy()
    dn_edges_shuffled["Postsynaptic_ID"] = shuffled_post

    # Compute DN sets for each modality
    shuf_dns = {}
    for mod_name, mod_neurons in modalities.items():
        edges = dn_edges_shuffled[dn_edges_shuffled["Presynaptic_ID"].isin(mod_neurons)]
        shuf_dns[mod_name] = set(edges["Postsynaptic_ID"].unique())

    for m1, m2 in focus_pairs:
        d1, d2 = shuf_dns[m1], shuf_dns[m2]
        union = d1 | d2
        j = len(d1 & d2) / len(union) if union else 0
        shuffled_jaccards[(m1, m2)].append(j)

print(f"\n  {'Pair':>40s}  {'Real':>8s}  {'Shuffled':>10s}  {'Ratio':>8s}")
for pair in focus_pairs:
    real_j = real_jaccards[pair]
    mean_shuf = np.mean(shuffled_jaccards[pair])
    ratio = mean_shuf / real_j if real_j > 0 else float('inf')
    print(f"  {pair[0]+' & '+pair[1]:>40s}  {real_j:>8.3f}  {mean_shuf:>10.3f}  {ratio:>7.1f}x")

# ==============================================================================
# 7. CHARACTERIZE AUDITORY DN TARGETS
# ==============================================================================
print("\n" + "=" * 80)
print("7. AUDITORY DN TARGETS --DETAILED CHARACTERIZATION")
print("=" * 80)

aud_dn_ids = modality_dns["auditory"]
print(f"\n  Auditory reaches {len(aud_dn_ids)} DNs at 1-hop")

if len(aud_dn_ids) > 0:
    dn_to_group = {}
    for gname, gids in group_names.items():
        for gid in gids:
            dn_to_group[gid] = gname

    aud_edges = modality_dn_edges["auditory"]

    print(f"\n  {'DN root_id':>22s}  {'Group':>12s}  {'Edges':>6s}  {'Syn':>6s}  {'cell_type':>20s}  {'flow':>12s}")
    for dn_id in sorted(aud_dn_ids):
        dn_e = aud_edges[aud_edges["Postsynaptic_ID"] == dn_id]
        n_edges = len(dn_e)
        n_syn = dn_e["Connectivity"].sum()
        dg = dn_to_group.get(dn_id, "not_readout")
        ann_row = ann[ann["root_id"] == dn_id]
        ct = str(ann_row["cell_type"].values[0]) if len(ann_row) > 0 else "N/A"
        fl = str(ann_row["flow"].values[0]) if len(ann_row) > 0 else "N/A"
        print(f"  {dn_id:>22d}  {dg:>12s}  {n_edges:>6d}  {n_syn:>6d}  {ct:>20s}  {fl:>12s}")

    # Which other modalities share these DNs?
    print(f"\n  Overlap of auditory DNs with other modalities:")
    for mod_name in ["somatosensory", "visual", "olfactory", "thermosensory", "hygrosensory"]:
        shared = aud_dn_ids & modality_dns[mod_name]
        print(f"    auditory & {mod_name}: {len(shared)} shared DNs")

# ==============================================================================
# 8. CHARACTERIZE THERMO/HYGRO DN TARGETS
# ==============================================================================
print("\n" + "=" * 80)
print("8. THERMOSENSORY & HYGROSENSORY DN TARGETS")
print("=" * 80)

for mod_name in ["thermosensory", "hygrosensory"]:
    dns = modality_dns[mod_name]
    print(f"\n  {mod_name} reaches {len(dns)} DNs at 1-hop")
    if len(dns) > 0:
        edges = modality_dn_edges[mod_name]
        print(f"  {'DN root_id':>22s}  {'Group':>12s}  {'Edges':>6s}  {'Syn':>6s}  {'cell_type':>20s}")
        for dn_id in sorted(dns):
            dn_e = edges[edges["Postsynaptic_ID"] == dn_id]
            n_edges = len(dn_e)
            n_syn = dn_e["Connectivity"].sum()
            dg = dn_to_group.get(dn_id, "not_readout") if 'dn_to_group' in dir() else "?"
            ann_row = ann[ann["root_id"] == dn_id]
            ct = str(ann_row["cell_type"].values[0]) if len(ann_row) > 0 else "N/A"
            print(f"  {dn_id:>22d}  {dg:>12s}  {n_edges:>6d}  {n_syn:>6d}  {ct:>20s}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: LABELED LINE ANALYSIS FOR NEW MODALITIES")
print("=" * 80)

print(f"\n  Modality populations:")
for name, neurons in modalities.items():
    dns = modality_dns[name]
    edges = modality_dn_edges[name]
    total_syn = edges["Connectivity"].sum() if len(edges) > 0 else 0
    print(f"    {name:>16s}: {len(neurons):>4d} neurons -> {len(dns):>4d} DNs ({total_syn:>8,} syn)")

print(f"\n  Key findings:")
# Auditory segregation
aud_vis_j = real_jaccards.get(("auditory", "visual"), 0)
aud_som_j = real_jaccards.get(("auditory", "somatosensory"), 0)
print(f"    Auditory-Visual Jaccard:         {aud_vis_j:.3f}")
print(f"    Auditory-Somatosensory Jaccard:  {aud_som_j:.3f}")

# Thermo segregation
thermo_som_j = real_jaccards.get(("thermosensory", "somatosensory"), 0)
thermo_vis_j = real_jaccards.get(("thermosensory", "visual"), 0)
print(f"    Thermo-Somatosensory Jaccard:    {thermo_som_j:.3f}")
print(f"    Thermo-Visual Jaccard:           {thermo_vis_j:.3f}")

# Hygro-thermo relationship
hygro_thermo_j = real_jaccards.get(("hygrosensory", "thermosensory"), 0)
print(f"    Hygro-Thermo Jaccard:            {hygro_thermo_j:.3f}")

print("\nDone.")
