"""
Olfactory DN sparsity check vs Aymanns et al. 2022 (eLife).

Aymanns et al. found that only 2-4 DNs per fly (~2-5% of recorded population)
showed odor-specific activity. The rest encoded behavior regardless of odor context.

This script checks: in our FlyWire connectome, how many of our readout DNs
receive olfactory input at 1-hop, 2-hop, and 3-hop?

If the connectome predicts a similarly sparse olfactory->DN pathway, that is
consistent with the Aymanns finding (structural sparsity -> functional sparsity).
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from pathlib import Path

# -- Paths -------------------------------------------------------------------
BASE = str(Path(__file__).resolve().parent.parent.parent)  # up to "Connectome Fly Brain"
CHANNEL_MAP = f"{BASE}/plastic-fly/data/channel_map_v4_looming.json"
READOUT_IDS = f"{BASE}/plastic-fly/data/readout_ids_v4_looming.npy"
DECODER_GROUPS = f"{BASE}/plastic-fly/data/decoder_groups_v4_looming.json"
CONNECTIVITY = f"{BASE}/brain-model/Connectivity_783.parquet"
COMPLETENESS = f"{BASE}/brain-model/Completeness_783.csv"

# -- Load data ---------------------------------------------------------------
print("=" * 80)
print("OLFACTORY DN SPARSITY CHECK vs AYMANNS et al. 2022")
print("=" * 80)

print("\nLoading data...")
t0 = time.time()

with open(CHANNEL_MAP) as f:
    channel_map = json.load(f)
with open(DECODER_GROUPS) as f:
    decoder_groups = json.load(f)

comp = pd.read_csv(COMPLETENESS)
root_ids_array = comp["Unnamed: 0"].values
valid_set = set(root_ids_array)

# Resolve readout IDs to root IDs
raw_ids = np.load(READOUT_IDS)
small_mask = raw_ids < 1_000_000
readout_rootids = set()
readout_rootids.update(raw_ids[~small_mask].tolist())
readout_rootids.update(root_ids_array[raw_ids[small_mask]].tolist())

print(f"Loading connectivity (15M rows)...")
conn = pd.read_parquet(CONNECTIVITY)

print(f"  Loaded in {time.time() - t0:.1f}s")

# -- Extract olfactory neuron IDs --------------------------------------------
olfactory_ids = set(channel_map["olfactory_left"] + channel_map["olfactory_right"])
print(f"\n  Olfactory sensory neurons (ORNs): {len(olfactory_ids)}")
print(f"  Readout DNs: {len(readout_rootids)}")
print(f"  Connectome edges: {conn.shape[0]:,}")

# -- Build adjacency for efficient multi-hop BFS -----------------------------
print("\nBuilding adjacency map...")
t1 = time.time()

# Build forward adjacency: pre -> set(post)
adj_forward = defaultdict(set)
pre_col = conn["Presynaptic_ID"].values
post_col = conn["Postsynaptic_ID"].values
syn_col = conn["Connectivity"].values

for i in range(len(pre_col)):
    adj_forward[pre_col[i]].add(post_col[i])

print(f"  Adjacency built in {time.time() - t1:.1f}s ({len(adj_forward):,} source neurons)")

# -- Decoder group membership ------------------------------------------------
group_names = {}
for k, v in decoder_groups.items():
    name = k.replace("_ids", "")
    group_names[name] = set(v)

all_readout = set()
for g in group_names.values():
    all_readout.update(g)
print(f"  Total readout DNs (from decoder groups): {len(all_readout)}")

# -- 1-HOP: Direct olfactory -> readout DN ------------------------------------
print("\n" + "=" * 80)
print("1-HOP: DIRECT OLFACTORY -> READOUT DN")
print("=" * 80)

hop1_targets = set()
for orn in olfactory_ids:
    hop1_targets.update(adj_forward.get(orn, set()))

hop1_dns = hop1_targets & all_readout
hop1_pct = 100 * len(hop1_dns) / len(all_readout) if all_readout else 0

print(f"\n  ORN direct downstream (all): {len(hop1_targets):,} neurons")
print(f"  Readout DNs reached at 1-hop: {len(hop1_dns)} / {len(all_readout)} ({hop1_pct:.1f}%)")

# Per-group breakdown
print(f"\n  Per-group breakdown:")
for gname in ["forward", "turn_left", "turn_right", "rhythm", "stance"]:
    gids = group_names[gname]
    reached = hop1_dns & gids
    pct = 100 * len(reached) / len(gids) if gids else 0
    print(f"    {gname:>12s}: {len(reached):>3d} / {len(gids):>3d} ({pct:.1f}%)")

# Synapse counts for 1-hop
dn_mask = conn["Postsynaptic_ID"].isin(all_readout) & conn["Presynaptic_ID"].isin(olfactory_ids)
olf_dn_edges = conn[dn_mask]
if len(olf_dn_edges) > 0:
    print(f"\n  Total olfactory->DN edges: {len(olf_dn_edges)}")
    print(f"  Total olfactory->DN synapses: {olf_dn_edges['Connectivity'].sum():,}")
    per_dn = olf_dn_edges.groupby("Postsynaptic_ID")["Connectivity"].sum()
    print(f"  Synapses per DN reached: mean={per_dn.mean():.1f}, max={per_dn.max()}")

# -- 2-HOP: Olfactory -> intermediary -> readout DN ----------------------------
print("\n" + "=" * 80)
print("2-HOP: OLFACTORY -> INTERMEDIARY -> READOUT DN")
print("=" * 80)

# hop1_targets already computed — these are all direct downstream of ORNs
# Remove ORNs and readout DNs from intermediary pool
intermediaries = hop1_targets - olfactory_ids - all_readout
print(f"\n  Intermediary neurons (1 hop from ORN, excluding ORN/DN): {len(intermediaries):,}")

hop2_targets = set()
for inter in intermediaries:
    hop2_targets.update(adj_forward.get(inter, set()))

hop2_dns = hop2_targets & all_readout
# Include 1-hop DNs too (reachable at <=2 hops)
hop2_dns_cumulative = hop1_dns | hop2_dns
hop2_pct = 100 * len(hop2_dns_cumulative) / len(all_readout) if all_readout else 0
hop2_only_pct = 100 * len(hop2_dns - hop1_dns) / len(all_readout) if all_readout else 0

print(f"  DNs reachable at exactly 2-hop (new): {len(hop2_dns - hop1_dns)}")
print(f"  DNs reachable at <=2-hop (cumulative): {len(hop2_dns_cumulative)} / {len(all_readout)} ({hop2_pct:.1f}%)")

# Per-group breakdown
print(f"\n  Per-group breakdown (cumulative <=2-hop):")
for gname in ["forward", "turn_left", "turn_right", "rhythm", "stance"]:
    gids = group_names[gname]
    reached = hop2_dns_cumulative & gids
    pct = 100 * len(reached) / len(gids) if gids else 0
    print(f"    {gname:>12s}: {len(reached):>3d} / {len(gids):>3d} ({pct:.1f}%)")

# -- 3-HOP: Olfactory -> inter1 -> inter2 -> readout DN ------------------------
print("\n" + "=" * 80)
print("3-HOP: OLFACTORY -> INTER1 -> INTER2 -> READOUT DN")
print("=" * 80)

# hop2_targets are all neurons reachable from intermediaries
# New intermediaries at hop 2 (excluding ORN, DN, and hop-1 intermediaries)
hop2_intermediaries = hop2_targets - olfactory_ids - all_readout - intermediaries
print(f"\n  Hop-2 intermediary neurons (new at distance 2): {len(hop2_intermediaries):,}")

hop3_targets = set()
for inter2 in hop2_intermediaries:
    hop3_targets.update(adj_forward.get(inter2, set()))

hop3_dns = hop3_targets & all_readout
hop3_dns_cumulative = hop1_dns | hop2_dns | hop3_dns
hop3_pct = 100 * len(hop3_dns_cumulative) / len(all_readout) if all_readout else 0
hop3_only = hop3_dns - hop2_dns_cumulative
hop3_only_pct = 100 * len(hop3_only) / len(all_readout) if all_readout else 0

print(f"  DNs reachable at exactly 3-hop (new): {len(hop3_only)}")
print(f"  DNs reachable at <=3-hop (cumulative): {len(hop3_dns_cumulative)} / {len(all_readout)} ({hop3_pct:.1f}%)")

# Per-group breakdown
print(f"\n  Per-group breakdown (cumulative <=3-hop):")
for gname in ["forward", "turn_left", "turn_right", "rhythm", "stance"]:
    gids = group_names[gname]
    reached = hop3_dns_cumulative & gids
    pct = 100 * len(reached) / len(gids) if gids else 0
    print(f"    {gname:>12s}: {len(reached):>3d} / {len(gids):>3d} ({pct:.1f}%)")

# -- COMPARISON WITH AYMANNS 2022 --------------------------------------------
print("\n" + "=" * 80)
print("COMPARISON WITH AYMANNS et al. 2022")
print("=" * 80)

print(f"""
  Aymanns et al. 2022 (eLife) finding:
    - Recorded from ~60-100 DNs in neck connective
    - Only 2-4 DNs per fly showed odor-specific responses
    - That's ~2-5% of the recorded DN population
    - The rest encoded behavior (walking, grooming) regardless of odor

  Our connectome analysis (FlyWire, {len(all_readout)} readout DNs):
    - 1-hop (direct ORN->DN):      {len(hop1_dns):>3d} / {len(all_readout)} = {100*len(hop1_dns)/len(all_readout):.1f}%
    - 2-hop (ORN->inter->DN):      {len(hop2_dns_cumulative):>3d} / {len(all_readout)} = {100*len(hop2_dns_cumulative)/len(all_readout):.1f}%
    - 3-hop (ORN->i1->i2->DN):     {len(hop3_dns_cumulative):>3d} / {len(all_readout)} = {100*len(hop3_dns_cumulative)/len(all_readout):.1f}%
""")

# Determine which hop count best matches Aymanns
aymanns_low, aymanns_high = 2.0, 5.0  # percent
results = {
    "1-hop": 100 * len(hop1_dns) / len(all_readout),
    "2-hop": 100 * len(hop2_dns_cumulative) / len(all_readout),
    "3-hop": 100 * len(hop3_dns_cumulative) / len(all_readout),
}

print(f"  Aymanns range: {aymanns_low}-{aymanns_high}% of DNs show odor responses")
print()
for hop, pct in results.items():
    if pct < aymanns_low:
        verdict = "BELOW Aymanns range (fewer DNs than expected)"
    elif pct <= aymanns_high:
        verdict = "WITHIN Aymanns range -- CONSISTENT"
    else:
        verdict = "ABOVE Aymanns range (more DNs than expected)"
    print(f"    {hop}: {pct:.1f}% -> {verdict}")

# -- SIGNAL DILUTION ANALYSIS ------------------------------------------------
print("\n" + "=" * 80)
print("SIGNAL DILUTION: OLFACTORY vs NON-OLFACTORY INPUT TO DNs")
print("=" * 80)

# For each DN reached at each hop, what fraction of its total input is olfactory?
print("\n  For 1-hop DNs: fraction of total synaptic input from olfactory sources")

if len(hop1_dns) > 0:
    for dn_id in sorted(hop1_dns):
        # Total input to this DN
        all_input = conn[conn["Postsynaptic_ID"] == dn_id]
        total_syn = all_input["Connectivity"].sum()
        # Olfactory input
        olf_input = all_input[all_input["Presynaptic_ID"].isin(olfactory_ids)]
        olf_syn = olf_input["Connectivity"].sum()
        frac = olf_syn / total_syn if total_syn > 0 else 0
        # Which group?
        dn_group = "none"
        for gname, gids in group_names.items():
            if dn_id in gids:
                dn_group = gname
                break
        print(f"    DN {dn_id}: olfactory={olf_syn}/{total_syn} ({100*frac:.2f}%) [{dn_group}]")

# -- SYNAPSE-WEIGHTED 2-HOP ANALYSIS ----------------------------------------
print("\n" + "=" * 80)
print("SYNAPSE-WEIGHTED 2-HOP ANALYSIS")
print("=" * 80)
print("\n  For 2-hop reachable DNs: estimating effective olfactory drive strength")
print("  (product of synapse weights along path, normalized)")

# Build weighted edges for the 2-hop paths
# ORN -> intermediary synapse weights
orn_to_inter = conn[conn["Presynaptic_ID"].isin(olfactory_ids) &
                     conn["Postsynaptic_ID"].isin(intermediaries)]
# intermediary -> DN synapse weights
inter_to_dn = conn[conn["Presynaptic_ID"].isin(intermediaries) &
                    conn["Postsynaptic_ID"].isin(all_readout)]

# For each DN, sum up path-product weights
dn_2hop_drive = defaultdict(float)
# Build intermediary lookup: inter_id -> total olfactory input synapses
inter_olf_input = defaultdict(float)
for _, row in orn_to_inter.iterrows():
    inter_olf_input[row["Postsynaptic_ID"]] += row["Connectivity"]

# For each inter->DN edge, multiply by olfactory input to that intermediary
for _, row in inter_to_dn.iterrows():
    inter_id = row["Presynaptic_ID"]
    dn_id = row["Postsynaptic_ID"]
    olf_in = inter_olf_input.get(inter_id, 0)
    if olf_in > 0:
        # Product of olfactory->inter weight and inter->DN weight
        dn_2hop_drive[dn_id] += olf_in * row["Connectivity"]

# Sort by drive and show top DNs
sorted_dns = sorted(dn_2hop_drive.items(), key=lambda x: -x[1])
max_drive = sorted_dns[0][1] if sorted_dns else 1

print(f"\n  2-hop DNs with olfactory drive (top 20 / {len(sorted_dns)} total):")
print(f"  {'Rank':>4s}  {'DN root_id':>22s}  {'Group':>12s}  {'Drive(norm)':>12s}  {'Drive(raw)':>12s}")
for rank, (dn_id, drive) in enumerate(sorted_dns[:20], 1):
    dn_group = "none"
    for gname, gids in group_names.items():
        if dn_id in gids:
            dn_group = gname
            break
    print(f"  {rank:>4d}  {dn_id:>22d}  {dn_group:>12s}  {drive/max_drive:>12.4f}  {drive:>12.0f}")

# How many DNs have >1% of max drive? >10%?
strong_1pct = sum(1 for _, d in sorted_dns if d > 0.01 * max_drive)
strong_10pct = sum(1 for _, d in sorted_dns if d > 0.10 * max_drive)
strong_1pct_pct = 100 * strong_1pct / len(all_readout)
strong_10pct_pct = 100 * strong_10pct / len(all_readout)

print(f"\n  DNs with >1% of max drive:  {strong_1pct} / {len(all_readout)} ({strong_1pct_pct:.1f}%)")
print(f"  DNs with >10% of max drive: {strong_10pct} / {len(all_readout)} ({strong_10pct_pct:.1f}%)")

# -- FINAL VERDICT -----------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

print(f"""
  STRUCTURAL SPARSITY SUMMARY:

  Hop distance    DNs reached    % of readout    Aymanns match?
  -------------   -----------    ------------    --------------
  1-hop (direct)  {len(hop1_dns):>11d}    {100*len(hop1_dns)/len(all_readout):>11.1f}%    {"YES" if aymanns_low <= 100*len(hop1_dns)/len(all_readout) <= aymanns_high else "NO"}
  <=2-hop         {len(hop2_dns_cumulative):>11d}    {100*len(hop2_dns_cumulative)/len(all_readout):>11.1f}%    {"YES" if aymanns_low <= 100*len(hop2_dns_cumulative)/len(all_readout) <= aymanns_high else "NO"}
  <=3-hop         {len(hop3_dns_cumulative):>11d}    {100*len(hop3_dns_cumulative)/len(all_readout):>11.1f}%    {"YES" if aymanns_low <= 100*len(hop3_dns_cumulative)/len(all_readout) <= aymanns_high else "NO"}

  Synapse-weighted (2-hop):
    DNs with >10% max drive: {strong_10pct} ({strong_10pct_pct:.1f}%)
    DNs with >1% max drive:  {strong_1pct} ({strong_1pct_pct:.1f}%)
""")

# Interpretation
best_match = None
for hop, pct in results.items():
    if aymanns_low <= pct <= aymanns_high:
        best_match = hop
        break

if best_match:
    print(f"  CONCLUSION: {best_match} connectivity best matches Aymanns 2-5% finding.")
    print(f"  This suggests that the structural wiring alone (at {best_match})")
    print(f"  can explain the functional sparsity observed electrophysiologically.")
else:
    # Find closest
    diffs = {hop: min(abs(pct - aymanns_low), abs(pct - aymanns_high))
             for hop, pct in results.items() if pct > 0}
    if diffs:
        closest = min(diffs, key=diffs.get)
        print(f"  CONCLUSION: No hop count falls exactly in the 2-5% range.")
        print(f"  Closest match is {closest} at {results[closest]:.1f}%.")

    # Check synapse-weighted
    if aymanns_low <= strong_10pct_pct <= aymanns_high:
        print(f"  However, synapse-weighted 2-hop drive (>10% threshold) = {strong_10pct_pct:.1f}%")
        print(f"  matches the Aymanns range, suggesting functional sparsity arises from")
        print(f"  signal dilution through weak multi-hop paths.")

print(f"\n  Total runtime: {time.time() - t0:.1f}s")
print("\nDone.")
