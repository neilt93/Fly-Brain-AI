"""
Synapse-weighted hop analysis for ALL 6 sensory modalities.

Extends the olfactory sparsity analysis (which matched Aymanns 2022 at 2.0%)
to the full set of sensory modalities:
  - Somatosensory (75 neurons: gustatory + proprioceptive + mechanosensory + vestibular)
  - Visual (310 neurons: LPLC2 + R7/R8)
  - Olfactory (100 neurons: ORNs)
  - Auditory (390 JO neurons)
  - Thermosensory (29 neurons)
  - Hygrosensory (74 neurons)

For each modality, computes:
  1-hop:  direct sensory -> DN connections
  2-hop topology:  sensory -> intermediary -> DN (any path exists)
  2-hop weighted:  sensory -> intermediary -> DN with >10% of max weighted drive
    Weight = sum over all 2-hop paths of (synapse_count_leg1 * synapse_count_leg2)
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from pathlib import Path


def _write_json_atomic(path, obj, **kwargs):
    """Write JSON atomically: write to tmp file then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, **kwargs)
    tmp.replace(path)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parent.parent.parent)  # up to "Connectome Fly Brain"
CHANNEL_MAP   = f"{BASE}/plastic-fly/data/channel_map_v4_looming.json"
READOUT_IDS   = f"{BASE}/plastic-fly/data/readout_ids_v4_looming.npy"
DECODER_GROUPS = f"{BASE}/plastic-fly/data/decoder_groups_v4_looming.json"
CONNECTIVITY  = f"{BASE}/brain-model/Connectivity_783.parquet"
COMPLETENESS  = f"{BASE}/brain-model/Completeness_783.csv"
ANNOTATIONS   = f"{BASE}/brain-model/flywire_annotations_matched.csv"

# ── Load data ────────────────────────────────────────────────────────────────
print("=" * 80)
print("ALL-MODALITY SYNAPSE-WEIGHTED HOP ANALYSIS")
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

# Also collect all readout from decoder groups (union)
all_readout = set()
group_names = {}
for k, v in decoder_groups.items():
    name = k.replace("_ids", "")
    gids = set(v)
    group_names[name] = gids
    all_readout.update(gids)

print(f"  Readout DNs (from decoder groups): {len(all_readout)}")

print("Loading connectivity (15M rows)...")
conn = pd.read_parquet(CONNECTIVITY)
print(f"  Connectivity: {conn.shape[0]:,} edges")

print("Loading annotations...")
ann = pd.read_csv(ANNOTATIONS, low_memory=False)
print(f"  Annotations: {ann.shape[0]:,} neurons")

print(f"  Loaded in {time.time() - t0:.1f}s")

# ── Build modality populations ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODALITY POPULATIONS")
print("=" * 80)

def get_channel_neurons(channels):
    s = set()
    for ch in channels:
        s.update(channel_map[ch])
    return s

# Somatosensory: 4 bridge channels
somatosensory = get_channel_neurons(["gustatory", "proprioceptive",
                                      "mechanosensory", "vestibular"])

# Visual: R7/R8 + LPLC2
visual = get_channel_neurons(["visual_left", "visual_right",
                               "lplc2_left", "lplc2_right"])

# Olfactory: bilateral ORNs
olfactory = get_channel_neurons(["olfactory_left", "olfactory_right"])

# Auditory: JO neurons from annotations
aud_mask = ((ann["cell_class"] == "mechanosensory") &
            (ann["cell_sub_class"] == "auditory"))
auditory = set(ann[aud_mask]["root_id"].values) & valid_set

# Thermosensory
thermo_mask = ann["cell_class"] == "thermosensory"
thermosensory = set(ann[thermo_mask]["root_id"].values) & valid_set

# Hygrosensory
hygro_mask = ann["cell_class"] == "hygrosensory"
hygrosensory = set(ann[hygro_mask]["root_id"].values) & valid_set

modalities = {
    "Somatosensory": somatosensory,
    "Visual":        visual,
    "Olfactory":     olfactory,
    "Auditory":      auditory,
    "Thermosensory": thermosensory,
    "Hygrosensory":  hygrosensory,
}

for name, neurons in modalities.items():
    print(f"  {name:>16s}: {len(neurons):>4d} neurons")
print(f"  {'Readout DNs':>16s}: {len(all_readout):>4d}")

# ── Build adjacency structures ───────────────────────────────────────────────
print("\nBuilding adjacency and weight maps...")
t1 = time.time()

pre_col  = conn["Presynaptic_ID"].values
post_col = conn["Postsynaptic_ID"].values
syn_col  = conn["Connectivity"].values

# Forward adjacency: pre -> set(post)
adj_forward = defaultdict(set)
# Weighted edges: (pre, post) -> synapse count
edge_weight = defaultdict(float)

for i in range(len(pre_col)):
    adj_forward[pre_col[i]].add(post_col[i])
    edge_weight[(pre_col[i], post_col[i])] += syn_col[i]

print(f"  Adjacency built in {time.time() - t1:.1f}s "
      f"({len(adj_forward):,} source neurons, {len(edge_weight):,} edges)")

# ── Per-modality analysis ────────────────────────────────────────────────────
results = []

for mod_name, mod_neurons in modalities.items():
    print(f"\n{'=' * 80}")
    print(f"MODALITY: {mod_name} ({len(mod_neurons)} neurons)")
    print(f"{'=' * 80}")

    # ── 1-HOP: direct sensory -> readout DN ──────────────────────────────
    hop1_targets = set()
    for nid in mod_neurons:
        hop1_targets.update(adj_forward.get(nid, set()))

    hop1_dns = hop1_targets & all_readout
    hop1_pct = 100 * len(hop1_dns) / len(all_readout)

    print(f"\n  1-hop (direct):")
    print(f"    All downstream: {len(hop1_targets):,}")
    print(f"    Readout DNs reached: {len(hop1_dns)} / {len(all_readout)} ({hop1_pct:.1f}%)")

    # ── 2-HOP topology: sensory -> intermediary -> DN ────────────────────
    intermediaries = hop1_targets - mod_neurons - all_readout

    hop2_targets = set()
    for inter in intermediaries:
        hop2_targets.update(adj_forward.get(inter, set()))

    hop2_dns = hop2_targets & all_readout
    hop2_dns_cumulative = hop1_dns | hop2_dns
    hop2_topo_pct = 100 * len(hop2_dns_cumulative) / len(all_readout)

    print(f"\n  2-hop topology (cumulative <=2-hop):")
    print(f"    Intermediary neurons: {len(intermediaries):,}")
    print(f"    DNs reached (new at 2-hop): {len(hop2_dns - hop1_dns)}")
    print(f"    DNs reached (cumulative): {len(hop2_dns_cumulative)} / {len(all_readout)} ({hop2_topo_pct:.1f}%)")

    # ── 2-HOP weighted: synapse-product drive ────────────────────────────
    # For each DN, compute: sum over all 2-hop paths of
    #   (sensory->inter synapse count) * (inter->DN synapse count)
    # plus direct 1-hop synapses (treated as weight directly)

    dn_drive = defaultdict(float)

    # Accumulate olfactory input to each intermediary
    inter_sensory_input = defaultdict(float)
    for nid in mod_neurons:
        for post in adj_forward.get(nid, set()):
            if post in intermediaries:
                inter_sensory_input[post] += edge_weight[(nid, post)]

    # For each intermediary with sensory input, multiply by inter->DN weight
    for inter_id, sensory_weight in inter_sensory_input.items():
        for post in adj_forward.get(inter_id, set()):
            if post in all_readout:
                dn_drive[post] += sensory_weight * edge_weight[(inter_id, post)]

    # Also add direct 1-hop synapses as drive (sum of direct edge weights)
    for nid in mod_neurons:
        for dn in adj_forward.get(nid, set()):
            if dn in all_readout:
                # For fair comparison with 2-hop, we add direct as a separate
                # entry. But the 2-hop weighted metric focuses on 2-hop paths.
                pass  # 1-hop already counted above; weighted metric is 2-hop only

    # Threshold analysis
    if dn_drive:
        sorted_dns = sorted(dn_drive.items(), key=lambda x: -x[1])
        max_drive = sorted_dns[0][1]

        strong_10pct = sum(1 for _, d in sorted_dns if d > 0.10 * max_drive)
        # Include 1-hop DNs in the weighted count (they have even stronger drive)
        weighted_dns = set(dn_id for dn_id, d in sorted_dns if d > 0.10 * max_drive)
        weighted_dns_cumulative = hop1_dns | weighted_dns
        hop2_weighted_pct = 100 * len(weighted_dns_cumulative) / len(all_readout)

        print(f"\n  2-hop weighted (>10% max drive):")
        print(f"    DNs with any 2-hop drive: {len(sorted_dns)}")
        print(f"    Max drive: {max_drive:,.0f}")
        print(f"    DNs above 10% threshold (2-hop only): {strong_10pct}")
        print(f"    DNs above 10% + 1-hop: {len(weighted_dns_cumulative)} / {len(all_readout)} ({hop2_weighted_pct:.1f}%)")

        # Show top 10
        print(f"\n    Top 10 DNs by weighted drive:")
        print(f"    {'Rank':>4s}  {'DN root_id':>22s}  {'Group':>12s}  {'Drive(norm)':>12s}  {'Drive(raw)':>12s}")
        for rank, (dn_id, drive) in enumerate(sorted_dns[:10], 1):
            dn_group = "none"
            for gname, gids in group_names.items():
                if dn_id in gids:
                    dn_group = gname
                    break
            print(f"    {rank:>4d}  {dn_id:>22d}  {dn_group:>12s}  "
                  f"{drive/max_drive:>12.4f}  {drive:>12.0f}")
    else:
        strong_10pct = 0
        weighted_dns_cumulative = hop1_dns
        hop2_weighted_pct = hop1_pct
        print(f"\n  2-hop weighted: no 2-hop paths found")

    # ── Per-decoder-group breakdown ──────────────────────────────────────
    print(f"\n  Per-group breakdown:")
    print(f"    {'Group':>12s}  {'1-hop':>8s}  {'2-hop topo':>11s}  {'2-hop wt':>10s}")
    for gname in ["forward", "turn_left", "turn_right", "rhythm", "stance"]:
        gids = group_names[gname]
        g_hop1   = len(hop1_dns & gids)
        g_hop2t  = len(hop2_dns_cumulative & gids)
        g_hop2w  = len(weighted_dns_cumulative & gids)
        print(f"    {gname:>12s}  "
              f"{g_hop1:>3d}/{len(gids):<3d}  "
              f"{g_hop2t:>3d}/{len(gids):<7d}  "
              f"{g_hop2w:>3d}/{len(gids):<3d}")

    results.append({
        "modality": mod_name,
        "n_neurons": len(mod_neurons),
        "hop1_dns": len(hop1_dns),
        "hop1_pct": hop1_pct,
        "hop2_topo_dns": len(hop2_dns_cumulative),
        "hop2_topo_pct": hop2_topo_pct,
        "hop2_weighted_dns": len(weighted_dns_cumulative),
        "hop2_weighted_pct": hop2_weighted_pct,
    })

# ── COMPARISON TABLE ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPARISON TABLE: ALL MODALITIES")
print("=" * 80)

total_dn = len(all_readout)
print(f"\n  Total readout DNs: {total_dn}")
print()

header = (f"  {'Modality':>16s}  {'Neurons':>7s}  "
          f"{'1-hop DNs':>10s}  {'(%)':>6s}  "
          f"{'2-hop topo':>10s}  {'(%)':>6s}  "
          f"{'2-hop wt':>10s}  {'(%)':>6s}")
print(header)
print("  " + "-" * (len(header) - 2))

for r in results:
    print(f"  {r['modality']:>16s}  {r['n_neurons']:>7d}  "
          f"{r['hop1_dns']:>10d}  {r['hop1_pct']:>5.1f}%  "
          f"{r['hop2_topo_dns']:>10d}  {r['hop2_topo_pct']:>5.1f}%  "
          f"{r['hop2_weighted_dns']:>10d}  {r['hop2_weighted_pct']:>5.1f}%")

# ── Markdown table for easy copy ─────────────────────────────────────────────
print(f"\n  Markdown format ({total_dn} readout DNs):")
print()
print("  | Modality       | Neurons | 1-hop DNs (%) | 2-hop topo (%) | 2-hop weighted (%) |")
print("  |----------------|--------:|--------------:|---------------:|-------------------:|")
for r in results:
    print(f"  | {r['modality']:<14s} | {r['n_neurons']:>7d} | "
          f"{r['hop1_dns']:>4d} ({r['hop1_pct']:>5.1f}%) | "
          f"{r['hop2_topo_dns']:>4d} ({r['hop2_topo_pct']:>5.1f}%) | "
          f"{r['hop2_weighted_dns']:>4d} ({r['hop2_weighted_pct']:>5.1f}%) |")

# ── SPARSITY GRADIENT ANALYSIS ──────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("SPARSITY GRADIENT: TOPOLOGY vs WEIGHTED FILTERING")
print("=" * 80)

print(f"\n  How much does synapse-weighting reduce the apparent connectivity?")
print(f"  (ratio = topology / weighted, higher = more reduction = sparser effective wiring)")
print()
print(f"  {'Modality':>16s}  {'Topo %':>7s}  {'Wt %':>6s}  {'Reduction':>10s}  {'Interpretation':>30s}")
print(f"  {'-'*16}  {'-'*7}  {'-'*6}  {'-'*10}  {'-'*30}")

for r in results:
    topo = r["hop2_topo_pct"]
    wt   = r["hop2_weighted_pct"]
    if wt > 0:
        ratio = topo / wt
    else:
        ratio = float('inf')

    if wt < 5:
        interp = "Very sparse (< 5%)"
    elif wt < 20:
        interp = "Moderately sparse"
    elif wt < 50:
        interp = "Broad drive"
    else:
        interp = "Very broad (> 50%)"

    print(f"  {r['modality']:>16s}  {topo:>6.1f}%  {wt:>5.1f}%  "
          f"{ratio:>9.1f}x  {interp:>30s}")

# ── AYMANNS COMPARISON FOR OLFACTORY ─────────────────────────────────────────
olf = next(r for r in results if r["modality"] == "Olfactory")
print(f"\n{'=' * 80}")
print("OLFACTORY BENCHMARK: AYMANNS et al. 2022")
print(f"{'=' * 80}")
print(f"\n  Aymanns et al. 2022 finding: 2-5% of DNs show odor-specific activity")
print(f"  Our 2-hop weighted result for olfactory: {olf['hop2_weighted_pct']:.1f}%")
if 2.0 <= olf["hop2_weighted_pct"] <= 5.0:
    print(f"  --> CONSISTENT with Aymanns (within 2-5% range)")
else:
    print(f"  --> Outside Aymanns 2-5% range")

# ── Save results to JSON ─────────────────────────────────────────────────────
output_path = f"{BASE}/plastic-fly/analysis/all_modality_dn_sparsity_results.json"
output = {
    "description": "Synapse-weighted hop analysis for all 6 sensory modalities",
    "total_readout_dns": total_dn,
    "threshold": "10% of max weighted drive",
    "modalities": results,
}
_write_json_atomic(output_path, output, indent=2)
print(f"\n  Results saved to: {output_path}")

print(f"\n  Total runtime: {time.time() - t0:.1f}s")
print("\nDone.")
