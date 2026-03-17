"""
Parse MaleCNS connectome data (Male Adult CNS, Janelia).

Loads Feather files from data/manc/ and reports:
  1. Basic stats: neuron counts, connection counts, neuron classes
  2. DN (descending neuron) and MN (motor neuron) populations
  3. Overlap between MaleCNS DN types and our brain model readout DNs
  4. Neurotransmitter distribution (excitatory/inhibitory)

Data source: https://male-cns.janelia.org/download/ (CC-BY 4.0)
Files are from MaleCNS v0.9 (brain + VNC in one volume, 166K neurons).
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parent.parent
MANC_DIR = ROOT / "data" / "manc"
DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------------
# Our brain model readout DN types (from Sturner FAFB mapping)
# These are the standard DN type names for our 47+ readout neurons.
# --------------------------------------------------------------------------
OUR_READOUT_ALL_DN_TYPES = {
    "DNa02", "DNg56", "DNpe030", "DNp44", "DNge149", "oviDNx", "DNg29",
    "DNge037", "DNge133", "DNg05_a", "DNge113", "DNp102", "DNg90", "DNp23",
    "DNge136", "DNge119", "DNge138", "DNp25", "DNp05", "DNpe045", "DNg93",
    "DNge129", "DNge132", "DNp104", "DNge047", "DNg17", "DNae007", "DNg83",
    "DNg23", "DNc01", "DNg62", "DNge039", "DNp19", "DNg86", "DNge044",
    "DNp62", "DNbe003", "DNc02", "DNg35", "DNp42", "DNg33", "DNge100",
    "DNp02", "DNp10", "DNp30", "DNg59", "DNge046", "DNb08", "DNge038",
    "DNd02", "DNg102", "DNg85", "oviDNd", "DNge148", "DNge135", "DNge067",
    "DNge054", "DNge041", "DNp06", "DNp29", "DNde005", "DNp37",
    "DNge142", "DNg05_b", "DNg47", "DNg15", "DNg87",
    "DNb05", "DNge075", "DNp35", "DNbe007", "DNge077", "DNge099", "DNpe049",
    "DNg81", "DNd04", "DNp12", "DNge172", "DNp54", "DNpe006",
    "DNp32", "DNg68", "DNge012", "DNpe017",
    "DNg57", "DNp34", "DNpe023", "DNp08", "DNge023",
    "DNge065", "DNpe003", "DNg20", "DNpe056", "DNg39",
    "DNg103", "DNge121", "DNp29", "DNae009", "DNpe043",
    "DNg84", "DNde002", "DNpe053", "DNge101", "DNp64", "DNp14", "DNg100",
    "DNp69", "DNg32", "DNp55", "DNg34",
    "DNp59", "DNde001", "DNpe048", "DNg105", "DNge057",
    "DNb09", "DNd03", "DNp49", "DNge018",
    "DNde003", "DNde006", "DNp46", "DNp102",
    "DNg37", "DNpe020",
}

# SEZ-name to Sturner DN type mapping (for the core annotated readout neurons)
SEZ_TO_DN = {
    "bobber":     "DNg57",
    "nagini":     "DNg15",
    "knees":      ["DNg05_a", "DNg05_b"],
    "mute":       "DNge172",
    # These SEZ names have no direct Sturner match:
    # mothership, lion, shark, aSG1, twirl, trogon, ruby, damsel, bridle
}


# Columns known to contain array/non-hashable values
ARRAY_COLUMNS = {"somaLocation", "tosomaLocation"}


def load_annotations():
    """Load body annotations feather file."""
    path = MANC_DIR / "body-annotations-male-cns-v0.9-minconf-0.5.feather"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return pd.DataFrame(feather.read_feather(str(path)))


def load_neurotransmitters():
    """Load neurotransmitter predictions feather file."""
    path = MANC_DIR / "body-neurotransmitters-male-cns-v0.9.feather"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return pd.DataFrame(feather.read_feather(str(path)))


def load_connectivity():
    """Load connectivity weights feather file (large, ~1.1 GB)."""
    path = MANC_DIR / "connectome-weights-male-cns-v0.9-minconf-0.5.feather"
    if not path.exists():
        print(f"WARNING: {path} not found (still downloading?)")
        return None
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < 500:
        print(f"WARNING: Connectivity file looks incomplete ({size_mb:.0f} MB, expected ~1100 MB)")
        return None
    print(f"Loading connectivity file ({size_mb:.0f} MB)...")
    df = pd.DataFrame(feather.read_feather(str(path)))
    print(f"Loaded {len(df):,} connections")
    return df


def safe_col_stats(series, col_name):
    """Get column stats, handling array-valued columns safely."""
    if col_name in ARRAY_COLUMNS:
        n_null = series.isna().sum()
        return f"  {col_name}: (array column), {n_null} null"
    try:
        n_unique = series.nunique()
        n_null = series.isna().sum()
        non_null = series.dropna()
        sample = non_null.iloc[:3].tolist() if len(non_null) > 0 else []
        return f"  {col_name}: {n_unique:,} unique, {n_null:,} null, sample: {sample}"
    except TypeError:
        n_null = series.isna().sum()
        return f"  {col_name}: (unhashable type), {n_null:,} null"


def analyze_annotations(ann):
    """Analyze neuron annotations."""
    print("=" * 70)
    print("MaleCNS BODY ANNOTATIONS")
    print("=" * 70)
    print(f"\nTotal rows: {len(ann):,}")
    print(f"Columns ({len(ann.columns)}): {list(ann.columns)}")
    print()

    # Column stats (skip array columns)
    for col in ann.columns:
        print(safe_col_stats(ann[col], col))
    print()

    # --- Superclass breakdown (this is the main classification) ---
    print("--- Superclass Breakdown ---")
    sc_counts = ann["superclass"].value_counts()
    for cls, count in sc_counts.items():
        print(f"  {cls}: {count:,}")
    total_annotated = sc_counts.sum()
    print(f"  [annotated]: {total_annotated:,}")
    print(f"  [unannotated]: {len(ann) - total_annotated:,}")
    print()

    # --- Descending Neurons ---
    dns = ann[ann["superclass"] == "descending_neuron"].copy()
    print(f"--- Descending Neurons (superclass=descending_neuron) ---")
    print(f"  Count: {len(dns):,}")
    dn_types = sorted(dns["type"].dropna().unique())
    print(f"  Unique types: {len(dn_types)}")
    print(f"  First 30: {dn_types[:30]}")
    # Also check flywireType column for cross-reference
    fw_types = sorted(dns["flywireType"].dropna().unique())
    print(f"  Unique flywireTypes: {len(fw_types)}")
    print(f"  First 30: {fw_types[:30]}")
    print()

    # Check if any DNs have MANC-specific IDs
    if "mancBodyid" in dns.columns:
        n_manc = dns["mancBodyid"].notna().sum()
        print(f"  DNs with MANC body IDs: {n_manc:,} / {len(dns):,}")
    if "mancType" in dns.columns:
        manc_types = sorted(dns["mancType"].dropna().unique())
        print(f"  DNs with MANC types: {len(manc_types)} unique")
        print(f"  First 20: {manc_types[:20]}")
    print()

    # --- Motor Neurons ---
    mns = ann[ann["superclass"] == "vnc_motor"].copy()
    cb_mns = ann[ann["superclass"] == "cb_motor"].copy()
    print(f"--- Motor Neurons ---")
    print(f"  VNC motor (vnc_motor): {len(mns):,}")
    print(f"  CB motor (cb_motor): {len(cb_mns):,}")
    mn_types = sorted(mns["type"].dropna().unique())
    print(f"  VNC MN unique types: {len(mn_types)}")
    if len(mn_types) > 0:
        print(f"  First 20: {mn_types[:20]}")
    # Neuromere breakdown
    if "somaNeuromere" in mns.columns:
        nm_counts = mns["somaNeuromere"].value_counts()
        print(f"  Soma neuromere breakdown:")
        for nm, count in nm_counts.items():
            print(f"    {nm}: {count}")
    print()

    # --- Sensory Neurons ---
    sn_classes = ["vnc_sensory", "ol_sensory", "cb_sensory"]
    for sc in sn_classes:
        sns = ann[ann["superclass"] == sc]
        print(f"  {sc}: {len(sns):,}")
    print()

    # --- Ascending Neurons ---
    ans = ann[ann["superclass"] == "ascending_neuron"]
    print(f"--- Ascending Neurons: {len(ans):,} ---")
    print()

    # --- VNC intrinsic (the bulk of VNC computation) ---
    vnc_in = ann[ann["superclass"] == "vnc_intrinsic"]
    print(f"--- VNC Intrinsic Neurons: {len(vnc_in):,} ---")
    print()

    return dns, mns


def analyze_neurotransmitters(nt):
    """Analyze neurotransmitter predictions."""
    print("=" * 70)
    print("NEUROTRANSMITTER PREDICTIONS")
    print("=" * 70)
    print(f"\nTotal rows: {len(nt):,}")
    print(f"Columns: {list(nt.columns)}")
    print()

    for col in nt.columns:
        print(safe_col_stats(nt[col], col))
    print()

    # Find the main neurotransmitter prediction column
    # Common patterns: predictedNt, neuropredicted, top_nt, etc.
    nt_col = None
    for col in nt.columns:
        if col in ARRAY_COLUMNS:
            continue
        try:
            vals = nt[col].dropna().unique()
            val_strs = [str(v).lower() for v in vals[:20]]
            if any(k in " ".join(val_strs) for k in ["acetylcholine", "gaba", "glutamate", "serotonin", "dopamine", "octopamine"]):
                nt_col = col
                break
        except (AttributeError, ValueError, TypeError):
            pass

    if nt_col:
        print(f"--- Neurotransmitter distribution ({nt_col}) ---")
        counts = nt[nt_col].value_counts()
        for val, count in counts.head(15).items():
            pct = 100 * count / len(nt)
            print(f"  {val}: {count:,} ({pct:.1f}%)")
    else:
        # Show all object columns that might be NT predictions
        print("  [Could not auto-detect NT prediction column]")
        for col in nt.select_dtypes(include=["object"]).columns:
            print(f"  {col}: {nt[col].value_counts().head(5).to_dict()}")
    print()


def analyze_connectivity(conn):
    """Analyze connectivity weights."""
    print("=" * 70)
    print("CONNECTIVITY WEIGHTS")
    print("=" * 70)
    print(f"\nTotal connections: {len(conn):,}")
    print(f"Columns: {list(conn.columns)}")
    print()

    for col in conn.columns:
        print(safe_col_stats(conn[col], col))
    print()

    # Identify pre/post columns
    pre_col = next((c for c in conn.columns if "pre" in c.lower() or "source" in c.lower()
                     or c == "bodyId_pre"), None)
    post_col = next((c for c in conn.columns if "post" in c.lower() or "target" in c.lower()
                      or c == "bodyId_post"), None)
    weight_col = next((c for c in conn.columns if "weight" in c.lower() or "syn" in c.lower()), None)

    if pre_col and post_col:
        n_pre = conn[pre_col].nunique()
        n_post = conn[post_col].nunique()
        n_neurons = len(set(conn[pre_col].unique()) | set(conn[post_col].unique()))
        print(f"  Pre column: {pre_col}")
        print(f"  Post column: {post_col}")
        print(f"  Unique presynaptic: {n_pre:,}")
        print(f"  Unique postsynaptic: {n_post:,}")
        print(f"  Unique neurons (union): {n_neurons:,}")

    if weight_col:
        w = conn[weight_col]
        print(f"\n  Weight column: {weight_col}")
        print(f"    min={w.min()}, max={w.max()}, mean={w.mean():.2f}, median={w.median():.1f}")
        print(f"    Total synaptic weight: {w.sum():,.0f}")
    print()


def map_dn_overlap(dns):
    """Check overlap between MaleCNS DN types and our brain model readout DNs."""
    print("=" * 70)
    print("DN MAPPING: Our Brain Readout <-> MaleCNS")
    print("=" * 70)

    if dns is None or len(dns) == 0:
        print("  No DN data available")
        return

    # Use 'type' column (MaleCNS types) and also check 'flywireType'
    manc_types = set(dns["type"].dropna().unique())
    fw_types = set(dns["flywireType"].dropna().unique())

    print(f"\nMaleCNS DN types (type column): {len(manc_types)}")
    print(f"MaleCNS DN flywireTypes: {len(fw_types)}")
    print(f"Our readout DN types: {len(OUR_READOUT_ALL_DN_TYPES)}")

    # -- Check overlap against type column --
    overlap_type = OUR_READOUT_ALL_DN_TYPES & manc_types
    print(f"\n--- Overlap with 'type' column: {len(overlap_type)} ---")
    for t in sorted(overlap_type):
        n = len(dns[dns["type"] == t])
        print(f"  {t}: {n} neurons in MaleCNS")

    # -- Check overlap against flywireType column --
    overlap_fw = OUR_READOUT_ALL_DN_TYPES & fw_types
    print(f"\n--- Overlap with 'flywireType' column: {len(overlap_fw)} ---")
    for t in sorted(overlap_fw):
        n = len(dns[dns["flywireType"] == t])
        print(f"  {t}: {n} neurons in MaleCNS (flywireType)")

    # Combined overlap
    all_manc = manc_types | fw_types
    combined_overlap = OUR_READOUT_ALL_DN_TYPES & all_manc
    print(f"\n--- Combined overlap (type OR flywireType): {len(combined_overlap)} / {len(OUR_READOUT_ALL_DN_TYPES)} ---")

    # Missing from both
    missing = OUR_READOUT_ALL_DN_TYPES - all_manc
    print(f"\nOur DN types NOT found in MaleCNS: {len(missing)}")

    # Try prefix matching for missing types
    resolved = {}
    still_missing = set()
    for t in sorted(missing):
        base = t.split("_")[0] if "_" in t else t
        type_matches = [m for m in all_manc if m.startswith(base) and m != t]
        if type_matches:
            resolved[t] = type_matches
        else:
            still_missing.add(t)

    if resolved:
        print(f"\n  Resolved by prefix matching: {len(resolved)}")
        for t, matches in sorted(resolved.items()):
            print(f"    {t} -> {matches[:5]}")

    if still_missing:
        print(f"\n  Truly missing (no match in MaleCNS): {len(still_missing)}")
        for t in sorted(still_missing):
            print(f"    {t}")

    # MaleCNS DN types NOT in our readout
    new_types = all_manc - OUR_READOUT_ALL_DN_TYPES
    print(f"\nMaleCNS DN types not in our readout: {len(new_types)} (potential expansion)")
    print(f"  Sample: {sorted(new_types)[:20]}")
    print()

    # --- Key walking DNs (from literature) ---
    print("--- Key Walking DNs (Pugliese et al. 2025 / Bidaye et al.) ---")
    walking_dns = {
        "DNg100": "walking command (Pugliese 2025)",
        "DNa01":  "forward walking (Bidaye 2020)",
        "DNa02":  "backward walking (Bidaye 2020)",
        "DNb01":  "leg coordination",
        "DNb02":  "walking speed",
        "DNp09":  "moonwalker (backward)",
        "DNg11":  "turning",
    }
    for dn, role in walking_dns.items():
        in_type = dn in manc_types
        in_fw = dn in fw_types
        in_ours = dn in OUR_READOUT_ALL_DN_TYPES
        n_type = len(dns[dns["type"] == dn]) if in_type else 0
        n_fw = len(dns[dns["flywireType"] == dn]) if in_fw else 0
        tags = []
        if in_type:
            tags.append(f"type({n_type})")
        if in_fw:
            tags.append(f"fwType({n_fw})")
        if in_ours:
            tags.append("OUR_READOUT")
        loc = " + ".join(tags) if tags else "NOT FOUND"
        print(f"  {dn} [{role}]: {loc}")
    print()

    # --- SEZ name mapping summary ---
    print("--- SEZ Name -> DN Type Mapping ---")
    sez_names = ["mothership", "lion", "bobber", "twirl", "trogon", "ruby",
                 "damsel", "shark", "mute", "aSG1", "nagini", "knees", "bridle"]
    for name in sez_names:
        if name in SEZ_TO_DN:
            dn = SEZ_TO_DN[name]
            if isinstance(dn, list):
                for d in dn:
                    in_manc = d in all_manc
                    print(f"  {name} -> {d}: {'FOUND' if in_manc else 'MISSING'} in MaleCNS")
            else:
                in_manc = dn in all_manc
                print(f"  {name} -> {dn}: {'FOUND' if in_manc else 'MISSING'} in MaleCNS")
        else:
            print(f"  {name} -> [no Sturner mapping]")


def main():
    print("MaleCNS Connectome Parser (v0.9)")
    print("Data: male-cns.janelia.org (CC-BY 4.0)")
    print()

    # 1. Annotations
    ann = load_annotations()
    dns, mns = analyze_annotations(ann)

    # 2. Neurotransmitters
    nt = load_neurotransmitters()
    analyze_neurotransmitters(nt)

    # 3. Connectivity (may still be downloading)
    conn = load_connectivity()
    if conn is not None:
        analyze_connectivity(conn)
    else:
        print("\n[Connectivity file not available -- skipping]\n")

    # 4. DN mapping
    map_dn_overlap(dns)

    # 5. Summary
    print("=" * 70)
    print("SUMMARY FOR VNC MODEL")
    print("=" * 70)

    # VNC-specific neurons
    vnc_classes = ["vnc_intrinsic", "vnc_motor", "vnc_sensory", "descending_neuron",
                   "ascending_neuron", "vnc_efferent"]
    print("\nVNC-relevant populations:")
    for sc in vnc_classes:
        n = len(ann[ann["superclass"] == sc])
        print(f"  {sc}: {n:,}")

    all_manc_dn = set(dns["type"].dropna().unique()) | set(dns["flywireType"].dropna().unique())
    overlap = len(OUR_READOUT_ALL_DN_TYPES & all_manc_dn)
    print(f"\n  Our readout DN types found in MaleCNS: {overlap} / {len(OUR_READOUT_ALL_DN_TYPES)}")
    if conn is not None:
        print(f"  Total connections: {len(conn):,}")
    print()


if __name__ == "__main__":
    main()
