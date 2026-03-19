"""
Extract 3x3 CPG weight matrix from MANC connectivity.

Identifies the 3-neuron E-E-I oscillator (Pugliese et al. 2025):
    E1: IN17A001 (ACh, excitatory)
    E2: INXXX466 (ACh, excitatory)
    I1: IN16B036 (Glu, inhibitory)

For each of 6 hemi-segments (T1-T3 x L/R), extracts pairwise synapse
counts and averages them to produce a canonical 3x3 weight matrix.

Output: data/cpg_weights.json

Usage:
    python scripts/extract_cpg_weights.py
    python scripts/extract_cpg_weights.py --exc-mult 0.03 --inh-mult 0.03
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANC_DIR = ROOT / "data" / "manc"

CPG_TYPES = {
    "E1": "IN17A001",
    "E2": "INXXX466",
    "I": "IN16B036",
}

HEMI_SEGMENTS = [
    ("T1", "L"), ("T1", "R"),
    ("T2", "L"), ("T2", "R"),
    ("T3", "L"), ("T3", "R"),
]


def main():
    parser = argparse.ArgumentParser(description="Extract CPG weights from MANC")
    parser.add_argument("--exc-mult", type=float, default=0.03,
                        help="Excitatory weight multiplier")
    parser.add_argument("--inh-mult", type=float, default=0.03,
                        help="Inhibitory weight multiplier")
    args = parser.parse_args()

    ann_path = MANC_DIR / "body-annotations-male-cns-v0.9-minconf-0.5.feather"
    con_path = MANC_DIR / "connectome-weights-male-cns-v0.9-minconf-0.5.feather"

    print("Loading MANC annotations...")
    ann = pd.read_feather(ann_path)
    print(f"  {len(ann)} neurons")

    print("Loading MANC connectivity...")
    con = pd.read_feather(con_path)
    print(f"  {len(con)} connections")

    # Find body IDs for each CPG neuron type per hemi-segment
    print("\nCPG neuron inventory:")
    segment_ids = {}  # (segment, side) -> {"E1": bodyId, "E2": bodyId, "I": bodyId}

    for nm, side in HEMI_SEGMENTS:
        segment_ids[(nm, side)] = {}
        for role, ntype in CPG_TYPES.items():
            mask = (ann["type"] == ntype) & (ann["somaNeuromere"] == nm) & (ann["somaSide"] == side)
            matches = ann[mask]["bodyId"].values
            if len(matches) == 1:
                segment_ids[(nm, side)][role] = int(matches[0])
                print(f"  {nm}{side} {role}({ntype}): bodyId={matches[0]}")
            elif len(matches) == 0:
                print(f"  {nm}{side} {role}({ntype}): NOT FOUND")
            else:
                # Take first match
                segment_ids[(nm, side)][role] = int(matches[0])
                print(f"  {nm}{side} {role}({ntype}): {len(matches)} matches, using {matches[0]}")

    # Extract pairwise synapse counts for each hemi-segment
    print("\nExtracting pairwise connectivity per hemi-segment:")
    roles = ["E1", "E2", "I"]
    all_matrices = []

    for nm, side in HEMI_SEGMENTS:
        ids = segment_ids[(nm, side)]
        if len(ids) < 3:
            print(f"  {nm}{side}: incomplete ({len(ids)}/3 neurons), skipping")
            continue

        # Raw synapse count matrix (3x3): raw[post_idx, pre_idx]
        raw = np.zeros((3, 3), dtype=np.float64)
        for i, pre_role in enumerate(roles):
            for j, post_role in enumerate(roles):
                if i == j:
                    continue
                pre_id = ids[pre_role]
                post_id = ids[post_role]
                mask = (con["body_pre"] == pre_id) & (con["body_post"] == post_id)
                w = con[mask]["weight"].sum()
                raw[j, i] = float(w)  # W[post, pre]

        print(f"  {nm}{side} synapse counts:")
        for i, role_i in enumerate(roles):
            for j, role_j in enumerate(roles):
                if raw[i, j] > 0:
                    print(f"    {role_j} -> {role_i}: {raw[i, j]:.0f}")

        all_matrices.append(raw)

    if not all_matrices:
        print("ERROR: No complete hemi-segments found!")
        sys.exit(1)

    # Average across hemi-segments
    avg_raw = np.mean(all_matrices, axis=0)
    print(f"\nAveraged synapse counts ({len(all_matrices)} hemi-segments):")
    for i, role_i in enumerate(roles):
        for j, role_j in enumerate(roles):
            if avg_raw[i, j] > 0:
                print(f"  {role_j} -> {role_i}: {avg_raw[i, j]:.1f}")

    # Apply E/I multipliers to get effective weight matrix
    # E1 (idx 0) and E2 (idx 1) are excitatory (ACh)
    # I (idx 2) is inhibitory (glutamate)
    W = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if avg_raw[i, j] == 0:
                continue
            if j == 2:  # Pre is I (inhibitory)
                W[i, j] = -avg_raw[i, j] * args.inh_mult
            else:  # Pre is E1 or E2 (excitatory)
                W[i, j] = avg_raw[i, j] * args.exc_mult

    print(f"\nEffective weight matrix W (exc_mult={args.exc_mult}, inh_mult={args.inh_mult}):")
    for i, role_i in enumerate(roles):
        row = " ".join(f"{W[i, j]:+8.3f}" for j in range(3))
        print(f"  {role_i}: [{row}]")

    # Save
    output = {
        "W": W.tolist(),
        "neuron_params": {
            "tau_ms": 20.0,
            "theta": 7.5,
            "a": 1.0,
            "R_max": 200.0,
            "exc_mult": args.exc_mult,
            "inh_mult": args.inh_mult,
            "drive_scale": 0.5,
            "drive_target": "E1",
        },
        "neuron_types": ["IN17A001", "INXXX466", "IN16B036"],
        "neuron_roles": ["E1 (excitatory)", "E2 (excitatory)", "I (inhibitory)"],
        "raw_synapse_counts_avg": avg_raw.tolist(),
        "n_hemi_segments_averaged": len(all_matrices),
        "per_segment_ids": {
            f"{nm}{side}": segment_ids[(nm, side)]
            for nm, side in HEMI_SEGMENTS
            if (nm, side) in segment_ids and len(segment_ids[(nm, side)]) == 3
        },
        "source": "MANC v0.9 (male adult CNS)",
        "reference": "Pugliese et al. 2025",
    }

    out_path = ROOT / "data" / "cpg_weights.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Quick validation: check that the matrix should produce oscillations
    eigenvalues = np.linalg.eigvals(W)
    print(f"\nW eigenvalues: {eigenvalues}")
    has_complex = any(np.abs(e.imag) > 1e-6 for e in eigenvalues)
    print(f"Has complex eigenvalues (oscillatory potential): {has_complex}")


if __name__ == "__main__":
    main()
