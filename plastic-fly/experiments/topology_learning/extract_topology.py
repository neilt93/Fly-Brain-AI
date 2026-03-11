"""
Extract compressed VNC topology from MANC data as PyTorch sparse tensors.

Produces a topology dict suitable for building ConnectomePolicy:
  - Sparse adjacency (COO indices + weights)
  - Neuron group indices (DN inputs, MN outputs, intrinsic)
  - MN-to-joint mapping for output layer

Compression: keep all DNs + all MNs + top-K intrinsic by degree.
"""

import json
import numpy as np
import torch
from pathlib import Path

from .config import TopologyConfig


# Neurotransmitter -> sign mapping (matches vnc_connectome.py)
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": +1.0,
    "dopamine": +1.0,
    "serotonin": +1.0,
    "octopamine": +1.0,
    "histamine": -1.0,
    "unclear": +1.0,
}


def extract_compressed_vnc(cfg: TopologyConfig = None) -> dict:
    """Extract compressed VNC topology from MANC feather files.

    Returns dict with:
        adj_indices: (2, E) LongTensor -- sparse COO indices
        adj_values: (E,) FloatTensor -- signed weights (synapse_count * nt_sign)
        n_neurons: int
        n_dn: int, n_mn: int, n_intrinsic: int
        dn_indices: list[int] -- neuron indices for DN inputs
        mn_indices: list[int] -- neuron indices for MN outputs
        mn_body_ids: list[int] -- MANC body IDs for MN decoder lookup
        joint_mapping: dict -- MN neuron index -> (joint_idx, direction)
        metadata: dict
    """
    import pandas as pd

    if cfg is None:
        cfg = TopologyConfig()

    manc_dir = cfg.manc_dir

    # --- Load MANC data ---
    ann = pd.read_feather(manc_dir / "body-annotations-male-cns-v0.9-minconf-0.5.feather")
    nt_df = pd.read_feather(manc_dir / "body-neurotransmitters-male-cns-v0.9.feather")
    conn = pd.read_feather(manc_dir / "connectome-weights-male-cns-v0.9-minconf-0.5.feather")

    # --- Build NT lookup ---
    nt_map = {}
    for _, row in nt_df.iterrows():
        bid = int(row["body"])
        nt = str(row.get("predicted_nt", "unclear")).lower().strip()
        nt_map[bid] = NT_SIGN.get(nt, +1.0)

    # --- Select neurons: DN + MN + top-K intrinsic ---
    # DNs: superclass == "descending_neuron"
    dn_mask = ann["superclass"] == "descending_neuron"
    dn_ids = set(ann.loc[dn_mask, "bodyId"].astype(int))

    # MNs: superclass == "vnc_motor" AND in thoracic neuromeres (leg MNs)
    thoracic = {"T1", "T2", "T3"}
    mn_mask = (
        (ann["superclass"] == "vnc_motor") &
        ann["somaNeuromere"].isin(thoracic)
    )
    mn_ids = set(ann.loc[mn_mask, "bodyId"].astype(int))

    # Intrinsic: thoracic VNC neurons, not DN, not MN
    intrinsic_mask = (
        ann["somaNeuromere"].isin(thoracic) &
        ~dn_mask & ~mn_mask
    )
    all_intrinsic_ids = set(ann.loc[intrinsic_mask, "bodyId"].astype(int))

    # Filter connectivity to neurons we might include
    candidate_ids = dn_ids | mn_ids | all_intrinsic_ids
    conn_filtered = conn[
        conn["body_pre"].isin(candidate_ids) &
        conn["body_post"].isin(candidate_ids)
    ].copy()

    # Compute degree for intrinsic neurons (vectorized for speed)
    degree = {bid: 0 for bid in all_intrinsic_ids}
    pre_vals = conn_filtered["body_pre"].values
    post_vals = conn_filtered["body_post"].values
    for i in range(len(pre_vals)):
        pre, post = int(pre_vals[i]), int(post_vals[i])
        if pre in degree:
            degree[pre] += 1
        if post in degree:
            degree[post] += 1

    # Select top-K intrinsic by degree
    sorted_intrinsic = sorted(degree.items(), key=lambda x: -x[1])
    top_intrinsic_ids = set(bid for bid, _ in sorted_intrinsic[:cfg.top_k_intrinsic])

    # Final neuron set
    selected_ids = sorted(dn_ids | mn_ids | top_intrinsic_ids)
    id_to_idx = {bid: i for i, bid in enumerate(selected_ids)}
    n_neurons = len(selected_ids)

    # Classify indices
    dn_indices = sorted(id_to_idx[bid] for bid in dn_ids if bid in id_to_idx)
    mn_indices = sorted(id_to_idx[bid] for bid in mn_ids if bid in id_to_idx)
    intrinsic_indices = sorted(id_to_idx[bid] for bid in top_intrinsic_ids if bid in id_to_idx)
    mn_body_ids = sorted(bid for bid in mn_ids if bid in id_to_idx)

    # --- Build sparse adjacency ---
    selected_set = set(selected_ids)
    conn_sub = conn_filtered[
        conn_filtered["body_pre"].isin(selected_set) &
        conn_filtered["body_post"].isin(selected_set)
    ]

    pre_idx = []
    post_idx = []
    values = []

    sub_pre = conn_sub["body_pre"].values
    sub_post = conn_sub["body_post"].values
    sub_weight = conn_sub["weight"].values

    for i in range(len(sub_pre)):
        pre_bid = int(sub_pre[i])
        post_bid = int(sub_post[i])
        weight = float(sub_weight[i])
        sign = nt_map.get(pre_bid, +1.0)
        pre_idx.append(id_to_idx[pre_bid])
        post_idx.append(id_to_idx[post_bid])
        values.append(weight * sign)

    adj_indices = torch.tensor([pre_idx, post_idx], dtype=torch.long)
    adj_values = torch.tensor(values, dtype=torch.float32)

    # --- MN joint mapping ---
    joint_mapping = {}  # mn_neuron_idx -> (joint_idx, direction)
    if cfg.mn_mapping_path.exists():
        with open(cfg.mn_mapping_path) as f:
            mn_map_raw = json.load(f)
        for bid_str, info in mn_map_raw.items():
            bid = int(bid_str)
            if bid in id_to_idx:
                joint_mapping[id_to_idx[bid]] = {
                    "joint_idx": int(info["joint_idx"]),
                    "direction": float(info["direction"]),
                }

    n_dn = len(dn_indices)
    n_mn = len(mn_indices)
    n_intr = len(intrinsic_indices)
    n_edges = adj_indices.shape[1]
    sparsity = n_edges / (n_neurons * n_neurons)

    metadata = {
        "n_neurons": n_neurons,
        "n_dn": n_dn,
        "n_mn": n_mn,
        "n_intrinsic": n_intr,
        "n_edges": n_edges,
        "sparsity": sparsity,
        "top_k_intrinsic": cfg.top_k_intrinsic,
    }

    print(f"Compressed VNC topology: {n_neurons} neurons "
          f"({n_dn} DN + {n_mn} MN + {n_intr} intrinsic), "
          f"{n_edges} edges, sparsity={sparsity:.4f}")

    return {
        "adj_indices": adj_indices,
        "adj_values": adj_values,
        "n_neurons": n_neurons,
        "n_dn": n_dn,
        "n_mn": n_mn,
        "n_intrinsic": n_intr,
        "dn_indices": dn_indices,
        "mn_indices": mn_indices,
        "mn_body_ids": mn_body_ids,
        "intrinsic_indices": intrinsic_indices,
        "joint_mapping": joint_mapping,
        "id_to_idx": id_to_idx,
        "metadata": metadata,
    }


def create_shuffled_topology(topo: dict, seed: int = 42) -> dict:
    """Create a shuffled version: same degree distribution, permuted targets."""
    rng = np.random.RandomState(seed)
    indices = topo["adj_indices"].clone()
    indices[1] = torch.tensor(rng.permutation(indices[1].numpy()), dtype=torch.long)
    out = dict(topo)
    out["adj_indices"] = indices
    return out


def create_random_sparse_topology(topo: dict, seed: int = 42) -> dict:
    """Create random Erdos-Renyi sparse topology with same density."""
    rng = np.random.RandomState(seed)
    n = topo["n_neurons"]
    n_edges = topo["adj_indices"].shape[1]

    # Sample random edges (no self-loops)
    edges = set()
    while len(edges) < n_edges:
        pre = rng.randint(0, n)
        post = rng.randint(0, n)
        if pre != post:
            edges.add((pre, post))

    edges = sorted(edges)
    pre_idx = [e[0] for e in edges]
    post_idx = [e[1] for e in edges]

    # Random E/I signs matching the original distribution
    orig_signs = torch.sign(topo["adj_values"])
    frac_inh = (orig_signs < 0).float().mean().item()
    signs = torch.where(
        torch.tensor(rng.random(n_edges) < frac_inh),
        torch.tensor(-1.0), torch.tensor(+1.0)
    )

    # Random weights from same distribution
    orig_abs = topo["adj_values"].abs()
    weight_indices = rng.randint(0, len(orig_abs), size=n_edges)
    weights = orig_abs[weight_indices] * signs

    out = dict(topo)
    out["adj_indices"] = torch.tensor([pre_idx, post_idx], dtype=torch.long)
    out["adj_values"] = weights
    return out


if __name__ == "__main__":
    topo = extract_compressed_vnc()
    print(f"\nTopology extracted:")
    for k, v in topo["metadata"].items():
        print(f"  {k}: {v}")
    print(f"  joint_mapping entries: {len(topo['joint_mapping'])}")
