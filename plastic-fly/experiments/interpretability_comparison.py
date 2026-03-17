"""
Paper 3: Interpretability comparison — connectome vs random sparse.

Both architectures achieve EQUAL performance on forward locomotion.
This experiment quantifies the INTERPRETABILITY advantage of the connectome:
  1. Ablation specificity — targeted neuron removal causes predictable deficits
  2. Functional modularity — neurons specialize for distinct functions
  3. Pathway bottlenecks — information flows through identifiable chokepoints
  4. Ablation consistency — same ablation -> same effect across seeds

Usage:
    # Full analysis (~2-3 hours)
    python -m experiments.interpretability_comparison

    # Quick proof (~20 min)
    python -m experiments.interpretability_comparison --quick

    # Single metric
    python -m experiments.interpretability_comparison --analysis group_ablation
"""

import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.topology_learning.config import TopologyConfig
from experiments.topology_learning.extract_topology import extract_compressed_vnc
from experiments.topology_learning.vnc_policy import (
    SparseRecurrentPolicy,
    build_connectome_policy,
    build_random_sparse_policy,
    build_shuffled_policy,
)
from experiments.topology_learning.flygym_env import FlyGymLocomotionEnv


def _write_json_atomic(path: Path, payload, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, **kwargs)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Ablation primitives
# ---------------------------------------------------------------------------

def ablate_neurons(policy: SparseRecurrentPolicy, neuron_indices: list):
    """Silence neurons by zeroing their recurrent weights (in-place).

    Returns saved state for restoration.
    """
    saved = {}
    with torch.no_grad():
        w = policy.W_rec.weight
        for idx in neuron_indices:
            saved[idx] = (w[idx, :].clone(), w[:, idx].clone())
            w[idx, :] = 0.0
            w[:, idx] = 0.0
    return saved


def restore_neurons(policy: SparseRecurrentPolicy, saved: dict):
    """Restore neurons from saved state."""
    with torch.no_grad():
        w = policy.W_rec.weight
        for idx, (row, col) in saved.items():
            w[idx, :] = row
            w[:, idx] = col


def evaluate_locomotion(policy, env, n_episodes=3, episode_length=1000):
    """Evaluate forward distance and turning angle over n episodes."""
    import flygym

    distances = []
    headings = []
    rewards = []

    for ep in range(n_episodes):
        if env.sim is None:
            env._create_sim()
        obs = env.reset()
        policy.reset_hidden()
        total_r = 0.0

        start_pos = env._prev_pos.copy()

        for step in range(episode_length):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action = policy(obs_t).numpy()
            obs, r, done = env.step(action)
            total_r += r
            if done:
                break

        end_pos = env._prev_pos.copy()
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dist = float(np.sqrt(dx**2 + dy**2))
        heading = float(np.degrees(np.arctan2(dy, dx)))

        distances.append(dist)
        headings.append(heading)
        rewards.append(total_r)

    return {
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "mean_heading": float(np.mean(headings)),
        "std_heading": float(np.std(headings)),
        "mean_reward": float(np.mean(rewards)),
        "distances": distances,
        "headings": headings,
    }


# ---------------------------------------------------------------------------
# Analysis 1: Group ablation specificity
# ---------------------------------------------------------------------------

def run_group_ablation(policy, topo, env, episode_length=1000, n_episodes=2):
    """Ablate functional neuron groups and measure deficits.

    Groups: DN (all), MN (all), intrinsic (all), DN subsets (random halves).
    For connectome: DN/MN/intrinsic map to real neuron classes.
    For random_sparse: same indices but wiring is random.
    """
    results = {}

    # Baseline (intact)
    baseline = evaluate_locomotion(policy, env, n_episodes, episode_length)
    results["intact"] = baseline
    print(f"    Intact: dist={baseline['mean_distance']:.2f}mm, "
          f"heading={baseline['mean_heading']:+.1f}°")

    # Define ablation groups
    groups = {
        "all_dn": topo["dn_indices"],
        "all_mn": topo["mn_indices"],
        "all_intrinsic": topo["intrinsic_indices"],
        "dn_first_half": topo["dn_indices"][:len(topo["dn_indices"])//2],
        "dn_second_half": topo["dn_indices"][len(topo["dn_indices"])//2:],
        "mn_first_half": topo["mn_indices"][:len(topo["mn_indices"])//2],
        "mn_second_half": topo["mn_indices"][len(topo["mn_indices"])//2:],
    }

    for name, indices in groups.items():
        saved = ablate_neurons(policy, indices)
        result = evaluate_locomotion(policy, env, n_episodes, episode_length)
        restore_neurons(policy, saved)

        deficit = 1.0 - result["mean_distance"] / max(baseline["mean_distance"], 1e-6)
        result["deficit_fraction"] = deficit
        result["n_ablated"] = len(indices)
        results[name] = result
        print(f"    {name} ({len(indices)} neurons): "
              f"dist={result['mean_distance']:.2f}mm, deficit={deficit:.1%}")

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Single-neuron ablation entropy (functional distribution)
# ---------------------------------------------------------------------------

def run_single_neuron_ablations(policy, topo, env, n_sample=100,
                                 episode_length=500, n_episodes=1):
    """Ablate individual neurons, measure effect size distribution.

    Metrics:
    - effect_entropy: how distributed are the deficits? (high = diffuse, low = concentrated)
    - top_k_fraction: fraction of total deficit in top-10 neurons
    - zero_effect_fraction: neurons whose removal has <1% effect
    """
    rng = np.random.RandomState(42)
    all_indices = list(range(topo["n_neurons"]))
    sample = sorted(rng.choice(all_indices, size=min(n_sample, len(all_indices)),
                                replace=False))

    # Baseline
    baseline = evaluate_locomotion(policy, env, n_episodes, episode_length)
    base_dist = baseline["mean_distance"]

    effects = {}
    for i, idx in enumerate(sample):
        saved = ablate_neurons(policy, [idx])
        result = evaluate_locomotion(policy, env, n_episodes, episode_length)
        restore_neurons(policy, saved)

        deficit = base_dist - result["mean_distance"]
        effects[idx] = {
            "deficit_mm": float(deficit),
            "deficit_frac": float(deficit / max(base_dist, 1e-6)),
            "distance": result["mean_distance"],
            "heading": result["mean_heading"],
        }

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(sample)} neurons ablated...")

    # Compute summary metrics
    deficits = np.array([e["deficit_frac"] for e in effects.values()])
    abs_deficits = np.abs(deficits)

    # Entropy of effect distribution (higher = more diffuse)
    p = abs_deficits / (abs_deficits.sum() + 1e-10)
    entropy = float(-np.sum(p * np.log(p + 1e-10)))
    max_entropy = float(np.log(len(p)))

    # Top-10 concentration
    sorted_effects = np.sort(abs_deficits)[::-1]
    top10_frac = float(sorted_effects[:10].sum() / (abs_deficits.sum() + 1e-10))

    # Zero-effect fraction (<1% deficit)
    zero_frac = float(np.mean(abs_deficits < 0.01))

    # Classify neurons by type
    dn_set = set(topo["dn_indices"])
    mn_set = set(topo["mn_indices"])
    type_effects = {"dn": [], "mn": [], "intrinsic": []}
    for idx, eff in effects.items():
        if idx in dn_set:
            type_effects["dn"].append(eff["deficit_frac"])
        elif idx in mn_set:
            type_effects["mn"].append(eff["deficit_frac"])
        else:
            type_effects["intrinsic"].append(eff["deficit_frac"])

    return {
        "baseline_distance": base_dist,
        "n_sampled": len(sample),
        "effect_entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / max(max_entropy, 1e-10),
        "top10_concentration": top10_frac,
        "zero_effect_fraction": zero_frac,
        "mean_deficit": float(np.mean(abs_deficits)),
        "std_deficit": float(np.std(abs_deficits)),
        "type_mean_deficits": {
            k: float(np.mean(v)) if v else 0.0
            for k, v in type_effects.items()
        },
        "per_neuron": effects,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Graph modularity (spectral, on W_rec)
# ---------------------------------------------------------------------------

def compute_graph_modularity(policy):
    """Compute Newman modularity Q of the learned W_rec graph.

    Uses spectral bisection on the modularity matrix B = A - k_i*k_j/(2m).
    Higher Q = more modular structure.
    """
    with torch.no_grad():
        W = policy.W_rec.weight.cpu().numpy()
        mask = policy.rec_mask.cpu().numpy()

    # Binary adjacency (learned weights, thresholded by mask)
    A = (np.abs(W) > 1e-6) & (mask > 0.5)
    A = A.astype(float)

    n = A.shape[0]
    k = A.sum(axis=1)  # degree
    m = A.sum() / 2.0
    if m < 1:
        return {"Q": 0.0, "n_communities": 1}

    # Modularity matrix
    B = A - np.outer(k, k) / (2.0 * m)

    # Leading eigenvector for bisection
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        B_sparse = csr_matrix(B)
        eigenvalues, eigenvectors = eigsh(B_sparse, k=1, which="LA")
        s = np.sign(eigenvectors[:, 0])
    except Exception:
        # Fallback: random partition
        s = np.ones(n)
        s[:n//2] = -1.0

    # Compute Q
    Q = float(s.T @ B @ s / (4.0 * m))

    # Count communities in this bisection
    n_pos = int((s > 0).sum())
    n_neg = int((s < 0).sum())

    return {
        "Q": Q,
        "n_communities": 2,
        "community_sizes": [n_pos, n_neg],
    }


# ---------------------------------------------------------------------------
# Analysis 4: Pathway bottleneck density (BFS from DN to MN)
# ---------------------------------------------------------------------------

def compute_pathway_bottlenecks(policy, topo):
    """Trace signal paths from DN->MN through the recurrent graph.

    Metrics:
    - mean_path_length: average shortest path DN->MN
    - bottleneck_ratio: fraction of intrinsic neurons on ANY DN->MN shortest path
    - path_diversity: average number of distinct shortest paths per DN-MN pair
    """
    with torch.no_grad():
        W = policy.W_rec.weight.cpu().numpy()
        mask = policy.rec_mask.cpu().numpy()

    # Build adjacency list (active edges only)
    adj = defaultdict(list)
    n = W.shape[0]
    for i in range(n):
        for j in range(n):
            if mask[i, j] > 0.5 and abs(W[i, j]) > 1e-6:
                adj[j].append(i)  # j->i edge (W convention: W[i,j] = j->i)

    dn_set = set(topo["dn_indices"])
    mn_set = set(topo["mn_indices"])

    # BFS from each DN to all reachable MNs
    path_lengths = []
    intermediates_on_paths = set()
    n_paths_per_pair = []

    # Sample DNs (all would be expensive for 1314 DNs)
    rng = np.random.RandomState(42)
    sample_dns = sorted(rng.choice(topo["dn_indices"],
                                    size=min(50, len(topo["dn_indices"])),
                                    replace=False))

    for dn in sample_dns:
        # BFS
        visited = {dn: 0}
        parents = defaultdict(list)
        queue = [dn]
        layer = 0
        found_mns = {}

        while queue and layer < 10:  # max depth
            next_queue = []
            for node in queue:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited[neighbor] = layer + 1
                        parents[neighbor].append(node)
                        next_queue.append(neighbor)
                        if neighbor in mn_set and neighbor not in found_mns:
                            found_mns[neighbor] = layer + 1
                    elif visited[neighbor] == layer + 1:
                        parents[neighbor].append(node)
            queue = next_queue
            layer += 1

        for mn, dist in found_mns.items():
            path_lengths.append(dist)

            # Trace back to find intermediates
            to_trace = [mn]
            traced = set()
            while to_trace:
                node = to_trace.pop()
                if node in traced:
                    continue
                traced.add(node)
                if node != dn and node != mn and node not in dn_set and node not in mn_set:
                    intermediates_on_paths.add(node)
                for p in parents.get(node, []):
                    to_trace.append(p)

    total_intrinsic = len(topo["intrinsic_indices"])
    bottleneck_ratio = (len(intermediates_on_paths) / max(total_intrinsic, 1))

    return {
        "mean_path_length": float(np.mean(path_lengths)) if path_lengths else float("inf"),
        "std_path_length": float(np.std(path_lengths)) if path_lengths else 0.0,
        "n_dn_mn_pairs_reached": len(path_lengths),
        "n_intermediates_on_paths": len(intermediates_on_paths),
        "total_intrinsic": total_intrinsic,
        "bottleneck_ratio": bottleneck_ratio,
        "path_length_histogram": {
            str(k): int(v)
            for k, v in zip(*np.unique(path_lengths, return_counts=True))
        } if path_lengths else {},
    }


# ---------------------------------------------------------------------------
# Analysis 5: Weight structure — functional weight magnitude
# ---------------------------------------------------------------------------

def compute_weight_structure(policy, topo):
    """Analyze weight magnitude patterns in the learned recurrent layer.

    Connectome should show structured weight patterns (high-weight clusters
    along biological pathways). Random sparse should be uniform.
    """
    with torch.no_grad():
        W = policy.W_rec.weight.cpu().numpy()
        mask = policy.rec_mask.cpu().numpy()

    active = W[mask > 0.5]
    if len(active) == 0:
        return {"n_active": 0}

    # Weight statistics
    abs_active = np.abs(active)

    # Gini coefficient (inequality of weight magnitudes)
    sorted_w = np.sort(abs_active)
    n = len(sorted_w)
    index = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(index * sorted_w) / (n * np.sum(sorted_w))) - (n + 1.0) / n)

    # DN->intrinsic vs intrinsic->MN weight ratios
    dn_set = set(topo["dn_indices"])
    mn_set = set(topo["mn_indices"])

    dn_to_any = []
    any_to_mn = []
    intra_intrinsic = []

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if mask[i, j] > 0.5 and abs(W[i, j]) > 1e-6:
                w = abs(W[i, j])
                if j in dn_set:
                    dn_to_any.append(w)
                if i in mn_set:
                    any_to_mn.append(w)
                if j not in dn_set and j not in mn_set and i not in dn_set and i not in mn_set:
                    intra_intrinsic.append(w)

    return {
        "n_active": int(np.sum(mask > 0.5)),
        "mean_abs_weight": float(np.mean(abs_active)),
        "std_abs_weight": float(np.std(abs_active)),
        "max_abs_weight": float(np.max(abs_active)),
        "weight_gini": gini,
        "dn_outgoing_mean": float(np.mean(dn_to_any)) if dn_to_any else 0.0,
        "mn_incoming_mean": float(np.mean(any_to_mn)) if any_to_mn else 0.0,
        "intrinsic_mean": float(np.mean(intra_intrinsic)) if intra_intrinsic else 0.0,
        "dn_outgoing_count": len(dn_to_any),
        "mn_incoming_count": len(any_to_mn),
        "intrinsic_count": len(intra_intrinsic),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_policy(arch, seed, topo, cfg):
    """Load a trained policy by architecture and seed."""
    from bridge.mn_decoder import _JOINT_PARAMS

    builders = {
        "connectome": lambda: build_connectome_policy(
            topo, cfg.obs_dim, cfg.act_dim, cfg.recurrence_steps, _JOINT_PARAMS),
        "random_sparse": lambda: build_random_sparse_policy(
            topo, seed=99, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim,
            recurrence_steps=cfg.recurrence_steps, joint_params=_JOINT_PARAMS),
        "shuffled": lambda: build_shuffled_policy(
            topo, seed=99, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim,
            recurrence_steps=cfg.recurrence_steps, joint_params=_JOINT_PARAMS),
    }

    policy = builders[arch]()
    params_path = cfg.output_dir / f"{arch}_s{seed}_params.npy"
    if not params_path.exists():
        raise FileNotFoundError(f"No trained params: {params_path}")
    policy.set_flat_params(np.load(params_path))
    return policy


def run_comparison(architectures, topo, cfg, analyses, episode_length=500,
                   n_sample_neurons=100, n_episodes=2):
    """Run full interpretability comparison across architectures."""
    log_dir = cfg.output_dir / "interpretability"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = FlyGymLocomotionEnv(
        episode_length=episode_length,
        warmup_steps=300,
        timestep=1e-4,
        stability_weight=0.1,
        energy_weight=0.01,
    )

    all_results = {}

    for arch, seed in architectures:
        print(f"\n{'='*60}")
        print(f"  {arch.upper()} (seed={seed})")
        print(f"{'='*60}")

        try:
            policy = load_policy(arch, seed, topo, cfg)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        results = {"arch": arch, "seed": seed}

        if "group_ablation" in analyses:
            print(f"\n  [1/5] Group ablation specificity...")
            results["group_ablation"] = run_group_ablation(
                policy, topo, env, episode_length, n_episodes)

        if "single_neuron" in analyses:
            print(f"\n  [2/5] Single-neuron ablation entropy...")
            results["single_neuron"] = run_single_neuron_ablations(
                policy, topo, env, n_sample_neurons, episode_length, 1)

        if "modularity" in analyses:
            print(f"\n  [3/5] Graph modularity...")
            results["modularity"] = compute_graph_modularity(policy)
            print(f"    Q = {results['modularity']['Q']:.4f}")

        if "pathways" in analyses:
            print(f"\n  [4/5] Pathway bottlenecks...")
            results["pathways"] = compute_pathway_bottlenecks(policy, topo)
            print(f"    Mean path length: {results['pathways']['mean_path_length']:.2f}")
            print(f"    Bottleneck ratio: {results['pathways']['bottleneck_ratio']:.3f}")

        if "weight_structure" in analyses:
            print(f"\n  [5/5] Weight structure...")
            results["weight_structure"] = compute_weight_structure(policy, topo)
            print(f"    Gini: {results['weight_structure']['weight_gini']:.4f}")

        key = f"{arch}_s{seed}"
        all_results[key] = results

        # Save incrementally
        _write_json_atomic(log_dir / f"{key}_interpretability.json", results)
        print(f"\n  Saved: {log_dir / f'{key}_interpretability.json'}")

    env.close()
    return all_results


def print_comparison_summary(all_results):
    """Print comparison table across architectures."""
    print(f"\n\n{'='*80}")
    print("INTERPRETABILITY COMPARISON SUMMARY")
    print(f"{'='*80}")

    # Group ablation summary
    print(f"\n--- Group Ablation Deficits ---")
    print(f"{'Arch':<20} {'DN deficit':>12} {'MN deficit':>12} {'Intrinsic':>12}")
    print("-" * 60)
    for key, r in all_results.items():
        if "group_ablation" in r:
            ga = r["group_ablation"]
            dn_d = ga.get("all_dn", {}).get("deficit_fraction", 0)
            mn_d = ga.get("all_mn", {}).get("deficit_fraction", 0)
            in_d = ga.get("all_intrinsic", {}).get("deficit_fraction", 0)
            print(f"{key:<20} {dn_d:>11.1%} {mn_d:>11.1%} {in_d:>11.1%}")

    # Single neuron summary
    print(f"\n--- Single-Neuron Ablation Distribution ---")
    print(f"{'Arch':<20} {'Entropy':>10} {'Top10 conc':>12} {'Zero-eff %':>12}")
    print("-" * 58)
    for key, r in all_results.items():
        if "single_neuron" in r:
            sn = r["single_neuron"]
            print(f"{key:<20} {sn['normalized_entropy']:>9.4f} "
                  f"{sn['top10_concentration']:>11.1%} "
                  f"{sn['zero_effect_fraction']:>11.1%}")

    # Graph metrics
    print(f"\n--- Graph Structure ---")
    print(f"{'Arch':<20} {'Modularity':>12} {'Gini':>10} {'Bottleneck':>12} {'Path len':>10}")
    print("-" * 68)
    for key, r in all_results.items():
        Q = r.get("modularity", {}).get("Q", 0)
        gini = r.get("weight_structure", {}).get("weight_gini", 0)
        bn = r.get("pathways", {}).get("bottleneck_ratio", 0)
        pl = r.get("pathways", {}).get("mean_path_length", 0)
        print(f"{key:<20} {Q:>11.4f} {gini:>9.4f} {bn:>11.3f} {pl:>9.2f}")

    print(f"\n{'='*80}")
    print("Interpretation:")
    print("  Higher modularity Q -> more structured, interpretable")
    print("  Higher Gini -> weight inequality -> structured pathways")
    print("  Lower bottleneck ratio -> fewer relay neurons -> clearer pathways")
    print("  Higher top-10 concentration -> fewer critical neurons -> targeted ablation")
    print("  Lower entropy -> effects concentrated in few neurons -> interpretable")


def main():
    parser = argparse.ArgumentParser(description="Interpretability comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Quick proof: fewer neurons, shorter episodes")
    parser.add_argument("--analysis", nargs="+", default=None,
                        choices=["group_ablation", "single_neuron", "modularity",
                                 "pathways", "weight_structure"],
                        help="Which analyses to run (default: all)")
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--n-sample", type=int, default=100,
                        help="Neurons to sample for single-neuron ablation")
    parser.add_argument("--n-episodes", type=int, default=2)
    args = parser.parse_args()

    if args.quick:
        args.episode_len = 300
        args.n_sample = 30
        args.n_episodes = 1
        args.analysis = ["group_ablation", "modularity", "weight_structure"]

    analyses = args.analysis or [
        "group_ablation", "single_neuron", "modularity",
        "pathways", "weight_structure",
    ]

    cfg = TopologyConfig()
    print("Extracting VNC topology...")
    topo = extract_compressed_vnc(cfg)

    # Find trained architectures
    log_dir = cfg.output_dir
    architectures = []
    for arch in ["connectome", "random_sparse", "shuffled"]:
        for params_file in sorted(log_dir.glob(f"{arch}_s*_params.npy")):
            stem = params_file.stem.replace("_params", "")
            parts = stem.rsplit("_s", maxsplit=1)
            seed = int(parts[1])
            architectures.append((arch, seed))

    if not architectures:
        print("No trained models found. Run run_learning_speed.py first.")
        return

    print(f"Found {len(architectures)} trained models: "
          f"{[f'{a}(s{s})' for a, s in architectures]}")

    t_start = time.time()
    all_results = run_comparison(
        architectures, topo, cfg, analyses,
        episode_length=args.episode_len,
        n_sample_neurons=args.n_sample,
        n_episodes=args.n_episodes,
    )

    print_comparison_summary(all_results)

    # Save full results
    out_path = log_dir / "interpretability" / "comparison_summary.json"
    _write_json_atomic(out_path, {
        "results": {k: v for k, v in all_results.items()},
        "analyses": analyses,
        "config": {
            "episode_length": args.episode_len,
            "n_sample_neurons": args.n_sample,
            "n_episodes": args.n_episodes,
        },
        "total_time_s": time.time() - t_start,
    })
    print(f"\nFull results: {out_path}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
