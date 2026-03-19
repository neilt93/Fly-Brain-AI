"""
Claim 1: Connectome topology accelerates learning.

Trains 4 architectures on FlyGym forward locomotion using ES:
  1. Connectome — real MANC VNC topology
  2. Dense — fully connected, same hidden size
  3. Random sparse — Erdos-Renyi, same density
  4. Shuffled — same degree distribution, permuted targets

Produces learning curves: reward vs generation for each architecture.

Usage:
    # Quick proof (2 architectures, 100 generations, ~30 min)
    python -m experiments.topology_learning.run_learning_speed --quick

    # Full overnight (4 architectures x 2 seeds, 400 generations)
    python -m experiments.topology_learning.run_learning_speed

    # Single architecture test
    python -m experiments.topology_learning.run_learning_speed --arch connectome --gens 50
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.topology_learning.config import TopologyConfig
from experiments.topology_learning.extract_topology import extract_compressed_vnc
from experiments.topology_learning.vnc_policy import (
    build_connectome_policy, build_dense_policy,
    build_random_sparse_policy, build_shuffled_policy,
)
from experiments.topology_learning.es_optimizer import OpenAIES


def _write_json_atomic(path: Path, payload, **kwargs):
    """Write JSON atomically so checkpoints stay readable if the run is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, **kwargs)
    tmp_path.replace(path)


# Joint params from mn_decoder.py for safe output clamping
def _load_joint_params():
    from bridge.mn_decoder import _JOINT_PARAMS
    return _JOINT_PARAMS


# --- Worker process globals (initialized once per worker, not per evaluation) ---
_worker_policy = None
_worker_env = None


def _worker_init(policy_config, env_config):
    """Initialize worker: build policy and env once (called by pool initializer)."""
    global _worker_policy, _worker_env

    from experiments.topology_learning.flygym_env import FlyGymLocomotionEnv
    from experiments.topology_learning.vnc_policy import SparseRecurrentPolicy
    import torch

    mask = (torch.tensor(policy_config["mask"], dtype=torch.float32)
            if policy_config["mask"] is not None else None)
    joint_rest = (np.array(policy_config["joint_rest"])
                  if policy_config["joint_rest"] is not None else None)
    joint_amp = (np.array(policy_config["joint_amp"])
                 if policy_config["joint_amp"] is not None else None)

    _worker_policy = SparseRecurrentPolicy(
        obs_dim=policy_config["obs_dim"],
        act_dim=policy_config["act_dim"],
        hidden_dim=policy_config["hidden_dim"],
        mask=mask,
        dn_indices=policy_config["dn_indices"],
        mn_indices=policy_config["mn_indices"],
        joint_rest=joint_rest,
        joint_amp=joint_amp,
        recurrence_steps=policy_config["recurrence_steps"],
        n_evals=policy_config.get("n_evals", 2),
    )
    _worker_env = FlyGymLocomotionEnv(
        episode_length=env_config["episode_length"],
        warmup_steps=env_config["warmup_steps"],
        timestep=env_config["timestep"],
        stability_weight=env_config["stability_weight"],
        energy_weight=env_config["energy_weight"],
    )


def _evaluate_single(flat_params):
    """Evaluate one parameter vector using pre-initialized worker policy/env.

    Runs n_evals episodes and returns the mean reward (reduces crash noise).
    """
    global _worker_policy, _worker_env
    _worker_policy.set_flat_params(flat_params)
    total = 0.0
    n = _worker_policy.n_evals
    for _ in range(n):
        total += _worker_env.evaluate(_worker_policy)
    return total / n


def _policy_to_config(policy) -> dict:
    """Serialize policy config for worker initializer (sent once per pool, not per eval)."""
    mask = policy.rec_mask.cpu().numpy().tolist() if policy.rec_mask is not None else None
    joint_rest = policy.joint_rest.cpu().numpy().tolist() if policy.joint_rest is not None else None
    joint_amp = policy.joint_amp.cpu().numpy().tolist() if policy.joint_amp is not None else None
    return {
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
        "hidden_dim": policy.hidden_dim,
        "mask": mask,
        "dn_indices": policy.dn_idx.cpu().tolist(),
        "mn_indices": policy.mn_idx.cpu().tolist(),
        "joint_rest": joint_rest,
        "joint_amp": joint_amp,
        "recurrence_steps": policy.recurrence_steps,
        "n_evals": policy.n_evals,
    }


def _build_result(arch_name, seed, n_params, n_total, policy, cfg, curve,
                   total_time=0.0):
    """Build result dict from curve data."""
    total_episodes = len(curve) * cfg.pop_size
    return {
        "arch": arch_name,
        "seed": seed,
        "n_params": n_params,
        "n_total_params": n_total,
        "hidden_dim": policy.hidden_dim,
        "n_generations": len(curve),
        "pop_size": cfg.pop_size,
        "total_episodes": total_episodes,
        "total_time_s": total_time,
        "curve": curve,
        "final_mean_reward": curve[-1]["mean_reward"] if curve else 0.0,
        "final_max_reward": curve[-1]["max_reward"] if curve else 0.0,
        "best_mean_reward": max((c["mean_reward"] for c in curve), default=0.0),
    }


def run_training(
    arch_name: str,
    policy,
    cfg: TopologyConfig,
    seed: int = 42,
) -> dict:
    """Train one architecture with ES, return learning curve."""
    np.random.seed(seed)

    n_params = policy.n_params  # ES-optimizable (active only)
    n_total = policy.n_total_params
    print(f"\n{'='*60}")
    print(f"  {arch_name.upper()} | seed={seed}")
    print(f"  params: {n_total:,} total, {n_params:,} ES-active")
    print(f"  hidden_dim: {policy.hidden_dim}, pop_size: {cfg.pop_size}")
    print(f"{'='*60}")

    es = OpenAIES(
        n_params=n_params,
        sigma=cfg.sigma,
        lr=cfg.lr,
        pop_size=cfg.pop_size,
        antithetic=cfg.antithetic,
        weight_decay=cfg.weight_decay,
    )
    es.set_params(policy.get_flat_params())

    policy_config = _policy_to_config(policy)
    env_config = {
        "episode_length": cfg.episode_length,
        "warmup_steps": cfg.warmup_steps,
        "timestep": cfg.timestep,
        "stability_weight": cfg.stability_weight,
        "energy_weight": cfg.energy_weight,
    }

    curve = []
    t_start = time.time()
    checkpoint_dir = cfg.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_gen = 0

    # --- Resume from checkpoint if available ---
    ckpt_path = checkpoint_dir / f"{arch_name}_s{seed}_ckpt.json"
    theta_path = checkpoint_dir / f"{arch_name}_s{seed}_theta.npy"
    if ckpt_path.exists() and theta_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        if ckpt["n_params"] == n_params and ckpt["pop_size"] == cfg.pop_size:
            start_gen = ckpt["gen"] + 1
            curve = ckpt["curve"]
            saved_theta = np.load(theta_path)
            es.set_params(saved_theta)
            es.generation = start_gen
            print(f"  RESUMED from checkpoint: gen {ckpt['gen']}, "
                  f"mean={curve[-1]['mean_reward']:+.4f}")
        else:
            print(f"  Checkpoint incompatible (params/pop changed), starting fresh")

    if start_gen >= cfg.n_generations:
        print(f"  Already completed {cfg.n_generations} generations, skipping")
        return _build_result(arch_name, seed, n_params, n_total, policy, cfg, curve)

    # Create pool once (not per generation) — workers keep policy/env alive
    pool = None
    if cfg.n_workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=cfg.n_workers,
            initializer=_worker_init,
            initargs=(policy_config, env_config),
        )

    try:
        for gen in range(start_gen, cfg.n_generations):
            population = es.ask()

            # Evaluate population (parallel or sequential)
            if pool is not None:
                rewards = list(pool.map(_evaluate_single, population))
            else:
                rewards = []
                for p in population:
                    policy.set_flat_params(p)
                    from experiments.topology_learning.flygym_env import FlyGymLocomotionEnv
                    env = FlyGymLocomotionEnv(
                        episode_length=cfg.episode_length,
                        warmup_steps=cfg.warmup_steps,
                    )
                    r = env.evaluate(policy)
                    env.close()
                    rewards.append(r)

            es.tell(rewards)

            mean_r = float(np.mean(rewards))
            max_r = float(np.max(rewards))
            min_r = float(np.min(rewards))
            elapsed = time.time() - t_start

            curve.append({
                "gen": gen,
                "mean_reward": mean_r,
                "max_reward": max_r,
                "min_reward": min_r,
                "elapsed_s": elapsed,
            })

            if gen % cfg.log_interval == 0:
                eps_per_s = (gen - start_gen + 1) * cfg.pop_size / max(elapsed, 0.1)
                print(f"  gen {gen:4d}: mean={mean_r:+.4f} max={max_r:+.4f} "
                      f"({elapsed:.0f}s, {eps_per_s:.1f} eps/s)")

                # Checkpoint: save curve + ES params every log_interval
                ckpt = {
                    "arch": arch_name, "seed": seed,
                    "gen": gen, "curve": curve,
                    "n_params": n_params, "n_total_params": n_total,
                    "hidden_dim": policy.hidden_dim,
                    "pop_size": cfg.pop_size,
                }
                ckpt_path = checkpoint_dir / f"{arch_name}_s{seed}_ckpt.json"
                _write_json_atomic(ckpt_path, ckpt)
                np.save(
                    checkpoint_dir / f"{arch_name}_s{seed}_theta.npy",
                    es.best_params,
                )

    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    total_time = time.time() - t_start
    best_params = es.best_params

    # Save best params
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    params_path = cfg.output_dir / f"{arch_name}_s{seed}_params.npy"
    np.save(params_path, best_params)

    result = _build_result(arch_name, seed, n_params, n_total, policy, cfg, curve,
                           total_time=total_time)
    print(f"\n  Done: {result['total_episodes']} episodes in {total_time:.0f}s")
    print(f"  Final: mean={curve[-1]['mean_reward']:+.4f}, "
          f"max={curve[-1]['max_reward']:+.4f}")
    print(f"  Best mean: {result['best_mean_reward']:+.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Topology learning speed experiment")
    parser.add_argument("--arch", nargs="+", default=None,
                        choices=["connectome", "dense", "random_sparse", "shuffled"],
                        help="Architectures to train (default: all 4)")
    parser.add_argument("--seeds", type=int, default=2, help="Seeds per architecture")
    parser.add_argument("--gens", type=int, default=400, help="Generations per run")
    parser.add_argument("--pop", type=int, default=32, help="Population size")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--episode-len", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--top-k", type=int, default=500, help="Top-K intrinsic neurons")
    parser.add_argument("--quick", action="store_true",
                        help="Quick proof: connectome vs random_sparse, 100 gens, 1 seed")
    args = parser.parse_args()

    if args.quick:
        args.arch = ["connectome", "random_sparse"]
        args.seeds = 1
        args.gens = 100
        args.workers = 2

    cfg = TopologyConfig(
        n_generations=args.gens,
        pop_size=args.pop,
        n_workers=args.workers,
        episode_length=args.episode_len,
        top_k_intrinsic=args.top_k,
    )

    architectures = args.arch or ["connectome", "dense", "random_sparse", "shuffled"]
    seed_list = [42 + i * 37 for i in range(args.seeds)]

    # --- Extract topology ---
    print("Extracting compressed VNC topology from MANC data...")
    t0 = time.time()
    topo = extract_compressed_vnc(cfg)
    print(f"  Extraction: {time.time()-t0:.1f}s")

    joint_params = _load_joint_params()

    # --- Build policies (all share same I/O constraint: input→DN, output→MN) ---
    builders = {
        "connectome": lambda run_seed: build_connectome_policy(
            topo, cfg.obs_dim, cfg.act_dim, cfg.recurrence_steps, joint_params),
        "dense": lambda run_seed: build_dense_policy(
            topo["n_neurons"], cfg.obs_dim, cfg.act_dim, cfg.recurrence_steps,
            joint_params, dn_indices=topo["dn_indices"], mn_indices=topo["mn_indices"]),
        "random_sparse": lambda run_seed: build_random_sparse_policy(
            topo, seed=run_seed, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim,
            recurrence_steps=cfg.recurrence_steps, joint_params=joint_params),
        "shuffled": lambda run_seed: build_shuffled_policy(
            topo, seed=run_seed, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim,
            recurrence_steps=cfg.recurrence_steps, joint_params=joint_params),
    }

    # --- Run training ---
    all_results = []
    t_total = time.time()

    for arch_name in architectures:
        for seed in seed_list:
            policy = builders[arch_name](seed)
            result = run_training(arch_name, policy, cfg, seed=seed)
            all_results.append(result)

            # Save incrementally
            out_path = cfg.output_dir / f"{arch_name}_s{seed}_curve.json"
            _write_json_atomic(out_path, result)
            print(f"  Saved: {out_path}")

    # --- Summary ---
    total_time = time.time() - t_total
    print(f"\n\n{'='*70}")
    print(f"TOPOLOGY LEARNING SUMMARY ({len(all_results)} runs, {total_time:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Arch':<16} {'Seed':>4} {'ES params':>10} {'Total':>10} "
          f"{'Final mean':>12} {'Best mean':>12} {'Time':>8}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['arch']:<16} {r['seed']:>4} {r['n_params']:>10,} {r['n_total_params']:>10,} "
              f"{r['final_mean_reward']:>+12.4f} {r['best_mean_reward']:>+12.4f} "
              f"{r['total_time_s']:>7.0f}s")

    # Aggregate by architecture
    print(f"\n  Per-architecture averages:")
    for arch in architectures:
        arch_results = [r for r in all_results if r["arch"] == arch]
        if arch_results:
            mean_final = np.mean([r["final_mean_reward"] for r in arch_results])
            mean_best = np.mean([r["best_mean_reward"] for r in arch_results])
            print(f"    {arch:<16}: final={mean_final:+.4f}, best={mean_best:+.4f}")

    # Save full summary
    summary_path = cfg.output_dir / "summary.json"
    _write_json_atomic(summary_path, {
        "results": all_results,
        "config": {
            "architectures": architectures,
            "seeds": seed_list,
            "n_generations": cfg.n_generations,
            "pop_size": cfg.pop_size,
            "episode_length": cfg.episode_length,
            "top_k_intrinsic": cfg.top_k_intrinsic,
        },
        "total_time_s": total_time,
    })
    print(f"\n  Full summary: {summary_path}")


if __name__ == "__main__":
    main()
