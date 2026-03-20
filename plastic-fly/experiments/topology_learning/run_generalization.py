"""
Claim 2: Connectome topology enables generalization.

Loads policies trained on forward locomotion (from run_learning_speed.py)
and evaluates zero-shot on held-out tasks:
  1. Turning: reward = heading change toward target
  2. Stability: reward = time standing without falling on uneven terrain

The connectome's modality routing should enable transfer that random
topologies can't match.

Usage:
    python -m experiments.topology_learning.run_generalization
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _write_json_atomic(path: Path, payload, **kwargs):
    """Write JSON atomically so checkpoints stay readable if the run is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, **kwargs)
    for _attempt in range(5):
        try:
            tmp_path.replace(path)
            break
        except PermissionError:
            if _attempt < 4:
                import time; time.sleep(0.05 * (_attempt + 1))
            else:
                import shutil; shutil.copy2(str(tmp_path), str(path))
                tmp_path.unlink(missing_ok=True)


from experiments.topology_learning.config import TopologyConfig
from experiments.topology_learning.extract_topology import extract_compressed_vnc
from experiments.topology_learning.vnc_policy import SparseRecurrentPolicy


def evaluate_turning(policy, policy_config, n_episodes=5, episode_length=1000):
    """Evaluate zero-shot turning ability.

    Inject a bias in the observation (asymmetric contact forces) to simulate
    a lateral stimulus. Measure heading change.
    """
    import flygym

    rewards = []
    for ep in range(n_episodes):
        fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
        arena = flygym.arena.FlatTerrain()
        sim = flygym.SingleFlySimulation(fly=fly, arena=arena, timestep=1e-4)
        obs, _ = sim.reset()

        # Warmup
        init_joints = np.array(obs["joints"][0], dtype=np.float32)
        for _ in range(300):
            action = {"joints": init_joints, "adhesion": np.ones(6, dtype=np.float32)}
            obs, _, t, tr, _ = sim.step(action)
            if t or tr:
                break

        start_pos = np.array(obs["fly"][0])
        policy.reset_hidden()

        for step in range(episode_length):
            # Extract obs
            joints = np.array(obs["joints"])
            parts = [joints[0].flatten().astype(np.float32)]
            parts.append((joints[1].flatten() * 0.01).astype(np.float32))

            # Inject lateral bias: boost left-side contact, suppress right-side
            cf = np.array(obs["contact_forces"])
            magnitudes = np.linalg.norm(cf, axis=1) if cf.ndim == 2 else cf
            per_leg = np.array([
                np.clip(magnitudes[i * 5:(i + 1) * 5].max() / 10.0, 0.0, 1.0)
                for i in range(6)
            ], dtype=np.float32)
            contacts = per_leg
            # Bias: amplify left legs (0,1,2), suppress right (3,4,5)
            contacts[:3] *= 2.0
            contacts[3:] *= 0.5
            parts.append(contacts)

            obs_vec = np.concatenate(parts)
            if len(obs_vec) < 90:
                obs_vec = np.pad(obs_vec, (0, 90 - len(obs_vec)))

            with torch.no_grad():
                obs_t = torch.tensor(obs_vec, dtype=torch.float32)
                action_vec = policy(obs_t).numpy()

            action = {
                "joints": action_vec[:42].astype(np.float32),
                "adhesion": action_vec[42:48].astype(np.float32),
            }

            try:
                obs, _, term, trunc, _ = sim.step(action)
                if term or trunc:
                    break
            except Exception:
                break

        end_pos = np.array(obs["fly"][0])
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        heading = np.degrees(np.arctan2(dy, dx))

        # Reward: heading change (positive = turned left, which matches the bias)
        rewards.append(heading)
        sim.close()

    return {
        "mean_heading": float(np.mean(rewards)),
        "std_heading": float(np.std(rewards)),
        "headings": [float(h) for h in rewards],
    }


def evaluate_endurance(policy, policy_config, n_episodes=5, episode_length=2000):
    """Evaluate endurance: how long can the policy keep the fly standing?"""
    import flygym

    durations = []
    distances = []

    for ep in range(n_episodes):
        fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
        arena = flygym.arena.FlatTerrain()
        sim = flygym.SingleFlySimulation(fly=fly, arena=arena, timestep=1e-4)
        obs, _ = sim.reset()

        init_joints = np.array(obs["joints"][0], dtype=np.float32)
        for _ in range(300):
            action = {"joints": init_joints, "adhesion": np.ones(6, dtype=np.float32)}
            obs, _, t, tr, _ = sim.step(action)
            if t or tr:
                break

        start_pos = np.array(obs["fly"][0])
        policy.reset_hidden()
        survived = 0

        for step in range(episode_length):
            joints = np.array(obs["joints"])
            parts = [joints[0].flatten().astype(np.float32)]
            parts.append((joints[1].flatten() * 0.01).astype(np.float32))
            cf = np.array(obs["contact_forces"])
            per_leg = np.array([
                np.clip(np.linalg.norm(cf[i * 5:(i + 1) * 5]) / 10.0, 0.0, 1.0)
                for i in range(6)
            ], dtype=np.float32)
            parts.append(per_leg)
            obs_vec = np.concatenate(parts)
            if len(obs_vec) < 90:
                obs_vec = np.pad(obs_vec, (0, 90 - len(obs_vec)))

            with torch.no_grad():
                obs_t = torch.tensor(obs_vec, dtype=torch.float32)
                action_vec = policy(obs_t).numpy()

            action = {
                "joints": action_vec[:42].astype(np.float32),
                "adhesion": action_vec[42:48].astype(np.float32),
            }

            try:
                obs, _, term, trunc, _ = sim.step(action)
                z = obs["fly"][0][2]
                if term or trunc or z < 0.5:
                    break
                survived += 1
            except Exception:
                break

        end_pos = np.array(obs["fly"][0])
        dist = float(np.linalg.norm(end_pos - start_pos))
        durations.append(survived)
        distances.append(dist)
        sim.close()

    return {
        "mean_duration": float(np.mean(durations)),
        "mean_distance": float(np.mean(distances)),
        "durations": [int(d) for d in durations],
        "distances": [float(d) for d in distances],
    }


def main():
    cfg = TopologyConfig()
    log_dir = cfg.output_dir

    # Find all trained models
    param_files = sorted(log_dir.glob("*_params.npy"))
    curve_files = sorted(log_dir.glob("*_curve.json"))

    if not param_files:
        print(f"No trained models found in {log_dir}")
        print("Run run_learning_speed.py first.")
        return

    print(f"Found {len(param_files)} trained models")

    # Extract topology once (1.1GB feather read)
    topo = extract_compressed_vnc(cfg)
    from experiments.topology_learning.vnc_policy import (
        build_connectome_policy, build_dense_policy,
        build_random_sparse_policy, build_shuffled_policy,
    )
    from bridge.mn_decoder import _JOINT_PARAMS
    joint_params = _JOINT_PARAMS

    results = []
    for pf in param_files:
        # Parse arch and seed from filename: e.g. "random_sparse_s42_params"
        stem = pf.stem.replace("_params", "")  # "random_sparse_s42"
        # Split on last "_s" to handle underscores in arch name
        parts = stem.rsplit("_s", maxsplit=1)
        arch = parts[0]
        seed = int(parts[1])

        # Load curve to get policy config
        cf = log_dir / f"{arch}_s{seed}_curve.json"
        if not cf.exists():
            print(f"  Skipping {arch} s{seed}: no curve file")
            continue

        with open(cf) as f:
            curve_data = json.load(f)

        hidden_dim = curve_data["hidden_dim"]

        print(f"\n  Evaluating {arch} (seed={seed}, hidden={hidden_dim})...")

        builders = {
            "connectome": lambda: build_connectome_policy(
                topo, cfg.obs_dim, cfg.act_dim, cfg.recurrence_steps, joint_params),
            "dense": lambda: build_dense_policy(
                topo["n_neurons"], cfg.obs_dim, cfg.act_dim, cfg.recurrence_steps, joint_params,
                dn_indices=topo["dn_indices"], mn_indices=topo["mn_indices"]),
            "random_sparse": lambda: build_random_sparse_policy(
                topo, seed=seed, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim,
                recurrence_steps=cfg.recurrence_steps, joint_params=joint_params),
            "shuffled": lambda: build_shuffled_policy(
                topo, seed=seed, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim,
                recurrence_steps=cfg.recurrence_steps, joint_params=joint_params),
        }

        policy = builders[arch]()
        params = np.load(pf)
        policy.set_flat_params(params)

        # Zero-shot turning
        turn_result = evaluate_turning(policy, None, n_episodes=3, episode_length=1000)
        print(f"    Turning: heading={turn_result['mean_heading']:+.1f}° ± {turn_result['std_heading']:.1f}°")

        # Endurance
        endurance_result = evaluate_endurance(policy, None, n_episodes=3, episode_length=2000)
        print(f"    Endurance: {endurance_result['mean_duration']:.0f} steps, "
              f"{endurance_result['mean_distance']:.2f}mm")

        results.append({
            "arch": arch,
            "seed": seed,
            "forward_reward": curve_data["final_mean_reward"],
            "turning": turn_result,
            "endurance": endurance_result,
        })

    # Summary
    print(f"\n\n{'='*70}")
    print("GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Arch':<16} {'Fwd reward':>12} {'Turn heading':>14} {'Endurance':>10} {'Distance':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['arch']:<16} {r['forward_reward']:>+12.4f} "
              f"{r['turning']['mean_heading']:>+12.1f}° "
              f"{r['endurance']['mean_duration']:>9.0f} "
              f"{r['endurance']['mean_distance']:>9.2f}mm")

    # Save
    out_path = log_dir / "generalization.json"
    _write_json_atomic(out_path, results)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
