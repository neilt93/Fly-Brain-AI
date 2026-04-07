"""
FlyGym locomotion environment wrapper for ES training.

Provides a simple episodic interface:
  env.reset() -> obs
  env.step(action) -> obs, reward, done
  env.evaluate(policy) -> total_reward
"""

import numpy as np
from typing import Optional


class FlyGymLocomotionEnv:
    """Episodic FlyGym wrapper for locomotion RL."""

    def __init__(
        self,
        episode_length: int = 1000,
        warmup_steps: int = 300,
        timestep: float = 1e-4,
        stability_weight: float = 0.1,
        energy_weight: float = 0.01,
    ):
        self.episode_length = episode_length
        self.warmup_steps = warmup_steps
        self.timestep = timestep
        self.stability_weight = stability_weight
        self.energy_weight = energy_weight

        self.sim = None
        self.fly = None
        self._step_count = 0
        self._prev_pos = None
        self._prev_joints = None

    def _create_sim(self):
        """Create FlyGym simulation (lazy init for multiprocessing)."""
        import flygym

        self.fly = flygym.Fly(
            enable_adhesion=True,
            init_pose="stretch",
            control="position",
        )
        arena = flygym.arena.FlatTerrain()
        self.sim = flygym.SingleFlySimulation(
            fly=self.fly, arena=arena, timestep=self.timestep
        )

    def reset(self) -> np.ndarray:
        """Reset environment, run warmup, return initial observation."""
        if self.sim is None:
            self._create_sim()

        obs, _ = self.sim.reset()
        self._step_count = 0

        # Warmup: hold init pose with slight ramp
        init_joints = np.array(obs["joints"][0], dtype=np.float32)
        for i in range(self.warmup_steps):
            action = {
                "joints": init_joints,
                "adhesion": np.ones(6, dtype=np.float32),
            }
            try:
                obs, _, term, trunc, _ = self.sim.step(action)
                if term or trunc:
                    break
            except (RuntimeError, ValueError):  # MuJoCo physics instability
                self.sim = None  # Force recreation on next reset
                break

        self._prev_pos = np.array(obs["fly"][0], dtype=np.float64)
        self._prev_joints = np.array(obs["joints"][0], dtype=np.float32)
        return self._extract_obs(obs)

    def step(self, action_vec: np.ndarray) -> tuple:
        """Take one step.

        Args:
            action_vec: (48,) -- joints(42) + adhesion(6)

        Returns:
            obs (90,), reward (float), done (bool)
        """
        joints = action_vec[:42].astype(np.float32)
        adhesion = action_vec[42:48].astype(np.float32)

        action = {"joints": joints, "adhesion": adhesion}

        try:
            obs, _, terminated, truncated, _ = self.sim.step(action)
        except (RuntimeError, ValueError):  # MuJoCo physics instability
            # Force sim recreation on next reset to avoid corrupted state
            self.sim = None
            return np.zeros(90, dtype=np.float32), -1.0, True

        self._step_count += 1
        done = terminated or truncated or (self._step_count >= self.episode_length)

        # Compute reward
        cur_pos = np.array(obs["fly"][0], dtype=np.float64)
        cur_joints = np.array(obs["joints"][0], dtype=np.float32)

        # Forward displacement this step
        dx = cur_pos[0] - self._prev_pos[0]
        reward = float(dx)  # Forward distance (mm)

        # Stability: penalize if fly is falling (z drops below threshold)
        z = cur_pos[2]
        if z < 0.5:
            reward -= 0.1
            done = True  # Fell over

        # Contact stability bonus: reward having 3+ legs in contact
        cf = np.array(obs["contact_forces"])
        magnitudes = np.linalg.norm(cf, axis=1) if cf.ndim == 2 else np.abs(cf)
        legs_in_contact = sum(
            1 for i in range(6)
            if magnitudes[i * 5:(i + 1) * 5].max() > 0.1
        )
        if legs_in_contact >= 3:
            reward += self.stability_weight * 0.001

        # Energy penalty: discourage jittery joint motion
        joint_vel = np.abs(cur_joints - self._prev_joints)
        energy = float(joint_vel.mean())
        reward -= self.energy_weight * energy

        self._prev_pos = cur_pos
        self._prev_joints = cur_joints

        return self._extract_obs(obs), reward, done

    def _extract_obs(self, obs: dict) -> np.ndarray:
        """Extract 90-dim observation vector from FlyGym obs dict."""
        parts = []

        # Joint angles (42)
        joints = np.array(obs["joints"])
        if joints.ndim == 2 and joints.shape[0] >= 2:
            parts.append(joints[0].flatten().astype(np.float32))
            # Joint velocities (42), scaled
            parts.append((joints[1].flatten() * 0.01).astype(np.float32))
        else:
            parts.append(joints.flatten().astype(np.float32))
            parts.append(np.zeros(42, dtype=np.float32))

        # Per-leg contact forces (6)
        cf = np.array(obs["contact_forces"])
        magnitudes = np.linalg.norm(cf, axis=1) if cf.ndim == 2 else cf
        per_leg = np.array([
            magnitudes[i * 5:(i + 1) * 5].max() for i in range(6)
        ])
        per_leg = np.clip(per_leg / 10.0, 0.0, 1.0).astype(np.float32)
        parts.append(per_leg)

        vec = np.concatenate(parts)
        # Ensure exactly 90 dims
        if len(vec) < 90:
            vec = np.pad(vec, (0, 90 - len(vec)))
        elif len(vec) > 90:
            vec = vec[:90]

        return vec

    def evaluate(self, policy, device="cpu") -> float:
        """Run full episode with policy, return total reward."""
        import torch

        obs = self.reset()
        policy.reset_hidden()
        total_reward = 0.0

        for _ in range(self.episode_length):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                action = policy(obs_t).cpu().numpy()

            obs, reward, done = self.step(action)
            total_reward += reward
            if done:
                break

        return total_reward

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
