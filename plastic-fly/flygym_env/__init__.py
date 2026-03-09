"""
FlyGym environment wrapper for the plastic fly controller project.

Provides a unified interface around flygym.SingleFlySimulation with:
- Configurable terrain (flat, gapped, blocks, mixed)
- Observation/action logging per step
- Built-in video recording
"""

import numpy as np
import pickle
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import flygym
    import flygym.arena
    HAS_FLYGYM = True
except ImportError:
    HAS_FLYGYM = False


@dataclass
class StepRecord:
    """Single timestep record."""
    step: int
    obs: dict
    action: np.ndarray
    reward: float = 0.0
    info: dict = field(default_factory=dict)


def make_terrain(name: str):
    """Create a flygym arena object by name."""
    name = name.lower()
    if name == "flat":
        return flygym.arena.FlatTerrain()
    elif name == "gapped":
        return flygym.arena.GappedTerrain()
    elif name == "blocks":
        return flygym.arena.BlocksTerrain()
    elif name == "mixed":
        return flygym.arena.MixedTerrain()
    else:
        raise ValueError(f"Unknown terrain: {name}")


class FlyEnv:
    """Wrapper around flygym.SingleFlySimulation for experiments.

    Observation layout (flygym v1.2):
        obs['joints']:         (3, 42)  — angles, velocities, torques
        obs['fly']:            (4, 3)   — position, velocity, orientation, ang_vel
        obs['contact_forces']: (30, 3)  — 30 tarsal sensors x 3D force
        obs['end_effectors']:  (6, 3)   — 6 leg tip positions
        obs['fly_orientation']: (3,)    — roll, pitch, yaw
        obs['cardinal_vectors']: (3, 3) — forward, right, up vectors

    Action format:
        {'joints': np.ndarray(42)}  — target joint angles (position control)
    """

    def __init__(
        self,
        terrain: str = "flat",
        timestep: float = 1e-4,
        enable_adhesion: bool = False,
    ):
        if not HAS_FLYGYM:
            raise ImportError("flygym is not installed. Run: pip install flygym")

        self.terrain_name = terrain
        self.timestep = timestep
        self.history: list[StepRecord] = []
        self._step_count = 0

        arena = make_terrain(terrain)
        self.fly_obj = flygym.Fly(enable_adhesion=enable_adhesion)
        self.sim = flygym.SingleFlySimulation(
            fly=self.fly_obj,
            arena=arena,
            timestep=timestep,
        )

        self.num_dofs = len(self.fly_obj.actuated_joints)
        self.joint_names = list(self.fly_obj.actuated_joints)

    def reset(self):
        """Reset the environment and clear history."""
        obs, info = self.sim.reset()
        self.history.clear()
        self._step_count = 0
        return obs, info

    def step(self, action: np.ndarray):
        """Step the environment and log.

        Args:
            action: raw array of 42 joint targets. Wrapped in {'joints': action}.
        """
        action_dict = {"joints": action}
        obs, reward, terminated, truncated, info = self.sim.step(action_dict)
        record = StepRecord(
            step=self._step_count,
            obs={k: np.array(v) if hasattr(v, '__array__') else v
                 for k, v in obs.items()},
            action=np.array(action),
            reward=float(reward),
            info=info,
        )
        self.history.append(record)
        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def save_log(self, path: str):
        """Save step history to pickle."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def close(self):
        """Close the environment."""
        self.sim.close()

    def get_fly_position(self, obs: dict) -> np.ndarray:
        """Extract fly position (x, y, z) from observation."""
        return np.array(obs["fly"][0])

    def get_fly_velocity(self, obs: dict) -> np.ndarray:
        """Extract fly velocity from observation."""
        return np.array(obs["fly"][1])
