"""Configuration for topology learning experiments."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TopologyConfig:
    # --- Network ---
    obs_dim: int = 90           # joints(42) + joint_vel(42) + contacts(6)
    act_dim: int = 48           # joints(42) + adhesion_logits(6)
    top_k_intrinsic: int = 500  # compressed VNC: keep top-K by degree
    recurrence_steps: int = 3   # unrolled sparse recurrence per forward pass

    # --- ES optimizer ---
    pop_size: int = 32
    sigma: float = 0.02         # perturbation noise std
    lr: float = 0.01
    weight_decay: float = 0.005
    antithetic: bool = True     # mirrored sampling

    # --- Environment ---
    episode_length: int = 1000  # body steps per episode
    warmup_steps: int = 300     # warmup with ramped neutral pose
    timestep: float = 1e-4      # FlyGym physics timestep (0.1ms)

    # --- Reward ---
    stability_weight: float = 0.1
    energy_weight: float = 0.01

    # --- Training ---
    n_generations: int = 400
    n_workers: int = 4          # parallel episode evaluators
    log_interval: int = 10      # print every N generations

    # --- Paths ---
    manc_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "manc")
    mn_mapping_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "mn_joint_mapping.json")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "logs" / "topology_learning")
