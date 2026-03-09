"""
Fixed sparse recurrent controller for NeuroMechFly.

Architecture:
- CPG base gait from flygym's PreprogrammedSteps (stable walking)
- Sparse recurrent network modulates CPG amplitude per-leg
- Weights are frozen after initialization — no online adaptation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from flygym.examples.locomotion import PreprogrammedSteps
from flygym.examples.locomotion.cpg_controller import CPGNetwork
from flygym.preprogrammed import get_cpg_biases


LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]


class SparseRecurrentNet(nn.Module):
    """Sparse recurrent network for modulating locomotion.

    Input: proprioceptive obs (joint angles + velocities) + contact sensors
    Hidden: sparse recurrent layer with fixed connectivity mask
    Output: per-leg amplitude modulation (6 values)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 6,
        sparsity: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        rng = torch.Generator().manual_seed(seed)

        self.W_in = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        mask = (torch.rand(hidden_dim, hidden_dim, generator=rng) > sparsity).float()
        self.register_buffer("rec_mask", mask)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
            nn.init.xavier_uniform_(self.W_rec.weight, gain=0.3)
            nn.init.xavier_uniform_(self.W_out.weight, gain=0.1)
            self.W_rec.weight.mul_(mask)
            # Bias output toward 0 (no modulation by default)
            self.W_out.bias.zero_()

        self.register_buffer("h", torch.zeros(1, hidden_dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_rec = self.W_rec.weight * self.rec_mask
        self.h = torch.tanh(
            self.W_in(x) + torch.mm(self.h, masked_rec.T)
        )
        return self.W_out(self.h)

    def reset_state(self):
        self.h.zero_()


class FixedController:
    """Fixed (non-adapting) controller for NeuroMechFly.

    Uses flygym's CPGNetwork + PreprogrammedSteps for stable base gait.
    A sparse recurrent network modulates per-leg amplitude based on
    sensory feedback, but weights are frozen.
    """

    def __init__(
        self,
        num_dofs: int = 42,
        obs_dim: int = 90,
        hidden_dim: int = 64,
        sparsity: float = 0.8,
        cpg_freq: float = 12.0,
        cpg_amplitude: float = 1.0,
        timestep: float = 1e-4,
        modulation_scale: float = 0.15,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.num_dofs = num_dofs
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.modulation_scale = modulation_scale
        self.timestep = timestep

        # CPG network (6 oscillators, one per leg)
        phase_biases = get_cpg_biases("tripod")
        coupling_weights = (phase_biases > 0).astype(float) * 10
        self.cpg = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=np.ones(6) * cpg_freq,
            intrinsic_amps=np.ones(6) * cpg_amplitude,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=np.ones(6) * 20,
            seed=seed,
        )

        # Preprogrammed step patterns
        self.steps = PreprogrammedSteps()

        # Recurrent network for sensory modulation
        self.net = SparseRecurrentNet(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=6,  # per-leg amplitude modulation
            sparsity=sparsity,
            seed=seed,
        ).to(device)

        # Freeze all weights
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

    def _build_obs_vector(self, obs: dict) -> torch.Tensor:
        """Extract observation vector from flygym obs dict.

        obs['joints']:         (3, 42) — angles, velocities, torques
        obs['contact_forces']: (30, 3) — 30 tarsal sensors x 3D force

        Returns: (1, obs_dim) tensor with angles(42) + velocities(42) + per-leg contact(6) = 90
        """
        parts = []

        if "joints" in obs:
            joints = np.array(obs["joints"])
            if joints.ndim == 2 and joints.shape[0] >= 2:
                parts.append(joints[0].flatten())
                parts.append(joints[1].flatten() * 0.01)
            else:
                parts.append(joints.flatten())

        if "contact_forces" in obs:
            cf = np.array(obs["contact_forces"])
            magnitudes = np.linalg.norm(cf, axis=1)
            per_leg = np.array([
                magnitudes[i*5:(i+1)*5].max() for i in range(6)
            ])
            per_leg = np.clip(per_leg / 10.0, 0.0, 1.0)
            parts.append(per_leg)

        if not parts:
            return torch.zeros(1, self.obs_dim, device=self.device)

        vec = np.concatenate(parts).astype(np.float32)
        if len(vec) < self.obs_dim:
            vec = np.pad(vec, (0, self.obs_dim - len(vec)))
        elif len(vec) > self.obs_dim:
            vec = vec[:self.obs_dim]

        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def get_action(self, obs: dict, timestep: float = 1e-4) -> np.ndarray:
        """Compute joint angle targets using CPG + network modulation.

        Returns raw array of 42 joint angles (caller wraps in {'joints': ...}).
        """
        # Step CPG
        self.cpg.step()

        # Get network modulation
        obs_vec = self._build_obs_vector(obs)
        modulation = self.net(obs_vec).cpu().numpy().flatten()  # (6,)
        modulation = np.tanh(modulation) * self.modulation_scale

        # Generate joint angles from CPG phases + modulated magnitudes
        joints_angles = []
        for i, leg in enumerate(LEGS):
            mag = self.cpg.curr_magnitudes[i] + modulation[i]
            mag = np.clip(mag, 0.0, 1.5)
            angles = self.steps.get_joint_angles(
                leg, self.cpg.curr_phases[i], mag
            )
            joints_angles.append(angles)

        return np.concatenate(joints_angles)

    def get_adhesion(self) -> np.ndarray:
        """Get adhesion on/off for each leg based on CPG phase."""
        adhesion = []
        for i, leg in enumerate(LEGS):
            adhesion.append(
                self.steps.get_adhesion_onoff(leg, self.cpg.curr_phases[i])
            )
        return np.array(adhesion).astype(int)

    def reset(self):
        """Reset controller state."""
        self.cpg.reset()
        self.net.reset_state()

    def get_weights(self) -> dict:
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def get_connectivity_stats(self) -> dict:
        mask = self.net.rec_mask
        total = mask.numel()
        nonzero = mask.sum().item()
        return {
            "total_recurrent": total,
            "active_connections": int(nonzero),
            "sparsity": 1.0 - nonzero / total,
        }
