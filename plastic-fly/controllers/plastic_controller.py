"""
Plastic sparse recurrent controller for NeuroMechFly.

Same architecture as fixed_controller, but with a local Hebbian-style
learning rule that updates recurrent weights online — no backprop.

Plasticity rule:
  Δw_ij = η * error_scale * (pre_i * post_j - λ * (w_ij - w_ij_init))
  (Hebbian with homeostatic decay toward initial weights)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from flygym.examples.locomotion import PreprogrammedSteps
from flygym.examples.locomotion.cpg_controller import CPGNetwork
from flygym.preprogrammed import get_cpg_biases


LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]


class PlasticRecurrentNet(nn.Module):
    """Sparse recurrent network with local Hebbian plasticity."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 6,
        sparsity: float = 0.8,
        learning_rate: float = 1e-5,
        weight_decay: float = 1.0,
        plasticity_cap: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.plasticity_cap = plasticity_cap

        rng = torch.Generator().manual_seed(seed)

        # Input projection (fixed)
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=True)

        # Recurrent weights — these are plastic
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        mask = (torch.rand(hidden_dim, hidden_dim, generator=rng) > sparsity).float()
        self.register_buffer("rec_mask", mask)

        # Output projection (fixed)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
            nn.init.xavier_uniform_(self.W_rec.weight, gain=0.3)
            nn.init.xavier_uniform_(self.W_out.weight, gain=0.1)
            self.W_rec.weight.mul_(mask)
            self.W_out.bias.zero_()

        # Store initial recurrent weights for homeostatic decay
        self.register_buffer("W_rec_init", self.W_rec.weight.data.clone(), persistent=False)

        # Hidden state
        self.register_buffer("h", torch.zeros(1, hidden_dim), persistent=False)
        self.register_buffer("h_prev", torch.zeros(1, hidden_dim), persistent=False)

        # Freeze non-plastic parameters
        for param in self.W_in.parameters():
            param.requires_grad = False
        for param in self.W_out.parameters():
            param.requires_grad = False
        self.W_rec.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_rec = self.W_rec.weight * self.rec_mask

        h_pre = self.h.clone()
        h_new = torch.tanh(
            self.W_in(x) + torch.mm(self.h, masked_rec.T)
        )

        # Prediction error: surprise signal modulates plasticity
        pred_error = (h_new - self.h_prev).abs().mean().item()

        # Apply Hebbian plasticity
        self._update_weights(h_pre, h_new, pred_error)

        self.h_prev = self.h.clone()
        self.h = h_new

        return self.W_out(self.h)

    def _update_weights(self, pre: torch.Tensor, post: torch.Tensor, pred_error: float):
        """Local Hebbian update with homeostatic decay."""
        # Clamp prediction error to prevent runaway learning
        pred_error = min(pred_error, 1.0)
        effective_lr = self.lr * (1.0 + pred_error)

        # Hebbian outer product (normalized by hidden dim for stability)
        hebbian = (pre.T @ post) / self.hidden_dim

        # Homeostatic decay toward initial weights
        decay = self.weight_decay * (self.W_rec.weight.data - self.W_rec_init)

        delta = effective_lr * hebbian - self.lr * decay
        delta *= self.rec_mask

        self.W_rec.weight.data += delta
        self.W_rec.weight.data.clamp_(-self.plasticity_cap, self.plasticity_cap)

    def reset_state(self):
        self.h.zero_()
        self.h_prev.zero_()

    def reset_weights(self):
        self.W_rec.weight.data.copy_(self.W_rec_init)

    def get_weight_drift(self) -> float:
        diff = self.W_rec.weight.data - self.W_rec_init
        return (diff * self.rec_mask).norm().item()

    def get_plasticity_stats(self) -> dict:
        w = self.W_rec.weight.data * self.rec_mask
        active = self.rec_mask.bool()
        return {
            "weight_drift": self.get_weight_drift(),
            "weight_mean": w[active].mean().item(),
            "weight_std": w[active].std().item(),
            "weight_max": w[active].abs().max().item(),
            "active_connections": self.rec_mask.sum().item(),
        }


class PlasticController:
    """Plastic (online-adapting) controller for NeuroMechFly.

    Same interface as FixedController but with Hebbian plasticity
    on the recurrent layer.
    """

    def __init__(
        self,
        num_dofs: int = 42,
        obs_dim: int = 90,
        hidden_dim: int = 64,
        sparsity: float = 0.8,
        learning_rate: float = 1e-5,
        weight_decay: float = 1.0,
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

        # CPG network
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

        self.steps = PreprogrammedSteps()

        # Plastic recurrent network
        self.net = PlasticRecurrentNet(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=6,
            sparsity=sparsity,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        ).to(device)

    def _build_obs_vector(self, obs: dict) -> torch.Tensor:
        """Extract observation vector from flygym obs dict."""
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
        """Compute action with online plasticity."""
        self.cpg.step()

        obs_vec = self._build_obs_vector(obs)
        # forward() includes the plasticity update
        modulation = self.net(obs_vec).cpu().numpy().flatten()
        modulation = np.tanh(modulation) * self.modulation_scale

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
        adhesion = []
        for i, leg in enumerate(LEGS):
            adhesion.append(
                self.steps.get_adhesion_onoff(leg, self.cpg.curr_phases[i])
            )
        return np.array(adhesion).astype(int)

    def reset(self):
        """Reset controller state (keep adapted weights)."""
        self.cpg.reset()
        self.net.reset_state()

    def hard_reset(self):
        """Full reset including weights."""
        self.reset()
        self.net.reset_weights()

    def get_weights(self) -> dict:
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def get_weight_drift(self) -> float:
        return self.net.get_weight_drift()

    def get_plasticity_stats(self) -> dict:
        return self.net.get_plasticity_stats()

    def get_connectivity_stats(self) -> dict:
        mask = self.net.rec_mask
        total = mask.numel()
        nonzero = mask.sum().item()
        return {
            "total_recurrent": total,
            "active_connections": int(nonzero),
            "sparsity": 1.0 - nonzero / total,
        }
