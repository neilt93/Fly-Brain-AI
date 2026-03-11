"""
Policy networks with different topology constraints.

All architectures share the same interface:
  - forward(obs) -> action
  - reset_hidden()
  - get_flat_params() / set_flat_params()
  - n_params property

The only difference is the sparsity mask on the recurrent layer.
I/O is constrained: input feeds DN neurons only, output reads MN neurons only.
Information must flow through the recurrent topology to reach the output.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SparseRecurrentPolicy(nn.Module):
    """Recurrent policy with sparse masked hidden layer and constrained I/O.

    Architecture:
        obs -> W_in -> h[dn_indices] += input
        h = tanh((W_rec * mask) @ h_prev + input_injection)
        action = W_out @ h[mn_indices]

    Input only drives DN neurons. Output only reads MN neurons.
    The recurrent mask determines how information flows DN -> intrinsic -> MN.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        mask: Optional[torch.Tensor] = None,
        dn_indices: Optional[list] = None,
        mn_indices: Optional[list] = None,
        joint_rest: Optional[np.ndarray] = None,
        joint_amp: Optional[np.ndarray] = None,
        recurrence_steps: int = 3,
        n_evals: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.recurrence_steps = recurrence_steps
        self.n_evals = n_evals

        n_dn = len(dn_indices) if dn_indices is not None else hidden_dim
        n_mn = len(mn_indices) if mn_indices is not None else hidden_dim

        # Input projection: obs -> DN neurons only
        self.W_in = nn.Linear(obs_dim, n_dn, bias=True)

        # Recurrent layer: hidden -> hidden (sparse)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection: MN neurons only -> action
        self.W_out = nn.Linear(n_mn, act_dim, bias=True)

        # Index buffers for I/O routing
        if dn_indices is not None:
            self.register_buffer("dn_idx", torch.tensor(dn_indices, dtype=torch.long))
        else:
            self.register_buffer("dn_idx", torch.arange(hidden_dim))

        if mn_indices is not None:
            self.register_buffer("mn_idx", torch.tensor(mn_indices, dtype=torch.long))
        else:
            self.register_buffer("mn_idx", torch.arange(hidden_dim))

        # Sparsity mask (fixed)
        if mask is None:
            mask = torch.ones(hidden_dim, hidden_dim)
        self.register_buffer("rec_mask", mask.float())

        # Joint angle bounds for output clamping
        if joint_rest is not None:
            self.register_buffer("joint_rest", torch.tensor(joint_rest, dtype=torch.float32))
            self.register_buffer("joint_amp", torch.tensor(joint_amp, dtype=torch.float32))
        else:
            self.joint_rest = None
            self.joint_amp = None

        # Hidden state
        self.register_buffer("h", torch.zeros(hidden_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
            nn.init.zeros_(self.W_in.bias)
            nn.init.xavier_uniform_(self.W_rec.weight, gain=0.3)
            self.W_rec.weight.mul_(self.rec_mask)
            nn.init.xavier_uniform_(self.W_out.weight, gain=0.1)
            nn.init.zeros_(self.W_out.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (obs_dim,) -> action (act_dim,)"""
        masked_rec = self.W_rec.weight * self.rec_mask

        # Input injection: obs -> DN neurons only
        dn_input = self.W_in(obs)  # (n_dn,)
        x_in = torch.zeros(self.hidden_dim, device=obs.device)
        x_in[self.dn_idx] = dn_input

        # Recurrent update
        h = self.h
        for _ in range(self.recurrence_steps):
            h = torch.tanh(x_in + F.linear(h, masked_rec))

        self.h = h.detach()

        # Output: read MN neurons only -> action
        mn_activations = h[self.mn_idx]  # (n_mn,)
        raw = self.W_out(mn_activations)

        # Split: joints (42) + adhesion logits (6)
        joint_raw = raw[:42]
        adhesion_logits = raw[42:]

        # Clamp joints to safe range
        if self.joint_rest is not None:
            joints = self.joint_rest + self.joint_amp * torch.tanh(joint_raw)
        else:
            joints = joint_raw

        adhesion = (adhesion_logits > 0.0).float()

        return torch.cat([joints, adhesion])

    def reset_hidden(self):
        self.h.zero_()

    @property
    def n_params(self) -> int:
        """ES-optimizable parameter count (active only, excluding masked recurrent)."""
        n = 0
        for name, p in self.named_parameters():
            if name == "W_rec.weight":
                n += int(self.rec_mask.sum().item())
            else:
                n += p.numel()
        return n

    @property
    def n_total_params(self) -> int:
        """Total parameter count including masked-out recurrent weights."""
        return sum(p.numel() for p in self.parameters())

    @property
    def n_active_params(self) -> int:
        """Alias for n_params (backward compat)."""
        return self.n_params

    def get_flat_params(self) -> np.ndarray:
        """Return active parameters as a flat array (masked recurrent excluded)."""
        params = []
        for name, p in self.named_parameters():
            if name == "W_rec.weight":
                mask_flat = self.rec_mask.flatten().bool()
                params.append(p.data.cpu().flatten()[mask_flat].numpy())
            else:
                params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat: np.ndarray):
        """Set active parameters from a flat array (masked recurrent excluded)."""
        offset = 0
        for name, p in self.named_parameters():
            if name == "W_rec.weight":
                mask_flat = self.rec_mask.flatten().bool()
                n_active = int(mask_flat.sum().item())
                full = torch.zeros(p.numel(), dtype=p.dtype, device=p.device)
                full[mask_flat] = torch.tensor(
                    flat[offset:offset + n_active], dtype=p.dtype
                )
                p.data.copy_(full.reshape(p.shape))
                offset += n_active
            else:
                n = p.numel()
                p.data.copy_(torch.tensor(
                    flat[offset:offset + n].reshape(p.shape),
                    dtype=p.dtype, device=p.device
                ))
                offset += n


def _joint_arrays(joint_params):
    """Extract rest/amp arrays from joint_params dict."""
    if joint_params is None:
        return None, None
    return (
        np.array([joint_params[j][0] for j in range(42)]),
        np.array([joint_params[j][1] for j in range(42)]),
    )


def build_connectome_policy(
    topo: dict,
    obs_dim: int = 90,
    act_dim: int = 48,
    recurrence_steps: int = 3,
    joint_params: Optional[dict] = None,
) -> SparseRecurrentPolicy:
    """Build policy with real connectome topology as sparsity mask."""
    n = topo["n_neurons"]
    mask = torch.zeros(n, n)
    indices = topo["adj_indices"]
    # mask[post, pre] = 1 so that W[i,j]*mask[i,j] gates the j→i connection,
    # matching PyTorch nn.Linear convention (W[out, in] = weight from in→out).
    mask[indices[1], indices[0]] = 1.0
    joint_rest, joint_amp = _joint_arrays(joint_params)

    return SparseRecurrentPolicy(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=n,
        mask=mask,
        dn_indices=topo["dn_indices"],
        mn_indices=topo["mn_indices"],
        joint_rest=joint_rest, joint_amp=joint_amp,
        recurrence_steps=recurrence_steps,
    )


def build_dense_policy(
    hidden_dim: int,
    obs_dim: int = 90,
    act_dim: int = 48,
    recurrence_steps: int = 3,
    joint_params: Optional[dict] = None,
    dn_indices: Optional[list] = None,
    mn_indices: Optional[list] = None,
) -> SparseRecurrentPolicy:
    """Build fully connected policy (no sparsity mask), same I/O constraint."""
    joint_rest, joint_amp = _joint_arrays(joint_params)

    return SparseRecurrentPolicy(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
        mask=None,
        dn_indices=dn_indices,
        mn_indices=mn_indices,
        joint_rest=joint_rest, joint_amp=joint_amp,
        recurrence_steps=recurrence_steps,
    )


def build_random_sparse_policy(
    topo: dict,
    seed: int = 42,
    obs_dim: int = 90,
    act_dim: int = 48,
    recurrence_steps: int = 3,
    joint_params: Optional[dict] = None,
) -> SparseRecurrentPolicy:
    """Build policy with random Erdos-Renyi sparsity, same density + I/O as connectome."""
    n = topo["n_neurons"]
    n_edges = topo["adj_indices"].shape[1]
    density = n_edges / (n * n)

    rng = np.random.RandomState(seed)
    mask = torch.tensor(rng.random((n, n)) < density, dtype=torch.float32)
    mask.fill_diagonal_(0.0)
    joint_rest, joint_amp = _joint_arrays(joint_params)

    return SparseRecurrentPolicy(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=n,
        mask=mask,
        dn_indices=topo["dn_indices"],
        mn_indices=topo["mn_indices"],
        joint_rest=joint_rest, joint_amp=joint_amp,
        recurrence_steps=recurrence_steps,
    )


def build_shuffled_policy(
    topo: dict,
    seed: int = 42,
    obs_dim: int = 90,
    act_dim: int = 48,
    recurrence_steps: int = 3,
    joint_params: Optional[dict] = None,
) -> SparseRecurrentPolicy:
    """Build policy with shuffled connectome (same degree distribution, permuted targets)."""
    n = topo["n_neurons"]
    rng = np.random.RandomState(seed)
    indices = topo["adj_indices"].clone()
    indices[1] = torch.tensor(rng.permutation(indices[1].numpy()), dtype=torch.long)

    mask = torch.zeros(n, n)
    # mask[post, pre] = 1 — same transpose convention as build_connectome_policy
    mask[indices[1], indices[0]] = 1.0
    joint_rest, joint_amp = _joint_arrays(joint_params)

    return SparseRecurrentPolicy(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=n,
        mask=mask,
        dn_indices=topo["dn_indices"],
        mn_indices=topo["mn_indices"],
        joint_rest=joint_rest, joint_amp=joint_amp,
        recurrence_steps=recurrence_steps,
    )
