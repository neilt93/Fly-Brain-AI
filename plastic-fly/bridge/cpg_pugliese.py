"""
Pugliese et al. 2025 firing-rate CPG model.

3-neuron E-E-I oscillator per hemi-segment, derived from MANC wiring:
    E1 (IN17A001, ACh) — excitatory interneuron
    E2 (INXXX466, ACh) — excitatory interneuron
    I1 (IN16B036, Glu) — inhibitory interneuron

Rate model ODE (Pugliese et al. 2025):
    dR/dt = (-R + R_max * max(0, tanh((W@R + I - theta) * a / R_max))) / tau

6 hemi-segments with tripod phase offsets [0, pi, 0, pi, 0, pi].
External drive from forward_drive replaces the sine oscillator in VNCBridge.

Usage:
    from bridge.cpg_pugliese import PuglieseCPG
    cpg = PuglieseCPG.from_json("data/cpg_weights.json")
    for t in range(n_steps):
        leg_mod = cpg.step(dt_s=1e-4, forward_drive=20.0)
        ext_mod, flex_mod = cpg.get_leg_modulation(leg_idx=0)
"""

import json
import numpy as np
from pathlib import Path


# Tuned parameters: MANC weights use exc/inh_mult=0.01, giving W values
# that produce 10-20Hz oscillation with R_max=50 and tau=15ms.
_DEFAULT_PARAMS = {
    "tau_ms": 25.0,       # Time constant (ms) — tuned for ~12Hz at moderate drive
    "theta": 3.0,         # Activation threshold
    "a": 1.0,             # Gain
    "R_max": 50.0,        # Maximum firing rate (Hz)
    "exc_mult": 0.01,     # Excitatory weight multiplier (synapse count -> effective weight)
    "inh_mult": 0.01,     # Inhibitory weight multiplier
    "drive_scale": 2.0,   # forward_drive (Hz) -> I_ext scaling
    "drive_target": "E1", # Which neurons receive external drive: "E1", "E1E2", "all"
}

# Tripod phase offsets: LF, LM, LH, RF, RM, RH
_TRIPOD_PHASES = np.array([0, np.pi, 0, np.pi, 0, np.pi])


class PuglieseCPG:
    """3-neuron E-E-I firing-rate CPG, 6 hemi-segments.

    Each hemi-segment contains 3 rate units:
        0: E1 (IN17A001, excitatory)
        1: E2 (INXXX466, excitatory)
        2: I1 (IN16B036, inhibitory)

    The weight matrix W (3x3) encodes recurrent connectivity from MANC.
    External drive from forward_drive enters as I_ext to E1 (default).

    Tripod gait: legs [0,2,4] start at phase 0, legs [1,3,5] at phase pi.
    Phase offset is implemented via staggered initial conditions.

    Output normalization: E1 rate is adaptively normalized to [0, 1] using
    an exponential moving average of min/max, ensuring stable modulation
    depth regardless of absolute rate magnitudes.
    """

    def __init__(
        self,
        W: np.ndarray,
        neuron_params: dict | None = None,
        n_legs: int = 6,
    ):
        params = dict(_DEFAULT_PARAMS)
        if neuron_params:
            params.update(neuron_params)

        self.W = np.asarray(W, dtype=np.float64)
        assert self.W.shape == (3, 3), f"W must be 3x3, got {self.W.shape}"

        self.tau_s = params["tau_ms"] / 1000.0
        self.theta = float(params["theta"])
        self.a = float(params["a"])
        self.R_max = float(params["R_max"])
        self.drive_scale = float(params["drive_scale"])
        self.drive_target = params.get("drive_target", "E1")

        self.n_legs = n_legs
        self.n_neurons = 3

        # State: (n_legs, 3) firing rates
        self.R = np.zeros((n_legs, 3), dtype=np.float64)

        # Adaptive normalization: track running min/max of E1 per leg
        self._e1_min = np.full(n_legs, 0.0)
        self._e1_max = np.full(n_legs, 1.0)
        self._norm_alpha = 0.01  # EMA smoothing for min/max tracking

        self._init_tripod_phases()
        self._time_s = 0.0

    def _init_tripod_phases(self):
        """Set initial conditions with tripod phase offsets.

        Since all 6 CPGs are identical independent oscillators, we establish
        the tripod pattern by pre-running the pi-phase legs for half a period.
        This creates a persistent phase offset because the oscillators are
        autonomous (no inter-leg coupling to pull them into sync).
        """
        # Start all legs at the same high-E1 state
        for leg in range(self.n_legs):
            self.R[leg, 0] = self.R_max * 0.8
            self.R[leg, 1] = self.R_max * 0.1
            self.R[leg, 2] = self.R_max * 0.2

        # Pre-run pi-phase legs for half a period (~40ms at 12Hz)
        # using moderate drive to establish oscillation
        half_period_s = 0.5 / 12.0  # half period at 12Hz
        n_pre_steps = int(half_period_s / 0.0001)
        I_ext = np.zeros(3, dtype=np.float64)
        I_ext[0] = 20.0 * self.drive_scale  # moderate drive

        for _ in range(n_pre_steps):
            for leg in range(self.n_legs):
                if _TRIPOD_PHASES[leg] < 0.01:
                    continue  # Skip phase-0 legs
                R_leg = self.R[leg]
                total_input = self.W @ R_leg + I_ext - self.theta
                R_inf = self._activation(total_input)
                self.R[leg] = R_leg + 0.0001 * (-R_leg + R_inf) / self.tau_s
                np.clip(self.R[leg], 0.0, self.R_max, out=self.R[leg])

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear activation: R_max * max(0, tanh(x * a / R_max))."""
        return self.R_max * np.maximum(0.0, np.tanh(x * self.a / self.R_max))

    def step(self, dt_s: float, forward_drive: float) -> np.ndarray:
        """Advance CPG by one timestep.

        Args:
            dt_s: Time step in seconds (typically 1e-4).
            forward_drive: Forward drive rate in Hz from DescendingDecoder.

        Returns:
            (n_legs,) modulation signals in [0, 1], adaptively normalized.
            Values near 1 = E1 at its peak (swing), near 0 = E1 at trough (stance).
        """
        I_ext = np.zeros(3, dtype=np.float64)
        drive = forward_drive * self.drive_scale

        if self.drive_target == "E1":
            I_ext[0] = drive
        elif self.drive_target == "E1E2":
            I_ext[0] = drive
            I_ext[1] = drive * 0.5
        elif self.drive_target == "all":
            I_ext[:] = drive
        else:
            I_ext[0] = drive

        # Sub-step for stability (0.1ms substeps)
        n_substeps = max(1, int(np.ceil(dt_s / 0.0001)))
        sub_dt = dt_s / n_substeps

        for _ in range(n_substeps):
            for leg in range(self.n_legs):
                R_leg = self.R[leg]
                total_input = self.W @ R_leg + I_ext - self.theta
                R_inf = self._activation(total_input)
                self.R[leg] = R_leg + sub_dt * (-R_leg + R_inf) / self.tau_s
                np.clip(self.R[leg], 0.0, self.R_max, out=self.R[leg])

        self._time_s += dt_s

        # Adaptive normalization of E1 rates
        e1_rates = self.R[:, 0]
        alpha = self._norm_alpha
        self._e1_min = self._e1_min * (1.0 - alpha) + e1_rates * alpha
        self._e1_max = np.maximum(self._e1_max * (1.0 - alpha) + e1_rates * alpha,
                                  self._e1_min + 0.1)  # Prevent zero range

        # Track true min/max more aggressively
        self._e1_min = np.minimum(self._e1_min, e1_rates)
        self._e1_max = np.maximum(self._e1_max, e1_rates)

        # Normalize to [0, 1]
        span = self._e1_max - self._e1_min
        span = np.maximum(span, 0.1)  # Prevent division by zero
        mod = (e1_rates - self._e1_min) / span
        return np.clip(mod, 0.0, 1.0)

    def get_leg_modulation(self, leg_idx: int) -> tuple[float, float]:
        """Get extensor/flexor modulation for a specific leg.

        Returns:
            (ext_mod, flex_mod): Both in [0, 1].
            ext_mod high during swing (E1 high), flex_mod high during stance.
        """
        e1 = self.R[leg_idx, 0]
        span = max(self._e1_max[leg_idx] - self._e1_min[leg_idx], 0.1)
        norm = np.clip((e1 - self._e1_min[leg_idx]) / span, 0.0, 1.0)
        return (float(norm), float(1.0 - norm))

    def get_osc_signal(self, leg_idx: int) -> float:
        """Get oscillatory signal in [-1, 1] (drop-in for np.sin in _apply_rhythm).

        E1 at peak -> +1 (swing), E1 at trough -> -1 (stance).
        """
        e1 = self.R[leg_idx, 0]
        span = max(self._e1_max[leg_idx] - self._e1_min[leg_idx], 0.1)
        norm = np.clip((e1 - self._e1_min[leg_idx]) / span, 0.0, 1.0)
        return float(2.0 * norm - 1.0)

    def reset(self):
        """Reset CPG to initial conditions."""
        self.R[:] = 0.0
        self._e1_min[:] = 0.0
        self._e1_max[:] = 1.0
        self._time_s = 0.0
        self._init_tripod_phases()

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def state(self) -> np.ndarray:
        """Full state array (n_legs, 3)."""
        return self.R.copy()

    @classmethod
    def from_json(cls, path: str | Path, neuron_params: dict | None = None, n_legs: int = 6):
        """Load CPG from weight file (e.g. data/cpg_weights.json)."""
        with open(path) as f:
            data = json.load(f)

        W = np.array(data["W"], dtype=np.float64)

        file_params = data.get("neuron_params", {})
        if neuron_params:
            file_params.update(neuron_params)

        return cls(W=W, neuron_params=file_params, n_legs=n_legs)
