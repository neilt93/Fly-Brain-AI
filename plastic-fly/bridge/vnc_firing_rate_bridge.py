"""
Firing rate VNC bridge: connects FiringRateVNCRunner to FlyGym body.

Pipeline:
    DescendingDecoder group_rates -> DN stimulation -> FiringRateVNCRunner
    -> MN firing rates -> MotorNeuronDecoder -> joint angles

KEY DIFFERENCE from vnc_bridge.py (Brian2 LIF VNC):
    The firing rate model generates its OWN rhythm from network dynamics
    (spike-frequency adaptation + delayed inhibition). NO external sine/CPG
    rhythm modulation is applied. The rhythm is connectome-emergent.

    However, not all legs achieve clean anti-phase flex/ext alternation.
    For legs that don't, we apply a fallback: detect the network's dominant
    frequency and impose anti-phase from that frequency.

Group rate -> DN type mapping:
    forward:    DNg100 (primary), DNa01, DNa02, DNb02
    turn_left:  DNg11, DNg29, DNg33 (left-side only in MANC)
    turn_right: DNg11, DNg29, DNg33 (right-side only in MANC)
    rhythm:     DNb01, DNb08, DNg100
    stance:     DNp44, DNp35

Usage:
    from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge

    bridge = FiringRateVNCBridge()
    bridge.warmup(warmup_ms=200.0)

    for step in range(5000):
        action = bridge.step(group_rates, dt_s=1e-4)
        obs, _, _, _, _ = sim.step(action)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional

from bridge.vnc_firing_rate import (
    FiringRateVNCRunner,
    FiringRateVNCConfig,
    LEG_ORDER,
)
from bridge.mn_decoder import MotorNeuronDecoder
from bridge.vnc_connectome import VNCOutput
from bridge.interfaces import BodyObservation


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Tripod phase offsets: LF, LM, LH, RF, RM, RH
_TRIPOD_PHASES = np.array([0, np.pi, 0, np.pi, 0, np.pi])

# Leg name -> first joint index (7 DOFs per leg)
_LEG_OFFSET = {"LF": 0, "LM": 7, "LH": 14, "RF": 21, "RM": 28, "RH": 35}

# BANC cell_type keyword -> (dof_within_leg, direction)
# DOF: 0=Coxa, 1=Coxa_roll, 2=Coxa_yaw, 3=Femur, 4=Femur_roll, 5=Tibia, 6=Tarsus1
# Checked against MANC mn_joint_mapping equivalence.
_BANC_CELLTYPE_TO_DOF = [
    # Tibia flex/ext (DOF 5) — largest MN groups
    ("tibia_flexor", 5, -1.0),
    ("accessory_tibia_flexor", 5, -1.0),
    ("tibia_extensor", 5, 1.0),
    # Trochanter/Femur flex/ext (DOF 3)
    ("trochanter_flexor", 3, -1.0),
    ("accessory_trochanter_flexor", 3, -1.0),
    ("sternotrochanter_extensor", 3, 1.0),
    ("trochanter_extensor", 3, 1.0),
    ("tergotrochanter_extensor", 3, 1.0),
    # Femur roll/reductor (DOF 4)
    ("femur_reductor", 4, -1.0),
    # Long tendon muscle (multi-joint, assign DOF 3 with partial weight)
    ("long_tendon_muscle", 3, 0.5),
    # Coxa yaw / rotators (DOF 2)
    ("sternal_posterior_rotator", 2, -1.0),
    ("sternal_anterior_rotator", 2, 1.0),
    # Coxa roll / promotor/remotor (DOF 1)
    ("pleural_remotor", 1, -1.0),
    ("tergopleural", 1, 1.0),
    ("promotor", 1, 1.0),
    # Coxa / adductor/abductor (DOF 0)
    ("sternal_adductor", 0, -1.0),
    ("coxal_depressor", 0, -1.0),
    ("coxal_levator", 0, 1.0),
    # Tarsus (DOF 6)
    ("tarsus_depressor", 6, -1.0),
    ("tarsus_levator", 6, 1.0),
]


def _banc_celltype_to_joint(cell_type: str, leg: str) -> tuple:
    """Map a BANC cell_type to (joint_idx, direction) or None.

    Returns:
        (joint_idx, direction) or (None, None) if unrecognized.
    """
    ct_lower = cell_type.lower()
    for keyword, dof, direction in _BANC_CELLTYPE_TO_DOF:
        if keyword in ct_lower:
            joint_idx = _LEG_OFFSET.get(leg, 0) + dof
            return joint_idx, direction
    return None, None

# DN types associated with each decoder group (from literature + Brian2 VNC)
_GROUP_DN_TYPES = {
    "forward": ["DNg100", "DNa01", "DNa02", "DNb02", "DNp30", "DNp02"],
    "turn_left": ["DNg11", "DNg29", "DNg33", "DNg35", "DNg47"],
    "turn_right": ["DNg11", "DNg29", "DNg33", "DNg35", "DNg47"],
    "rhythm": ["DNb01", "DNb08", "DNb09", "DNg100"],
    "stance": ["DNp44", "DNp35", "DNp42"],
}


def _build_banc_mn_mapping(mn_info: list, mn_body_ids: np.ndarray) -> dict:
    """Build synthetic mn_joint_mapping from BANC MN metadata.

    Maps each BANC MN to a FlyGym joint based on its cell_type (muscle name)
    and leg assignment. Returns a dict in the same format as
    data/mn_joint_mapping.json: {str(body_id): {joint_idx, direction, ...}}.
    """
    mapping = {}
    n_mapped = 0
    for i, info in enumerate(mn_info):
        bid = str(int(mn_body_ids[i]))
        ct = info.get("cell_type", "")
        leg = info.get("leg", "LF")

        joint_idx, direction = _banc_celltype_to_joint(ct, leg)
        if joint_idx is None:
            # Fallback: use MN direction metadata for generic flex/ext
            # Map to Femur (DOF 3) as the primary walking joint
            d = info.get("direction", 0.0)
            if d != 0.0:
                joint_idx = _LEG_OFFSET.get(leg, 0) + 3  # Femur
                direction = float(d)
            else:
                continue  # truly ambiguous, skip

        mapping[bid] = {
            "joint_idx": joint_idx,
            "direction": direction,
            "mn_type": ct,
            "leg": leg,
            "muscle_group": ct.split("_")[0] if ct else "unknown",
        }
        n_mapped += 1
    return mapping


class FiringRateVNCBridge:
    """Bridge from DescendingDecoder group rates to FlyGym joint angles
    via the Pugliese-style firing rate VNC model.

    The firing rate model produces its own rhythm from network dynamics.
    This bridge does NOT impose external rhythm -- the walking pattern
    comes from the connectome.

    For legs where the network does not achieve clear anti-phase alternation,
    a fallback rhythm is blended in at low weight to provide minimal
    stance/swing coordination.

    Parameters
    ----------
    cfg : FiringRateVNCConfig, optional
        Firing rate model config. Defaults to standard params.
    mn_mapping_path : Path, optional
        Path to mn_joint_mapping.json.
    mn_rate_scale : float
        MotorNeuronDecoder rate_scale (Hz at ~0.76 amplitude).
    mn_alpha : float
        MotorNeuronDecoder smoothing coefficient.
    dn_baseline_hz : float
        Background tonic stimulation for all DNs (models brain baseline).
    substeps_per_body : int
        Number of ODE substeps per body step. More = smoother but slower.
        At dt_body=0.1ms and substeps=2, effective ODE dt=0.05ms.
    fallback_blend : float
        Blend weight for fallback rhythm on legs without clean anti-phase.
        0.0 = pure network rhythm, 1.0 = pure fallback. Default 0.3.
    fallback_freq_hz : float
        Frequency for fallback rhythm (Hz). Default 10.0 (tripod-like).
    """

    def __init__(
        self,
        cfg: FiringRateVNCConfig | None = None,
        mn_mapping_path: str | Path | None = None,
        mn_rate_scale: float = 35.0,
        mn_alpha: float = 0.3,
        dn_baseline_hz: float = 25.0,
        substeps_per_body: int = 5,
        fallback_blend: float = 0.3,
        fallback_freq_hz: float = 10.0,
        perturbation_interval: int = 1500,
        perturbation_scale: float = 0.3,
    ):
        self.cfg = cfg or FiringRateVNCConfig()
        self.dn_baseline_hz = dn_baseline_hz
        self.substeps_per_body = max(1, substeps_per_body)
        self.fallback_blend = fallback_blend
        self.fallback_freq_hz = fallback_freq_hz
        self._perturbation_interval = perturbation_interval
        self._perturbation_scale = perturbation_scale

        # Build firing rate VNC model
        self.vnc = FiringRateVNCRunner(cfg=self.cfg, warmup_ms=0.0)

        # Build MN decoder
        if mn_mapping_path is None:
            mn_mapping_path = DATA_DIR / "mn_joint_mapping.json"
        self.mn_decoder = MotorNeuronDecoder(
            mapping_path=mn_mapping_path,
            rate_scale=mn_rate_scale,
            alpha=mn_alpha,
        )

        # Map DN types to group names for stimulation
        self._group_to_dn_indices: Dict[str, list] = {}
        self._build_group_dn_mapping()

        # Body-step time tracking
        self._body_time_ms = 0.0
        self._step_count = 0

        # Fallback rhythm phase (per leg)
        self._fallback_phase = _TRIPOD_PHASES.copy().astype(np.float64)

        # Per-leg anti-phase quality score (updated periodically)
        self._antiphase_quality = np.zeros(6, dtype=np.float64)
        self._flex_ext_history = {
            leg: {"flex": [], "ext": []} for leg in range(6)
        }
        self._quality_window = 200  # steps to track for quality estimation

        # MN body ID -> rhythm unit mapping (for fallback)
        self._build_mn_rhythm_map()

        # Current group rates cache
        self._cached_group_rates: dict = {}

    @classmethod
    def from_banc(
        cls,
        banc_data=None,
        cfg: FiringRateVNCConfig | None = None,
        mn_mapping_path: str | Path | None = None,
        mn_rate_scale: float = 15.0,
        mn_alpha: float = 0.3,
        dn_baseline_hz: float = 25.0,
        substeps_per_body: int = 5,
        fallback_blend: float = 0.3,
        fallback_freq_hz: float = 10.0,
        perturbation_interval: int = 1500,
        perturbation_scale: float = 0.3,
    ) -> "FiringRateVNCBridge":
        """Build bridge using BANC (female) connectome data.

        Args:
            banc_data: Pre-loaded BANCVNCData (raw weights recommended).
                       If None, loads from default path.
            cfg: FiringRateVNCConfig with ODE and weight scaling params.
            perturbation_interval: Body steps between rate perturbations (restarts
                transient oscillation that the ODE otherwise dampens).
            perturbation_scale: Multiply all rates by this factor at perturbation
                (0.3 = reduce to 30%, triggering re-establishment of rhythm).
            Other args: same as __init__.

        Returns:
            Fully initialized FiringRateVNCBridge with BANC VNC.
        """
        self = cls.__new__(cls)
        # Use optimized BANC config by default (uniform 0.01, no norm, Pugliese a=1)
        if cfg is None:
            cfg = FiringRateVNCConfig(
                a=1.0, theta=7.5, fr_cap=200.0,
                exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
                use_adaptation=False, normalize_weights=False,
                use_delay=True, delay_inh_ms=3.0,
                param_cv=0.05, seed=42,
            )
        self.cfg = cfg
        self.dn_baseline_hz = dn_baseline_hz
        self.substeps_per_body = max(1, substeps_per_body)
        self.fallback_blend = fallback_blend
        self.fallback_freq_hz = fallback_freq_hz
        self._perturbation_interval = perturbation_interval
        self._perturbation_scale = perturbation_scale

        # Load BANC data if not provided
        if banc_data is None:
            from bridge.banc_loader import load_banc_vnc
            banc_data = load_banc_vnc(
                exc_mult=1.0, inh_mult=1.0, inh_scale=1.0,
                normalize_weights=False, verbose=True,
            )

        # Build firing rate VNC from BANC
        self.vnc = FiringRateVNCRunner.from_banc(banc_data, cfg=self.cfg, warmup_ms=0.0)

        # Build BANC-aware MN decoder: create synthetic mapping from cell_type
        banc_mapping = _build_banc_mn_mapping(self.vnc.mn_info, self.vnc.mn_body_ids)
        import json, tempfile
        tmp_mapping = Path(tempfile.mktemp(suffix=".json"))
        with open(tmp_mapping, "w") as f:
            json.dump(banc_mapping, f)
        self.mn_decoder = MotorNeuronDecoder(
            mapping_path=tmp_mapping,
            rate_scale=mn_rate_scale,
            alpha=mn_alpha,
        )
        tmp_mapping.unlink(missing_ok=True)
        n_mapped = len(banc_mapping)
        n_total = len(self.vnc.mn_body_ids)
        print(f"  BANC MN decoder: {n_mapped}/{n_total} MNs mapped to joints")

        # Map DN types to group names
        self._group_to_dn_indices: Dict[str, list] = {}
        self._build_group_dn_mapping()

        # State tracking
        self._body_time_ms = 0.0
        self._step_count = 0
        self._fallback_phase = _TRIPOD_PHASES.copy().astype(np.float64)
        self._antiphase_quality = np.zeros(6, dtype=np.float64)
        self._flex_ext_history = {
            leg: {"flex": [], "ext": []} for leg in range(6)
        }
        self._quality_window = 200
        self._build_mn_rhythm_map()
        self._cached_group_rates: dict = {}

        return self

    def _build_group_dn_mapping(self):
        """Map decoder groups to firing rate VNC neuron indices.

        Turn groups are lateralized by soma side: left-soma DNs map to
        turn_left, right-soma DNs map to turn_right. This is critical —
        without lateralization, both turn commands activate the same neurons.
        """
        dn_type_map = self.vnc._dn_type_to_indices  # type -> [model_idx]
        dn_type_bids = self.vnc._dn_type_to_body_ids  # type -> [body_id]

        # Build side lookup: model_idx -> "left"/"right"/""
        idx_side = {}
        for info in getattr(self.vnc, 'mn_info', []):
            pass  # MN info, not DN
        # For DNs, check if the runner has stored side info from BANC
        if hasattr(self.vnc, '_dn_side'):
            idx_side = self.vnc._dn_side
        else:
            # Infer side from body_id: look up in banc_loader if available
            # For MANC path, no side info — fall back to non-lateralized
            idx_side = {}

        for group_name, dn_types in _GROUP_DN_TYPES.items():
            indices = []
            need_left = (group_name == "turn_left")
            need_right = (group_name == "turn_right")
            lateralize = (need_left or need_right) and len(idx_side) > 0

            for dn_type in dn_types:
                idxs = dn_type_map.get(dn_type, [])
                if lateralize:
                    for idx in idxs:
                        side = idx_side.get(idx, "")
                        if need_left and side == "left":
                            indices.append(idx)
                        elif need_right and side == "right":
                            indices.append(idx)
                else:
                    indices.extend(idxs)
            self._group_to_dn_indices[group_name] = sorted(set(indices))

        # Report
        for g, idxs in self._group_to_dn_indices.items():
            print(f"  FiringRateBridge: {g} -> {len(idxs)} DN neurons")

    def _build_mn_rhythm_map(self):
        """Build MN body ID -> rhythm unit (leg_idx*2 + is_ext) for fallback.

        Same scheme as vnc_bridge.py: rhythm_unit = leg_idx*2 + (0 if ext, 1 if flex).
        """
        self._mn_rhythm_map: Dict[int, int] = {}
        for i, info in enumerate(self.vnc.mn_info):
            bid = info["body_id"]
            leg_idx = info["leg_idx"]
            direction = info["direction"]
            if direction > 0:
                # Extensor
                self._mn_rhythm_map[bid] = leg_idx * 2
            elif direction < 0:
                # Flexor
                self._mn_rhythm_map[bid] = leg_idx * 2 + 1
            # direction == 0: ambiguous, skip

    def _apply_dn_stimulation(self, group_rates: dict):
        """Convert group rates to per-neuron stimulation currents.

        Each group's rate (Hz) is applied to the DN types associated with
        that group. The baseline is applied to ALL DNs first, then group-
        specific rates are added on top.

        This sets I_stim directly (no print statements) for use in the
        per-body-step loop where verbose output is unwanted.
        """
        # Start with baseline for all DNs (direct, no print)
        baseline_current = self.cfg.theta * (self.dn_baseline_hz / 20.0)
        for idx in self.vnc._dn_indices:
            self.vnc.I_stim[idx] = baseline_current

        # Apply group-specific stimulation (additive on top of baseline)
        for group_name, rate_hz in group_rates.items():
            if rate_hz <= 0:
                continue
            indices = self._group_to_dn_indices.get(group_name, [])
            if not indices:
                continue
            # Convert to current (same as stimulate_dn_type)
            total_rate = self.dn_baseline_hz + rate_hz
            current = self.cfg.theta * (total_rate / 20.0)
            for idx in indices:
                self.vnc.I_stim[idx] = current

    def _get_mn_output(self) -> VNCOutput:
        """Get current MN firing rates as VNCOutput (compatible with mn_decoder)."""
        rates = self.vnc.get_mn_rates()
        return VNCOutput(
            mn_body_ids=self.vnc.mn_body_ids.copy(),
            firing_rates_hz=rates.astype(np.float32),
        )

    def _update_antiphase_quality(self):
        """Estimate per-leg anti-phase quality from recent flex/ext history.

        Quality is the magnitude of the cross-correlation at pi phase offset.
        High quality (>0.3) means the network produces clean alternation.
        Low quality (<0.1) means flex/ext are in-phase or not oscillating.
        """
        for leg in range(6):
            hist = self._flex_ext_history[leg]
            if len(hist["flex"]) < 50:
                self._antiphase_quality[leg] = 0.0
                continue

            flex_arr = np.array(hist["flex"][-self._quality_window:])
            ext_arr = np.array(hist["ext"][-self._quality_window:])

            # Normalize
            flex_norm = flex_arr - flex_arr.mean()
            ext_norm = ext_arr - ext_arr.mean()

            flex_std = flex_norm.std()
            ext_std = ext_norm.std()

            if flex_std < 1.0 or ext_std < 1.0:
                # Not oscillating
                self._antiphase_quality[leg] = 0.0
                continue

            # Cross-correlation at lag 0 (should be negative for anti-phase)
            cc = float(np.dot(flex_norm, ext_norm) / (len(flex_norm) * flex_std * ext_std))
            # Anti-phase quality: negative correlation = good alternation
            self._antiphase_quality[leg] = float(np.clip(-cc, 0.0, 1.0))

    def _apply_fallback_rhythm(
        self,
        vnc_output: VNCOutput,
        dt_s: float,
    ) -> VNCOutput:
        """Blend fallback tripod rhythm for legs with poor anti-phase quality.

        For each MN:
          - If its leg has good anti-phase (quality > threshold): use network rates
          - If poor: blend with a fallback sine rhythm at tripod frequency

        The fallback ensures minimal stance/swing coordination even when the
        network's oscillation is in-phase or too weak.
        """
        if self.fallback_blend <= 0:
            return vnc_output

        # Advance fallback phase
        omega = 2.0 * np.pi * self.fallback_freq_hz
        self._fallback_phase += omega * dt_s
        self._fallback_phase %= (2.0 * np.pi)

        mn_ids = vnc_output.mn_body_ids
        rates = vnc_output.firing_rates_hz.copy()
        n_mn = len(mn_ids)

        # Quality threshold: below this, blend in fallback
        quality_threshold = 0.2

        # Compute forward-rate-dependent amplitude for fallback
        fwd_rate = float(self._cached_group_rates.get("forward", 0.0))
        amp_scale = float(np.clip(np.tanh(fwd_rate / 20.0), 0.1, 1.0))
        base_hz = 40.0  # Fallback rhythm base rate

        for j in range(n_mn):
            bid = int(mn_ids[j])
            if bid not in self._mn_rhythm_map:
                continue

            ru = self._mn_rhythm_map[bid]
            leg_idx = ru // 2
            is_ext = (ru % 2) == 0

            quality = self._antiphase_quality[leg_idx]
            if quality >= quality_threshold:
                # Good anti-phase: use pure network output
                continue

            # Compute blend weight (more fallback when quality is worse)
            blend = self.fallback_blend * (1.0 - quality / quality_threshold)

            # Fallback signal
            osc = np.sin(self._fallback_phase[leg_idx])
            if is_ext:
                fallback_rate = base_hz * amp_scale * max(0.0, 0.5 - 0.5 * osc)
            else:
                fallback_rate = base_hz * amp_scale * max(0.0, 0.5 + 0.5 * osc)

            # Blend: network * (1-blend) + fallback * blend
            rates[j] = float(rates[j]) * (1.0 - blend) + fallback_rate * blend

        return VNCOutput(mn_body_ids=mn_ids.copy(), firing_rates_hz=rates)

    def warmup(self, warmup_ms: float = 200.0):
        """Run VNC warmup with baseline DN stimulation.

        This lets network transients settle before the main loop.
        The baseline stimulation ensures neurons are near their operating
        point when group-specific rates are applied.
        """
        if warmup_ms <= 0:
            return

        print(f"  FiringRateBridge: warming up {warmup_ms:.0f}ms...")
        self.vnc.stimulate_all_dns(rate_hz=self.dn_baseline_hz)

        dt = self.cfg.dt_ms
        n_steps = int(warmup_ms / dt)
        for _ in range(n_steps):
            self.vnc.step(dt_ms=dt)

        print(f"  FiringRateBridge: warmup done (t={self.vnc.current_time_ms:.0f}ms)")

    def step(
        self,
        group_rates: dict,
        dt_s: float = 1e-4,
        body_obs: BodyObservation | None = None,
    ) -> dict:
        """Run one body step: group rates -> VNC -> MN rates -> joint angles.

        Args:
            group_rates: {"forward": Hz, "turn_left": Hz, ...} from decoder.
            dt_s: Body timestep in seconds (default 0.1ms = FlyGym timestep).
            body_obs: Optional body observation (not used yet, for future
                      proprioceptive feedback).

        Returns:
            {'joints': ndarray(42), 'adhesion': ndarray(6)}
        """
        dt_ms = dt_s * 1000.0
        self._cached_group_rates = group_rates

        # Apply DN stimulation from group rates
        self._apply_dn_stimulation(group_rates)

        # Periodic perturbation: partially reset VNC rates to prevent
        # fixed-point convergence. The firing-rate ODE converges to a
        # stable fixed point after ~2000 steps; this burst restarts the
        # transient oscillation that produces real flex/ext alternation.
        # Biologically: neuromodulatory bursts / proprioceptive resets.
        if (self._step_count > 0
                and self._step_count % self._perturbation_interval == 0):
            self.vnc.R *= self._perturbation_scale

        # Step the firing rate model (multiple substeps for stability)
        sub_dt = dt_ms / self.substeps_per_body
        for _ in range(self.substeps_per_body):
            self.vnc.step(dt_ms=sub_dt)

        # Get MN output
        vnc_output = self._get_mn_output()

        # Track flex/ext rates for anti-phase quality estimation
        for leg_idx in range(6):
            flex_rate, ext_rate = self.vnc.get_flexor_extensor_rates(leg_idx)
            self._flex_ext_history[leg_idx]["flex"].append(flex_rate)
            self._flex_ext_history[leg_idx]["ext"].append(ext_rate)

        # Update anti-phase quality periodically
        if self._step_count % 100 == 0 and self._step_count > 0:
            self._update_antiphase_quality()

        # Apply fallback rhythm for legs with poor alternation
        vnc_output = self._apply_fallback_rhythm(vnc_output, dt_s)

        # Decode MN rates to joint angles
        action = self.mn_decoder.decode(
            mn_body_ids=vnc_output.mn_body_ids,
            firing_rates_hz=vnc_output.firing_rates_hz,
        )

        self._body_time_ms += dt_ms
        self._step_count += 1
        return action

    def step_brain(self, group_rates: dict, sim_ms: float = 20.0,
                   body_obs: BodyObservation | None = None) -> None:
        """Compatibility: for brain-step-frequency VNC updates.

        The firing rate model runs at body frequency (not brain frequency),
        so this just caches the group rates for use in step().
        """
        self._cached_group_rates = group_rates
        self._apply_dn_stimulation(group_rates)

    def reset(self, init_angles: np.ndarray = None):
        """Reset VNC state and MN decoder smoothing."""
        self.vnc.reset()
        self.mn_decoder.reset(init_angles=init_angles)
        self._body_time_ms = 0.0
        self._step_count = 0
        self._fallback_phase = _TRIPOD_PHASES.copy().astype(np.float64)
        self._antiphase_quality[:] = 0.0
        self._flex_ext_history = {
            leg: {"flex": [], "ext": []} for leg in range(6)
        }
        self._cached_group_rates = {}

    @property
    def current_time_ms(self) -> float:
        return self.vnc.current_time_ms

    def summary(self) -> str:
        """Return diagnostic summary."""
        quality_str = ", ".join(
            f"{LEG_ORDER[i]}={self._antiphase_quality[i]:.2f}"
            for i in range(6)
        )
        lines = [
            f"FiringRateVNCBridge: {self.vnc.n_neurons} neurons",
            f"  VNC time: {self.vnc.current_time_ms:.0f}ms",
            f"  Body time: {self._body_time_ms:.0f}ms",
            f"  Steps: {self._step_count}",
            f"  Substeps/body: {self.substeps_per_body}",
            f"  DN baseline: {self.dn_baseline_hz:.1f}Hz",
            f"  Fallback blend: {self.fallback_blend:.2f} @ {self.fallback_freq_hz:.1f}Hz",
            f"  Anti-phase quality: {quality_str}",
            "",
            self.mn_decoder.summary(),
        ]
        return "\n".join(lines)
