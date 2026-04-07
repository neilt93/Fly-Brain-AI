"""
Motor neuron decoder: converts MANC motor neuron firing rates to FlyGym
joint angles (42 DOFs) and per-leg adhesion signals (6).

Maps ~328 identified leg motor neurons from the Male Adult Nerve Cord
(MANC v0.9) to 42 FlyGym actuated joints across 6 legs x 7 DOFs.

Anatomy reference (Baek & Mann 2009, Azevedo et al. 2024 MANC):
  - T1 (prothoracic)  -> front legs  (LF, RF)
  - T2 (mesothoracic) -> middle legs (LM, RM)
  - T3 (metathoracic) -> hind legs   (LH, RH)

FlyGym joint order per leg (7 DOFs):
  0: Coxa       (thorax-coxa pitch / elevation-depression)
  1: Coxa_roll  (thorax-coxa pro/remotion = forward/backward swing)
  2: Coxa_yaw   (thorax-coxa rotation about long axis)
  3: Femur      (coxa-trochanter-femur flexion/extension)
  4: Femur_roll (femur rotation / reductor)
  5: Tibia      (femur-tibia flexion/extension)
  6: Tarsus1    (tibia-tarsus flexion)

Joint angle computation per joint j:
    net_drive_j = sum_over_MNs( direction_i * rate_i )
    angle_j     = rest_angle_j + amplitude_j * tanh(net_drive_j / rate_scale)

Adhesion per leg: 1 if tibia angle > rest_angle (net extension), else 0.

Exponential smoothing prevents jerky joint motion:
    angle_t = alpha * new_angle + (1 - alpha) * angle_{t-1}

Known limitations:
  - 4 joints unmapped (Tarsus1 for LM, LH, RM, RH) -> held at rest angle
  - Many-to-one mapping: multiple MNs drive the same joint (pool coding)
  - MN body IDs not in the mapping are silently ignored
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


# Default mapping file location (relative to this module)
_DEFAULT_MAPPING = Path(__file__).resolve().parent.parent / "data" / "mn_joint_mapping.json"

# FlyGym joint names in order (for reference / debugging)
FLYGYM_JOINTS = [
    "joint_LFCoxa", "joint_LFCoxa_roll", "joint_LFCoxa_yaw",
    "joint_LFFemur", "joint_LFFemur_roll", "joint_LFTibia", "joint_LFTarsus1",
    "joint_LMCoxa", "joint_LMCoxa_roll", "joint_LMCoxa_yaw",
    "joint_LMFemur", "joint_LMFemur_roll", "joint_LMTibia", "joint_LMTarsus1",
    "joint_LHCoxa", "joint_LHCoxa_roll", "joint_LHCoxa_yaw",
    "joint_LHFemur", "joint_LHFemur_roll", "joint_LHTibia", "joint_LHTarsus1",
    "joint_RFCoxa", "joint_RFCoxa_roll", "joint_RFCoxa_yaw",
    "joint_RFFemur", "joint_RFFemur_roll", "joint_RFTibia", "joint_RFTarsus1",
    "joint_RMCoxa", "joint_RMCoxa_roll", "joint_RMCoxa_yaw",
    "joint_RMFemur", "joint_RMFemur_roll", "joint_RMTibia", "joint_RMTarsus1",
    "joint_RHCoxa", "joint_RHCoxa_roll", "joint_RHCoxa_yaw",
    "joint_RHFemur", "joint_RHFemur_roll", "joint_RHTibia", "joint_RHTarsus1",
]

# Leg names in FlyGym order
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]

# Leg base index in the 42-element joint array
LEG_OFFSET = {"LF": 0, "LM": 7, "LH": 14, "RF": 21, "RM": 28, "RH": 35}

# Joint names within each leg
JOINT_NAMES_PER_LEG = [
    "Coxa", "Coxa_roll", "Coxa_yaw",
    "Femur", "Femur_roll", "Tibia", "Tarsus1",
]

# ----- Per-joint parameters calibrated from FlyGym PreprogrammedSteps -----
# Each leg segment has different rest angles due to fly anatomy.
# Values measured from 1000-step CPG trajectory (forward_drive=1.0).
# Format: (rest_angle, amplitude) per absolute joint index [0..41]
_JOINT_PARAMS = {
    # Symmetrized: L/R rest angles averaged per segment so the decoder
    # does not introduce a heading bias.  Mirrored joints (roll, yaw)
    # use the mean magnitude with opposite signs.
    #
    # LF (T1 left): joints 0-6
    0:  (+0.450, 0.28),   # Coxa          avg(0.475, 0.425)
    1:  (+0.993, 0.34),   # Coxa_roll     avg(|1.012|, |0.974|) = 0.993
    2:  (+0.045, 0.18),   # Coxa_yaw      avg(|0.042|, |0.047|) = 0.045
    3:  (-2.230, 0.50),   # Femur         avg(2.237, 2.223) = 2.230
    4:  (+0.815, 0.22),   # Femur_roll    avg(|0.809|, |0.821|) = 0.815
    5:  (+1.729, 0.45),   # Tibia         avg(1.760, 1.697) = 1.729
    6:  (-0.661, 0.24),   # Tarsus1       avg(0.638, 0.683) = 0.661
    # LM (T2 left): joints 7-13
    7:  (+0.082, 0.30),   # Coxa          avg(0.087, 0.076)
    8:  (+1.839, 0.08),   # Coxa_roll     avg(|1.849|, |1.829|) = 1.839
    9:  (+0.793, 0.13),   # Coxa_yaw      avg(|0.785|, |0.801|) = 0.793
    10: (-1.492, 0.30),   # Femur         avg(1.485, 1.498) = 1.492
    11: (+0.057, 0.46),   # Femur_roll    avg(|0.011|, |0.102|) = 0.057
    12: (+2.051, 0.18),   # Tibia         avg(2.030, 2.071) = 2.051
    13: (-0.747, 0.26),   # Tarsus1       avg(0.772, 0.722) = 0.747 (unmapped)
    # LH (T3 left): joints 14-20
    14: (+0.535, 0.25),   # Coxa          avg(0.570, 0.500)
    15: (+2.454, 0.22),   # Coxa_roll     avg(|2.476|, |2.431|) = 2.454
    16: (+0.365, 0.12),   # Coxa_yaw      avg(|0.370|, |0.360|) = 0.365
    17: (-1.708, 0.37),   # Femur         avg(1.651, 1.764) = 1.708
    18: (-0.291, 0.10),   # Femur_roll    avg(|0.291|, |0.290|) = 0.291
    19: (+1.829, 0.55),   # Tibia         avg(1.772, 1.886) = 1.829
    20: (-0.074, 0.19),   # Tarsus1       avg(0.082, 0.065) = 0.074 (unmapped)
    # RF (T1 right): joints 21-27
    21: (+0.450, 0.28),   # Coxa          (same as LF)
    22: (-0.993, 0.34),   # Coxa_roll     (mirrored)
    23: (-0.045, 0.18),   # Coxa_yaw      (mirrored)
    24: (-2.230, 0.50),   # Femur         (same as LF)
    25: (-0.815, 0.22),   # Femur_roll    (mirrored)
    26: (+1.729, 0.45),   # Tibia         (same as LF)
    27: (-0.661, 0.24),   # Tarsus1       (same as LF)
    # RM (T2 right): joints 28-34
    28: (+0.082, 0.30),   # Coxa          (same as LM)
    29: (-1.839, 0.08),   # Coxa_roll     (mirrored)
    30: (-0.793, 0.13),   # Coxa_yaw      (mirrored)
    31: (-1.492, 0.30),   # Femur         (same as LM)
    32: (-0.057, 0.46),   # Femur_roll    (mirrored from LM)
    33: (+2.051, 0.18),   # Tibia         (same as LM)
    34: (-0.747, 0.26),   # Tarsus1       (same as LM, unmapped)
    # RH (T3 right): joints 35-41
    35: (+0.535, 0.25),   # Coxa          (same as LH)
    36: (-2.454, 0.22),   # Coxa_roll     (mirrored)
    37: (-0.365, 0.12),   # Coxa_yaw      (mirrored)
    38: (-1.708, 0.37),   # Femur         (same as LH)
    39: (+0.291, 0.10),   # Femur_roll    (mirrored, same mag)
    40: (+1.829, 0.55),   # Tibia         (same as LH)
    41: (-0.074, 0.19),   # Tarsus1       (same as LH, unmapped)
}

# Tibia joint-type index within a leg (used for adhesion)
_TIBIA_WITHIN_LEG = 5

# Missing joints: Tarsus1 for LM(13), LH(20), RM(34), RH(41)
_MISSING_JOINTS = {13, 20, 34, 41}

# Path to connectome-derived params (generated by scripts/derive_mn_decoder_params.py)
_DERIVED_PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "mn_decoder_params.json"


def _load_derived_params() -> dict | None:
    """Load connectome-derived rest angles + amplitudes if available.

    Returns a dict mapping joint index (0-41) to (rest_angle, amplitude),
    or None if the params file doesn't exist.
    """
    if not _DERIVED_PARAMS_PATH.exists():
        return None
    with open(_DERIVED_PARAMS_PATH) as f:
        params = json.load(f)
    result = {}
    rest = params.get("rest_angles", {})
    amps = params.get("amplitudes", {})
    for j in range(42):
        r = float(rest.get(str(j), _JOINT_PARAMS[j][0]))
        a = float(amps.get(str(j), _JOINT_PARAMS[j][1]))
        result[j] = (r, a)
    return result


class MotorNeuronDecoder:
    """Convert motor neuron firing rates into FlyGym joint angles and adhesion.

    Each MANC motor neuron is mapped to one of 42 FlyGym joints with a signed
    direction indicating agonist (+) or antagonist (-) action.  For each joint
    the decoder computes:

        net_drive = sum( direction_i * rate_i )   over all MNs for this joint
        angle     = rest + amplitude * tanh(net_drive / rate_scale)

    Exponential smoothing is applied across time steps to prevent abrupt jumps.
    Adhesion per leg is derived from tibia extension state.

    Parameters
    ----------
    mapping_path : Path or str
        Path to ``mn_joint_mapping.json``.
    rate_scale : float
        Firing rate (Hz) at which tanh reaches ~0.76 of max amplitude.
    alpha : float
        Exponential smoothing coefficient in [0, 1].  Higher = less smoothing.
    """

    def __init__(
        self,
        mapping_path: Optional[Union[str, Path]] = None,
        rate_scale: float = 50.0,
        alpha: float = 0.4,
    ):
        path = Path(mapping_path) if mapping_path else _DEFAULT_MAPPING
        if not path.exists():
            raise FileNotFoundError(
                f"MN-joint mapping file not found: {path}\n"
                "This file maps MANC motor neurons to FlyGym joints.\n"
                "Expected at: data/mn_joint_mapping.json"
            )
        with open(path) as f:
            raw: dict = json.load(f)

        self.rate_scale = rate_scale
        self.alpha = alpha

        # ---- Parse mapping into fast lookup structures ----
        self._body_ids: list[str] = []
        self._joint_indices: list[int] = []
        self._directions: list[float] = []
        self._mn_types: list[str] = []
        self._legs: list[str] = []
        self._muscle_groups: list[str] = []

        for body_id, info in raw.items():
            self._body_ids.append(body_id)
            self._joint_indices.append(int(info["joint_idx"]))
            self._directions.append(float(info["direction"]))
            self._mn_types.append(info["mn_type"])
            self._legs.append(info["leg"])
            self._muscle_groups.append(info["muscle_group"])

        self._joint_indices_arr = np.array(self._joint_indices, dtype=np.int32)
        self._directions_arr = np.array(self._directions, dtype=np.float64)
        self.n_mns = len(self._body_ids)

        # Body-ID -> internal index (for dict-based input)
        self._id_to_idx: Dict[str, int] = {
            bid: i for i, bid in enumerate(self._body_ids)
        }

        # Precompute per-joint positive/negative MN counts for pool-normalized decoding.
        # Without normalization, joints with many more flexors than extensors
        # (e.g., 15 flexors vs 4 extensors for tibia) are biased toward flexion
        # when all MNs fire at similar rates.
        real_pos_count = np.zeros(42, dtype=np.float64)
        real_neg_count = np.zeros(42, dtype=np.float64)
        for i, (j, d) in enumerate(zip(self._joint_indices, self._directions)):
            if d > 0:
                real_pos_count[j] += abs(d)
            elif d < 0:
                real_neg_count[j] += abs(d)

        # Symmetrize MN pool counts between corresponding L/R joints.
        # Store both real and symmetrized counts — the per-MN weight correction
        # factor ensures that asymmetric MANC pool sizes don't bias the decoder.
        self._joint_pos_count = real_pos_count.copy()
        self._joint_neg_count = real_neg_count.copy()
        for li, leg_l in enumerate(["LF", "LM", "LH"]):
            ri = li + 3  # RF, RM, RH
            off_l, off_r = LEG_OFFSET[leg_l], LEG_OFFSET[LEGS[ri]]
            for dof in range(7):
                jl, jr = off_l + dof, off_r + dof
                avg_pos = (real_pos_count[jl] + real_pos_count[jr]) / 2.0
                avg_neg = (real_neg_count[jl] + real_neg_count[jr]) / 2.0
                self._joint_pos_count[jl] = self._joint_pos_count[jr] = avg_pos
                self._joint_neg_count[jl] = self._joint_neg_count[jr] = avg_neg

        # Per-MN weight correction: scale each MN's contribution so that the
        # sum for a pool with uniform rates equals sym_count * rate (not real_count * rate).
        # correction = sym_count / real_count for that MN's pool.
        self._mn_weight = np.ones(len(self._body_ids), dtype=np.float64)
        for i, (j, d) in enumerate(zip(self._joint_indices, self._directions)):
            if d > 0 and real_pos_count[j] > 0:
                self._mn_weight[i] = self._joint_pos_count[j] / real_pos_count[j]
            elif d < 0 and real_neg_count[j] > 0:
                self._mn_weight[i] = self._joint_neg_count[j] / real_neg_count[j]

        # Build per-joint amplitude and rest-angle arrays (42,)
        # Try connectome-derived params first, fall back to CPG-calibrated
        derived = _load_derived_params()
        joint_params = derived if derived is not None else _JOINT_PARAMS

        self._amplitude = np.zeros(42, dtype=np.float64)
        self._rest_angle = np.zeros(42, dtype=np.float64)
        for j in range(42):
            rest, amp = joint_params[j]
            self._rest_angle[j] = rest
            self._amplitude[j] = amp

        # Tibia joint indices for each leg (for adhesion)
        self._tibia_indices = np.array(
            [LEG_OFFSET[leg] + _TIBIA_WITHIN_LEG for leg in LEGS],
            dtype=np.int32,
        )

        # Smoothing state
        self._prev_angles: Optional[np.ndarray] = None

    # ---- Public API ----

    def decode(
        self,
        mn_body_ids: np.ndarray,
        firing_rates_hz: np.ndarray,
    ) -> dict:
        """Convert MN firing rates to FlyGym action dict.

        Parameters
        ----------
        mn_body_ids : ndarray of int or str
            MANC body IDs for the motor neurons that fired.
        firing_rates_hz : ndarray of float
            Firing rate (Hz) for each corresponding body ID.

        Returns
        -------
        dict
            ``{'joints': ndarray(42,), 'adhesion': ndarray(6,)}``.
        """
        mn_body_ids = np.asarray(mn_body_ids)
        firing_rates_hz = np.asarray(firing_rates_hz, dtype=np.float64)

        if mn_body_ids.shape[0] != firing_rates_hz.shape[0]:
            raise ValueError(
                f"mn_body_ids length ({mn_body_ids.shape[0]}) != "
                f"firing_rates_hz length ({firing_rates_hz.shape[0]})"
            )

        # ---- Step 1: pool-normalized drive per joint ----
        # Compute mean firing rate for positive-direction (extensor) and
        # negative-direction (flexor) MN pools separately, then take the
        # difference.  This prevents pool-size imbalance from biasing the
        # joint angle (e.g. 15 flexors vs 4 extensors for tibia).
        pos_sum = np.zeros(42, dtype=np.float64)
        neg_sum = np.zeros(42, dtype=np.float64)

        for k in range(mn_body_ids.shape[0]):
            bid = str(mn_body_ids[k])
            idx = self._id_to_idx.get(bid)
            if idx is None:
                continue
            j = self._joint_indices_arr[idx]
            d = self._directions_arr[idx]
            rate = firing_rates_hz[k]
            w = self._mn_weight[idx]  # pool-composition correction
            if d > 0:
                pos_sum[j] += abs(d) * rate * w
            elif d < 0:
                neg_sum[j] += abs(d) * rate * w

        # Mean rate per pool (avoid /0)
        pos_mean = np.divide(pos_sum, self._joint_pos_count,
                             out=np.zeros(42), where=self._joint_pos_count > 0)
        neg_mean = np.divide(neg_sum, self._joint_neg_count,
                             out=np.zeros(42), where=self._joint_neg_count > 0)

        # Net drive computation:
        # - For joints with both agonist (+) and antagonist (-) MNs:
        #   net_drive = mean_agonist - mean_antagonist
        # - For joints with only one direction:
        #   Use deviation from baseline (50Hz) so the joint can move both ways.
        #   net_drive = (mean_rate - baseline) * direction
        baseline_hz = 50.0
        net_drive = np.zeros(42, dtype=np.float64)
        for j in range(42):
            has_pos = self._joint_pos_count[j] > 0
            has_neg = self._joint_neg_count[j] > 0
            if has_pos and has_neg:
                net_drive[j] = pos_mean[j] - neg_mean[j]
            elif has_pos:
                net_drive[j] = pos_mean[j] - baseline_hz
            elif has_neg:
                net_drive[j] = baseline_hz - neg_mean[j]
            # else: no MNs, net_drive stays 0

        # ---- Step 2: nonlinear activation -> joint angles ----
        raw_angles = self._rest_angle + self._amplitude * np.tanh(
            net_drive / self.rate_scale
        )

        # Force missing joints to rest angle (not 0.0 — avoids tarsus
        # being driven away from its natural pose)
        for j in _MISSING_JOINTS:
            raw_angles[j] = self._rest_angle[j]

        # ---- Step 3: exponential smoothing ----
        if self._prev_angles is None:
            self._prev_angles = raw_angles.copy()
        else:
            self._prev_angles = (
                self.alpha * raw_angles + (1.0 - self.alpha) * self._prev_angles
            )
        smoothed = self._prev_angles.copy()

        # Force missing joints to rest angle after smoothing too
        for j in _MISSING_JOINTS:
            smoothed[j] = self._rest_angle[j]

        # ---- Step 4: adhesion from tibia angles ----
        # Adhesion ON when tibia angle is above rest (flexed toward ground).
        # In FlyGym, positive tibia angle = more flexed = foot pressed down.
        tibia_angles = smoothed[self._tibia_indices]
        tibia_rest = self._rest_angle[self._tibia_indices]
        adhesion = (tibia_angles > tibia_rest).astype(np.float64)

        return {
            "joints": smoothed.astype(np.float32),
            "adhesion": adhesion.astype(np.float32),
        }

    def reset(self, init_angles: np.ndarray = None) -> None:
        """Clear or set smoothing state.

        Args:
            init_angles: If provided, initialize smoothing to this 42-element
                         array (e.g. FlyGym's init pose). Otherwise clear.
        """
        if init_angles is not None:
            self._prev_angles = np.asarray(init_angles, dtype=np.float64).copy()
        else:
            self._prev_angles = None

    # ---- Alternative constructors ----

    @classmethod
    def from_biomechanics(
        cls,
        mapping_path: Optional[Union[str, Path]] = None,
        params_path: Optional[Union[str, Path]] = None,
        rate_scale: float = 50.0,
        alpha: float = 0.4,
    ) -> "MotorNeuronDecoder":
        """Create decoder with connectome-derived rest angles and amplitudes.

        Uses params from ``data/mn_decoder_params.json`` (generated by
        ``scripts/derive_mn_decoder_params.py``).  Falls back to CPG-calibrated
        params if the file doesn't exist.

        Parameters
        ----------
        mapping_path : Path or str, optional
            Path to ``mn_joint_mapping.json``.
        params_path : Path or str, optional
            Path to ``mn_decoder_params.json``.  If None, uses the default
            location (``data/mn_decoder_params.json``).
        rate_scale : float
            Firing rate (Hz) at which tanh reaches ~0.76 of max amplitude.
        alpha : float
            Exponential smoothing coefficient in [0, 1].
        """
        decoder = cls(mapping_path=mapping_path, rate_scale=rate_scale, alpha=alpha)

        if params_path is not None:
            p = Path(params_path)
            if p.exists():
                with open(p) as f:
                    params = json.load(f)
                for j in range(42):
                    r = float(params["rest_angles"].get(str(j), decoder._rest_angle[j]))
                    a = float(params["amplitudes"].get(str(j), decoder._amplitude[j]))
                    decoder._rest_angle[j] = r
                    decoder._amplitude[j] = a

        return decoder

    # ---- Properties ----

    @property
    def body_ids(self) -> list[str]:
        """List of MANC body IDs in mapping order."""
        return list(self._body_ids)

    @property
    def mapped_joints(self) -> set[int]:
        """Set of joint indices (0-41) that have at least one mapped MN."""
        return set(self._joint_indices)

    @property
    def unmapped_joints(self) -> set[int]:
        """Set of joint indices (0-41) with no mapped MN."""
        return set(range(42)) - self.mapped_joints

    # ---- Diagnostics ----

    def summary(self) -> str:
        """Return a human-readable summary of the mapping."""
        lines = [
            f"MotorNeuronDecoder: {self.n_mns} MNs -> 42 joints",
            f"  Mapped joints: {len(self.mapped_joints)}/42",
            f"  Unmapped joints: {sorted(self.unmapped_joints)}",
            f"  rate_scale: {self.rate_scale} Hz, alpha: {self.alpha}",
            "",
            "  Per-leg breakdown:",
        ]
        for leg in LEGS:
            offset = LEG_OFFSET[leg]
            leg_mns = [
                i for i, j in enumerate(self._joint_indices)
                if offset <= j < offset + 7
            ]
            joint_detail = []
            for jj in range(7):
                j_abs = offset + jj
                n_pos = sum(
                    1 for i in leg_mns
                    if self._joint_indices[i] == j_abs and self._directions[i] > 0
                )
                n_neg = sum(
                    1 for i in leg_mns
                    if self._joint_indices[i] == j_abs and self._directions[i] < 0
                )
                if n_pos + n_neg > 0:
                    joint_detail.append(
                        f"{JOINT_NAMES_PER_LEG[jj]}(+{n_pos}/-{n_neg})"
                    )
            lines.append(f"    {leg} ({len(leg_mns)} MNs): {', '.join(joint_detail)}")

        # Muscle group counts
        from collections import Counter
        mg_counts = Counter(self._muscle_groups)
        lines.append("")
        lines.append("  Muscle groups:")
        for mg, count in sorted(mg_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {mg}: {count}")

        return "\n".join(lines)


def load_mapping_dataframe():
    """Load the MN-joint mapping as a pandas DataFrame (for analysis).

    Returns a DataFrame with columns:
        body_id, leg, joint_idx, direction, mn_type, muscle_group,
        neuromere, side, joint_name
    """
    import pandas as pd

    if not _DEFAULT_MAPPING.exists():
        raise FileNotFoundError(
            f"MN-joint mapping file not found: {_DEFAULT_MAPPING}\n"
            "Expected at: data/mn_joint_mapping.json"
        )
    with open(_DEFAULT_MAPPING) as f:
        raw = json.load(f)

    rows = []
    for body_id, info in raw.items():
        leg = info["leg"]
        j_abs = info["joint_idx"]
        j_within = j_abs - LEG_OFFSET[leg]
        joint_name = f"joint_{leg}{JOINT_NAMES_PER_LEG[j_within]}"
        rows.append({
            "body_id": body_id,
            "leg": leg,
            "joint_idx": j_abs,
            "joint_within_leg": j_within,
            "direction": info["direction"],
            "mn_type": info["mn_type"],
            "muscle_group": info["muscle_group"],
            "neuromere": info["neuromere"],
            "side": info["side"],
            "joint_name": joint_name,
        })

    return pd.DataFrame(rows)
