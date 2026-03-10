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
  - 4 joints unmapped (Tarsus1 for LM, LH, RM, RH) -> held at 0.0
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
    # LF (T1 left): joints 0-6
    0:  (+0.475, 0.28),   # Coxa
    1:  (+1.012, 0.34),   # Coxa_roll
    2:  (+0.042, 0.18),   # Coxa_yaw
    3:  (-2.237, 0.50),   # Femur
    4:  (+0.809, 0.22),   # Femur_roll
    5:  (+1.760, 0.45),   # Tibia
    6:  (-0.638, 0.24),   # Tarsus1
    # LM (T2 left): joints 7-13
    7:  (+0.087, 0.30),   # Coxa
    8:  (+1.849, 0.08),   # Coxa_roll
    9:  (+0.785, 0.13),   # Coxa_yaw
    10: (-1.485, 0.30),   # Femur
    11: (+0.011, 0.46),   # Femur_roll
    12: (+2.030, 0.18),   # Tibia
    13: (-0.772, 0.26),   # Tarsus1 (unmapped — held at rest)
    # LH (T3 left): joints 14-20
    14: (+0.570, 0.25),   # Coxa
    15: (+2.476, 0.22),   # Coxa_roll
    16: (+0.370, 0.12),   # Coxa_yaw
    17: (-1.651, 0.37),   # Femur
    18: (-0.291, 0.10),   # Femur_roll
    19: (+1.772, 0.55),   # Tibia
    20: (-0.082, 0.19),   # Tarsus1 (unmapped)
    # RF (T1 right): joints 21-27
    21: (+0.425, 0.28),   # Coxa
    22: (-0.974, 0.34),   # Coxa_roll (mirrored sign from LF)
    23: (-0.047, 0.18),   # Coxa_yaw
    24: (-2.223, 0.50),   # Femur
    25: (-0.821, 0.22),   # Femur_roll (mirrored)
    26: (+1.697, 0.45),   # Tibia
    27: (-0.683, 0.24),   # Tarsus1
    # RM (T2 right): joints 28-34
    28: (+0.076, 0.30),   # Coxa
    29: (-1.829, 0.08),   # Coxa_roll (mirrored)
    30: (-0.801, 0.13),   # Coxa_yaw (mirrored)
    31: (-1.498, 0.30),   # Femur
    32: (+0.102, 0.46),   # Femur_roll
    33: (+2.071, 0.18),   # Tibia
    34: (-0.722, 0.26),   # Tarsus1 (unmapped)
    # RH (T3 right): joints 35-41
    35: (+0.500, 0.25),   # Coxa
    36: (-2.431, 0.22),   # Coxa_roll (mirrored)
    37: (-0.360, 0.12),   # Coxa_yaw (mirrored)
    38: (-1.764, 0.37),   # Femur
    39: (+0.290, 0.10),   # Femur_roll (mirrored)
    40: (+1.886, 0.55),   # Tibia
    41: (-0.065, 0.19),   # Tarsus1 (unmapped)
}

# Tibia joint-type index within a leg (used for adhesion)
_TIBIA_WITHIN_LEG = 5

# Missing joints: Tarsus1 for LM(13), LH(20), RM(34), RH(41)
_MISSING_JOINTS = {13, 20, 34, 41}


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
        self._joint_pos_count = np.zeros(42, dtype=np.float64)
        self._joint_neg_count = np.zeros(42, dtype=np.float64)
        for i, (j, d) in enumerate(zip(self._joint_indices, self._directions)):
            if d > 0:
                self._joint_pos_count[j] += 1.0
            elif d < 0:
                self._joint_neg_count[j] += 1.0

        # Build per-joint amplitude and rest-angle arrays (42,)
        # Calibrated from actual CPG/PreprogrammedSteps output per leg segment
        self._amplitude = np.zeros(42, dtype=np.float64)
        self._rest_angle = np.zeros(42, dtype=np.float64)
        for j in range(42):
            rest, amp = _JOINT_PARAMS[j]
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
            if d > 0:
                pos_sum[j] += rate
            elif d < 0:
                neg_sum[j] += rate

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

        # Force missing joints to 0.0
        for j in _MISSING_JOINTS:
            raw_angles[j] = 0.0

        # ---- Step 3: exponential smoothing ----
        if self._prev_angles is None:
            self._prev_angles = raw_angles.copy()
        else:
            self._prev_angles = (
                self.alpha * raw_angles + (1.0 - self.alpha) * self._prev_angles
            )
        smoothed = self._prev_angles.copy()

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
