"""
VNC adapter: bridges the brain descending decoder to the VNC connectome model,
and maps VNC motor neuron output to FlyGym joint angles.

Pipeline:
    Brain BrainOutput
    → DescendingDecoder.get_group_rates() → {forward, turn_L, turn_R, rhythm, stance}
    → VNCAdapter.brain_to_dn_rates()       → {dn_body_id: rate_hz, ...}
    → Brian2VNCRunner.step()                 → {mn_body_id: rate_hz, ...}
    → VNCAdapter.mn_rates_to_action()      → FlyGym action dict

This module handles two cross-domain mappings:
    1. FlyWire brain readout DN types → MANC DN body IDs (for VNC input)
    2. MANC MN firing rates → FlyGym joint angles + adhesion (for body)
"""

import numpy as np
from pathlib import Path
from typing import Optional

from bridge.interfaces import BrainOutput, BodyObservation
from bridge.vnc_connectome import (
    Brian2VNCRunner, FakeVNCRunner, VNCConfig,
    OUR_READOUT_DN_TYPES,
    MN_TYPE_TO_JOINT_GROUP, JOINT_GROUP_TO_DOF,
    SEGMENT_SIDE_TO_LEG, LEG_ORDER, JOINTS_PER_LEG, N_JOINTS,
)


class VNCAdapter:
    """Adapts brain decoder output for VNC input and VNC output for FlyGym.

    Manages:
        1. DN type → MANC body ID cross-mapping (FlyWire → MANC)
        2. Decoder group rates → per-DN firing rates
        3. MN firing rates → joint angle offsets
        4. CPG-free joint angle generation from MN population activity

    Args:
        vnc: Brian2VNCRunner or FakeVNCRunner instance.
        decoder_groups_path: path to decoder_groups.json (FlyWire IDs).
        dn_type_map: optional pre-built {dn_type: [manc_body_ids]} dict.
            If None, loads from vnc.get_dn_type_to_body_ids().
        base_joint_angles: optional (42,) resting joint angles.
            If None, uses zeros (neutral pose).
    """

    def __init__(
        self,
        vnc: "Brian2VNCRunner | FakeVNCRunner",
        decoder_groups_path: Optional[str | Path] = None,
        dn_type_map: Optional[dict] = None,
        base_joint_angles: Optional[np.ndarray] = None,
    ):
        self.vnc = vnc

        # --- Build DN type → MANC body ID mapping ---
        if dn_type_map is not None:
            self._dn_type_to_manc = dn_type_map
        elif hasattr(vnc, "get_dn_type_to_body_ids"):
            self._dn_type_to_manc = vnc.get_dn_type_to_body_ids()
        else:
            self._dn_type_to_manc = {}

        # --- Load FlyWire decoder groups (maps group -> FlyWire IDs) ---
        # We need this to know which FlyWire readout IDs correspond to which
        # decoder groups, so we can assign group-level rates to DNs.
        self._decoder_groups = {}
        self._flywire_to_dn_type = {}
        if decoder_groups_path is not None:
            self._load_decoder_groups(decoder_groups_path)

        # --- Build FlyWire ID → MANC body IDs mapping ---
        self._build_flywire_to_manc_map()

        # --- MN metadata for joint mapping ---
        self._build_mn_joint_map()

        # --- Joint angle generation state ---
        self._base_angles = (
            base_joint_angles.copy() if base_joint_angles is not None
            else np.zeros(N_JOINTS, dtype=np.float32)
        )
        self._prev_angles = self._base_angles.copy()

        # Phase oscillator for CPG-free locomotion
        self._phase = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi])  # tripod
        self._phase_freq = 12.0  # Hz base frequency

        # Smoothing
        self._smooth_alpha = 0.3  # EMA smoothing for joint angles

        # Rate normalization
        self._mn_rate_scale = 80.0  # Hz that maps to full joint deflection

        # Running average of MN activity (persists between steps)
        self._activity_ema = 0.5  # start at moderate activity
        self._activity_ema_alpha = 0.3  # EMA smoothing factor for activity

    def _load_decoder_groups(self, path: str | Path):
        """Load decoder group JSON mapping FlyWire IDs to functional groups."""
        import json
        with open(path) as f:
            groups = json.load(f)

        self._decoder_groups = {
            "forward": set(map(int, groups.get("forward_ids", []))),
            "turn_left": set(map(int, groups.get("turn_left_ids", []))),
            "turn_right": set(map(int, groups.get("turn_right_ids", []))),
            "rhythm": set(map(int, groups.get("rhythm_ids", []))),
            "stance": set(map(int, groups.get("stance_ids", []))),
        }

        # Build reverse map: FlyWire ID → group name
        self._flywire_to_group = {}
        for group_name, ids in self._decoder_groups.items():
            for fid in ids:
                self._flywire_to_group[fid] = group_name

    def _build_flywire_to_manc_map(self):
        """Build mapping from FlyWire readout → MANC DN body IDs.

        Strategy: FlyWire IDs → DN type names (from decoder annotations)
                  → MANC body IDs (from MANC type/flywireType columns).

        Since we don't have a direct FlyWire→MANC body ID table, we use
        DN type names as the bridge. Each FlyWire readout neuron is assigned
        a DN type; that type maps to MANC body IDs with the same type label.
        """
        # For now, we map at the group level: each decoder group's mean rate
        # gets distributed to all MANC DNs of matching types.
        # The per-FlyWire-ID mapping requires the readout version's type
        # annotations, which we don't load here. Instead, we map by group.
        self._group_to_manc_dns = {
            "forward": [],
            "turn_left": [],
            "turn_right": [],
            "rhythm": [],
            "stance": [],
        }

        # Collect all MANC DN body IDs that correspond to our readout types
        all_readout_manc_ids = set()
        for dn_type in OUR_READOUT_DN_TYPES:
            manc_ids = self._dn_type_to_manc.get(dn_type, [])
            all_readout_manc_ids.update(manc_ids)

        # Since we don't know which specific DN types map to which decoder
        # groups without the full type annotation pipeline, we use a simpler
        # approach: all matched MANC DNs receive the mean of all group rates,
        # modulated by the group's dominance.
        self._readout_manc_ids = sorted(all_readout_manc_ids)
        print(f"  VNCAdapter: {len(self._readout_manc_ids)} MANC DNs matched "
              f"to brain readout ({len(OUR_READOUT_DN_TYPES)} types)")

    def _build_mn_joint_map(self):
        """Build mapping from MN body IDs to FlyGym joint indices.

        Each MN has: (body_id, segment, side, type) → (leg, dof_index, sign)
        """
        self._mn_to_joint = {}  # body_id -> (joint_idx_in_42, sign)

        if hasattr(self.vnc, "mn_info"):
            mn_info = self.vnc.mn_info
        else:
            mn_info = []

        mapped = 0
        unmapped = 0
        for entry in mn_info:
            if len(entry) == 6:
                bid, seg, side, mn_type, joint_group, sign = entry
            elif len(entry) == 4:
                bid, seg, side, mn_type = entry
                jg_entry = MN_TYPE_TO_JOINT_GROUP.get(mn_type, None)
                if jg_entry:
                    joint_group, sign = jg_entry
                else:
                    joint_group, sign = "unmapped", 0

            if joint_group == "unmapped" or sign == 0:
                unmapped += 1
                continue

            dof_entry = JOINT_GROUP_TO_DOF.get(joint_group, None)
            if dof_entry is None:
                unmapped += 1
                continue

            dof_within_leg, dof_sign = dof_entry
            leg = SEGMENT_SIDE_TO_LEG.get((str(seg), str(side)), None)
            if leg is None:
                unmapped += 1
                continue

            leg_idx = LEG_ORDER.index(leg)
            joint_idx = leg_idx * JOINTS_PER_LEG + dof_within_leg
            effective_sign = sign * dof_sign

            self._mn_to_joint[int(bid)] = (joint_idx, effective_sign)
            mapped += 1

        print(f"  VNCAdapter: {mapped} MNs mapped to joints, {unmapped} unmapped "
              f"(flight/abdominal/neck MNs)")

    def brain_to_dn_rates(
        self,
        group_rates: dict,
        brain_output: Optional[BrainOutput] = None,
    ) -> dict:
        """Convert decoder group rates to per-DN firing rates for VNC input.

        Args:
            group_rates: dict from DescendingDecoder.get_group_rates()
                keys: forward, turn_left, turn_right, rhythm, stance (Hz)
            brain_output: optional raw BrainOutput for per-neuron rates

        Returns:
            dict mapping MANC DN body_id (int) -> firing rate (Hz)
        """
        dn_rates = {}

        # Compute a composite rate for each MANC DN
        # Strategy: weighted sum of group rates, since different DN types
        # carry different functional signals
        fwd_rate = group_rates.get("forward", 0.0)
        turn_l = group_rates.get("turn_left", 0.0)
        turn_r = group_rates.get("turn_right", 0.0)
        rhythm_rate = group_rates.get("rhythm", 0.0)
        stance_rate = group_rates.get("stance", 0.0)

        # Mean descending drive (all groups contribute)
        mean_rate = (fwd_rate + turn_l + turn_r + rhythm_rate + stance_rate) / 5.0

        # Assign to all matched MANC DNs
        # DNs that specifically match a known walking-related type get
        # group-specific rates; others get the mean.
        walking_forward_types = {
            "DNa01", "DNa02", "DNg100", "DNb02", "DNp30",
        }
        walking_turn_types = {
            "DNg11", "DNa02", "DNg29",  # turning-related DNs
        }
        walking_rhythm_types = {
            "DNb01", "DNg100",  # coordination/rhythm DNs
        }

        for dn_type, manc_ids in self._dn_type_to_manc.items():
            if dn_type not in OUR_READOUT_DN_TYPES:
                continue

            # Determine which group rate to use for this DN type
            if dn_type in walking_forward_types:
                rate = fwd_rate * 0.6 + mean_rate * 0.4
            elif dn_type in walking_turn_types:
                rate = (turn_l + turn_r) * 0.3 + mean_rate * 0.4
            elif dn_type in walking_rhythm_types:
                rate = rhythm_rate * 0.5 + mean_rate * 0.5
            else:
                rate = mean_rate

            # Clip to reasonable range
            rate = float(np.clip(rate, 0.0, 200.0))

            for bid in manc_ids:
                dn_rates[int(bid)] = rate

        return dn_rates

    def mn_rates_to_action(
        self,
        mn_rates: dict,
        dt_s: float = 0.01,
        body_obs: Optional[BodyObservation] = None,
    ) -> dict:
        """Convert MN firing rates to FlyGym action dict.

        This is the core MN→joint mapping. MN firing rates drive joint angles
        through a rate-coded scheme:
            - Each MN maps to a specific joint DOF with a sign
            - Multiple MNs can map to the same joint (pool summation)
            - Higher firing rate = larger joint deflection
            - Antagonistic MN pools (flexor vs extensor) compete

        The output includes a phase-driven oscillatory component that provides
        basic rhythmic leg movement, modulated by the MN population activity.

        Args:
            mn_rates: dict {mn_body_id: firing_rate_hz}
            dt_s: timestep in seconds
            body_obs: optional body observation for feedback

        Returns:
            dict {"joints": ndarray(42), "adhesion": ndarray(6)}
        """
        # --- Accumulate MN drive per joint ---
        joint_drive = np.zeros(N_JOINTS, dtype=np.float64)
        joint_count = np.zeros(N_JOINTS, dtype=np.float64)

        for bid, rate in mn_rates.items():
            bid = int(bid)
            if bid not in self._mn_to_joint:
                continue
            joint_idx, sign = self._mn_to_joint[bid]
            # Normalize rate: tanh(rate / scale) gives [-1, 1] range
            normalized = np.tanh(rate / self._mn_rate_scale)
            joint_drive[joint_idx] += sign * normalized
            joint_count[joint_idx] += 1.0

        # Average where multiple MNs map to same joint
        active_mask = joint_count > 0
        joint_drive[active_mask] /= joint_count[active_mask]

        # --- Phase oscillator for rhythmic component ---
        # This provides the basic swing/stance alternation that MN activity
        # modulates. Without this, the fly would just hold a static pose.
        self._phase += 2.0 * np.pi * self._phase_freq * dt_s

        # Compute global activity level from MN rates (smoothed EMA)
        all_rates = list(mn_rates.values())
        mean_mn_rate = float(np.mean(all_rates)) if all_rates else 0.0
        instant_activity = np.tanh(mean_mn_rate / self._mn_rate_scale)
        # Update running average — this prevents oscillator from dying
        # when MN activity is sparse (only ~12% fire per step)
        self._activity_ema = (
            self._activity_ema_alpha * instant_activity +
            (1.0 - self._activity_ema_alpha) * self._activity_ema
        )
        # Use the EMA as the activity level, with a floor
        activity_level = max(self._activity_ema, 0.3)

        # Phase-driven oscillatory joint angles (simplified CPG replacement)
        # Each leg gets a phase-offset sinusoidal drive
        osc_angles = np.zeros(N_JOINTS, dtype=np.float64)
        for leg_i in range(6):
            phase = self._phase[leg_i]
            base_offset = leg_i * JOINTS_PER_LEG

            # Coxa protraction/retraction: swing phase
            osc_angles[base_offset + 0] = 0.3 * np.sin(phase) * activity_level
            # Coxa roll: small lateral component
            osc_angles[base_offset + 1] = 0.05 * np.sin(phase + 0.5) * activity_level
            # Coxa yaw: minimal
            osc_angles[base_offset + 2] = 0.02 * np.cos(phase) * activity_level
            # Femur: lift during swing
            osc_angles[base_offset + 3] = 0.25 * np.sin(phase) * activity_level
            # Femur roll: minimal
            osc_angles[base_offset + 4] = 0.02 * np.sin(phase) * activity_level
            # Tibia: flex/extend during swing
            osc_angles[base_offset + 5] = 0.4 * np.sin(phase + np.pi / 4) * activity_level
            # Tarsus: contact-dependent
            osc_angles[base_offset + 6] = 0.1 * np.cos(phase) * activity_level

        # --- Frequency modulation from MN activity ---
        # Higher overall MN activity = faster stepping
        freq_mod = 1.0 + 0.5 * activity_level
        self._phase_freq = 12.0 * freq_mod

        # --- Combine: MN-driven modulation + oscillatory base ---
        # MN drive provides the descending modulation (amplitude, asymmetry)
        # Oscillator provides the rhythmic timing
        mn_weight = 0.4  # how much MN activity overrides oscillator
        osc_weight = 0.6  # how much oscillator contributes

        raw_angles = (
            self._base_angles +
            mn_weight * joint_drive * 0.5 +  # MN modulation (scaled)
            osc_weight * osc_angles            # rhythmic oscillation
        )

        # --- Exponential moving average smoothing ---
        alpha = self._smooth_alpha
        smoothed = alpha * raw_angles + (1.0 - alpha) * self._prev_angles
        self._prev_angles = smoothed.copy()

        # --- Adhesion: stance legs stick, swing legs release ---
        adhesion = np.zeros(6, dtype=int)
        for leg_i in range(6):
            phase = self._phase[leg_i] % (2.0 * np.pi)
            # Stance phase: pi to 2*pi (leg on ground)
            adhesion[leg_i] = 1 if (np.pi < phase < 2.0 * np.pi) else 0

        return {
            "joints": smoothed.astype(np.float32),
            "adhesion": adhesion,
        }

    def step(
        self,
        group_rates: dict,
        dt_ms: float = 10.0,
        brain_output: Optional[BrainOutput] = None,
        body_obs: Optional[BodyObservation] = None,
    ) -> dict:
        """Full pipeline: decoder group rates → VNC → FlyGym action.

        This is the main entry point that replaces LocomotionBridge.step().

        Args:
            group_rates: dict from DescendingDecoder.get_group_rates()
            dt_ms: brain simulation window in milliseconds
            brain_output: optional raw brain output
            body_obs: optional body observation for feedback

        Returns:
            FlyGym action dict {"joints": ndarray(42), "adhesion": ndarray(6)}
        """
        # 1. Convert decoder groups → per-DN rates
        dn_rates = self.brain_to_dn_rates(group_rates, brain_output)

        # 2. Run VNC simulation
        mn_rates = self.vnc.step(dn_rates, dt_ms=dt_ms)

        # 3. Convert MN rates → joint angles
        action = self.mn_rates_to_action(
            mn_rates, dt_s=dt_ms / 1000.0, body_obs=body_obs
        )

        return action

    def step_body(
        self,
        action: dict,
        body_obs: Optional[BodyObservation] = None,
    ) -> dict:
        """Generate action for body steps between brain steps.

        Between brain steps, the VNC doesn't re-simulate — we just advance
        the phase oscillator and interpolate joint angles.

        Args:
            action: previous action from step()
            body_obs: current body observation

        Returns:
            FlyGym action dict for the intermediate body step
        """
        # Advance phase for this body step
        body_dt = 1e-4  # FlyGym default timestep
        self._phase += 2.0 * np.pi * self._phase_freq * body_dt

        # Interpolate: use smoothed previous angles with slight phase advance
        # This keeps the legs moving between brain steps
        activity_level = np.tanh(
            np.mean(np.abs(self._prev_angles)) / 0.3
        )
        activity_level = float(np.clip(activity_level, 0.1, 1.0))

        osc_delta = np.zeros(N_JOINTS, dtype=np.float64)
        for leg_i in range(6):
            phase = self._phase[leg_i]
            base_offset = leg_i * JOINTS_PER_LEG
            osc_delta[base_offset + 0] = 0.3 * np.sin(phase) * activity_level
            osc_delta[base_offset + 3] = 0.25 * np.sin(phase) * activity_level
            osc_delta[base_offset + 5] = 0.4 * np.sin(phase + np.pi / 4) * activity_level

        raw = self._base_angles + 0.6 * osc_delta + 0.4 * (self._prev_angles - self._base_angles)

        alpha = self._smooth_alpha
        smoothed = alpha * raw + (1.0 - alpha) * self._prev_angles
        self._prev_angles = smoothed.copy()

        adhesion = np.zeros(6, dtype=int)
        for leg_i in range(6):
            phase = self._phase[leg_i] % (2.0 * np.pi)
            adhesion[leg_i] = 1 if (np.pi < phase < 2.0 * np.pi) else 0

        return {
            "joints": smoothed.astype(np.float32),
            "adhesion": adhesion,
        }


def create_vnc_adapter(
    vnc: "Brian2VNCRunner | FakeVNCRunner",
    decoder_groups_path: Optional[str | Path] = None,
    base_joint_angles: Optional[np.ndarray] = None,
) -> VNCAdapter:
    """Factory for VNCAdapter.

    Args:
        vnc: VNC model instance (real or fake).
        decoder_groups_path: path to decoder_groups.json.
        base_joint_angles: optional resting pose.

    Returns:
        Configured VNCAdapter instance.
    """
    from bridge.config import BridgeConfig
    cfg = BridgeConfig()

    if decoder_groups_path is None:
        decoder_groups_path = cfg.decoder_groups_path

    return VNCAdapter(
        vnc=vnc,
        decoder_groups_path=decoder_groups_path,
        base_joint_angles=base_joint_angles,
    )
