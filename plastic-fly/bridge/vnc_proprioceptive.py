"""
Proprioceptive encoder for the BANC VNC pipeline.

Maps FlyGym body observations to firing rates for BANC proprioceptive sensory
neuron populations, then injects corresponding currents into VNC premotor
interneurons via the connectivity graph.

BANC proprioceptive neuron types:
    chordotonal / FeCO  — femoral chordotonal organ; encodes joint angle
                          (position) and joint velocity (movement)
    campaniform sensilla — cuticular strain sensors; encode load/force
    hair plate           — joint-movement detectors at articulations

Numbers (BANC): 2,305 proprioceptive neurons making 148,869 synapses
onto 5,792 VNC premotor interneurons.

Encoding model (tuning curves):
    campaniform   -> contact_forces (6,)  — monotonic load response
    chordotonal   -> joint_angles (42,)   — position (bell-shaped around rest)
                   + joint_velocities (42,) — velocity (rectified linear)
    hair plate    -> joint_velocities (42,) — movement detector (abs velocity)

The encoder does NOT inject spikes directly into sensory neurons.  Instead it
computes weighted current contributions to downstream VNC premotor interneurons,
exploiting the known BANC sensory->premotor connectivity.  This avoids requiring
the sensory neurons to be part of the VNC Brian2/rate model.

Usage:
    from bridge.vnc_proprioceptive import ProprioceptiveEncoder
    from bridge.banc_loader import BANCLoader, BANCVNCData, load_banc_vnc

    loader = BANCLoader()
    banc_data = load_banc_vnc()
    encoder = ProprioceptiveEncoder(banc_data, vnc_runner, loader=loader)

    # Each body step:
    encoder.inject(vnc_runner, body_obs)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from bridge.interfaces import BodyObservation


# ============================================================================
# Constants
# ============================================================================

# FlyGym leg order and joint layout
LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]
JOINTS_PER_LEG = 7
N_JOINTS = 42  # 6 legs x 7 joints

# Leg offsets into the 42-element joint arrays
LEG_OFFSETS = {
    "LF": 0, "LM": 7, "LH": 14,
    "RF": 21, "RM": 28, "RH": 35,
}

# Key proprioceptive joint indices within a leg's 7 DOFs
# Coxa=0, Coxa_roll=1, Coxa_yaw=2, Femur=3, Femur_roll=4, Tibia=5, Tarsus1=6
FEMUR_DOF = 3
TIBIA_DOF = 5

# BANC cell_sub_class patterns for proprioceptive leg assignment
SUB_CLASS_TO_LEG_PREFIX = {
    "front_leg":  {"T1"},
    "middle_leg": {"T2"},
    "hind_leg":   {"T3"},
}

# Segment x side -> FlyGym leg name
SEGMENT_SIDE_TO_LEG = {
    ("T1", "left"):  "LF", ("T1", "right"): "RF",
    ("T2", "left"):  "LM", ("T2", "right"): "RM",
    ("T3", "left"):  "LH", ("T3", "right"): "RH",
    ("T1", "L"): "LF", ("T1", "R"): "RF",
    ("T2", "L"): "LM", ("T2", "R"): "RM",
    ("T3", "L"): "LH", ("T3", "R"): "RH",
}

# BANC nerve -> segment mapping (subset relevant to leg proprioceptors)
NERVE_TO_SEGMENT = {
    "ProLN": "T1", "MesoLN": "T2", "MetaLN": "T3",
    "left_prothoracic_leg_nerve": "T1",
    "right_prothoracic_leg_nerve": "T1",
    "left_mesothoracic_leg_nerve": "T2",
    "right_mesothoracic_leg_nerve": "T2",
    "left_metathoracic_leg_nerve": "T3",
    "right_metathoracic_leg_nerve": "T3",
}

# Proprioceptive cell_class keywords (case-insensitive matching)
CHORDOTONAL_KEYWORDS = ["chordotonal", "feco"]
CAMPANIFORM_KEYWORDS = ["campaniform"]
HAIR_PLATE_KEYWORDS = ["hair_plate", "hair plate", "hairplate"]


# ============================================================================
# Encoding parameters
# ============================================================================

@dataclass
class ProprioceptiveParams:
    """Tunable parameters for the proprioceptive encoding model."""

    # Baseline and maximum firing rates (Hz)
    baseline_hz: float = 5.0
    max_hz: float = 100.0

    # Campaniform sensilla: force -> rate
    # rate = baseline + gain * clip(force, 0, 1)
    campaniform_gain: float = 80.0   # Hz per unit force

    # Chordotonal / FeCO position channel:
    # Bell-shaped tuning around rest angle (0 rad).
    # rate = baseline + gain * exp(-angle^2 / (2 * sigma^2))
    chordotonal_pos_gain: float = 60.0   # Hz peak above baseline
    chordotonal_pos_sigma: float = 0.8   # radians; width of tuning curve

    # Chordotonal / FeCO velocity channel:
    # Rectified linear: rate = baseline + gain * clip(|velocity|, 0, max_vel) / max_vel
    chordotonal_vel_gain: float = 50.0   # Hz at max velocity
    chordotonal_vel_max: float = 10.0    # rad/s normalization

    # Hair plate: absolute velocity -> rate (movement detector)
    # rate = baseline + gain * tanh(|velocity| / scale)
    hair_plate_gain: float = 60.0   # Hz at saturation
    hair_plate_scale: float = 5.0   # rad/s half-max

    # Current injection scaling: sensory rate (Hz) -> I_stim units
    # For firing-rate model: I = rate * current_scale * synapse_weight
    # For Brian2 model: injected via sensory_rates dict (Hz directly)
    current_scale: float = 0.01     # Converts Hz to activation units

    # Weight normalization: divide total current per target by fan-in count
    # to prevent high-fan-in neurons from saturating
    normalize_by_fanin: bool = True


# ============================================================================
# SensoryPopulation: one group of proprioceptive neurons for a given type+leg
# ============================================================================

@dataclass
class SensoryPopulation:
    """A group of proprioceptive sensory neurons of one type in one leg."""
    sensory_type: str           # "chordotonal", "campaniform", "hair_plate"
    leg: str                    # "LF", "LM", "LH", "RF", "RM", "RH"
    body_ids: List[int] = field(default_factory=list)
    # Downstream VNC targets: target_idx -> total synapse weight from this group
    target_weights: Dict[int, float] = field(default_factory=dict)
    # Number of sensory neurons projecting to each target (for normalization)
    target_fanin: Dict[int, int] = field(default_factory=dict)


# ============================================================================
# ProprioceptiveEncoder
# ============================================================================

class ProprioceptiveEncoder:
    """Encode FlyGym body state into proprioceptive current for VNC neurons.

    Loads BANC proprioceptive sensory neurons (chordotonal, campaniform,
    hair plate), groups them by type and leg, looks up their downstream
    VNC premotor targets from the connectivity graph, and computes per-target
    current injections based on body state.

    Two injection modes:
        inject()  -- adds current to vnc_runner.I_stim (firing-rate model)
        encode()  -- returns (n_vnc_neurons,) array of additional current

    Args:
        banc_data: BANCVNCData with bodyid_to_idx mapping for the VNC model.
        vnc_runner: The VNC runner (FiringRateVNCRunner or Brian2VNCRunner).
            Used only for sizing (n_neurons). May be None if only encode()
            is needed.
        loader: BANCLoader instance for accessing neuron annotations and
            connectivity. If None, a default BANCLoader is created.
        params: Encoding parameters. Defaults to ProprioceptiveParams().
        verbose: Print diagnostic messages during initialization.
    """

    def __init__(
        self,
        banc_data,
        vnc_runner=None,
        loader=None,
        params: ProprioceptiveParams | None = None,
        verbose: bool = True,
    ):
        self.params = params or ProprioceptiveParams()
        self._verbose = verbose
        self._n_vnc = banc_data.n_neurons
        self._bodyid_to_idx = banc_data.bodyid_to_idx

        # Sensory populations grouped by (type, leg)
        self.populations: List[SensoryPopulation] = []

        # Precomputed injection matrix: sparse representation
        # _target_indices[i] and _target_currents_per_hz[i] are parallel arrays
        # giving the VNC model index and per-Hz current weight for each
        # (sensory_population, target) pair.  Populated after connectivity lookup.
        self._inject_target_idx: np.ndarray = np.array([], dtype=np.int32)
        self._inject_weight: np.ndarray = np.array([], dtype=np.float32)
        self._inject_pop_idx: np.ndarray = np.array([], dtype=np.int32)
        self._n_pops = 0

        # Build from BANC data
        self._build(banc_data, loader)

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _build(self, banc_data, loader):
        """Identify proprioceptive neurons, group by type+leg, map targets."""
        from bridge.banc_loader import BANCLoader

        if loader is None:
            loader = BANCLoader()

        if not loader.is_available():
            self._log("BANC database not available; proprioceptive encoder "
                      "will be a no-op")
            return

        neurons = loader.load_neurons()
        connectivity = loader.load_connectivity()

        # -- Step 1: Select proprioceptive neurons by cell_class -----------

        proprio_mask = self._make_proprioceptive_mask(neurons)
        proprio_df = neurons[proprio_mask].copy()

        if len(proprio_df) == 0:
            self._log("No proprioceptive neurons found in BANC; encoder is a no-op")
            return

        self._log(f"Found {len(proprio_df)} proprioceptive neurons in BANC")

        # -- Step 2: Classify by type and assign to legs -------------------

        pop_map: Dict[tuple, SensoryPopulation] = {}  # (type, leg) -> pop

        for _, row in proprio_df.iterrows():
            bid = int(row["body_id"])
            cell_class = str(row.get("cell_class", "")).lower()
            cell_sub = str(row.get("cell_sub_class", "")).lower()
            side = str(row.get("side", "")).lower()
            nerve = str(row.get("nerve", "")).strip()

            # Classify type
            stype = self._classify_type(cell_class)
            if stype is None:
                continue

            # Assign to leg
            leg = self._assign_leg(cell_sub, side, nerve)
            if leg is None:
                continue

            key = (stype, leg)
            if key not in pop_map:
                pop_map[key] = SensoryPopulation(sensory_type=stype, leg=leg)
            pop_map[key].body_ids.append(bid)

        self.populations = list(pop_map.values())
        self._n_pops = len(self.populations)

        if self._n_pops == 0:
            self._log("No proprioceptive populations could be assigned to legs")
            return

        # Summary
        type_counts = {}
        for pop in self.populations:
            type_counts[pop.sensory_type] = (
                type_counts.get(pop.sensory_type, 0) + len(pop.body_ids)
            )
        self._log(f"Proprioceptive populations: {self._n_pops} groups "
                  f"({type_counts})")

        # -- Step 3: Find downstream VNC targets ---------------------------

        all_sensory_ids = set()
        for pop in self.populations:
            all_sensory_ids.update(pop.body_ids)

        # Filter connectivity: sensory -> any VNC model neuron
        vnc_ids_set = set(self._bodyid_to_idx.keys())
        sensory_edges = connectivity[
            connectivity["pre_id"].isin(all_sensory_ids)
            & connectivity["post_id"].isin(vnc_ids_set)
        ]

        self._log(f"Sensory->VNC edges: {len(sensory_edges):,} "
                  f"(from {len(all_sensory_ids)} sensory neurons)")

        # Build per-population target maps
        # Create fast lookup: sensory_body_id -> list of population indices
        bid_to_pops: Dict[int, List[int]] = {}
        for pi, pop in enumerate(self.populations):
            for bid in pop.body_ids:
                bid_to_pops.setdefault(bid, []).append(pi)

        for _, edge in sensory_edges.iterrows():
            pre_id = int(edge["pre_id"])
            post_id = int(edge["post_id"])
            weight = int(edge.get("weight", 1))

            post_idx = self._bodyid_to_idx.get(post_id)
            if post_idx is None:
                continue

            for pi in bid_to_pops.get(pre_id, []):
                pop = self.populations[pi]
                pop.target_weights[post_idx] = (
                    pop.target_weights.get(post_idx, 0.0) + float(weight)
                )
                pop.target_fanin[post_idx] = (
                    pop.target_fanin.get(post_idx, 0) + 1
                )

        # Count targeted VNC neurons
        all_targets = set()
        total_syn = 0
        for pop in self.populations:
            all_targets.update(pop.target_weights.keys())
            total_syn += sum(pop.target_weights.values())
        self._log(f"Targets: {len(all_targets)} VNC neurons, "
                  f"{int(total_syn)} total synapse weight")

        # -- Step 4: Build flat injection arrays ---------------------------

        self._build_injection_arrays()

    def _make_proprioceptive_mask(self, neurons):
        """Create boolean mask for proprioceptive sensory neurons."""
        cc = neurons["cell_class"].str.lower()

        masks = []
        for kw_list in [CHORDOTONAL_KEYWORDS, CAMPANIFORM_KEYWORDS,
                        HAIR_PLATE_KEYWORDS]:
            for kw in kw_list:
                masks.append(cc.str.contains(kw, na=False))

        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m

        # Also require sensory flow (if available) or VNC region
        # to avoid picking up central neurons with similar names
        is_sensory = (
            (neurons["flow"].str.lower() == "afferent")
            | (neurons["super_class"].str.lower() == "sensory")
        )

        return combined & is_sensory

    def _classify_type(self, cell_class: str) -> Optional[str]:
        """Classify a neuron's proprioceptive subtype from cell_class."""
        cell_class = cell_class.lower()
        for kw in CHORDOTONAL_KEYWORDS:
            if kw in cell_class:
                return "chordotonal"
        for kw in CAMPANIFORM_KEYWORDS:
            if kw in cell_class:
                return "campaniform"
        for kw in HAIR_PLATE_KEYWORDS:
            if kw in cell_class:
                return "hair_plate"
        return None

    def _assign_leg(self, cell_sub_class: str, side: str,
                    nerve: str) -> Optional[str]:
        """Assign a sensory neuron to a FlyGym leg.

        Priority:
            1. cell_sub_class (e.g., 'front_leg_chordotonal')
            2. nerve -> segment + side
            3. Fallback: None (unassigned)
        """
        # Strategy 1: cell_sub_class contains leg prefix
        for prefix, segments in SUB_CLASS_TO_LEG_PREFIX.items():
            if prefix in cell_sub_class:
                seg = next(iter(segments))
                s = "left" if side in ("left", "l") else "right"
                leg = SEGMENT_SIDE_TO_LEG.get((seg, s))
                if leg is not None:
                    return leg

        # Strategy 2: nerve -> segment
        seg = NERVE_TO_SEGMENT.get(nerve)
        if seg is not None:
            s = "left" if side in ("left", "l") else "right"
            leg = SEGMENT_SIDE_TO_LEG.get((seg, s))
            if leg is not None:
                return leg

        return None

    def _build_injection_arrays(self):
        """Flatten population target maps into parallel numpy arrays.

        This precomputes the sparse injection structure so that encode()
        is a fast vectorized operation rather than a dict iteration.
        """
        target_idxs = []
        weights = []
        pop_idxs = []

        params = self.params

        for pi, pop in enumerate(self.populations):
            for tidx, syn_weight in pop.target_weights.items():
                fanin = pop.target_fanin.get(tidx, 1)

                # Effective weight: synapse count, optionally normalized
                if params.normalize_by_fanin and fanin > 0:
                    w = syn_weight / fanin
                else:
                    w = syn_weight

                target_idxs.append(tidx)
                weights.append(w * params.current_scale)
                pop_idxs.append(pi)

        self._inject_target_idx = np.array(target_idxs, dtype=np.int32)
        self._inject_weight = np.array(weights, dtype=np.float32)
        self._inject_pop_idx = np.array(pop_idxs, dtype=np.int32)

        self._log(f"Injection array: {len(target_idxs)} (pop, target) pairs")

    # ------------------------------------------------------------------ #
    # Encoding: body state -> sensory firing rates per population
    # ------------------------------------------------------------------ #

    def _encode_population_rates(self, body_obs: BodyObservation) -> np.ndarray:
        """Compute firing rate (Hz) for each sensory population.

        Returns:
            (n_pops,) array of firing rates in Hz.
        """
        rates = np.full(self._n_pops, self.params.baseline_hz, dtype=np.float32)
        p = self.params

        for pi, pop in enumerate(self.populations):
            leg = pop.leg
            leg_idx = LEG_ORDER.index(leg)
            offset = LEG_OFFSETS[leg]

            if pop.sensory_type == "campaniform":
                # Campaniform sensilla: load-dependent rate from contact forces
                force = float(np.clip(body_obs.contact_forces[leg_idx], 0.0, 1.0))
                rates[pi] = p.baseline_hz + p.campaniform_gain * force

            elif pop.sensory_type == "chordotonal":
                # Chordotonal / FeCO: position + velocity encoding
                # Position: bell-shaped tuning around femur-tibia rest angle
                femur_angle = body_obs.joint_angles[offset + FEMUR_DOF]
                tibia_angle = body_obs.joint_angles[offset + TIBIA_DOF]
                mean_angle = 0.5 * (femur_angle + tibia_angle)
                pos_rate = p.chordotonal_pos_gain * np.exp(
                    -mean_angle**2 / (2.0 * p.chordotonal_pos_sigma**2)
                )

                # Velocity: rectified linear encoding
                femur_vel = body_obs.joint_velocities[offset + FEMUR_DOF]
                tibia_vel = body_obs.joint_velocities[offset + TIBIA_DOF]
                abs_vel = 0.5 * (abs(femur_vel) + abs(tibia_vel))
                vel_rate = p.chordotonal_vel_gain * float(
                    np.clip(abs_vel / p.chordotonal_vel_max, 0.0, 1.0)
                )

                rates[pi] = p.baseline_hz + pos_rate + vel_rate

            elif pop.sensory_type == "hair_plate":
                # Hair plate: movement detector, responds to absolute velocity
                femur_vel = body_obs.joint_velocities[offset + FEMUR_DOF]
                tibia_vel = body_obs.joint_velocities[offset + TIBIA_DOF]
                abs_vel = 0.5 * (abs(femur_vel) + abs(tibia_vel))
                rate = p.hair_plate_gain * float(
                    np.tanh(abs_vel / p.hair_plate_scale)
                )
                rates[pi] = p.baseline_hz + rate

        # Clip to [0, max_hz]
        np.clip(rates, 0.0, p.max_hz, out=rates)
        return rates

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def encode(self, body_obs: BodyObservation) -> np.ndarray:
        """Compute proprioceptive current injection for all VNC neurons.

        Maps body state through the sensory encoding model and the BANC
        sensory->premotor connectivity to produce a per-neuron current
        array suitable for adding to vnc_runner.I_stim.

        Args:
            body_obs: Current body state from FlyGym.

        Returns:
            (n_vnc_neurons,) float32 array of additional current to inject.
            Zero for neurons that receive no proprioceptive input.
        """
        current = np.zeros(self._n_vnc, dtype=np.float32)

        if self._n_pops == 0 or len(self._inject_target_idx) == 0:
            return current

        # Compute per-population firing rates
        pop_rates = self._encode_population_rates(body_obs)

        # Scatter-add weighted rates to target neurons
        # current[target] += pop_rate[pop] * weight
        contribution = pop_rates[self._inject_pop_idx] * self._inject_weight
        np.add.at(current, self._inject_target_idx, contribution)

        return current

    def inject(self, vnc_runner, body_obs: BodyObservation):
        """Add proprioceptive current to vnc_runner.I_stim in-place.

        This is the primary interface for the closed-loop pipeline.
        Call once per VNC step, before the VNC model advances.

        For the firing-rate model (FiringRateVNCRunner):
            Adds directly to vnc_runner.I_stim (activation units).

        For Brian2 VNC (Brian2VNCRunner):
            If the runner has a sensory_group, encodes as sensory_rates
            dict and updates via the sensory PoissonGroup. Otherwise
            falls back to I_stim if available.

        Args:
            vnc_runner: VNC model instance with I_stim or sensory_group.
            body_obs: Current body state from FlyGym.
        """
        if self._n_pops == 0:
            return

        # Firing-rate model path: direct I_stim injection
        if hasattr(vnc_runner, "I_stim"):
            current = self.encode(body_obs)
            vnc_runner.I_stim += current
            return

        # Brian2 path: encode as sensory_rates dict (body_id -> Hz)
        # This is used if the VNC runner accepts sensory rates via VNCInput
        if hasattr(vnc_runner, "sensory_group") and vnc_runner.sensory_group is not None:
            rates_dict = self.encode_as_rates_dict(body_obs)
            # Update PoissonGroup rates directly if accessible
            if hasattr(vnc_runner, "_sensory_bodyid_to_input_idx"):
                n_sens = len(vnc_runner._sensory_brian_idx)
                rates_arr = np.full(n_sens, 10.0, dtype=np.float64)
                for bid, rate in rates_dict.items():
                    idx = vnc_runner._sensory_bodyid_to_input_idx.get(int(bid))
                    if idx is not None:
                        rates_arr[idx] = float(rate)
                vnc_runner.sensory_group.rates = rates_arr * vnc_runner._Hz

    def encode_as_rates_dict(self, body_obs: BodyObservation) -> Dict[int, float]:
        """Encode body state as a dict of sensory body_id -> firing rate (Hz).

        This format is compatible with VNCInput.sensory_rates and the
        Brian2VNCRunner sensory PoissonGroup interface.

        Args:
            body_obs: Current body state from FlyGym.

        Returns:
            Dict mapping BANC sensory neuron body_id -> rate in Hz.
        """
        if self._n_pops == 0:
            return {}

        pop_rates = self._encode_population_rates(body_obs)
        rates: Dict[int, float] = {}

        for pi, pop in enumerate(self.populations):
            rate = float(pop_rates[pi])
            for bid in pop.body_ids:
                rates[bid] = rate

        return rates

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a human-readable summary of the encoder state."""
        lines = [f"ProprioceptiveEncoder: {self._n_pops} populations"]

        if self._n_pops == 0:
            lines.append("  (no proprioceptive neurons loaded)")
            return "\n".join(lines)

        # Per-type counts
        type_neurons: Dict[str, int] = {}
        type_legs: Dict[str, set] = {}
        for pop in self.populations:
            type_neurons[pop.sensory_type] = (
                type_neurons.get(pop.sensory_type, 0) + len(pop.body_ids)
            )
            type_legs.setdefault(pop.sensory_type, set()).add(pop.leg)

        for stype in sorted(type_neurons):
            n = type_neurons[stype]
            legs = sorted(type_legs[stype])
            lines.append(f"  {stype}: {n} neurons across {len(legs)} legs")

        # Target stats
        all_targets = set()
        total_weight = 0.0
        for pop in self.populations:
            all_targets.update(pop.target_weights.keys())
            total_weight += sum(pop.target_weights.values())
        lines.append(f"  Targets: {len(all_targets)} VNC neurons, "
                     f"{int(total_weight)} total synapse weight")
        lines.append(f"  Injection pairs: {len(self._inject_target_idx)}")

        return "\n".join(lines)

    @property
    def n_sensory_neurons(self) -> int:
        """Total number of proprioceptive sensory neurons loaded."""
        return sum(len(pop.body_ids) for pop in self.populations)

    @property
    def n_target_neurons(self) -> int:
        """Number of unique VNC neurons receiving proprioceptive input."""
        targets = set()
        for pop in self.populations:
            targets.update(pop.target_weights.keys())
        return len(targets)

    @property
    def sensory_body_ids(self) -> Set[int]:
        """All proprioceptive sensory neuron body IDs."""
        ids: Set[int] = set()
        for pop in self.populations:
            ids.update(pop.body_ids)
        return ids

    def _log(self, msg: str):
        """Print diagnostic message if verbose."""
        if self._verbose:
            print(f"  ProprioceptiveEncoder: {msg}")
