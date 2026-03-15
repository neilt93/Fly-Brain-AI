"""
VNC connectome model: Brian2 LIF simulation of the Drosophila VNC using MANC data.

Two implementations:
    FakeVNCRunner     -- oscillatory MN output for testing without Brian2
    Brian2VNCRunner   -- real MANC-based Brian2 LIF network (~12K neurons)

Use create_vnc_runner() factory to pick the right one.

Data source: Male Adult CNS (MaleCNS) v0.9, Janelia (CC-BY 4.0)
    - body-annotations-male-cns-v0.9-minconf-0.5.feather
    - body-neurotransmitters-male-cns-v0.9.feather
    - connectome-weights-male-cns-v0.9-minconf-0.5.feather

Architecture:
    DN input (PoissonGroup) -> VNC interneurons -> Motor neurons (SpikeMonitor)
    ~261 matched DNs + ~500 leg MNs + ~11K intrinsic interneurons
"""

import json
import sys
import numpy as np
from pathlib import Path
from time import time
from dataclasses import dataclass, field


# ============================================================================
# Data paths
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
MANC_DIR = ROOT / "data" / "manc"
DATA_DIR = ROOT / "data"


# ============================================================================
# VNCInput / VNCOutput dataclasses
# ============================================================================

@dataclass
class VNCInput:
    """DN group firing rates from brain decoder."""
    group_rates: dict  # {"forward": Hz, "turn_left": Hz, "turn_right": Hz, "rhythm": Hz, "stance": Hz}


@dataclass
class VNCOutput:
    """Motor neuron firing rates from VNC."""
    mn_body_ids: np.ndarray     # MANC body IDs for MNs (int64)
    firing_rates_hz: np.ndarray  # Hz, one per MN (float32)


# ============================================================================
# VNCConfig
# ============================================================================

@dataclass
class VNCConfig:
    """Tunable parameters for the VNC connectome model."""
    # LIF parameters (matched to brain_runner.py)
    v_rest_mV: float = -52.0
    v_reset_mV: float = -52.0
    v_threshold_mV: float = -45.0
    tau_membrane_ms: float = 20.0
    tau_syn_ms: float = 5.0
    t_refrac_ms: float = 2.2
    t_delay_ms: float = 1.8

    # Spike-frequency adaptation (mild: prevents runaway without killing
    # sustained tonic output needed for post-hoc rhythm architecture)
    tau_adapt_ms: float = 100.0     # Adaptation time constant
    b_adapt_mV: float = 0.3        # Adaptation increment per spike (mild)

    # Tonic drive: pushes neurons closer to threshold for excitability
    I_tonic_mV: float = 3.0        # Constant depolarizing current (mV)

    # Background noise: Poisson input to all neurons (breaks symmetry)
    bg_rate_hz: float = 5.0        # Background Poisson rate per neuron
    bg_weight_mV: float = 1.5      # Background synapse weight (mV)

    # Inhibitory synapse scaling
    inh_scale: float = 1.5         # Multiplier for inhibitory (GABA) synapses

    # Post-hoc rhythm modulation: applied to MN output rates after Brian2 step.
    # The Brian2 VNC computes a tonic "gain profile" for each MN based on
    # actual MANC connectivity. Rhythm modulation then creates the temporal
    # pattern (tripod gait with flexor/extensor alternation).
    # This models the biological reality: rhythm emerges from interaction of
    # multiple circuit elements, not from the interneuron network alone.
    rhythm_freq_hz: float = 12.0   # Stepping frequency (Hz)
    rhythm_base_hz: float = 100.0  # Baseline oscillation amplitude (Hz)
    rhythm_depth: float = 0.92     # Modulation depth (0=tonic, 1=full on/off)

    # Synapse parameters
    w_syn_mV: float = 0.275         # Base synaptic weight (mV per synapse)
    w_input_scale: float = 250.0    # Weight multiplier for DN PoissonGroup input

    # DN baseline: non-readout DNs get this rate (biological: tonic background)
    dn_baseline_hz: float = 15.0

    # Warmup
    warmup_ms: float = 200.0
    warmup_rate_hz: float = 50.0

    # Paths
    manc_dir: Path = field(default_factory=lambda: MANC_DIR)

    @property
    def annotations_path(self) -> Path:
        return self.manc_dir / "body-annotations-male-cns-v0.9-minconf-0.5.feather"

    @property
    def neurotransmitters_path(self) -> Path:
        return self.manc_dir / "body-neurotransmitters-male-cns-v0.9.feather"

    @property
    def connectivity_path(self) -> Path:
        return self.manc_dir / "connectome-weights-male-cns-v0.9-minconf-0.5.feather"

    @property
    def mn_joint_mapping_path(self) -> Path:
        return DATA_DIR / "mn_joint_mapping.json"

    @property
    def decoder_groups_path(self) -> Path:
        return DATA_DIR / "decoder_groups_v5_steering.json"


# ============================================================================
# Motor neuron classification
# ============================================================================

# MANC MN type -> (joint_group, sign)
# Based on Baek & Mann 2009, Namiki et al. 2018, Azevedo et al. 2024
MN_TYPE_TO_JOINT_GROUP = {
    # Tibia
    "Ti flexor MN":          ("tibia_flexion", +1),
    "Ti extensor MN":        ("tibia_extension", -1),
    "Acc. ti flexor MN":     ("tibia_flexion", +1),
    # Trochanter-femur
    "Tr flexor MN":          ("femur_flexion", +1),
    "Tr extensor MN":        ("femur_extension", -1),
    "Acc. tr flexor MN":     ("femur_flexion", +1),
    "Fe reductor MN":        ("femur_rotation", +1),
    "Sternotrochanter MN":   ("femur_flexion", +1),
    # Coxa (thorax-coxa joint)
    "Tergopleural/Pleural promotor MN": ("coxa_protraction", +1),
    "Pleural remotor/abductor MN":      ("coxa_retraction", -1),
    "Sternal adductor MN":              ("coxa_rotation", +1),
    "Sternal anterior rotator MN":      ("coxa_protraction", +1),
    "Sternal posterior rotator MN":     ("coxa_retraction", -1),
    "Tergotr. MN":                      ("coxa_rotation", -1),
    # Tarsus
    "Ta depressor MN":       ("tarsus_depression", +1),
    "Ta levator MN":         ("tarsus_levation", -1),
    # Long tendon muscles
    "ltm MN":                ("tibia_flexion", +1),
    "ltm1-tibia MN":         ("tibia_flexion", +1),
    "ltm2-femur MN":         ("femur_flexion", +1),
}

# Joint group -> FlyGym DOF index within a leg's 7 joints
# Leg joints: [Coxa(0), Coxa_roll(1), Coxa_yaw(2), Femur(3), Femur_roll(4), Tibia(5), Tarsus1(6)]
JOINT_GROUP_TO_DOF = {
    "coxa_protraction":  (0, +1),
    "coxa_retraction":   (0, -1),
    "coxa_rotation":     (1, +1),
    "femur_flexion":     (3, +1),
    "femur_extension":   (3, -1),
    "femur_rotation":    (4, +1),
    "tibia_flexion":     (5, +1),
    "tibia_extension":   (5, -1),
    "tarsus_depression": (6, +1),
    "tarsus_levation":   (6, -1),
}

# MANC segment (T1/T2/T3) x side (L/R) -> FlyGym leg name
SEGMENT_SIDE_TO_LEG = {
    ("T1", "L"): "LF", ("T1", "R"): "RF",
    ("T2", "L"): "LM", ("T2", "R"): "RM",
    ("T3", "L"): "LH", ("T3", "R"): "RH",
}

LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]
JOINTS_PER_LEG = 7
N_JOINTS = 42  # 6 legs x 7 joints


# ============================================================================
# Neurotransmitter -> E/I sign
# ============================================================================

NT_SIGN = {
    "acetylcholine": +1.0,   # Excitatory
    "glutamate":     +1.0,   # Excitatory in vertebrates; mixed in Drosophila (GluCl inhibitory,
                              # but many VNC glutamatergic MNs are excitatory). Default positive.
    "gaba":          -1.0,   # Inhibitory
    "dopamine":      +1.0,   # Modulatory, treat as excitatory
    "serotonin":     +1.0,   # Modulatory, treat as excitatory
    "octopamine":    +1.0,   # Modulatory, treat as excitatory
    "histamine":     -1.0,   # Inhibitory (photoreceptors)
    "unclear":       +1.0,   # Default to excitatory
}


# ============================================================================
# DN type names (FlyWire readout types -> MANC body IDs)
# ============================================================================

OUR_READOUT_DN_TYPES = {
    "DNa02", "DNg56", "DNpe030", "DNp44", "DNge149", "oviDNx", "DNg29",
    "DNge037", "DNge133", "DNg05_a", "DNge113", "DNp102", "DNg90", "DNp23",
    "DNge136", "DNge119", "DNge138", "DNp25", "DNp05", "DNpe045", "DNg93",
    "DNge129", "DNge132", "DNp104", "DNge047", "DNg17", "DNae007", "DNg83",
    "DNg23", "DNc01", "DNg62", "DNge039", "DNp19", "DNg86", "DNge044",
    "DNp62", "DNbe003", "DNc02", "DNg35", "DNp42", "DNg33", "DNge100",
    "DNp02", "DNp10", "DNp30", "DNg59", "DNge046", "DNb08", "DNge038",
    "DNd02", "DNg102", "DNg85", "oviDNd", "DNge148", "DNge135", "DNge067",
    "DNge054", "DNge041", "DNp06", "DNp29", "DNde005", "DNp37",
    "DNge142", "DNg05_b", "DNg47", "DNg15", "DNg87",
    "DNb05", "DNge075", "DNp35", "DNbe007", "DNge077", "DNge099", "DNpe049",
    "DNg81", "DNd04", "DNp12", "DNge172", "DNp54", "DNpe006",
    "DNp32", "DNg68", "DNge012", "DNpe017",
    "DNg57", "DNp34", "DNpe023", "DNp08", "DNge023",
    "DNge065", "DNpe003", "DNg20", "DNpe056", "DNg39",
    "DNg103", "DNge121", "DNae009", "DNpe043",
    "DNg84", "DNde002", "DNpe053", "DNge101", "DNp64", "DNp14", "DNg100",
    "DNp69", "DNg32", "DNp55", "DNg34",
    "DNp59", "DNde001", "DNpe048", "DNg105", "DNge057",
    "DNb09", "DNd03", "DNp49", "DNge018",
    "DNde003", "DNde006", "DNp46",
    "DNg37", "DNpe020",
}


# ============================================================================
# FakeVNCRunner
# ============================================================================

class FakeVNCRunner:
    """Oscillatory MN output for testing the loop without Brian2.

    Produces tripod-like oscillatory MN firing patterns:
        - 12Hz sine wave per leg
        - Tripod phase offsets [0, pi, 0, pi, 0, pi] for 6 legs
        - "forward" group rate modulates amplitude
        - "turn_left"/"turn_right" reduce amplitude on turning side
        - Each MN oscillates between 0 and ~100Hz
    """

    def __init__(self, cfg: VNCConfig | None = None):
        self.cfg = cfg or VNCConfig()
        self._rng = np.random.RandomState(42)
        self._time_ms = 0.0
        self._freq_hz = 12.0

        # Tripod phases: LF, LM, LH, RF, RM, RH
        self._tripod_phases = np.array([0, np.pi, 0, np.pi, 0, np.pi])

        # MN metadata
        self.mn_body_ids = np.array([], dtype=np.int64)
        self.mn_info = []
        self.n_mn = 0
        self._mn_leg_idx = np.array([], dtype=int)  # leg index [0-5] per MN

        # Load MN metadata from MANC annotations (or mn_joint_mapping.json)
        self._load_mn_metadata()

    def _load_mn_metadata(self):
        """Load motor neuron metadata for realistic oscillatory output."""
        # Try mn_joint_mapping.json first (lighter weight)
        mapping_path = self.cfg.mn_joint_mapping_path
        if mapping_path.exists():
            with open(mapping_path) as f:
                mn_map = json.load(f)

            body_ids = []
            leg_indices = []
            directions = []
            info_list = []
            for bid_str, entry in mn_map.items():
                bid = int(bid_str)
                leg = entry.get("leg", "LF")
                if leg in LEG_ORDER:
                    leg_idx = LEG_ORDER.index(leg)
                else:
                    leg_idx = 0
                body_ids.append(bid)
                leg_indices.append(leg_idx)
                directions.append(float(entry.get("direction", 1.0)))
                info_list.append((
                    bid,
                    entry.get("neuromere", "T1"),
                    entry.get("side", "L"),
                    entry.get("mn_type", "unknown"),
                ))

            self.mn_body_ids = np.array(body_ids, dtype=np.int64)
            self._mn_leg_idx = np.array(leg_indices, dtype=int)
            self._mn_direction = np.array(directions, dtype=float)
            self.mn_info = info_list
            self.n_mn = len(body_ids)
            print(f"FakeVNCRunner: loaded {self.n_mn} MNs from mn_joint_mapping.json")
            return

        # Fallback: load from MANC annotations
        if self.cfg.annotations_path.exists():
            import pandas as pd
            import pyarrow.feather as feather

            ann = pd.DataFrame(feather.read_feather(str(self.cfg.annotations_path)))
            mns = ann[
                (ann["superclass"] == "vnc_motor")
                & (ann["somaNeuromere"].isin(["T1", "T2", "T3"]))
            ]

            body_ids = []
            leg_indices = []
            info_list = []
            for _, row in mns.iterrows():
                bid = int(row["bodyId"])
                seg = str(row["somaNeuromere"])
                side = str(row["somaSide"])
                mn_type = str(row["type"]) if pd.notna(row.get("type")) else "unknown"
                leg = SEGMENT_SIDE_TO_LEG.get((seg, side), "LF")
                leg_idx = LEG_ORDER.index(leg) if leg in LEG_ORDER else 0

                body_ids.append(bid)
                leg_indices.append(leg_idx)
                info_list.append((bid, seg, side, mn_type))

            self.mn_body_ids = np.array(body_ids, dtype=np.int64)
            self._mn_leg_idx = np.array(leg_indices, dtype=int)
            self._mn_direction = np.ones(len(body_ids), dtype=float)
            self.mn_info = info_list
            self.n_mn = len(body_ids)
            print(f"FakeVNCRunner: loaded {self.n_mn} MNs from MANC annotations")
            return

        print("FakeVNCRunner: no MN data found, using empty MN list")

    def step(self, vnc_input: VNCInput, sim_ms: float = 20.0) -> VNCOutput:
        """Produce tripod-phased oscillatory MN firing rates.

        Args:
            vnc_input: DN group firing rates from brain decoder.
            sim_ms: simulation window in milliseconds.

        Returns:
            VNCOutput with MN body IDs and oscillatory firing rates.
        """
        self._time_ms += sim_ms
        t_s = self._time_ms / 1000.0

        gr = vnc_input.group_rates
        fwd = gr.get("forward", 20.0)
        turn_l = gr.get("turn_left", 0.0)
        turn_r = gr.get("turn_right", 0.0)
        rhythm = gr.get("rhythm", 0.0)
        stance = gr.get("stance", 0.0)

        # Forward rate modulates overall amplitude (tanh-normalized)
        # Higher amplitude = stronger push-pull between extensors/flexors
        amplitude = 80.0 * np.clip(np.tanh(fwd / 30.0), 0.2, 1.0)

        # Rhythm modulates frequency: higher rhythm -> faster stepping
        freq = self._freq_hz * (1.0 + 0.5 * np.tanh(rhythm / 40.0))

        # Turn asymmetry: reduce amplitude on the side the fly turns toward
        # turn_left reduces LEFT legs, turn_right reduces RIGHT legs
        turn_asym_l = np.clip(1.0 - np.tanh(turn_l / 40.0) * 0.5, 0.3, 1.0)
        turn_asym_r = np.clip(1.0 - np.tanh(turn_r / 40.0) * 0.5, 0.3, 1.0)

        # Stance gain: higher stance slightly boosts baseline rate
        stance_boost = 10.0 * np.tanh(stance / 40.0)

        # Per-MN firing rates with direction-aware phase
        # Extensors (direction > 0) fire high during stance phase,
        # flexors (direction < 0) fire high during swing phase.
        # This produces the push-pull pattern needed for locomotion.
        rates = np.zeros(self.n_mn, dtype=np.float32)
        for i in range(self.n_mn):
            leg_idx = self._mn_leg_idx[i]
            phase = self._tripod_phases[leg_idx]
            direction = self._mn_direction[i] if hasattr(self, '_mn_direction') else 1.0

            # Sine oscillation at tripod phase
            osc = np.sin(2.0 * np.pi * freq * t_s + phase)

            # direction > 0 = protraction/flexion muscles: HIGH during swing (osc < 0)
            # direction < 0 = retraction/extension muscles: HIGH during stance (osc > 0)
            # So we NEGATE the sign: directed_osc = osc * (-sign(direction))
            directed_osc = -osc * np.sign(direction) if direction != 0 else osc

            # Side-dependent amplitude
            if leg_idx < 3:  # Left legs
                side_scale = turn_asym_l
            else:  # Right legs
                side_scale = turn_asym_r

            # MN rate: baseline + directed oscillation * amplitude
            rate = 50.0 + stance_boost + amplitude * side_scale * directed_osc

            # Add small noise for realism
            rate += self._rng.normal(0, 2.0)

            rates[i] = float(np.clip(rate, 0.0, 100.0))

        return VNCOutput(
            mn_body_ids=self.mn_body_ids.copy(),
            firing_rates_hz=rates,
        )

    @property
    def current_time_ms(self) -> float:
        return self._time_ms

    def get_dn_type_to_body_ids(self) -> dict:
        """Return empty mapping (no real DN neurons in fake mode)."""
        return {}


# ============================================================================
# Brian2VNCRunner
# ============================================================================

class Brian2VNCRunner:
    """Real MANC-based Brian2 LIF simulation of the Drosophila VNC.

    Loads the MaleCNS connectome, selects thoracic neurons (DNs + leg MNs +
    VNC intrinsic interneurons), and builds a Brian2 spiking network.

    Neuron selection:
        - ~1,314 DNs (all descending neurons in MaleCNS)
        - ~500 leg MNs (T1/T2/T3)
        - ~11,287 VNC intrinsic interneurons (T1/T2/T3)
        Total: ~13K neurons

    DN input is driven by a PoissonGroup. The 5 brain decoder groups
    (forward, turn_left, turn_right, rhythm, stance) are mapped to specific
    DN body IDs via the decoder_groups.json and MANC annotations.

    Usage:
        runner = Brian2VNCRunner()
        vnc_input = VNCInput(group_rates={"forward": 30, ...})
        vnc_output = runner.step(vnc_input, sim_ms=20.0)
    """

    def __init__(
        self,
        cfg: VNCConfig | None = None,
        warmup: bool = True,
        shuffle_seed: int | None = None,
    ):
        self.cfg = cfg or VNCConfig()
        self.shuffle_seed = shuffle_seed
        self._check_data_files()

        label = "SHUFFLED (seed=%d)" % shuffle_seed if shuffle_seed is not None else "real"
        print(f"Brian2VNCRunner: building MANC VNC network ({label})...")
        t0 = time()

        self._load_data()
        self._select_neurons()
        self._build_dn_group_mapping()
        self._build_mn_rhythm_mapping()
        self._build_network()

        build_time = time() - t0
        print(f"  Network ready in {build_time:.1f}s: "
              f"{self.n_neurons} neurons ({self.n_dn} DN, {self.n_mn} MN, "
              f"{self.n_intrinsic} intrinsic), {self.n_synapses:,} synapses")

        if warmup and self.cfg.warmup_ms > 0:
            self._warmup()

    # ---- Data loading -------------------------------------------------------

    def _check_data_files(self):
        """Verify MANC feather files exist."""
        missing = []
        for p in [self.cfg.annotations_path, self.cfg.neurotransmitters_path,
                   self.cfg.connectivity_path]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "MANC data files not found:\n"
                + "\n".join(f"  {m}" for m in missing)
                + "\nDownload from: https://male-cns.janelia.org/download/"
            )

    def _load_data(self):
        """Load annotations, neurotransmitters, and connectivity."""
        import pandas as pd
        import pyarrow.feather as feather

        t0 = time()
        self._ann = pd.DataFrame(feather.read_feather(str(self.cfg.annotations_path)))
        print(f"  Annotations: {len(self._ann):,} rows ({time()-t0:.1f}s)")

        t0 = time()
        nt_df = pd.DataFrame(feather.read_feather(str(self.cfg.neurotransmitters_path)))
        # Get per-body neurotransmitter prediction using consensus_nt
        # Keep first occurrence per body (consensus is usually identical across rows)
        nt_unique = nt_df.drop_duplicates(subset="body", keep="first")
        self._nt_map = dict(zip(
            nt_unique["body"].values,
            nt_unique["consensus_nt"].values,
        ))
        print(f"  Neurotransmitters: {len(self._nt_map):,} unique bodies ({time()-t0:.1f}s)")

        t0 = time()
        size_gb = self.cfg.connectivity_path.stat().st_size / 1e9
        print(f"  Loading connectivity ({size_gb:.1f} GB)...")
        self._conn = pd.DataFrame(feather.read_feather(str(self.cfg.connectivity_path)))
        print(f"  Connectivity: {len(self._conn):,} edges ({time()-t0:.1f}s)")

    # ---- Neuron selection ---------------------------------------------------

    def _select_neurons(self):
        """Select thoracic VNC neurons: DNs + leg MNs + intrinsic interneurons."""
        import pandas as pd

        ann = self._ann

        # --- Descending neurons (all, not just thoracic -- DNs project to VNC) ---
        dn_mask = ann["superclass"] == "descending_neuron"
        self._dn_df = ann[dn_mask].copy()
        dn_ids = set(self._dn_df["bodyId"].values)

        # --- Leg motor neurons (thoracic only: T1/T2/T3) ---
        mn_mask = (
            (ann["superclass"] == "vnc_motor")
            & (ann["somaNeuromere"].isin(["T1", "T2", "T3"]))
        )
        self._mn_df = ann[mn_mask].copy()
        mn_ids = set(self._mn_df["bodyId"].values)

        # --- VNC intrinsic interneurons (thoracic: T1/T2/T3) ---
        intrinsic_mask = (
            (ann["superclass"] == "vnc_intrinsic")
            & (ann["somaNeuromere"].isin(["T1", "T2", "T3"]))
        )
        self._intrinsic_df = ann[intrinsic_mask].copy()
        intrinsic_ids = set(self._intrinsic_df["bodyId"].values)

        # --- Combined neuron set ---
        all_ids = dn_ids | mn_ids | intrinsic_ids
        self.n_dn = len(dn_ids)
        self.n_mn = len(mn_ids)
        self.n_intrinsic = len(intrinsic_ids)
        self.n_neurons = len(all_ids)

        print(f"  Selected neurons: {self.n_dn} DN + {self.n_mn} MN + "
              f"{self.n_intrinsic} intrinsic = {self.n_neurons}")

        # Create body_id <-> brian2 index mapping
        all_ids_sorted = sorted(all_ids)
        self._bodyid_to_idx = {bid: i for i, bid in enumerate(all_ids_sorted)}
        self._idx_to_bodyid = {i: bid for bid, i in self._bodyid_to_idx.items()}

        # Store sets
        self._dn_ids = dn_ids
        self._mn_ids = mn_ids
        self._intrinsic_ids = intrinsic_ids

        # DN brian2 indices
        self._dn_brian_idx = np.array(
            [self._bodyid_to_idx[bid] for bid in sorted(dn_ids)], dtype=int
        )
        # MN brian2 indices and body IDs (sorted)
        mn_sorted = sorted(mn_ids)
        self._mn_brian_idx = np.array(
            [self._bodyid_to_idx[bid] for bid in mn_sorted], dtype=int
        )
        self._mn_body_ids = np.array(mn_sorted, dtype=np.int64)

        # Build MN metadata for joint mapping
        self._build_mn_metadata()

        # --- Filter connectivity to subnet ---
        print(f"  Filtering {len(self._conn):,} connections to {self.n_neurons} neurons...")
        t0 = time()
        all_ids_set = set(all_ids)

        # Use pandas isin for fast filtering on the 151M row table
        pre_in = self._conn["body_pre"].isin(all_ids_set)
        post_in = self._conn["body_post"].isin(all_ids_set)
        subnet_mask = pre_in & post_in
        self._subnet_conn = self._conn[subnet_mask].copy()
        self.n_synapses = len(self._subnet_conn)
        print(f"  Filtered to {self.n_synapses:,} intra-VNC edges ({time()-t0:.1f}s)")

    def _build_mn_metadata(self):
        """Build motor neuron metadata for joint angle mapping."""
        import pandas as pd

        self.mn_info = []
        self.mn_body_ids = self._mn_body_ids.copy()

        for _, row in self._mn_df.iterrows():
            bid = int(row["bodyId"])
            seg = str(row["somaNeuromere"])
            side = str(row["somaSide"]) if pd.notna(row.get("somaSide")) else "L"
            mn_type = str(row["type"]) if pd.notna(row.get("type")) else "unknown"

            jg_entry = MN_TYPE_TO_JOINT_GROUP.get(mn_type, None)
            if jg_entry is not None:
                joint_group, sign = jg_entry
            else:
                joint_group = "unmapped"
                sign = 0

            self.mn_info.append((bid, seg, side, mn_type, joint_group, sign))

    # ---- DN group -> MANC body ID mapping -----------------------------------

    def _build_dn_group_mapping(self):
        """Map the 5 brain decoder groups to MANC DN body IDs.

        Strategy:
            1. Load decoder_groups.json: group -> list of FlyWire neuron IDs
            2. Those FlyWire IDs correspond to DN types (from brain model)
            3. Find MANC body IDs with matching DN types (via 'type' or 'flywireType')
            4. Result: group name -> list of MANC DN body IDs
        """
        import pandas as pd

        # Step 1: Load decoder groups (FlyWire IDs per group)
        decoder_path = self.cfg.decoder_groups_path
        if not decoder_path.exists():
            # Try fallback paths
            for alt in ["decoder_groups_v5_steering.json",
                        "decoder_groups_v4_looming.json",
                        "decoder_groups_v3.json",
                        "decoder_groups_v2.json",
                        "decoder_groups.json"]:
                alt_path = DATA_DIR / alt
                if alt_path.exists():
                    decoder_path = alt_path
                    break

        if decoder_path.exists():
            with open(decoder_path) as f:
                raw = json.load(f)
            flywire_groups = {
                "forward": set(map(int, raw.get("forward_ids", []))),
                "turn_left": set(map(int, raw.get("turn_left_ids", []))),
                "turn_right": set(map(int, raw.get("turn_right_ids", []))),
                "rhythm": set(map(int, raw.get("rhythm_ids", []))),
                "stance": set(map(int, raw.get("stance_ids", []))),
            }
            total_fw = sum(len(v) for v in flywire_groups.values())
            print(f"  Decoder groups: {total_fw} FlyWire IDs from {decoder_path.name}")
        else:
            flywire_groups = {}
            print("  WARNING: No decoder_groups.json found -- DN group mapping disabled")

        # Step 2: Build DN type -> MANC body ID mapping from annotations
        dn_type_to_manc = {}
        for _, row in self._dn_df.iterrows():
            bid = int(row["bodyId"])
            for col in ["type", "flywireType"]:
                val = row.get(col, None)
                if val is not None and isinstance(val, str) and val.strip():
                    dn_type = val.strip()
                    if dn_type not in dn_type_to_manc:
                        dn_type_to_manc[dn_type] = []
                    if bid not in dn_type_to_manc[dn_type]:
                        dn_type_to_manc[dn_type].append(bid)

        self._dn_type_to_manc = dn_type_to_manc

        # Step 3: Collect all MANC DNs matching our readout types
        readout_manc_ids = set()
        for dn_type in OUR_READOUT_DN_TYPES:
            for bid in dn_type_to_manc.get(dn_type, []):
                readout_manc_ids.add(bid)

        self._readout_manc_ids = sorted(readout_manc_ids)

        # Map MANC DN body_id -> PoissonGroup input index
        # We use ALL descending neurons for the PoissonGroup
        dn_sorted = sorted(self._dn_ids)
        self._dn_bodyid_to_input_idx = {
            int(bid): i for i, bid in enumerate(dn_sorted)
        }

        # For group-specific targeting, partition readout DNs by type heuristics
        # Known walking-related DN types per group (from literature)
        self._group_dn_types = {
            "forward": {"DNa01", "DNa02", "DNg100", "DNb02", "DNp30", "DNp02",
                        "DNp10", "DNp06", "DNp05"},
            "turn_left": {"DNg11", "DNg29", "DNg33", "DNg35", "DNg47",
                          "DNpe006", "DNge039", "DNge044"},
            "turn_right": {"DNg11", "DNg29", "DNg33", "DNg35", "DNg47",
                           "DNpe006", "DNge039", "DNge044"},
            "rhythm": {"DNb01", "DNb08", "DNb09", "DNg100", "DNd02", "DNd03"},
            "stance": {"DNp44", "DNp35", "DNp42", "DNp23", "DNp25", "DNp19"},
        }

        # Build DN body_id -> somaSide map for lateralization
        dn_side_map = {}
        for _, row in self._dn_df.iterrows():
            bid = int(row["bodyId"])
            side = str(row.get("somaSide", "")) if pd.notna(row.get("somaSide")) else ""
            dn_side_map[bid] = side

        # Build group -> list of MANC DN body IDs
        # For turning groups, lateralize by somaSide (ipsilateral control):
        #   turn_left uses LEFT-side DNs only
        #   turn_right uses RIGHT-side DNs only
        self._group_to_manc_dn_ids = {}
        for group_name, types in self._group_dn_types.items():
            group_ids = []
            for dt in types:
                for bid in dn_type_to_manc.get(dt, []):
                    if bid not in self._dn_ids:
                        continue
                    # Lateralize turning groups
                    if group_name == "turn_left" and dn_side_map.get(bid) == "R":
                        continue
                    if group_name == "turn_right" and dn_side_map.get(bid) == "L":
                        continue
                    group_ids.append(bid)
            self._group_to_manc_dn_ids[group_name] = sorted(set(group_ids))

        # All readout DNs not assigned to a specific group get the mean rate
        assigned = set()
        for ids in self._group_to_manc_dn_ids.values():
            assigned.update(ids)
        self._unassigned_readout_dns = [
            bid for bid in self._readout_manc_ids if bid not in assigned
        ]

        n_assigned = len(assigned)
        n_unassigned = len(self._unassigned_readout_dns)
        print(f"  DN mapping: {len(self._readout_manc_ids)} matched readout DNs "
              f"({n_assigned} group-assigned, {n_unassigned} mean-rate)")

    # ---- Rhythm input mapping ------------------------------------------------

    def _build_mn_rhythm_mapping(self):
        """Map MN body IDs to leg index and direction for rhythm input.

        Creates 12 rhythm units (6 legs x 2 phases: ext/flex).
        Each MN is assigned to one rhythm unit based on its leg and direction.
        """
        mapping_path = self.cfg.mn_joint_mapping_path
        if not mapping_path.exists():
            self._rhythm_map = {}
            print("  WARNING: mn_joint_mapping.json not found, rhythm input disabled")
            return

        with open(mapping_path) as f:
            mn_map = json.load(f)

        # Build: MN body_id -> (leg_idx, direction)
        self._rhythm_map = {}
        for bid_str, entry in mn_map.items():
            bid = int(bid_str)
            if bid not in self._mn_ids:
                continue
            leg = entry.get("leg", "LF")
            if leg in LEG_ORDER:
                leg_idx = LEG_ORDER.index(leg)
            else:
                continue
            direction = float(entry.get("direction", 0.0))
            if direction == 0:
                continue
            # Rhythm unit index: 2*leg_idx + (0 for extensor, 1 for flexor)
            phase_idx = 0 if direction > 0 else 1
            rhythm_unit = 2 * leg_idx + phase_idx
            self._rhythm_map[bid] = rhythm_unit

        print(f"  Rhythm mapping: {len(self._rhythm_map)} MNs -> "
              f"12 rhythm units (6 legs x 2 phases)")

    # ---- E/I sign -----------------------------------------------------------

    def _get_nt_sign(self, body_id: int) -> float:
        """Get excitatory/inhibitory sign for a neuron from NT predictions."""
        nt = self._nt_map.get(int(body_id), "unclear")
        if isinstance(nt, float) or nt is None:
            return 1.0
        return NT_SIGN.get(str(nt).lower().strip(), 1.0)

    # ---- Brian2 network build -----------------------------------------------

    def _build_network(self):
        """Build Brian2 LIF network with spike-frequency adaptation.

        Adaptation is critical for CPG-like rhythm generation:
        - Active neurons accumulate adaptation current (w_a), reducing excitability
        - This causes fatigue → reciprocal inhibition lets opposing pool take over
        - Result: alternating flexor/extensor activity at ~10-15Hz (walking rhythm)

        Without adaptation, the network reaches a static E/I balance.
        """
        from brian2 import (
            NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
            mV, ms, Hz, second,
        )

        cfg = self.cfg

        # --- LIF + adaptation parameters ---
        params = {
            "v_0": cfg.v_rest_mV * mV,
            "v_rst": cfg.v_reset_mV * mV,
            "v_th": cfg.v_threshold_mV * mV,
            "t_mbr": cfg.tau_membrane_ms * ms,
            "tau": cfg.tau_syn_ms * ms,
            "t_rfc": cfg.t_refrac_ms * ms,
            "t_dly": cfg.t_delay_ms * ms,
            "w_syn": cfg.w_syn_mV * mV,
            "tau_a": cfg.tau_adapt_ms * ms,
            "b_a": cfg.b_adapt_mV * mV,
            "I_ton": cfg.I_tonic_mV * mV,
        }

        # Adaptive LIF: adaptation current (w_a) subtracts from drive,
        # tonic current (I_ton) adds constant depolarization
        eqs = """
            dv/dt = (v_0 - v + g - w_a + I_ton) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau               : volt (unless refractory)
            dw_a/dt = -w_a / tau_a         : volt
            rfc                            : second
        """

        print(f"  Building Brian2 NeuronGroup ({self.n_neurons} neurons, "
              f"adaptation tau={cfg.tau_adapt_ms}ms, b={cfg.b_adapt_mV}mV, "
              f"tonic={cfg.I_tonic_mV}mV)...")
        self.neurons = NeuronGroup(
            N=self.n_neurons, model=eqs, method="euler",
            threshold="v > v_th",
            reset="v = v_rst; g = 0 * mV; w_a += b_a",
            refractory="rfc", name="vnc_neurons", namespace=params,
        )
        self.neurons.v = params["v_0"]
        self.neurons.g = 0
        self.neurons.w_a = 0 * mV
        self.neurons.rfc = params["t_rfc"]

        # --- Internal synapses (from MANC connectivity) ---
        print(f"  Building Synapses ({self.n_synapses:,} edges)...")
        t0 = time()

        pre_idx = np.array([
            self._bodyid_to_idx[bid]
            for bid in self._subnet_conn["body_pre"].values
        ], dtype=int)
        post_idx = np.array([
            self._bodyid_to_idx[bid]
            for bid in self._subnet_conn["body_post"].values
        ], dtype=int)
        weights_raw = self._subnet_conn["weight"].values.astype(float)

        # Shuffle postsynaptic targets if requested (destroys specific wiring)
        if self.shuffle_seed is not None:
            rng = np.random.RandomState(self.shuffle_seed)
            post_idx = rng.permutation(post_idx)

        # Get E/I sign per presynaptic neuron
        nt_signs = np.array([
            self._get_nt_sign(bid)
            for bid in self._subnet_conn["body_pre"].values
        ], dtype=float)

        self.synapses = Synapses(
            self.neurons, self.neurons, "w : volt",
            on_pre="g += w", delay=params["t_dly"], name="vnc_synapses",
        )
        self.synapses.connect(i=pre_idx, j=post_idx)
        # Apply inhibitory scaling: GABA synapses (sign=-1) get inh_scale multiplier
        # This strengthens reciprocal inhibition to break global synchrony
        # into anti-phase flexor/extensor oscillation.
        inh_mask = (nt_signs < 0).astype(float)
        scale = np.ones_like(nt_signs) + inh_mask * (cfg.inh_scale - 1.0)
        self.synapses.w = weights_raw * nt_signs * scale * params["w_syn"]
        n_inh = int(inh_mask.sum())
        n_exc = len(nt_signs) - n_inh
        print(f"  Synapses built in {time()-t0:.1f}s "
              f"({n_exc:,} exc, {n_inh:,} inh x{cfg.inh_scale:.1f})")

        # --- DN input via PoissonGroup ---
        n_dn = len(self._dn_brian_idx)
        self.input_group = PoissonGroup(
            n_dn, rates=np.zeros(n_dn) * Hz, name="dn_input"
        )
        self.input_syn = Synapses(
            self.input_group, self.neurons, "w : volt",
            on_pre="g += w", name="dn_input_syn",
        )
        self.input_syn.connect(i=np.arange(n_dn), j=self._dn_brian_idx)
        self.input_syn.w = params["w_syn"] * cfg.w_input_scale

        # Reduce refractory for DN neurons (they relay input from brain)
        for idx in self._dn_brian_idx:
            self.neurons[int(idx)].rfc = 0 * ms

        # --- Background noise PoissonGroup ---
        # Low-rate Poisson input to ALL neurons breaks symmetry and
        # helps neurons near threshold fire spontaneously, seeding oscillation.
        if cfg.bg_rate_hz > 0:
            self.bg_input = PoissonGroup(
                self.n_neurons,
                rates=np.ones(self.n_neurons) * cfg.bg_rate_hz * Hz,
                name="bg_noise",
            )
            self.bg_syn = Synapses(
                self.bg_input, self.neurons, "w : volt",
                on_pre="g += w", name="bg_syn",
            )
            self.bg_syn.connect(j='i')  # one-to-one
            self.bg_syn.w = cfg.bg_weight_mV * mV
            print(f"  Background noise: {cfg.bg_rate_hz}Hz, {cfg.bg_weight_mV}mV")
        else:
            self.bg_input = None
            self.bg_syn = None

        # Tripod phase offsets for post-hoc rhythm modulation
        self._tripod_phases = np.array([0, np.pi, 0, np.pi, 0, np.pi])

        # --- Spike monitor ---
        self.spike_mon = SpikeMonitor(self.neurons)

        # --- Assemble network ---
        net_objects = [
            self.neurons, self.synapses, self.spike_mon,
            self.input_group, self.input_syn,
        ]
        if self.bg_input is not None:
            net_objects.extend([self.bg_input, self.bg_syn])
        self.net = Network(*net_objects)

        # Store units for step()
        self._ms = ms
        self._Hz = Hz
        self._second = second

        # Free large DataFrames to save memory
        del self._conn
        del self._subnet_conn
        del self._ann

    # ---- Warmup -------------------------------------------------------------

    def _warmup(self):
        """Run network for warmup period at baseline DN rates."""
        print(f"  VNC warmup ({self.cfg.warmup_ms:.0f}ms at "
              f"{self.cfg.warmup_rate_hz:.0f}Hz)...")
        t0 = time()
        n_dn = len(self._dn_brian_idx)
        self.input_group.rates = (
            np.ones(n_dn) * self.cfg.warmup_rate_hz * self._Hz
        )
        self.net.run(self.cfg.warmup_ms * self._ms)
        print(f"  Warmup done in {time()-t0:.1f}s")

    # ---- Step ---------------------------------------------------------------

    def step(self, vnc_input: VNCInput, sim_ms: float = 20.0) -> VNCOutput:
        """Run one VNC simulation step.

        Maps the 5 decoder group rates to DN PoissonGroup input rates,
        runs the Brian2 network, and reads out MN firing rates.

        Args:
            vnc_input: DN group firing rates from brain decoder.
            sim_ms: simulation window in milliseconds.

        Returns:
            VNCOutput with MN body IDs and firing rates.
        """
        gr = vnc_input.group_rates

        # --- Set DN input rates ---
        n_dn = len(self._dn_brian_idx)
        # Start with baseline rate for all DNs (biological: tonic descending activity)
        new_rates = np.full(n_dn, self.cfg.dn_baseline_hz, dtype=np.float64)

        # Assign group-specific rates to known DN types
        for group_name, manc_ids in self._group_to_manc_dn_ids.items():
            rate = float(gr.get(group_name, 0.0))
            for bid in manc_ids:
                if bid in self._dn_bodyid_to_input_idx:
                    new_rates[self._dn_bodyid_to_input_idx[bid]] = rate

        # Assign mean rate to unassigned readout DNs
        mean_rate = float(np.mean([
            gr.get("forward", 0.0),
            gr.get("turn_left", 0.0),
            gr.get("turn_right", 0.0),
            gr.get("rhythm", 0.0),
            gr.get("stance", 0.0),
        ]))
        for bid in self._unassigned_readout_dns:
            if bid in self._dn_bodyid_to_input_idx:
                new_rates[self._dn_bodyid_to_input_idx[bid]] = mean_rate

        self.input_group.rates = new_rates * self._Hz

        # --- Run simulation ---
        n_before = self.spike_mon.num_spikes
        if n_before > 500000:
            print(f"  WARNING: VNC SpikeMonitor has {n_before} spikes -- memory growing")
        self.net.run(sim_ms * self._ms)

        # --- Read MN firing rates from new spikes ---
        tonic_rates = np.zeros(len(self._mn_brian_idx), dtype=np.float32)
        n_after = self.spike_mon.num_spikes
        window_s = sim_ms / 1000.0

        if n_after > n_before:
            new_i = np.array(self.spike_mon.i[n_before:])
            for j, brian_idx in enumerate(self._mn_brian_idx):
                count = int(np.sum(new_i == brian_idx))
                tonic_rates[j] = count / window_s if window_s > 0 else 0.0

        # Return tonic MN rates from Brian2 spikes.
        # Rhythm modulation is applied at body-step frequency by VNCBridge.
        return VNCOutput(
            mn_body_ids=self._mn_body_ids.copy(),
            firing_rates_hz=tonic_rates,
        )

    @property
    def current_time_ms(self) -> float:
        return float(self.net.t / self._ms)

    def get_dn_type_to_body_ids(self) -> dict:
        """Return mapping from DN type name -> list of MANC body IDs."""
        return dict(self._dn_type_to_manc)


# ============================================================================
# Factory
# ============================================================================

def create_vnc_runner(
    use_fake: bool = False,
    cfg: VNCConfig | None = None,
    shuffle_seed: int | None = None,
    minimal: bool = False,
) -> "FakeVNCRunner | Brian2VNCRunner":
    """Factory: returns FakeVNCRunner, Brian2VNCRunner, or MinimalVNCRunner."""
    if use_fake:
        return FakeVNCRunner(cfg=cfg)
    if minimal:
        from bridge.vnc_minimal import MinimalVNCRunner, MinimalVNCConfig
        if cfg is None:
            cfg = MinimalVNCConfig()
        elif not isinstance(cfg, MinimalVNCConfig):
            cfg = MinimalVNCConfig(**{
                k: v for k, v in cfg.__dict__.items()
                if k in MinimalVNCConfig.__dataclass_fields__
            })
        return MinimalVNCRunner(cfg=cfg, shuffle_seed=shuffle_seed)
    return Brian2VNCRunner(cfg=cfg, shuffle_seed=shuffle_seed)


# ============================================================================
# Backward-compatible aliases (used by vnc_adapter.py)
# ============================================================================

FakeVNC = FakeVNCRunner
VNCConnectome = Brian2VNCRunner
create_vnc = create_vnc_runner
