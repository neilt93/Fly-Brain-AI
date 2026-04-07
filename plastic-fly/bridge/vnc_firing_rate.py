"""
Pugliese-style firing rate VNC model from MANC connectome data.

Implements the rate-model ODE from Pugliese et al. 2025 on the actual MANC
VNC premotor subnetwork (~7,500 neurons: 1,314 DNs + 381 MNs + 5,844 premotor
interneurons), rather than a 3-neuron hand-picked circuit or an LIF network.

ODE per neuron i (with adaptation):
    tau_i * dR_i/dt = -R_i + f(W_exc @ R(t) + W_inh @ R(t-d) + I_stim_i - A_i)
    tau_adapt * dA_i/dt = -A_i + b_adapt * R_i
    f(x) = max(fr_cap_i * tanh((a_i / fr_cap_i) * (x - theta_i)), 0)

Weight matrix construction:
    W_ij = exc_mult * count_ij   if presynaptic neuron j is excitatory (ACh, Glu)
    W_ij = -inh_mult * inh_scale * count_ij  if inhibitory (GABA, histamine)
    + row-sum normalization to place operating point near activation threshold

Key features:
    - Spike-frequency adaptation for rhythmic oscillation (~6-9 Hz)
    - Delayed inhibition for phase dynamics
    - Per-neuron parameter heterogeneity (10% CV)
    - Sparse matrix algebra for fast simulation (~1600 steps/s)

Data source: Male Adult CNS (MaleCNS) v0.9, Janelia (CC-BY 4.0)

Usage:
    from bridge.vnc_firing_rate import FiringRateVNCRunner
    runner = FiringRateVNCRunner()
    runner.stimulate_all_dns(rate_hz=25.0)
    runner.stimulate_dn_type("DNg100", rate_hz=60.0)
    for t in range(4000):
        runner.step(dt_ms=0.5)
    rates = runner.get_mn_rates()
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

# ============================================================================
# Paths (same convention as vnc_connectome.py)
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
MANC_DIR = ROOT / "data" / "manc"
DATA_DIR = ROOT / "data"

# ============================================================================
# Neurotransmitter -> E/I sign
# ============================================================================

NT_SIGN = {
    "acetylcholine": +1.0,
    "glutamate":     +1.0,   # Mixed in Drosophila, but default excitatory
    "gaba":          -1.0,
    "dopamine":      +1.0,
    "serotonin":     +1.0,
    "octopamine":    +1.0,
    "histamine":     -1.0,
    "unclear":       +1.0,   # Default excitatory
}

# ============================================================================
# MANC segment x side -> FlyGym leg
# ============================================================================

SEGMENT_SIDE_TO_LEG = {
    ("T1", "L"): "LF", ("T1", "R"): "RF",
    ("T2", "L"): "LM", ("T2", "R"): "RM",
    ("T3", "L"): "LH", ("T3", "R"): "RH",
}

LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FiringRateVNCConfig:
    """Tunable parameters for the Pugliese-style firing rate VNC model."""

    # --- Rate model ODE parameters (Pugliese et al. 2025) ---
    tau_ms: float = 20.0          # Membrane time constant (ms)
    a: float = 10.0               # Activation gain. With fr_cap=200, effective gain near
                                  #   threshold is a/fr_cap = 0.05, so 2 units above theta
                                  #   gives ~20 Hz output. (Pugliese uses a=1 with R_max=50,
                                  #   giving same effective gain 1/50=0.02; we need higher a
                                  #   because fr_cap is higher.)
    theta: float = 3.0            # Activation threshold (low: network easily excitable;
                                  #   matched to PuglieseCPG default; oscillation from E/I balance)
    fr_cap: float = 200.0         # Maximum firing rate (Hz)

    # --- Weight scaling ---
    # exc_mult and inh_mult convert synapse counts to effective weights.
    # inh_scale boosts inhibitory weights relative to excitatory, promoting
    # oscillation by strengthening the push-pull between E and I populations.
    exc_mult: float = 0.005       # Excitatory weight multiplier
    inh_mult: float = 0.005       # Inhibitory weight multiplier
    inh_scale: float = 2.0        # Additional scaling for inhibitory weights

    # --- Half-center oscillator boost ---
    # Selectively boosts inhibitory connections that form half-center circuits:
    #   (a) Inhibitory premotor INs that synapse onto BOTH flexor and extensor MNs
    #       within the same leg (their projections to MNs get boosted)
    #   (b) Direct inhibitory connections from flexor MN pool to extensor MN pool
    #       and vice versa (rare in Drosophila, but checked)
    # The boost is MULTIPLICATIVE on top of inh_scale (so total = inh_mult * inh_scale * half_center_boost).
    # Set to 1.0 for no extra boost. Values of 5-20 are biologically plausible
    # (half-center oscillators require strong reciprocal inhibition).
    half_center_boost: float = 1.0

    # --- Segment-specific scaling ---
    # If set, overrides inh_scale / exc_mult per thoracic segment for
    # intra-segment connections (where BOTH pre and post are in the same segment).
    # Cross-segment connections keep the global multiplier.
    # Format: {"T1": val, "T2": val, "T3": val}. None = use uniform global value.
    segment_inh_scales: Dict[str, float] | None = None
    segment_exc_mults: Dict[str, float] | None = None

    # --- Spike-frequency adaptation ---
    # Negative feedback: high firing builds adaptation, suppressing rate.
    # ODE: tau_adapt * dA/dt = -A + b_adapt * R
    # Total input becomes: W @ R + I_stim - A
    # Produces network-wide rhythmic oscillation (~6 Hz with tau_adapt=30ms).
    use_adaptation: bool = True
    tau_adapt_ms: float = 30.0    # Adaptation time constant (ms); shorter = faster oscillation
    b_adapt: float = 0.25         # Adaptation coupling strength

    # --- Synaptic delay ---
    # Inhibitory synapses are delayed by delay_inh_ms relative to excitatory.
    # This temporal asymmetry helps break synchrony between competing
    # populations (e.g., flexor vs extensor premotor circuits).
    use_delay: bool = True
    delay_inh_ms: float = 3.0     # Inhibitory synaptic delay (ms)

    # --- Weight normalization ---
    # If True, rescale all weights after construction so the mean excitatory
    # row sum equals target_exc_sum. The inh_scale is applied before this
    # normalization, so it controls the E/I ratio, not absolute magnitude.
    normalize_weights: bool = True
    target_exc_sum: float = 0.6   # Target mean exc input per neuron at 1 Hz all-active

    # --- Parameter heterogeneity ---
    param_cv: float = 0.10        # Coefficient of variation for per-neuron params
    seed: int = 42                # RNG seed for reproducibility

    # --- Neuron selection ---
    segments: Tuple[str, ...] = ("T1", "T2", "T3")   # Which thoracic segments to include
    min_mn_synapses: int = 3      # Minimum synapse count for premotor identification
    include_all_dns: bool = True  # Include all DNs (not just those synapsing onto MNs)

    # --- Ongoing noise ---
    # Gaussian noise added to each neuron's input at every timestep.
    # Prevents fixed-point convergence: without noise, the network settles
    # to static rates after ~2s, killing rhythmic oscillation.
    # Biologically: ongoing synaptic noise from unmodeled inputs.
    noise_sigma: float = 0.5     # Std of Gaussian noise on total_input (0 = off)

    # --- Integration ---
    dt_ms: float = 0.5            # Default integration timestep (ms)

    # --- Data paths ---
    manc_dir: Path = field(default_factory=lambda: MANC_DIR)

    @staticmethod
    def pugliese_exact() -> "FiringRateVNCConfig":
        """Pugliese et al. 2025 exact parameters: raw synapse counts * 0.03,
        NO normalization, NO adaptation, a=1, theta=7.5, fr_cap=200."""
        return FiringRateVNCConfig(
            tau_ms=20.0,
            a=1.0,
            theta=7.5,
            fr_cap=200.0,
            exc_mult=0.03,
            inh_mult=0.03,
            inh_scale=1.0,        # No extra inhibitory boost
            use_adaptation=False,  # Pugliese base model has no adaptation
            use_delay=True,
            delay_inh_ms=3.0,
            normalize_weights=False,  # KEY: no row-sum normalization
            param_cv=0.10,
            seed=42,
        )

    @staticmethod
    def pugliese_no_norm() -> "FiringRateVNCConfig":
        """Our parameters but WITHOUT row-sum normalization."""
        return FiringRateVNCConfig(
            tau_ms=20.0,
            a=10.0,
            theta=3.0,
            fr_cap=200.0,
            exc_mult=0.005,
            inh_mult=0.005,
            inh_scale=2.0,
            use_adaptation=True,
            use_delay=True,
            delay_inh_ms=3.0,
            normalize_weights=False,  # KEY: no normalization
            param_cv=0.10,
            seed=42,
        )

    @staticmethod
    def pugliese_strong_inh() -> "FiringRateVNCConfig":
        """Pugliese exact but with stronger inhibition (inh_scale=3.0)."""
        return FiringRateVNCConfig(
            tau_ms=20.0,
            a=1.0,
            theta=7.5,
            fr_cap=200.0,
            exc_mult=0.03,
            inh_mult=0.03,
            inh_scale=3.0,        # Extra inhibitory boost
            use_adaptation=False,
            use_delay=True,
            delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10,
            seed=42,
        )

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


# ============================================================================
# FiringRateVNCRunner
# ============================================================================


class FiringRateVNCRunner:
    """Pugliese-style firing rate VNC model built from MANC connectome data.

    Extracts the VNC premotor subnetwork: all DNs + leg motor neurons + premotor
    interneurons (VNC intrinsic neurons synapsing onto MNs above a weight
    threshold). Builds a sparse weight matrix from synapse counts with E/I signs
    from neurotransmitter annotations, then integrates the rate-model ODE with
    Euler steps including adaptation and delayed inhibition.

    Neuron selection (default: all thoracic segments T1/T2/T3):
        - ~1,314 DNs (all descending neurons)
        - ~381 leg MNs (fl/ml/hl subclasses)
        - ~5,844 premotor interneurons (>= 3 synapses to any MN)
        Total: ~7,539 neurons, ~1M connections

    Results at default parameters (DNg100 @ 60 Hz, DN baseline @ 25 Hz):
        - All 6 legs show rhythmic MN activity at ~6 Hz
        - Spectral power concentrated in locomotor band (5-20 Hz)
        - Flexors and extensors oscillate in-phase (anti-phase requires
          circuit-specific tuning beyond the raw connectome weights)

    Interface mirrors Brian2VNCRunner: step(), get_mn_rates(), silence_neurons().
    """

    def __init__(
        self,
        cfg: FiringRateVNCConfig | None = None,
        warmup_ms: float = 0.0,
    ):
        self.cfg = cfg or FiringRateVNCConfig()
        self._rng = np.random.RandomState(self.cfg.seed)
        self._time_ms = 0.0

        self._check_data_files()

        print("FiringRateVNCRunner: building MANC firing rate network...")
        t0 = time()

        self._load_data()
        self._select_neurons()
        self._build_weight_matrix()
        self._init_neuron_params()
        self._build_mn_metadata()
        self._apply_half_center_boost()
        self._build_dn_type_map()
        self._init_state()

        build_time = time() - t0
        print(f"  Network ready in {build_time:.1f}s: "
              f"{self.n_neurons} neurons ({self.n_dn} DN, {self.n_mn} MN, "
              f"{self.n_premotor} premotor interneurons), "
              f"{self.n_synapses:,} connections")

        if warmup_ms > 0:
            self._warmup(warmup_ms)

    @classmethod
    def from_banc(
        cls,
        banc_data,  # BANCVNCData from banc_loader.load_banc_vnc()
        cfg: "FiringRateVNCConfig | None" = None,
        warmup_ms: float = 0.0,
    ) -> "FiringRateVNCRunner":
        """Build a FiringRateVNCRunner from pre-loaded BANC data.

        Accepts a BANCVNCData object (from bridge.banc_loader.load_banc_vnc())
        and wires it into the runner's internal state, bypassing the MANC
        feather file loading.

        IMPORTANT: load_banc_vnc() should be called with exc_mult=1.0,
        inh_mult=1.0, inh_scale=1.0, normalize_weights=False to produce
        raw synapse-count matrices. This method applies the cfg's weight
        scaling (exc_mult, inh_mult, inh_scale, normalize_weights) to those
        raw matrices, matching the MANC _build_weight_matrix() path.

        Args:
            banc_data: BANCVNCData with W_exc, W_inh (raw synapse counts
                       recommended), population indices, and MN metadata.
            cfg: FiringRateVNCConfig with ODE and weight scaling params.
            warmup_ms: Optional warmup period in ms.

        Returns:
            A fully initialized FiringRateVNCRunner ready for step()/stimulate().
        """
        from scipy.sparse import csr_matrix

        self = cls.__new__(cls)
        self.cfg = cfg or FiringRateVNCConfig()
        self._rng = np.random.RandomState(self.cfg.seed)
        self._time_ms = 0.0

        print("FiringRateVNCRunner.from_banc: wiring BANC data...")
        t0 = time()

        # --- Population sizes ---
        self.n_neurons = banc_data.n_neurons
        self.n_dn = banc_data.n_dn
        self.n_mn = banc_data.n_mn
        self.n_premotor = banc_data.n_premotor
        self.n_synapses = banc_data.n_synapses

        # --- Index mappings ---
        self._bodyid_to_idx = banc_data.bodyid_to_idx
        self._idx_to_bodyid = banc_data.idx_to_bodyid
        self._dn_ids = banc_data.dn_ids
        self._mn_ids = banc_data.mn_ids
        self._premotor_ids = banc_data.premotor_ids
        self._dn_indices = banc_data.dn_indices
        self._mn_indices = banc_data.mn_indices
        self._premotor_indices = banc_data.premotor_indices

        # --- DN type map ---
        self._dn_type_to_indices = banc_data.dn_type_to_indices
        self._dn_type_to_body_ids = banc_data.dn_type_to_body_ids

        # --- MN metadata ---
        self.mn_body_ids = banc_data.mn_body_ids
        self.mn_info = banc_data.mn_info
        self._mn_leg = banc_data.mn_leg
        self._mn_direction = banc_data.mn_direction
        self._mn_is_flexor = banc_data.mn_direction < 0
        self._mn_is_extensor = banc_data.mn_direction > 0

        # --- Assign per-neuron segment (T1/T2/T3) for segment-specific scaling ---
        # MNs: from mn_info. DNs: no segment (0). Premotor INs: infer from
        # which MN segment they project to most strongly.
        seg_code_map = {"T1": 1, "T2": 2, "T3": 3}
        neuron_seg = np.zeros(self.n_neurons, dtype=np.int8)

        # MNs: segment from mn_info
        mn_seg_lookup = {}
        for i, info in enumerate(banc_data.mn_info):
            seg_str = info.get("segment", "")
            code = seg_code_map.get(seg_str, 0)
            model_idx = self._mn_indices[i]
            neuron_seg[model_idx] = code
            mn_seg_lookup[model_idx] = code

        # Premotor INs: infer segment from MN target weights
        W_raw_full = (banc_data.W_exc + banc_data.W_inh).toarray()
        for pm_idx in self._premotor_indices:
            # Weight from this premotor (column) to each MN (row)
            seg_weight = {1: 0.0, 2: 0.0, 3: 0.0}
            for mn_idx in self._mn_indices:
                w = abs(float(W_raw_full[mn_idx, pm_idx]))
                seg_c = mn_seg_lookup.get(mn_idx, 0)
                if seg_c > 0:
                    seg_weight[seg_c] += w
            total = sum(seg_weight.values())
            if total > 0:
                best_seg = max(seg_weight, key=seg_weight.get)
                neuron_seg[pm_idx] = best_seg

        n_t1 = int((neuron_seg == 1).sum())
        n_t2 = int((neuron_seg == 2).sum())
        n_t3 = int((neuron_seg == 3).sum())
        n_none = int((neuron_seg == 0).sum())
        print(f"  Segment assignment: T1={n_t1}, T2={n_t2}, T3={n_t3}, "
              f"unassigned={n_none} (DNs)")
        self._neuron_segment = {i: {1: "T1", 2: "T2", 3: "T3"}.get(neuron_seg[i], "")
                                for i in range(self.n_neurons)}

        # --- Weight matrices: apply config scaling to raw counts ---
        max_abs = max(abs(W_raw_full.max()), abs(W_raw_full.min()))
        if max_abs > 10.0:
            W_pos = np.where(W_raw_full > 0, W_raw_full, 0)
            W_neg = np.where(W_raw_full < 0, W_raw_full, 0)

            seg_exc = self.cfg.segment_exc_mults
            seg_inh = self.cfg.segment_inh_scales
            if seg_exc is not None or seg_inh is not None:
                # Per-edge segment-specific scaling
                N = self.n_neurons
                nz_i, nz_j = np.nonzero(W_raw_full)
                pre_seg = neuron_seg[nz_j]
                post_seg = neuron_seg[nz_i]
                same_seg = (pre_seg == post_seg) & (pre_seg > 0)

                edge_exc_mult = np.full(len(nz_i), self.cfg.exc_mult, dtype=np.float32)
                edge_inh_scale = np.full(len(nz_i), self.cfg.inh_scale, dtype=np.float32)
                for seg_name, seg_c in seg_code_map.items():
                    mask = same_seg & (pre_seg == seg_c)
                    if seg_exc is not None and seg_name in seg_exc:
                        edge_exc_mult[mask] = seg_exc[seg_name]
                    if seg_inh is not None and seg_name in seg_inh:
                        edge_inh_scale[mask] = seg_inh[seg_name]

                signs = np.where(W_raw_full[nz_i, nz_j] > 0, 1.0, -1.0)
                W = np.zeros((N, N), dtype=np.float32)
                eff = np.where(
                    signs > 0,
                    edge_exc_mult * W_pos[nz_i, nz_j],
                    self.cfg.inh_mult * edge_inh_scale * W_neg[nz_i, nz_j],
                )
                W[nz_i, nz_j] = eff
                n_intra = int(same_seg.sum())
                print(f"  Segment-specific scaling: {n_intra:,} intra-seg, "
                      f"{len(nz_i)-n_intra:,} cross-seg edges")
                if seg_exc:
                    print(f"    exc_mults: global={self.cfg.exc_mult}, per-seg={seg_exc}")
                if seg_inh:
                    print(f"    inh_scales: global={self.cfg.inh_scale}, per-seg={seg_inh}")
            else:
                W = (self.cfg.exc_mult * W_pos
                     + self.cfg.inh_mult * self.cfg.inh_scale * W_neg)
                print(f"  Applied config scaling: exc_mult={self.cfg.exc_mult}, "
                      f"inh_mult={self.cfg.inh_mult}, inh_scale={self.cfg.inh_scale}")
        else:
            W = W_raw_full.copy()
            print(f"  Weights pre-scaled (max_abs={max_abs:.4f}), using as-is")

        print(f"  W range: [{W.min():.4f}, {W.max():.4f}]")

        # --- Weight normalization ---
        if self.cfg.normalize_weights:
            exc_row_sums = np.where(W > 0, W, 0).sum(axis=1)
            mean_exc_row = float(exc_row_sums.mean())
            if mean_exc_row > 1e-8:
                scale = self.cfg.target_exc_sum / mean_exc_row
                W *= scale
                new_exc = np.where(W > 0, W, 0).sum(axis=1).mean()
                new_inh = np.where(W < 0, W, 0).sum(axis=1).mean()
                print(f"  Weight normalization: scale={scale:.6f}, "
                      f"mean exc={new_exc:.4f}, mean inh={new_inh:.4f}")

        # Split into exc/inh sparse matrices
        self.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
        self.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
        self.W = W.astype(np.float32)

        # --- Per-neuron parameters ---
        self._init_neuron_params()

        # --- State vectors ---
        self._init_state()

        build_time = time() - t0
        print(f"  BANC network ready in {build_time:.1f}s: "
              f"{self.n_neurons} neurons ({self.n_dn} DN, {self.n_mn} MN, "
              f"{self.n_premotor} premotor), {self.n_synapses:,} connections")

        if warmup_ms > 0:
            self._warmup(warmup_ms)

        return self

    # ========================================================================
    # Data loading
    # ========================================================================

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
        """Load annotations, neurotransmitters, and connectivity from feather files."""
        import pandas as pd
        import pyarrow.feather as feather

        t0 = time()
        self._ann = pd.DataFrame(feather.read_feather(str(self.cfg.annotations_path)))
        print(f"  Annotations: {len(self._ann):,} rows ({time() - t0:.1f}s)")

        t0 = time()
        nt_df = pd.DataFrame(feather.read_feather(str(self.cfg.neurotransmitters_path)))
        nt_unique = nt_df.drop_duplicates(subset="body", keep="first")
        self._nt_map: Dict[int, str] = dict(zip(
            nt_unique["body"].values,
            nt_unique["consensus_nt"].values,
        ))
        print(f"  Neurotransmitters: {len(self._nt_map):,} unique bodies ({time() - t0:.1f}s)")

        t0 = time()
        size_gb = self.cfg.connectivity_path.stat().st_size / 1e9
        print(f"  Loading connectivity ({size_gb:.1f} GB)...")
        self._conn = pd.DataFrame(feather.read_feather(str(self.cfg.connectivity_path)))
        print(f"  Connectivity: {len(self._conn):,} edges ({time() - t0:.1f}s)")

    # ========================================================================
    # Neuron selection: DNs + leg MNs + premotor interneurons
    # ========================================================================

    def _select_neurons(self):
        """Select the T1 premotor subnetwork: DNs + leg MNs + premotor interneurons.

        Premotor = VNC intrinsic neurons in the selected segments that synapse
        onto leg MNs with at least min_mn_synapses synapses.
        """
        import pandas as pd

        ann = self._ann
        segs = list(self.cfg.segments)

        # --- Descending neurons (all, regardless of segment) ---
        dn_mask = ann["superclass"] == "descending_neuron"
        self._dn_df = ann[dn_mask].copy()
        dn_ids = set(self._dn_df["bodyId"].values)

        # --- Leg motor neurons (selected segments, leg subclass only) ---
        mn_mask = (
            (ann["superclass"] == "vnc_motor")
            & (ann["somaNeuromere"].isin(segs))
            & (ann["subclass"].isin(["fl", "ml", "hl"]))  # leg MNs only
        )
        self._mn_df = ann[mn_mask].copy()
        mn_ids = set(self._mn_df["bodyId"].values)

        # --- Premotor interneurons: VNC intrinsic that synapse onto leg MNs ---
        print(f"  Finding premotor interneurons (min {self.cfg.min_mn_synapses} syn to MN)...")
        t0 = time()

        # Filter connectivity to edges ending at MNs
        pre_to_mn = self._conn[self._conn["body_post"].isin(mn_ids)]

        # Apply weight threshold
        pre_to_mn_heavy = pre_to_mn[pre_to_mn["weight"] >= self.cfg.min_mn_synapses]

        # Candidate premotor body IDs (excluding MNs themselves)
        candidate_premotor = set(pre_to_mn_heavy["body_pre"].values) - mn_ids

        # Filter to VNC intrinsic neurons in selected segments
        intrinsic_mask = (
            (ann["superclass"] == "vnc_intrinsic")
            & (ann["somaNeuromere"].isin(segs))
        )
        intrinsic_ids = set(ann[intrinsic_mask]["bodyId"].values)

        premotor_ids = candidate_premotor & intrinsic_ids
        print(f"  Premotor interneurons: {len(premotor_ids)} ({time() - t0:.1f}s)")

        # --- Combine all neuron populations ---
        if self.cfg.include_all_dns:
            all_ids = dn_ids | mn_ids | premotor_ids
        else:
            # Only include DNs that actually project to this subnetwork
            dn_premotor = candidate_premotor & dn_ids
            all_ids = dn_premotor | mn_ids | premotor_ids

        self.n_dn = len(dn_ids & all_ids)
        self.n_mn = len(mn_ids)
        self.n_premotor = len(premotor_ids)
        self.n_neurons = len(all_ids)

        print(f"  Selected neurons: {self.n_dn} DN + {self.n_mn} MN + "
              f"{self.n_premotor} premotor = {self.n_neurons}")

        # Create body_id <-> index mapping (sorted for determinism)
        all_ids_sorted = sorted(all_ids)
        self._bodyid_to_idx = {int(bid): i for i, bid in enumerate(all_ids_sorted)}
        self._idx_to_bodyid = {i: int(bid) for bid, i in self._bodyid_to_idx.items()}

        # Store population sets
        self._dn_ids = dn_ids & all_ids
        self._mn_ids = mn_ids
        self._premotor_ids = premotor_ids

        # Index arrays for each population
        self._dn_indices = np.array(
            sorted(self._bodyid_to_idx[bid] for bid in self._dn_ids), dtype=np.int32
        )
        self._mn_indices = np.array(
            sorted(self._bodyid_to_idx[bid] for bid in self._mn_ids), dtype=np.int32
        )
        self._premotor_indices = np.array(
            sorted(self._bodyid_to_idx[bid] for bid in self._premotor_ids), dtype=np.int32
        )

        # --- Build neuron-to-segment mapping (for segment-specific scaling) ---
        # Maps model index -> segment string ("T1", "T2", "T3") or "" for DNs
        self._neuron_segment = {}  # model_idx -> segment
        for bid in all_ids:
            idx = self._bodyid_to_idx[int(bid)]
            row = ann[ann["bodyId"] == bid]
            if len(row) > 0:
                seg = str(row.iloc[0].get("somaNeuromere", ""))
                if seg in ("T1", "T2", "T3"):
                    self._neuron_segment[idx] = seg
                else:
                    self._neuron_segment[idx] = ""
            else:
                self._neuron_segment[idx] = ""
        seg_counts = {}
        for seg in self._neuron_segment.values():
            seg_counts[seg] = seg_counts.get(seg, 0) + 1
        print(f"  Neuron segments: {seg_counts}")

        # Filter connectivity to the subnetwork
        print(f"  Filtering {len(self._conn):,} connections to {self.n_neurons} neurons...")
        t0 = time()
        pre_in = self._conn["body_pre"].isin(all_ids)
        post_in = self._conn["body_post"].isin(all_ids)
        self._subnet_conn = self._conn[pre_in & post_in].copy()
        self.n_synapses = len(self._subnet_conn)
        print(f"  Filtered to {self.n_synapses:,} intra-subnet edges ({time() - t0:.1f}s)")

    # ========================================================================
    # Weight matrix construction
    # ========================================================================

    def _build_weight_matrix(self):
        """Build weight matrix W from synapse counts and E/I signs.

        W[i, j] = effective weight from neuron j to neuron i:
            = exc_mult * count  if neuron j is excitatory
            = -inh_mult * inh_scale * count  if neuron j is inhibitory

        After construction, weights are normalized so the mean excitatory
        row sum equals target_exc_sum. This places the network's operating
        point near the activation threshold, enabling both excitation and
        inhibition to have meaningful effects.

        The final matrix is split into W_exc and W_inh (sparse CSR) for
        efficient delayed inhibition in the Euler integrator.
        """
        print("  Building weight matrix...")
        t0 = time()

        N = self.n_neurons
        W = np.zeros((N, N), dtype=np.float32)

        exc_mult = self.cfg.exc_mult
        inh_mult = self.cfg.inh_mult

        # Pre-compute E/I sign for each neuron (indexed by model index)
        neuron_sign = np.ones(N, dtype=np.float32)
        for bid, idx in self._bodyid_to_idx.items():
            nt = self._nt_map.get(bid, "unclear")
            neuron_sign[idx] = NT_SIGN.get(str(nt).lower().strip(), +1.0)

        # Vectorized weight matrix construction using numpy advanced indexing
        pre_bids = self._subnet_conn["body_pre"].values
        post_bids = self._subnet_conn["body_post"].values
        weights = self._subnet_conn["weight"].values.astype(np.float32)

        # Map body IDs to indices (vectorized via a lookup array)
        # Build a dense lookup: body_id -> index. Use max body_id + 1 as size.
        max_bid = max(self._bodyid_to_idx.keys())
        bid_to_idx_arr = np.full(max_bid + 1, -1, dtype=np.int32)
        for bid, idx in self._bodyid_to_idx.items():
            bid_to_idx_arr[bid] = idx

        j_indices = bid_to_idx_arr[pre_bids]   # presynaptic -> column
        i_indices = bid_to_idx_arr[post_bids]   # postsynaptic -> row

        # Filter out any that didn't map (shouldn't happen but safety)
        valid = (j_indices >= 0) & (i_indices >= 0)
        j_indices = j_indices[valid]
        i_indices = i_indices[valid]
        weights = weights[valid]

        # Get E/I sign for each presynaptic neuron
        signs = neuron_sign[j_indices]

        # --- Segment-specific scaling ---
        # For intra-segment connections (both pre and post in same segment),
        # use segment-specific exc_mult / inh_scale if configured.
        seg_exc = self.cfg.segment_exc_mults
        seg_inh = self.cfg.segment_inh_scales

        if seg_exc is not None or seg_inh is not None:
            # Build per-neuron segment code: 1=T1, 2=T2, 3=T3, 0=other/DN
            seg_code_map = {"T1": 1, "T2": 2, "T3": 3}
            neuron_seg_code = np.zeros(N, dtype=np.int8)
            for idx_n, seg_str in self._neuron_segment.items():
                neuron_seg_code[idx_n] = seg_code_map.get(seg_str, 0)

            pre_seg = neuron_seg_code[j_indices]
            post_seg = neuron_seg_code[i_indices]
            # Intra-segment: both in same non-zero segment
            same_seg = (pre_seg == post_seg) & (pre_seg > 0)

            # Build per-edge exc_mult and inh_scale arrays
            edge_exc_mult = np.full(len(j_indices), exc_mult, dtype=np.float32)
            edge_inh_scale = np.full(len(j_indices), self.cfg.inh_scale, dtype=np.float32)

            for seg_name, seg_c in seg_code_map.items():
                mask_seg = same_seg & (pre_seg == seg_c)
                if seg_exc is not None and seg_name in seg_exc:
                    edge_exc_mult[mask_seg] = seg_exc[seg_name]
                if seg_inh is not None and seg_name in seg_inh:
                    edge_inh_scale[mask_seg] = seg_inh[seg_name]

            n_intra = int(same_seg.sum())
            n_cross = len(j_indices) - n_intra
            print(f"  Segment-specific scaling: {n_intra:,} intra-seg, {n_cross:,} cross-seg edges")
            if seg_exc is not None:
                print(f"    exc_mults: global={exc_mult}, per-seg={seg_exc}")
            if seg_inh is not None:
                print(f"    inh_scales: global={self.cfg.inh_scale}, per-seg={seg_inh}")

            eff_weights = np.where(
                signs >= 0,
                edge_exc_mult * weights,
                -self.cfg.inh_mult * edge_inh_scale * weights,
            )
        else:
            # Global multipliers (original path)
            eff_weights = np.where(
                signs >= 0,
                exc_mult * weights,
                -inh_mult * self.cfg.inh_scale * weights,
            )

        # Accumulate into weight matrix using np.add.at for duplicate handling
        np.add.at(W, (i_indices, j_indices), eff_weights)

        n_exc = int((signs >= 0).sum())
        n_inh = int((signs < 0).sum())

        # Store pre-normalization stats
        nonzero_pre = np.count_nonzero(W)
        print(f"  Raw weight matrix: {N}x{N}, {nonzero_pre:,} nonzero, "
              f"{n_exc:,} exc + {n_inh:,} inh edges ({time() - t0:.1f}s)")
        print(f"  Raw W range: [{W.min():.4f}, {W.max():.4f}], inh_scale={self.cfg.inh_scale}")

        # --- Weight normalization ---
        # Rescale so the mean excitatory row sum equals target_exc_sum.
        # This determines: at 1 Hz all-active, how much exc input per neuron.
        # At 50 Hz with target_exc_sum=0.15: mean input = 0.15*50 = 7.5 (above theta=3).
        if self.cfg.normalize_weights:
            exc_row_sums = np.where(W > 0, W, 0).sum(axis=1)
            mean_exc_row = float(exc_row_sums.mean())
            if mean_exc_row > 1e-8:
                scale = self.cfg.target_exc_sum / mean_exc_row
                W *= scale
                # Verify
                new_exc = np.where(W > 0, W, 0).sum(axis=1).mean()
                new_inh = np.where(W < 0, W, 0).sum(axis=1).mean()
                print(f"  Weight normalization: scale={scale:.6f}, "
                      f"mean exc row sum={new_exc:.4f}, "
                      f"mean inh row sum={new_inh:.4f}")
                print(f"  At 50Hz all-active: mean exc input={new_exc*50:.1f}, "
                      f"mean inh input={new_inh*50:.1f}, "
                      f"net={new_exc*50+new_inh*50:.1f} (theta={self.cfg.theta})")

        # Split into excitatory and inhibitory components for delayed inhibition
        # Use scipy sparse matrices for much faster matrix-vector products
        # (the weight matrix is only ~2% dense).
        from scipy.sparse import csr_matrix
        self.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
        self.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
        self.W = W  # Keep dense for compatibility / analysis

        # Final statistics
        nonzero = np.count_nonzero(W)
        density = nonzero / (N * N) * 100 if N > 0 else 0
        exc_w = W[W > 0]
        inh_w = W[W < 0]
        print(f"  Final W: {nonzero:,} nonzero ({density:.2f}% dense), "
              f"range [{W.min():.6f}, {W.max():.6f}]")
        if len(exc_w) > 0 and len(inh_w) > 0:
            print(f"  mean exc={exc_w.mean():.6f}, mean inh={inh_w.mean():.6f}")

    # ========================================================================
    # Per-neuron parameter initialization (heterogeneity)
    # ========================================================================

    def _init_neuron_params(self):
        """Initialize per-neuron parameters with truncated-normal heterogeneity."""
        N = self.n_neurons
        cv = self.cfg.param_cv

        def _sample_truncnorm(mean: float, n: int) -> np.ndarray:
            """Sample from truncated normal: mean +/- 3*sigma, sigma = cv * mean."""
            sigma = cv * mean
            samples = self._rng.normal(mean, sigma, size=n)
            lo = mean - 3 * sigma
            hi = mean + 3 * sigma
            return np.clip(samples, max(lo, 1e-6), hi).astype(np.float32)

        self.tau = _sample_truncnorm(self.cfg.tau_ms, N)      # ms
        self.a = _sample_truncnorm(self.cfg.a, N)              # gain
        self.theta = _sample_truncnorm(self.cfg.theta, N)      # threshold
        self.fr_cap = _sample_truncnorm(self.cfg.fr_cap, N)    # max rate

        print(f"  Per-neuron params (cv={cv:.0%}): "
              f"tau={self.tau.mean():.1f}+/-{self.tau.std():.1f}ms, "
              f"a={self.a.mean():.2f}+/-{self.a.std():.2f}, "
              f"theta={self.theta.mean():.2f}+/-{self.theta.std():.2f}, "
              f"fr_cap={self.fr_cap.mean():.0f}+/-{self.fr_cap.std():.0f}")

    # ========================================================================
    # Motor neuron metadata
    # ========================================================================

    def _build_mn_metadata(self):
        """Build motor neuron metadata for leg/direction classification.

        Uses mn_joint_mapping.json if available, otherwise infers from MANC
        type annotations.
        """
        import pandas as pd

        # Sorted MN body IDs (same order as _mn_indices)
        self.mn_body_ids = np.array(sorted(self._mn_ids), dtype=np.int64)

        # Per-MN metadata: (body_id, segment, side, mn_type, leg, direction)
        self.mn_info: List[dict] = []
        self._mn_leg: np.ndarray = np.zeros(self.n_mn, dtype=np.int32)       # leg index [0-5]
        self._mn_direction: np.ndarray = np.zeros(self.n_mn, dtype=np.float32)  # +1 ext, -1 flex
        self._mn_is_flexor: np.ndarray = np.zeros(self.n_mn, dtype=bool)
        self._mn_is_extensor: np.ndarray = np.zeros(self.n_mn, dtype=bool)

        # Try mn_joint_mapping.json first
        mn_map = {}
        if self.cfg.mn_joint_mapping_path.exists():
            with open(self.cfg.mn_joint_mapping_path) as f:
                mn_map = json.load(f)

        for i, bid in enumerate(self.mn_body_ids):
            bid_str = str(int(bid))

            if bid_str in mn_map:
                entry = mn_map[bid_str]
                leg = entry.get("leg", "LF")
                direction = float(entry.get("direction", 0.0))
                mn_type = entry.get("mn_type", "unknown")
                seg = entry.get("neuromere", "T1")
                side = entry.get("side", "L")
            else:
                # Fallback: infer from MANC annotations
                row = self._mn_df[self._mn_df["bodyId"] == bid]
                if len(row) > 0:
                    row = row.iloc[0]
                    seg = str(row["somaNeuromere"]) if pd.notna(row.get("somaNeuromere")) else "T1"
                    side = str(row["somaSide"]) if pd.notna(row.get("somaSide")) else "L"
                    mn_type = str(row["type"]) if pd.notna(row.get("type")) else "unknown"
                    leg = SEGMENT_SIDE_TO_LEG.get((seg, side), "LF")
                    # Infer direction from type name
                    lower = mn_type.lower()
                    if "extensor" in lower or "levator" in lower:
                        direction = 1.0
                    elif "flexor" in lower or "depressor" in lower or "remotor" in lower:
                        direction = -1.0
                    else:
                        direction = 0.0
                else:
                    seg, side, mn_type, leg = "T1", "L", "unknown", "LF"
                    direction = 0.0

            leg_idx = LEG_ORDER.index(leg) if leg in LEG_ORDER else 0
            self._mn_leg[i] = leg_idx
            self._mn_direction[i] = direction

            # Classify as flexor or extensor for alternation analysis
            # direction < 0 -> flexor-type (protraction, flexion, depression)
            # direction > 0 -> extensor-type (retraction, extension, levation)
            self._mn_is_flexor[i] = direction < 0
            self._mn_is_extensor[i] = direction > 0

            self.mn_info.append({
                "body_id": int(bid),
                "segment": seg,
                "side": side,
                "mn_type": mn_type,
                "leg": leg,
                "leg_idx": leg_idx,
                "direction": direction,
            })

        n_flex = self._mn_is_flexor.sum()
        n_ext = self._mn_is_extensor.sum()
        n_other = self.n_mn - n_flex - n_ext
        print(f"  MN metadata: {n_ext} extensors, {n_flex} flexors, {n_other} ambiguous")

    # ========================================================================
    # Half-center oscillator boost
    # ========================================================================

    def _apply_half_center_boost(self):
        """Selectively boost inhibitory connections forming half-center oscillators.

        Half-center oscillators require strong reciprocal inhibition between
        flexor and extensor motor pools. This method identifies two types of
        half-center connections and multiplies their weights by half_center_boost:

        Type A: Inhibitory premotor interneurons that synapse onto BOTH flexor
                and extensor MNs within the same leg. Their projections to ALL
                MNs in that leg get boosted (they are the reciprocal inhibition
                relay neurons).

        Type B: Direct inhibitory connections between flexor and extensor MN
                pools within the same leg (rare in Drosophila).

        (Segment-specific inh_scale/exc_mult is now handled in _build_weight_matrix.)
        """
        from scipy.sparse import csr_matrix

        boost = self.cfg.half_center_boost

        if boost <= 1.0:
            return

        W = self.W.copy()
        t0 = time()

        # --- Half-center boost ---
        if boost > 1.0:
            print(f"  Identifying half-center connections (boost={boost}x)...")

            # Build per-leg flexor/extensor MN index sets
            leg_flex_indices = {}  # leg_idx -> set of model indices
            leg_ext_indices = {}
            for i, bid in enumerate(self.mn_body_ids):
                model_idx = self._bodyid_to_idx[int(bid)]
                leg_idx = self._mn_leg[i]
                if self._mn_is_flexor[i]:
                    leg_flex_indices.setdefault(leg_idx, set()).add(model_idx)
                elif self._mn_is_extensor[i]:
                    leg_ext_indices.setdefault(leg_idx, set()).add(model_idx)

            # --- Type A: Inhibitory premotor INs with cross-pool connectivity ---
            # For each inhibitory premotor interneuron, check if it synapses
            # onto both flexor and extensor MNs within the same leg.
            n_halfcenter_ins = 0
            n_boosted_entries = 0
            halfcenter_ins_per_leg = {leg: 0 for leg in range(6)}

            for bid in self._premotor_ids:
                idx = self._bodyid_to_idx.get(int(bid))
                if idx is None:
                    continue

                # Must be inhibitory
                nt = self._nt_map.get(int(bid), "unclear")
                sign = NT_SIGN.get(str(nt).lower().strip(), +1.0)
                if sign >= 0:
                    continue

                # Check connectivity to MNs: which legs' flex/ext does it reach?
                col_weights = W[:, idx]  # outgoing connections from this IN

                for leg_idx in range(6):
                    flex_set = leg_flex_indices.get(leg_idx, set())
                    ext_set = leg_ext_indices.get(leg_idx, set())
                    if not flex_set or not ext_set:
                        continue

                    flex_arr = np.array(list(flex_set), dtype=np.int32)
                    ext_arr = np.array(list(ext_set), dtype=np.int32)

                    # Does this IN inhibit at least one flexor AND one extensor?
                    hits_flex = np.any(col_weights[flex_arr] < 0)
                    hits_ext = np.any(col_weights[ext_arr] < 0)

                    if hits_flex and hits_ext:
                        # This IN forms a half-center bridge -- boost its
                        # inhibitory connections to ALL MNs in this leg
                        all_mn_in_leg = np.array(list(flex_set | ext_set), dtype=np.int32)
                        inh_to_leg = col_weights[all_mn_in_leg] < 0
                        n_entries = int(inh_to_leg.sum())
                        if n_entries > 0:
                            W[all_mn_in_leg[inh_to_leg], idx] *= boost
                            n_boosted_entries += n_entries
                            n_halfcenter_ins += 1
                            halfcenter_ins_per_leg[leg_idx] += 1

            # --- Type B: Direct flex <-> ext inhibition ---
            n_direct_boosted = 0
            for leg_idx in range(6):
                flex_set = leg_flex_indices.get(leg_idx, set())
                ext_set = leg_ext_indices.get(leg_idx, set())
                if not flex_set or not ext_set:
                    continue
                flex_arr = np.array(list(flex_set), dtype=np.int32)
                ext_arr = np.array(list(ext_set), dtype=np.int32)

                # Flex -> ext connections (flex is presynaptic)
                for f_idx in flex_arr:
                    for e_idx in ext_arr:
                        if W[e_idx, f_idx] < 0:
                            W[e_idx, f_idx] *= boost
                            n_direct_boosted += 1
                # Ext -> flex connections (ext is presynaptic)
                for e_idx in ext_arr:
                    for f_idx in flex_arr:
                        if W[f_idx, e_idx] < 0:
                            W[f_idx, e_idx] *= boost
                            n_direct_boosted += 1

            per_leg_str = ", ".join(
                f"{LEG_ORDER[k]}={v}" for k, v in sorted(halfcenter_ins_per_leg.items())
            )
            print(f"  Half-center INs found: {n_halfcenter_ins} "
                  f"(per leg: {per_leg_str})")
            print(f"  Boosted {n_boosted_entries} IN->MN entries + "
                  f"{n_direct_boosted} direct MN->MN entries "
                  f"(total {n_boosted_entries + n_direct_boosted} at {boost}x)")

        # Rebuild sparse matrices
        self.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
        self.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
        self.W = W

        print(f"  Half-center boost applied in {time() - t0:.1f}s")
        print(f"  Updated W range: [{W.min():.6f}, {W.max():.6f}]")

    # ========================================================================
    # DN type -> body ID mapping
    # ========================================================================

    def _build_dn_type_map(self):
        """Build DN type name -> list of model indices for targeted stimulation."""
        import pandas as pd

        self._dn_type_to_indices: Dict[str, List[int]] = {}
        self._dn_type_to_body_ids: Dict[str, List[int]] = {}

        for _, row in self._dn_df.iterrows():
            bid = int(row["bodyId"])
            if bid not in self._bodyid_to_idx:
                continue
            idx = self._bodyid_to_idx[bid]

            for col in ["type", "flywireType"]:
                val = row.get(col, None)
                if val is not None and isinstance(val, str) and val.strip():
                    dn_type = val.strip()
                    if dn_type not in self._dn_type_to_indices:
                        self._dn_type_to_indices[dn_type] = []
                        self._dn_type_to_body_ids[dn_type] = []
                    if idx not in self._dn_type_to_indices[dn_type]:
                        self._dn_type_to_indices[dn_type].append(idx)
                        self._dn_type_to_body_ids[dn_type].append(bid)

        n_types = len(self._dn_type_to_indices)
        n_dns_mapped = sum(len(v) for v in self._dn_type_to_indices.values())
        print(f"  DN type map: {n_types} types, {n_dns_mapped} neurons")

    # ========================================================================
    # State initialization
    # ========================================================================

    def _init_state(self):
        """Initialize firing rate state vector, adaptation, delay buffer, and stimulation."""
        N = self.n_neurons
        self.R = np.zeros(N, dtype=np.float32)       # Firing rates (Hz)
        self.A = np.zeros(N, dtype=np.float32)        # Adaptation variable
        self.I_stim = np.zeros(N, dtype=np.float32)   # External stimulation current

        # Silenced neuron mask (True = active)
        self._active = np.ones(N, dtype=bool)

        # Ring buffer for delayed inhibition
        if self.cfg.use_delay:
            delay_steps = max(1, int(self.cfg.delay_inh_ms / self.cfg.dt_ms))
            self._delay_buffer = np.zeros((delay_steps, N), dtype=np.float32)
            self._delay_idx = 0
            self._delay_steps = delay_steps

    # ========================================================================
    # Warmup
    # ========================================================================

    def _warmup(self, warmup_ms: float):
        """Run warmup period with low background noise to settle transients."""
        print(f"  Warming up for {warmup_ms:.0f} ms...")
        t0 = time()
        dt = self.cfg.dt_ms
        n_steps = int(warmup_ms / dt)

        # Add small noise during warmup to break symmetry
        noise_scale = 0.5
        for _ in range(n_steps):
            noise = self._rng.normal(0, noise_scale, size=self.n_neurons).astype(np.float32)
            self.I_stim[:] = noise
            self._euler_step(dt)

        self.I_stim[:] = 0.0
        print(f"  Warmup done ({time() - t0:.1f}s)")

    # ========================================================================
    # Pugliese rate-model ODE
    # ========================================================================

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Pugliese activation function (vectorized over all neurons).

        f(x) = max(fr_cap * tanh((a / fr_cap) * (x - theta)), 0)
        """
        shifted = x - self.theta
        scaled = (self.a / self.fr_cap) * shifted
        return np.maximum(self.fr_cap * np.tanh(scaled), 0.0)

    def _euler_step(self, dt_ms: float):
        """Advance the ODE by one Euler step.

        Rate equation:
            tau_i * dR_i/dt = -R_i + f(W_exc @ R(t) + W_inh @ R(t-d) + I_stim - A)

        Adaptation equation (if enabled):
            tau_adapt * dA_i/dt = -A_i + b_adapt * R_i

        Key features:
        - Excitatory input is instantaneous (R at time t)
        - Inhibitory input is delayed by delay_inh_ms (R at time t-d)
        - Adaptation provides negative self-feedback for oscillation
        """
        # Excitatory input: instantaneous (sparse @ dense)
        exc_input = np.asarray(self.W_exc @ self.R).ravel()

        # Inhibitory input: delayed if enabled (sparse @ dense)
        if self.cfg.use_delay and hasattr(self, '_delay_buffer'):
            R_delayed = self._delay_buffer[self._delay_idx]
            inh_input = np.asarray(self.W_inh @ R_delayed).ravel()
            # Store current rates in buffer
            self._delay_buffer[self._delay_idx] = self.R.copy()
            self._delay_idx = (self._delay_idx + 1) % self._delay_steps
        else:
            inh_input = np.asarray(self.W_inh @ self.R).ravel()

        total_input = exc_input + inh_input + self.I_stim

        if self.cfg.use_adaptation:
            total_input -= self.A

        # Ongoing noise: prevents fixed-point convergence
        if self.cfg.noise_sigma > 0:
            total_input += self._rng.normal(0, self.cfg.noise_sigma, self.n_neurons)

        # Activation function (per-neuron heterogeneous params)
        R_inf = self._activation(total_input)

        # Euler update: dR = (dt / tau) * (-R + R_inf)
        dR = (dt_ms / self.tau) * (-self.R + R_inf)
        self.R += dR

        # Update adaptation
        if self.cfg.use_adaptation:
            dA = (dt_ms / self.cfg.tau_adapt_ms) * (-self.A + self.cfg.b_adapt * self.R)
            self.A += dA

        # Enforce non-negativity and cap
        np.clip(self.R, 0.0, self.fr_cap, out=self.R)

        # Enforce silencing
        self.R[~self._active] = 0.0
        if self.cfg.use_adaptation:
            self.A[~self._active] = 0.0

        self._time_ms += dt_ms

    # ========================================================================
    # Public API: step, stimulation, readout
    # ========================================================================

    def step(self, dt_ms: float | None = None, n_substeps: int = 1):
        """Advance the model by dt_ms milliseconds.

        Args:
            dt_ms: Integration timestep in ms. Defaults to cfg.dt_ms (0.5ms).
            n_substeps: Number of substeps per call (for stability at large dt).
        """
        dt = dt_ms if dt_ms is not None else self.cfg.dt_ms
        n_substeps = max(n_substeps, 1)
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            self._euler_step(sub_dt)

    def stimulate_dn_type(self, dn_type: str, rate_hz: float):
        """Set tonic stimulation current for all neurons of a given DN type.

        The rate_hz is converted to a current in activation-function units:
        I = theta * (rate_hz / 20.0), so 20 Hz of stimulation pushes
        the neuron to the activation threshold.

        Args:
            dn_type: DN type name (e.g., "DNg100", "DNa02").
            rate_hz: Stimulation strength in Hz (20 Hz = threshold-level).
        """
        indices = self._dn_type_to_indices.get(dn_type, [])
        if not indices:
            print(f"  WARNING: DN type '{dn_type}' not found in network")
            return
        # Convert rate to current in activation units
        # 20 Hz -> theta (just at threshold), 40 Hz -> 2*theta (strong)
        current = self.cfg.theta * (rate_hz / 20.0)
        for idx in indices:
            self.I_stim[idx] = current
        print(f"  Stimulating {dn_type}: {len(indices)} neurons, "
              f"rate={rate_hz:.1f}Hz -> I={current:.2f} (theta={self.cfg.theta})")

    def stimulate_all_dns(self, rate_hz: float = 15.0):
        """Apply tonic background stimulation to all DN neurons.

        This models the biological baseline: DNs receive tonic input from
        the brain even at rest. Similar to dn_baseline_hz in Brian2VNCRunner.

        Args:
            rate_hz: Background stimulation rate (15 Hz = default baseline).
        """
        current = self.cfg.theta * (rate_hz / 20.0)
        for idx in self._dn_indices:
            self.I_stim[idx] = current
        print(f"  DN baseline: {len(self._dn_indices)} neurons at {rate_hz:.1f}Hz "
              f"(I={current:.2f})")

    def stimulate_indices(self, indices: np.ndarray | list, rate_hz: float):
        """Set tonic stimulation for specific neuron indices.

        Args:
            indices: Neuron indices in the model.
            rate_hz: Rate in Hz (converted to current units via theta scaling).
        """
        current = self.cfg.theta * (rate_hz / 20.0)
        for idx in indices:
            self.I_stim[idx] = current

    def clear_stimulation(self):
        """Remove all external stimulation."""
        self.I_stim[:] = 0.0

    def silence_neurons(self, body_ids: list | set | np.ndarray):
        """Silence specific neurons (set their rates to 0, disable updates)."""
        for bid in body_ids:
            idx = self._bodyid_to_idx.get(int(bid))
            if idx is not None:
                self._active[idx] = False
                self.R[idx] = 0.0

    def silence_dn_type(self, dn_type: str):
        """Silence all neurons of a given DN type."""
        indices = self._dn_type_to_indices.get(dn_type, [])
        for idx in indices:
            self._active[idx] = False
            self.R[idx] = 0.0

    def unsilence_all(self):
        """Restore all silenced neurons."""
        self._active[:] = True

    def get_mn_rates(self) -> np.ndarray:
        """Get current firing rates for all motor neurons.

        Returns:
            (n_mn,) array of firing rates in Hz, ordered by sorted body ID.
        """
        return self.R[self._mn_indices].copy()

    def get_mn_rates_by_leg(self) -> Dict[str, np.ndarray]:
        """Get MN firing rates grouped by leg.

        Returns:
            Dict mapping leg name -> (n_mn_in_leg,) rates array.
        """
        rates = self.get_mn_rates()
        by_leg = {}
        for leg_idx, leg_name in enumerate(LEG_ORDER):
            mask = self._mn_leg == leg_idx
            by_leg[leg_name] = rates[mask]
        return by_leg

    def get_flexor_extensor_rates(self, leg_idx: int | None = None) -> Tuple[float, float]:
        """Get mean flexor and extensor rates, optionally for a specific leg.

        Args:
            leg_idx: Leg index [0-5] or None for all legs.

        Returns:
            (mean_flexor_rate, mean_extensor_rate) in Hz.
        """
        rates = self.get_mn_rates()

        if leg_idx is not None:
            leg_mask = self._mn_leg == leg_idx
            flex_mask = self._mn_is_flexor & leg_mask
            ext_mask = self._mn_is_extensor & leg_mask
        else:
            flex_mask = self._mn_is_flexor
            ext_mask = self._mn_is_extensor

        flex_rate = float(rates[flex_mask].mean()) if flex_mask.any() else 0.0
        ext_rate = float(rates[ext_mask].mean()) if ext_mask.any() else 0.0
        return (flex_rate, ext_rate)

    def get_dn_rates(self) -> np.ndarray:
        """Get current firing rates for all DN neurons."""
        return self.R[self._dn_indices].copy()

    def get_all_rates(self) -> np.ndarray:
        """Get current firing rates for all neurons."""
        return self.R.copy()

    @property
    def current_time_ms(self) -> float:
        return self._time_ms

    def get_dn_type_to_body_ids(self) -> Dict[str, List[int]]:
        """Return DN type -> body ID mapping."""
        return dict(self._dn_type_to_body_ids)

    def reset(self):
        """Reset state to initial conditions."""
        self._init_state()
        self._time_ms = 0.0


# ============================================================================
# Main: validation test
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 72)
    print("Pugliese-style Firing Rate VNC Model -- Validation")
    print("=" * 72)

    # Build network from MANC data
    cfg = FiringRateVNCConfig(
        segments=("T1", "T2", "T3"),
        min_mn_synapses=3,
    )
    runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)

    # Apply tonic baseline to all DNs (like Brian2 model's dn_baseline_hz).
    # Must be above 20 Hz so the current exceeds theta (20Hz -> I=theta).
    runner.stimulate_all_dns(rate_hz=25.0)

    # Stimulate DNg100 (walking command neuron) with stronger tonic input
    runner.stimulate_dn_type("DNg100", rate_hz=60.0)

    # Run simulation for 2000ms, recording MN rates every 1ms
    sim_ms = 2000.0
    dt = 0.5  # ms
    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)  # record every 1ms

    n_records = int(sim_ms / 1.0)
    time_axis = np.linspace(0, sim_ms, n_records)

    # Per-leg flexor/extensor traces
    flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}

    # Also record full MN population
    all_mn_traces = np.zeros((n_records, runner.n_mn), dtype=np.float32)

    print(f"\nRunning simulation: {sim_ms:.0f} ms, dt={dt} ms, {n_steps} steps...")
    t0 = time()

    rec_idx = 0
    for step_i in range(n_steps):
        runner.step(dt_ms=dt)

        if (step_i + 1) % record_every == 0 and rec_idx < n_records:
            mn_rates = runner.get_mn_rates()
            all_mn_traces[rec_idx] = mn_rates

            for leg_idx, leg_name in enumerate(LEG_ORDER):
                flex_rate, ext_rate = runner.get_flexor_extensor_rates(leg_idx)
                flex_traces[leg_name][rec_idx] = flex_rate
                ext_traces[leg_name][rec_idx] = ext_rate

            rec_idx += 1

    sim_time = time() - t0
    print(f"Simulation done in {sim_time:.1f}s ({n_steps / sim_time:.0f} steps/s)")

    # ---- Analysis: check for flexor/extensor alternation ----
    print("\n" + "=" * 72)
    print("Flexor-Extensor Alternation Analysis")
    print("=" * 72)

    # Use the last 1500ms (skip first 500ms transient)
    skip_ms = 500
    skip_idx = int(skip_ms / 1.0)

    results = {}
    for leg_name in LEG_ORDER:
        f = flex_traces[leg_name][skip_idx:]
        e = ext_traces[leg_name][skip_idx:]

        # Check for nonzero activity
        f_active = f.max() > 1.0
        e_active = e.max() > 1.0

        if f_active and e_active and f.std() > 1e-6 and e.std() > 1e-6:
            # Pearson correlation between flexor and extensor
            corr = float(np.corrcoef(f, e)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        results[leg_name] = {
            "flex_mean": f.mean(),
            "ext_mean": e.mean(),
            "flex_max": f.max(),
            "ext_max": e.max(),
            "flex_active": f_active,
            "ext_active": e_active,
            "correlation": corr,
        }

        status = "ALTERNATING" if corr < -0.1 else ("ACTIVE" if (f_active or e_active) else "SILENT")
        print(f"  {leg_name}: flex={f.mean():.1f}Hz (max {f.max():.1f}), "
              f"ext={e.mean():.1f}Hz (max {e.max():.1f}), "
              f"corr={corr:.3f} -> {status}")

    # Overall result
    n_alternating = sum(1 for r in results.values() if r["correlation"] < -0.1)
    n_active = sum(1 for r in results.values() if r["flex_active"] or r["ext_active"])

    print(f"\nSummary: {n_alternating}/6 legs alternating, {n_active}/6 legs active")
    if n_alternating >= 3:
        print("SUCCESS: Flexor/extensor alternation achieved!")
    elif n_active >= 3:
        print("PARTIAL: MNs active but not alternating (may need parameter tuning)")
    else:
        print("FAILURE: Insufficient MN activity")

    # ---- Spectral analysis: check for rhythmic oscillation ----
    print("\n" + "=" * 72)
    print("Spectral Analysis (looking for 5-20 Hz rhythm)")
    print("=" * 72)

    for leg_name in ["LF", "RF"]:
        f = flex_traces[leg_name][skip_idx:]
        if f.max() < 1.0:
            print(f"  {leg_name}: too quiet for spectral analysis")
            continue

        # Remove DC component
        f_centered = f - f.mean()

        # FFT
        n_samples = len(f_centered)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / 1000.0)  # 1 kHz sampling
        psd = np.abs(np.fft.rfft(f_centered)) ** 2

        # Find peak in 5-20 Hz range
        band_mask = (freqs >= 5.0) & (freqs <= 20.0)
        if band_mask.any():
            band_psd = psd[band_mask]
            band_freqs = freqs[band_mask]
            peak_idx = np.argmax(band_psd)
            peak_freq = band_freqs[peak_idx]
            peak_power = band_psd[peak_idx]
            total_power = psd[freqs > 1.0].sum()
            frac = peak_power / total_power if total_power > 0 else 0
            print(f"  {leg_name}: peak at {peak_freq:.1f} Hz "
                  f"(power fraction in 5-20Hz band: {frac:.1%})")
        else:
            print(f"  {leg_name}: no data in 5-20 Hz band")

    # ---- Plot if matplotlib available ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle("Pugliese Firing Rate VNC: MN Activity (DNg100 stim @ 60 Hz, DN baseline @ 25 Hz)")

        for i, leg_name in enumerate(LEG_ORDER):
            ax = axes[i // 2, i % 2]
            t = time_axis

            ax.plot(t, flex_traces[leg_name], "b-", alpha=0.7, label="Flexor (mean)")
            ax.plot(t, ext_traces[leg_name], "r-", alpha=0.7, label="Extensor (mean)")
            ax.axvline(skip_ms, color="gray", linestyle="--", alpha=0.3)

            corr = results[leg_name]["correlation"]
            ax.set_title(f"{leg_name} (corr={corr:.3f})")
            ax.set_ylabel("Rate (Hz)")
            if i >= 4:
                ax.set_xlabel("Time (ms)")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        out_path = ROOT / "figures" / "vnc_firing_rate_validation.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150)
        print(f"\nPlot saved to {out_path}")

    except ImportError:
        print("\nmatplotlib not available -- skipping plot")

    print("\nDone.")
