"""
BANC connectome data loader for the VNC firing-rate / LIF models.

Data source: Brain And Nerve Cord (BANC) — first complete brain+VNC connectome
of an adult female Drosophila melanogaster.
    - ~160K neurons, 214M synapses
    - doi:10.7910/DVN/8TFGGB (Harvard Dataverse, CC-BY 4.0)

Database: banc_626_data.sqlite (~684 MB)
    meta            — neuron annotations (root_id, flow, super_class, cell_class,
                      cell_type, region, side, nerve, neurotransmitter_predicted)
    edgelist_simple — connectivity (pre_pt_root_id, post_pt_root_id, n)

Classification hierarchy (FlyWire Codex annotation standard):
    flow:        afferent | efferent | intrinsic
    super_class: ascending | descending | motor | sensory | endocrine |
                 visual_projection | visual_centrifugal |
                 optic_lobe_intrinsic | central_brain_intrinsic
    cell_class:  ~106 fine categories (leg_motor_neuron, olfactory_receptor_neuron, ...)
    cell_type:   individual neuron names (DNge110, Ti flexor MN, ...)

VNC neuron identification:
    region = 'ventral_nerve_cord'
    Leg MNs: super_class='motor', cell_class contains 'leg_motor'
    DNs:     super_class='descending' OR flow='efferent'

Two layers:
    BANCLoader     — generic DB reader (neurons, connectivity, selections)
    BANCVNCData    — VNC-specific extraction for the firing-rate / LIF model
                     (DNs, leg MNs, premotor interneurons, weight matrix)

Usage:
    # Generic access
    from bridge.banc_loader import BANCLoader
    loader = BANCLoader()
    neurons = loader.load_neurons()
    connectivity = loader.load_connectivity()

    # VNC model data (compatible with vnc_firing_rate.py interface)
    from bridge.banc_loader import BANCVNCData, load_banc_vnc
    data = load_banc_vnc()
    print(data.n_dn, data.n_mn, data.n_premotor, data.n_neurons)
    print(data.W_exc.shape, data.W_inh.shape)
"""

from __future__ import annotations

import sqlite3
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Set, Optional


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
BANC_DIR = ROOT / "data" / "banc"
DATA_DIR = ROOT / "data"

# ============================================================================
# Neurotransmitter -> E/I sign (same as vnc_firing_rate.py / vnc_connectome.py)
# ============================================================================

NT_SIGN = {
    "acetylcholine": +1.0,
    "glutamate":     +1.0,   # Mixed in Drosophila; default excitatory
    "gaba":          -1.0,
    "dopamine":      +1.0,
    "serotonin":     +1.0,
    "octopamine":    +1.0,
    "histamine":     -1.0,
    "unclear":       +1.0,   # Default excitatory
}

# ============================================================================
# BANC region/side -> FlyGym leg mapping
# ============================================================================

# BANC uses 'ventral_nerve_cord' for region, and 'left'/'right' for side.
# Cell types in BANC follow the MANC naming convention for VNC neurons
# (e.g., "Ti flexor MN"). Thoracic segment is encoded in the cell_type
# or nerve field (e.g., nerve='ProLN' for T1).

# Nerve -> segment mapping (BANC verbose names + MANC short codes)
NERVE_TO_SEGMENT = {
    # MANC short codes
    "ProLN":  "T1",   # Prothoracic leg nerve
    "MesoLN": "T2",   # Mesothoracic leg nerve
    "MetaLN": "T3",   # Metathoracic leg nerve
    # BANC verbose names — primary leg nerves
    "left_prothoracic_leg_nerve":  "T1",
    "right_prothoracic_leg_nerve": "T1",
    "left_mesothoracic_leg_nerve":  "T2",
    "right_mesothoracic_leg_nerve": "T2",
    "left_metathoracic_leg_nerve":  "T3",
    "right_metathoracic_leg_nerve": "T3",
    # BANC verbose — accessory/secondary leg nerves (same segment)
    "left_prothoracic_accessory_nerve":  "T1",
    "right_prothoracic_accessory_nerve": "T1",
    "left_ventral_prothoracic_nerve":  "T1",
    "right_ventral_prothoracic_nerve": "T1",
    "left_dorsal_prothoracic_nerve":  "T1",
    "right_dorsal_prothoracic_nerve": "T1",
    "left_mesothoracic_accessory_nerve":  "T2",
    "right_mesothoracic_accessory_nerve": "T2",
    "left_anterior_dorsal_mesothoracic_nerve":  "T2",
    "right_anterior_dorsal_mesothoracic_nerve": "T2",
    "left_posterior_dorsal_mesothoracic_nerve":  "T2",
    "right_posterior_dorsal_mesothoracic_nerve": "T2",
    "left_dorsal_metathoracic_nerve":  "T3",
    "right_dorsal_metathoracic_nerve": "T3",
    "left_first_abdominal_nerve":  "T3",   # MetaAN — hind leg MNs exit here
    "right_first_abdominal_nerve": "T3",
    # Mixed-format entries found in BANC
    "MesoLN_L": "T2", "MesoLN_R": "T2",
    "VProN_L": "T1",
    "right T2 leg nerve": "T2",
}

# cell_sub_class -> segment mapping (most reliable for BANC leg MNs)
SUB_CLASS_TO_SEGMENT = {
    "front_leg_motor_neuron":  "T1",
    "middle_leg_motor_neuron": "T2",
    "hind_leg_motor_neuron":   "T3",
}

# Segment x side -> FlyGym leg
SEGMENT_SIDE_TO_LEG = {
    ("T1", "left"):  "LF", ("T1", "right"): "RF",
    ("T2", "left"):  "LM", ("T2", "right"): "RM",
    ("T3", "left"):  "LH", ("T3", "right"): "RH",
    # MANC-style L/R variants (for cross-reference compatibility)
    ("T1", "L"): "LF", ("T1", "R"): "RF",
    ("T2", "L"): "LM", ("T2", "R"): "RM",
    ("T3", "L"): "LH", ("T3", "R"): "RH",
}

LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]


# ============================================================================
# BANCLoader — generic SQLite reader
# ============================================================================

class BANCLoader:
    """Loader for BANC SQLite connectome data.

    Reads from Harvard Dataverse SQLite files. Provides a standardized
    DataFrame interface matching the rest of the bridge pipeline.

    Handles two filename conventions:
        banc_626_data.sqlite  (v626, from Dataverse file_id 11842995)
        banc_data.sqlite      (generic name for backward compat)
    """

    def __init__(
        self,
        banc_dir: Path | str | None = None,
        use_frankenbrain: bool = False,
    ):
        self.banc_dir = Path(banc_dir) if banc_dir else BANC_DIR
        self.use_frankenbrain = use_frankenbrain
        self._neurons_cache: Optional[pd.DataFrame] = None
        self._connectivity_cache: Optional[pd.DataFrame] = None

    @property
    def db_path(self) -> Path:
        """Resolve the SQLite database path, trying versioned name first."""
        if self.use_frankenbrain:
            return self.banc_dir / "frankenbrain_v1.1_data.sqlite"
        # Try versioned name first, then generic fallback
        versioned = self.banc_dir / "banc_626_data.sqlite"
        if versioned.exists():
            return versioned
        return self.banc_dir / "banc_data.sqlite"

    def is_available(self) -> bool:
        """Check if the SQLite database exists."""
        return self.db_path.exists()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the SQLite database."""
        if not self.is_available():
            raise FileNotFoundError(
                f"BANC database not found. Looked for:\n"
                f"  {self.banc_dir / 'banc_626_data.sqlite'}\n"
                f"  {self.banc_dir / 'banc_data.sqlite'}\n"
                f"\n"
                f"Download from Harvard Dataverse (doi:10.7910/DVN/8TFGGB):\n"
                f"  python scripts/download_banc.py\n"
                f"\n"
                f"Or manually download banc_626_data.sqlite (~684 MB) from:\n"
                f"  https://dataverse.harvard.edu/api/access/datafile/11842995\n"
                f"and place it in: {self.banc_dir}"
            )
        return sqlite3.connect(str(self.db_path))

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        con = self._connect()
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )
        con.close()
        return tables["name"].tolist()

    def load_neurons(self) -> pd.DataFrame:
        """Load neuron annotations from the ``meta`` table.

        Standardizes column names to a common interface:
            body_id     int64   — neuron root ID
            cell_type   str     — individual neuron name
            super_class str     — coarse class (descending, motor, sensory, ...)
            cell_class  str     — fine class (leg_motor_neuron, ...)
            flow        str     — afferent / efferent / intrinsic
            region      str     — brain / ventral_nerve_cord / ...
            side        str     — left / right
            nerve       str     — entry nerve (ProLN, MesoLN, MetaLN)
            nt          str     — predicted neurotransmitter

        The raw columns are preserved alongside the standardized ones.
        """
        if self._neurons_cache is not None:
            return self._neurons_cache

        con = self._connect()
        df = pd.read_sql("SELECT * FROM meta", con)
        con.close()

        # Standardize column names.
        # BANC v626/v821 schema uses: root_id, cell_type, super_class,
        # cell_class, flow, region, side, nerve, neurotransmitter_predicted.
        # Older test DBs may use: id, soma_side, modality, etc.
        col_map = {}
        for target, candidates in [
            ("body_id", ["root_id", "root_626", "id", "pt_root_id",
                         "bodyId", "body_id", "segment_id"]),
            ("cell_type", ["cell_type", "type", "cellType"]),
            ("super_class", ["super_class", "superclass"]),
            ("cell_class", ["cell_class", "cell_class_name"]),
            ("cell_sub_class", ["cell_sub_class"]),
            ("flow", ["flow"]),
            ("region", ["region", "neuropil", "compartment"]),
            ("side", ["side", "soma_side", "somaSide"]),
            ("nerve", ["nerve", "entryNerve"]),
            ("nt", ["neurotransmitter_predicted", "neurotransmitter_verified",
                     "consensus_nt"]),
            ("modality", ["modality"]),
            ("manc_match", ["manc_match"]),
            ("fafb_match", ["fafb_match"]),
        ]:
            for c in candidates:
                if c in df.columns and target not in col_map.values():
                    col_map[c] = target
                    break

        df = df.rename(columns=col_map)

        # If body_id still missing, use first integer column
        if "body_id" not in df.columns:
            if df.index.name and "id" in str(df.index.name).lower():
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "body_id"})
            else:
                for c in df.columns:
                    if df[c].dtype in [np.int64, np.int32, int]:
                        df = df.rename(columns={c: "body_id"})
                        break

        # Fill missing columns with defaults
        for col in ["cell_type", "super_class", "cell_class", "cell_sub_class",
                     "flow", "region", "side", "nerve", "nt", "modality",
                     "manc_match", "fafb_match"]:
            if col not in df.columns:
                df[col] = ""

        # Ensure string columns are strings (SQLite may return None)
        for col in ["cell_type", "super_class", "cell_class", "cell_sub_class",
                     "flow", "region", "side", "nerve", "nt", "modality",
                     "manc_match", "fafb_match"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        df["body_id"] = df["body_id"].astype(np.int64)
        self._neurons_cache = df
        return df

    def load_connectivity(self) -> pd.DataFrame:
        """Load connectivity from the ``edgelist_simple`` table.

        Returns DataFrame with standardized columns:
            pre_id  int64 — presynaptic neuron root ID
            post_id int64 — postsynaptic neuron root ID
            weight  int   — synapse count
        """
        if self._connectivity_cache is not None:
            return self._connectivity_cache

        con = self._connect()
        df = pd.read_sql("SELECT * FROM edgelist_simple", con)
        con.close()

        # Standardize column names.
        # BANC schema: pre_pt_root_id, post_pt_root_id, n
        # Older test DBs: pre, post, count
        col_map = {}
        for target, candidates in [
            ("pre_id", ["pre_pt_root_id", "pre", "pre_id", "body_pre"]),
            ("post_id", ["post_pt_root_id", "post", "post_id", "body_post"]),
            ("weight", ["n", "count", "weight", "syn_count", "n_synapses"]),
        ]:
            for c in candidates:
                if c in df.columns and target not in col_map.values():
                    col_map[c] = target
                    break

        df = df.rename(columns=col_map)
        df["pre_id"] = df["pre_id"].astype(np.int64)
        df["post_id"] = df["post_id"].astype(np.int64)
        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1).astype(int)

        self._connectivity_cache = df
        return df

    def load_neck_bridge(self) -> Optional[pd.DataFrame]:
        """Load neck_bridge table (Frankenbrain only).

        Maps BANC brain IDs <-> MANC VNC IDs across the neck connective.
        Returns None if table does not exist.
        """
        if not self.is_available():
            return None
        con = self._connect()
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )
        if "neck_bridge" not in tables["name"].values:
            con.close()
            return None
        df = pd.read_sql("SELECT * FROM neck_bridge", con)
        con.close()
        return df

    def load_cross_references(self) -> Dict[str, pd.DataFrame]:
        """Load cross-reference data (BANC <-> other datasets).

        Returns a dict of DataFrames keyed by dataset name.
        Uses the manc_match and fafb_match columns from the meta table.
        """
        neurons = self.load_neurons()
        refs = {}

        # MANC cross-references
        if "manc_match" in neurons.columns:
            manc = neurons[neurons["manc_match"] != ""][["body_id", "manc_match"]].copy()
            if len(manc) > 0:
                refs["manc"] = manc

        # FAFB/FlyWire cross-references
        if "fafb_match" in neurons.columns:
            fafb = neurons[neurons["fafb_match"] != ""][["body_id", "fafb_match"]].copy()
            if len(fafb) > 0:
                refs["flywire"] = fafb.rename(
                    columns={"fafb_match": "flywire_id"}
                )

        return refs

    # ---- Selection helpers --------------------------------------------------

    def select_brain_neurons(self) -> Set[int]:
        """Select brain neurons (excluding VNC)."""
        df = self.load_neurons()
        mask = (
            df["region"].str.lower().isin(["brain", "central_brain", "optic_lobe"])
            | df["super_class"].str.lower().isin([
                "central_brain_intrinsic", "optic_lobe_intrinsic",
                "visual_projection", "visual_centrifugal",
            ])
        )
        return set(df[mask]["body_id"].values)

    def select_vnc_neurons(self) -> Set[int]:
        """Select VNC neurons (region='ventral_nerve_cord')."""
        df = self.load_neurons()
        mask = (
            df["region"].str.lower().str.contains(
                "vnc|ventral_nerve_cord|t1|t2|t3", na=False
            )
            | df["super_class"].str.lower().isin(["motor", "ascending"])
        )
        return set(df[mask]["body_id"].values)

    def select_dns(self) -> Set[int]:
        """Select descending neurons (brain -> VNC).

        Uses super_class='descending' as primary criterion, with cell_type
        pattern DN[a-z] as fallback. Does NOT use flow='efferent' alone
        because motor neurons also have efferent flow.
        """
        df = self.load_neurons()
        mask = (
            (df["super_class"].str.lower() == "descending")
            | df["cell_type"].str.match(r"^DN[a-z]", na=False)
        )
        return set(df[mask]["body_id"].values)

    def select_mns(self) -> Set[int]:
        """Select motor neurons (VNC -> muscles)."""
        df = self.load_neurons()
        mask = df["super_class"].str.lower() == "motor"
        return set(df[mask]["body_id"].values)

    def select_leg_mns(self) -> Set[int]:
        """Select leg motor neurons specifically.

        Uses cell_class='leg_motor_neuron' if available, otherwise infers
        from cell_type patterns (Ti/Tr/Fe/Ta MN, ltm MN, etc.).
        """
        df = self.load_neurons()
        mask = df["cell_class"].str.lower().str.contains(
            "leg_motor", na=False
        )
        # Fallback: cell_type pattern matching for known MN types
        type_mask = df["cell_type"].str.contains(
            r"(?:Ti |Tr |Fe |Ta |ltm|Sternotrochanter|Tergopleural|"
            r"Pleural remotor|Sternal |Tergotr\.) *(?:flexor|extensor|"
            r"MN|motor|promotor|remotor|depressor|levator|adductor|rotator)",
            na=False, case=False, regex=True,
        )
        # Also match nerve-based identification (leg nerves only)
        nerve_mask = (
            (df["super_class"].str.lower() == "motor")
            & df["nerve"].str.lower().isin(["proln", "mesoln", "metaln"])
        )
        return set(df[mask | type_mask | nerve_mask]["body_id"].values)

    def select_by_modality(self, modality: str) -> Set[int]:
        """Select neurons by modality (e.g., 'olfactory', 'visual')."""
        df = self.load_neurons()
        mask = df["modality"].str.lower().str.contains(modality.lower(), na=False)
        return set(df[mask]["body_id"].values)

    def select_by_type(self, type_pattern: str) -> Set[int]:
        """Select neurons by cell type pattern (regex)."""
        df = self.load_neurons()
        mask = df["cell_type"].str.contains(type_pattern, na=False, regex=True)
        return set(df[mask]["body_id"].values)

    def summary(self) -> str:
        """Return summary statistics."""
        if not self.is_available():
            return (
                f"Data not available at {self.db_path}.\n"
                f"Run: python scripts/download_banc.py"
            )

        db_name = "Frankenbrain" if self.use_frankenbrain else "BANC"
        df = self.load_neurons()
        conn = self.load_connectivity()

        lines = [
            f"{db_name}: {len(df):,} neurons, {len(conn):,} connection pairs",
            f"  Total synapses: {conn['weight'].sum():,.0f}",
            f"  DNs: {len(self.select_dns()):,}",
            f"  MNs (all motor): {len(self.select_mns()):,}",
            f"  Leg MNs: {len(self.select_leg_mns()):,}",
        ]

        neck = self.load_neck_bridge()
        if neck is not None:
            lines.append(f"  Neck bridge: {len(neck):,} cross-references")

        # Super-class breakdown
        sc = df["super_class"].value_counts()
        if len(sc) > 0:
            lines.append(f"  super_class: {dict(sc.head(10))}")

        # Region breakdown
        rg = df["region"].value_counts()
        if len(rg) > 0:
            lines.append(f"  region: {dict(rg.head(6))}")

        return "\n".join(lines)


# ============================================================================
# BANCVNCData — VNC subnetwork for the firing-rate / LIF model
# ============================================================================

@dataclass
class BANCVNCData:
    """VNC subnetwork extracted from BANC, ready for the firing-rate model.

    Provides the same interface that FiringRateVNCRunner._load_data() and
    _select_neurons() produce from MANC feather files:
        - neuron index arrays (dn_indices, mn_indices, premotor_indices)
        - body_id <-> index mappings
        - sparse weight matrices (W_exc, W_inh in CSR format)
        - neurotransmitter sign per neuron
        - DN type -> index mapping
        - MN metadata (body_id, segment, side, cell_type, leg, direction)
    """
    # Population sizes
    n_neurons: int = 0
    n_dn: int = 0
    n_mn: int = 0
    n_premotor: int = 0
    n_synapses: int = 0

    # Body ID <-> index
    bodyid_to_idx: Dict[int, int] = field(default_factory=dict)
    idx_to_bodyid: Dict[int, int] = field(default_factory=dict)

    # Population ID sets
    dn_ids: Set[int] = field(default_factory=set)
    mn_ids: Set[int] = field(default_factory=set)
    premotor_ids: Set[int] = field(default_factory=set)

    # Population index arrays (sorted, int32)
    dn_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    mn_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    premotor_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # Neurotransmitter sign per neuron (+1 exc, -1 inh)
    neuron_sign: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # Sparse weight matrices (CSR, shape n_neurons x n_neurons)
    # W_exc[i, j] >= 0: excitatory weight from j to i
    # W_inh[i, j] <= 0: inhibitory weight from j to i
    W_exc: object = None  # scipy.sparse.csr_matrix
    W_inh: object = None  # scipy.sparse.csr_matrix

    # DN type name -> list of model indices
    dn_type_to_indices: Dict[str, List[int]] = field(default_factory=dict)
    dn_type_to_body_ids: Dict[str, List[int]] = field(default_factory=dict)

    # DN soma side: model_idx -> "left"/"right" (for turn lateralization)
    dn_side: Dict[int, str] = field(default_factory=dict)

    # MN metadata: list of dicts with keys:
    #   body_id, segment, side, cell_type, leg, leg_idx, direction
    mn_body_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    mn_info: List[dict] = field(default_factory=list)
    mn_leg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    mn_direction: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


def load_banc_vnc(
    banc_dir: Path | str | None = None,
    min_mn_synapses: int = 3,
    include_all_dns: bool = True,
    exc_mult: float = 0.005,
    inh_mult: float = 0.005,
    inh_scale: float = 2.0,
    normalize_weights: bool = True,
    target_exc_sum: float = 0.6,
    verbose: bool = True,
) -> BANCVNCData:
    """Extract VNC premotor subnetwork from BANC for the firing-rate model.

    Mirrors the logic in FiringRateVNCRunner._load_data() / _select_neurons() /
    _build_weight_matrix(), but reads from BANC SQLite instead of MANC feather.

    Neuron selection:
        1. Descending neurons (super_class='descending' or flow='efferent')
        2. Leg motor neurons (cell_class contains 'leg_motor', or type-matched)
        3. Premotor interneurons: VNC intrinsic neurons with >= min_mn_synapses
           synapses onto any leg MN

    Weight matrix:
        W[i, j] = exc_mult * count    if neuron j is excitatory (ACh, Glu)
        W[i, j] = -inh_mult * inh_scale * count   if inhibitory (GABA, histamine)
        Optional row-sum normalization to target_exc_sum.

    Args:
        banc_dir: Path to directory containing banc_626_data.sqlite.
        min_mn_synapses: Minimum synapse count for premotor identification.
        include_all_dns: If True, include all DNs; if False, only those that
            project to the selected subnetwork.
        exc_mult: Excitatory weight multiplier.
        inh_mult: Inhibitory weight multiplier.
        inh_scale: Additional scaling for inhibitory weights.
        normalize_weights: If True, rescale so mean exc row sum = target_exc_sum.
        target_exc_sum: Target mean excitatory row sum.
        verbose: Print progress messages.

    Returns:
        BANCVNCData with all arrays and matrices populated.
    """
    from scipy.sparse import csr_matrix

    loader = BANCLoader(banc_dir=banc_dir)
    data = BANCVNCData()

    def _print(msg: str):
        if verbose:
            print(msg)

    _print("BANCVNCData: loading BANC connectome...")
    t_total = time()

    # ---- Load raw data from SQLite ----------------------------------------

    t0 = time()
    neurons = loader.load_neurons()
    _print(f"  Neurons: {len(neurons):,} rows ({time() - t0:.1f}s)")

    t0 = time()
    conn = loader.load_connectivity()
    _print(f"  Connectivity: {len(conn):,} edges ({time() - t0:.1f}s)")

    # ---- Select neuron populations ----------------------------------------

    _print("  Selecting VNC subnetwork populations...")
    t0 = time()

    # 1. Descending neurons (super_class='descending' or cell_type starts with DN)
    # Do NOT use flow='efferent' alone — motor neurons also have efferent flow.
    dn_mask = (
        (neurons["super_class"].str.lower() == "descending")
        | neurons["cell_type"].str.match(r"^DN[a-z]", na=False)
    )
    dn_df = neurons[dn_mask].copy()
    dn_ids = set(dn_df["body_id"].values)

    # 2. Leg motor neurons
    leg_mn_mask = (
        neurons["cell_class"].str.lower().str.contains("leg_motor", na=False)
    )
    # Fallback: cell_type pattern matching for known MN types
    type_mn_mask = neurons["cell_type"].str.contains(
        r"(?:Ti |Tr |Fe |Ta |ltm|Sternotrochanter|Tergopleural|"
        r"Pleural remotor|Sternal |Tergotr\.) *(?:flexor|extensor|"
        r"MN|motor|promotor|remotor|depressor|levator|adductor|rotator)",
        na=False, case=False, regex=True,
    )
    # Nerve-based identification (leg nerves only)
    nerve_mn_mask = (
        (neurons["super_class"].str.lower() == "motor")
        & neurons["nerve"].str.lower().isin(["proln", "mesoln", "metaln"])
    )
    mn_mask = leg_mn_mask | type_mn_mask | nerve_mn_mask
    mn_df = neurons[mn_mask].copy()
    mn_ids = set(mn_df["body_id"].values)

    # 3. Premotor interneurons: VNC intrinsic neurons with significant
    #    synaptic output to leg MNs
    pre_to_mn = conn[conn["post_id"].isin(mn_ids)]
    pre_to_mn_heavy = pre_to_mn[pre_to_mn["weight"] >= min_mn_synapses]
    candidate_premotor = set(pre_to_mn_heavy["pre_id"].values) - mn_ids

    # Filter to VNC intrinsic neurons (region=ventral_nerve_cord, not DN/MN)
    vnc_intrinsic_mask = (
        neurons["region"].str.lower().str.contains(
            "vnc|ventral_nerve_cord", na=False
        )
        & ~neurons["body_id"].isin(dn_ids | mn_ids)
    )
    # Also accept neurons whose super_class is not DN/MN/sensory but are in VNC
    intrinsic_fallback = (
        ~neurons["super_class"].str.lower().isin([
            "descending", "motor", "sensory", "ascending",
            "visual_projection", "visual_centrifugal", "endocrine",
        ])
        & neurons["region"].str.lower().str.contains(
            "vnc|ventral_nerve_cord|t1|t2|t3", na=False
        )
    )
    intrinsic_ids = set(
        neurons[vnc_intrinsic_mask | intrinsic_fallback]["body_id"].values
    )
    premotor_ids = candidate_premotor & intrinsic_ids

    _print(f"  Populations: {len(dn_ids)} DN, {len(mn_ids)} leg MN, "
           f"{len(premotor_ids)} premotor ({time() - t0:.1f}s)")

    # ---- Combine neuron set -----------------------------------------------

    if include_all_dns:
        all_ids = dn_ids | mn_ids | premotor_ids
    else:
        # Only DNs that project to the subnetwork
        dn_projecting = candidate_premotor & dn_ids
        all_ids = dn_projecting | mn_ids | premotor_ids

    all_ids_sorted = sorted(all_ids)
    bodyid_to_idx = {int(bid): i for i, bid in enumerate(all_ids_sorted)}
    idx_to_bodyid = {i: int(bid) for bid, i in bodyid_to_idx.items()}

    N = len(all_ids_sorted)
    data.n_neurons = N
    data.n_dn = len(dn_ids & all_ids)
    data.n_mn = len(mn_ids)
    data.n_premotor = len(premotor_ids)
    data.bodyid_to_idx = bodyid_to_idx
    data.idx_to_bodyid = idx_to_bodyid
    data.dn_ids = dn_ids & all_ids
    data.mn_ids = mn_ids
    data.premotor_ids = premotor_ids

    data.dn_indices = np.array(
        sorted(bodyid_to_idx[bid] for bid in data.dn_ids), dtype=np.int32
    )
    data.mn_indices = np.array(
        sorted(bodyid_to_idx[bid] for bid in mn_ids if bid in bodyid_to_idx),
        dtype=np.int32,
    )
    data.premotor_indices = np.array(
        sorted(bodyid_to_idx[bid] for bid in premotor_ids if bid in bodyid_to_idx),
        dtype=np.int32,
    )

    _print(f"  Selected: {data.n_dn} DN + {data.n_mn} MN + "
           f"{data.n_premotor} premotor = {N} neurons")

    # ---- Filter connectivity to subnetwork --------------------------------

    _print(f"  Filtering {len(conn):,} edges to {N} neurons...")
    t0 = time()
    all_ids_set = set(all_ids)
    pre_in = conn["pre_id"].isin(all_ids_set)
    post_in = conn["post_id"].isin(all_ids_set)
    subnet_conn = conn[pre_in & post_in].copy()
    data.n_synapses = len(subnet_conn)
    _print(f"  Filtered to {data.n_synapses:,} intra-VNC edges ({time() - t0:.1f}s)")

    # ---- Neurotransmitter sign per neuron ---------------------------------

    neuron_sign = np.ones(N, dtype=np.float32)
    for _, row in neurons.iterrows():
        bid = int(row["body_id"])
        if bid in bodyid_to_idx:
            nt = str(row.get("nt", "unclear")).lower().strip()
            neuron_sign[bodyid_to_idx[bid]] = NT_SIGN.get(nt, +1.0)
    data.neuron_sign = neuron_sign

    # ---- Build weight matrix ----------------------------------------------

    _print("  Building weight matrix...")
    t0 = time()

    W = np.zeros((N, N), dtype=np.float32)

    pre_bids = subnet_conn["pre_id"].values
    post_bids = subnet_conn["post_id"].values
    weights = subnet_conn["weight"].values.astype(np.float32)

    # Map body_id -> index using pandas Series for large sparse IDs
    bid_series = pd.Series(bodyid_to_idx)
    j_indices = bid_series.reindex(pre_bids).values    # presynaptic -> column
    i_indices = bid_series.reindex(post_bids).values   # postsynaptic -> row

    # Filter out unmapped IDs (NaN -> -1)
    j_valid = ~np.isnan(j_indices.astype(float))
    i_valid = ~np.isnan(i_indices.astype(float))
    valid = j_valid & i_valid
    j_indices = j_indices[valid].astype(np.int32)
    i_indices = i_indices[valid].astype(np.int32)
    weights = weights[valid]

    signs = neuron_sign[j_indices]

    eff_weights = np.where(
        signs >= 0,
        exc_mult * weights,
        -inh_mult * inh_scale * weights,
    )

    np.add.at(W, (i_indices, j_indices), eff_weights)

    n_exc = int((signs >= 0).sum())
    n_inh = int((signs < 0).sum())
    _print(f"  Raw weight matrix: {N}x{N}, {np.count_nonzero(W):,} nonzero, "
           f"{n_exc:,} exc + {n_inh:,} inh edges ({time() - t0:.1f}s)")

    # Weight normalization
    if normalize_weights:
        exc_row_sums = np.where(W > 0, W, 0).sum(axis=1)
        mean_exc_row = float(exc_row_sums.mean())
        if mean_exc_row > 1e-8:
            scale = target_exc_sum / mean_exc_row
            W *= scale
            new_exc = np.where(W > 0, W, 0).sum(axis=1).mean()
            new_inh = np.where(W < 0, W, 0).sum(axis=1).mean()
            _print(f"  Weight normalization: scale={scale:.6f}, "
                   f"mean exc row sum={new_exc:.4f}, "
                   f"mean inh row sum={new_inh:.4f}")

    data.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
    data.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))

    # ---- DN type map ------------------------------------------------------

    _print("  Building DN type map...")
    dn_type_to_indices: Dict[str, List[int]] = {}
    dn_type_to_body_ids: Dict[str, List[int]] = {}

    for _, row in dn_df.iterrows():
        bid = int(row["body_id"])
        if bid not in bodyid_to_idx:
            continue
        idx = bodyid_to_idx[bid]
        ct = str(row.get("cell_type", ""))
        if ct:
            if ct not in dn_type_to_indices:
                dn_type_to_indices[ct] = []
                dn_type_to_body_ids[ct] = []
            if idx not in dn_type_to_indices[ct]:
                dn_type_to_indices[ct].append(idx)
                dn_type_to_body_ids[ct].append(bid)

    data.dn_type_to_indices = dn_type_to_indices
    data.dn_type_to_body_ids = dn_type_to_body_ids

    # DN soma side for turn lateralization
    dn_side: Dict[int, str] = {}
    for _, row in dn_df.iterrows():
        bid = int(row["body_id"])
        if bid in bodyid_to_idx:
            side = str(row.get("side", "")).lower().strip()
            if side in ("left", "right"):
                dn_side[bodyid_to_idx[bid]] = side
    data.dn_side = dn_side

    n_types = len(dn_type_to_indices)
    n_dns_mapped = sum(len(v) for v in dn_type_to_indices.values())
    n_sided = sum(1 for s in dn_side.values() if s)
    _print(f"  DN type map: {n_types} types, {n_dns_mapped} neurons, {n_sided} with side info")

    # ---- MN metadata ------------------------------------------------------

    _print("  Building MN metadata...")
    mn_sorted = sorted(mn_ids)
    data.mn_body_ids = np.array(mn_sorted, dtype=np.int64)

    mn_leg = np.zeros(data.n_mn, dtype=np.int32)
    mn_direction = np.zeros(data.n_mn, dtype=np.float32)
    mn_info_list: List[dict] = []

    # Try to load mn_joint_mapping.json for MN metadata (MANC body IDs).
    # For BANC, we primarily use cell_type and nerve fields.
    mn_joint_map = {}
    mn_joint_path = DATA_DIR / "mn_joint_mapping.json"
    if mn_joint_path.exists():
        with open(mn_joint_path) as f:
            mn_joint_map = json.load(f)

    for i, bid in enumerate(mn_sorted):
        row = mn_df[mn_df["body_id"] == bid]
        if len(row) > 0:
            row = row.iloc[0]
            ct = str(row.get("cell_type", ""))
            side = str(row.get("side", "left"))
            nerve = str(row.get("nerve", ""))
            sub_class = str(row.get("cell_sub_class", ""))

            # Determine segment: cell_sub_class is the most reliable source
            # for BANC leg MNs (works even when nerve is None)
            segment = SUB_CLASS_TO_SEGMENT.get(sub_class, "")
            if not segment:
                # Fallback to nerve-based mapping (expanded for BANC verbose names)
                segment = NERVE_TO_SEGMENT.get(nerve, "T1")

            # Determine leg from segment + side
            side_short = side[0].upper() if side else "L"
            leg = SEGMENT_SIDE_TO_LEG.get(
                (segment, side), SEGMENT_SIDE_TO_LEG.get(
                    (segment, side_short), "LF"
                )
            )
            leg_idx = LEG_ORDER.index(leg) if leg in LEG_ORDER else 0

            # Infer direction from cell type name
            lower = ct.lower()
            if any(k in lower for k in ["extensor", "levator", "remotor",
                                         "retract", "posterior rotator"]):
                direction = 1.0
            elif any(k in lower for k in ["flexor", "depressor", "promotor",
                                           "protract", "anterior rotator",
                                           "adductor"]):
                direction = -1.0
            else:
                direction = 0.0

            # Cross-check with MANC mapping if the neuron has a manc_match
            manc_match = str(row.get("manc_match", ""))
            if manc_match and manc_match in mn_joint_map:
                entry = mn_joint_map[manc_match]
                leg = entry.get("leg", leg)
                leg_idx = LEG_ORDER.index(leg) if leg in LEG_ORDER else leg_idx
                direction = float(entry.get("direction", direction))
        else:
            ct, segment, side, leg, leg_idx, direction = (
                "unknown", "T1", "left", "LF", 0, 0.0
            )

        mn_leg[i] = leg_idx
        mn_direction[i] = direction
        mn_info_list.append({
            "body_id": int(bid),
            "segment": segment,
            "side": side,
            "cell_type": ct,
            "leg": leg,
            "leg_idx": leg_idx,
            "direction": direction,
        })

    data.mn_leg = mn_leg
    data.mn_direction = mn_direction
    data.mn_info = mn_info_list

    n_flex = int((mn_direction < 0).sum())
    n_ext = int((mn_direction > 0).sum())
    n_other = data.n_mn - n_flex - n_ext
    _print(f"  MN metadata: {n_ext} extensors, {n_flex} flexors, {n_other} ambiguous")

    # Per-leg distribution
    leg_counts = {}
    for info in mn_info_list:
        leg = info["leg"]
        leg_counts[leg] = leg_counts.get(leg, 0) + 1
    _print(f"  MN per leg: {leg_counts}")

    build_time = time() - t_total
    _print(f"  BANC VNC data ready in {build_time:.1f}s: "
           f"{N} neurons, {data.n_synapses:,} connections")

    return data
