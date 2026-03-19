"""
BANC / Frankenbrain connectome data loader.

Data source: Harvard Dataverse SQLite files (no authentication needed).
Two databases:
    banc_data.sqlite         — 160K neurons, full BANC connectome (single female fly)
    frankenbrain_v1.1_data.sqlite — FAFB brain + MANC VNC bridged via neck_bridge table

Schema:
    meta           — neuron annotations (cell_type, modality, super_class, ...)
    edgelist_simple — connectivity (pre, post, count)
    neck_bridge    — BANC↔MANC ID mapping (Frankenbrain only)

Usage:
    from bridge.banc_loader import BANCLoader
    loader = BANCLoader()                           # uses banc_data.sqlite
    loader = BANCLoader(use_frankenbrain=True)       # uses frankenbrain
    neurons = loader.load_neurons()
    connectivity = loader.load_connectivity()
    dns = loader.select_dns()
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BANC_DIR = ROOT / "data" / "banc"


class BANCLoader:
    """Loader for BANC/Frankenbrain SQLite connectome data.

    Reads from Harvard Dataverse SQLite files. Provides standardized
    DataFrame interface matching the rest of the bridge pipeline.
    """

    def __init__(
        self,
        banc_dir: Path | str | None = None,
        use_frankenbrain: bool = False,
    ):
        self.banc_dir = Path(banc_dir) if banc_dir else BANC_DIR
        self.use_frankenbrain = use_frankenbrain
        self._neurons_cache = None
        self._connectivity_cache = None

    @property
    def db_path(self) -> Path:
        if self.use_frankenbrain:
            return self.banc_dir / "frankenbrain_v1.1_data.sqlite"
        return self.banc_dir / "banc_data.sqlite"

    def is_available(self) -> bool:
        """Check if the SQLite database exists."""
        return self.db_path.exists()

    def _connect(self) -> sqlite3.Connection:
        if not self.is_available():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                "Run: python scripts/download_banc.py"
            )
        return sqlite3.connect(str(self.db_path))

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        con = self._connect()
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        con.close()
        return tables["name"].tolist()

    def load_neurons(self) -> pd.DataFrame:
        """Load neuron annotations from the `meta` table.

        Returns DataFrame with standardized columns:
            body_id: int64
            cell_type: str
            super_class: str
            modality: str
            soma_side: str
            region: str
        """
        if self._neurons_cache is not None:
            return self._neurons_cache

        con = self._connect()
        df = pd.read_sql("SELECT * FROM meta", con)
        con.close()

        # Standardize column names to our convention
        col_map = {}
        for target, candidates in [
            ("body_id", ["id", "root_id", "pt_root_id", "bodyId", "body_id", "segment_id"]),
            ("cell_type", ["cell_type", "type", "cellType"]),
            ("super_class", ["super_class", "superclass", "cell_class", "class"]),
            ("modality", ["modality"]),
            ("soma_side", ["soma_side", "somaSide", "side"]),
            ("region", ["region", "neuropil", "compartment"]),
        ]:
            for c in candidates:
                if c in df.columns and target not in col_map.values():
                    col_map[c] = target
                    break

        df = df.rename(columns=col_map)

        # If body_id still missing, use index or first column
        if "body_id" not in df.columns:
            if df.index.name and "id" in str(df.index.name).lower():
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "body_id"})
            else:
                # Use first integer column as body_id
                for c in df.columns:
                    if df[c].dtype in [np.int64, np.int32, int]:
                        df = df.rename(columns={c: "body_id"})
                        break

        # Fill missing columns with defaults
        for col, default in [("cell_type", ""), ("super_class", ""),
                             ("modality", ""), ("soma_side", ""), ("region", "")]:
            if col not in df.columns:
                df[col] = default

        # Ensure string columns are strings (SQLite may return None)
        for col in ["cell_type", "super_class", "modality", "soma_side", "region"]:
            df[col] = df[col].fillna("").astype(str)

        df["body_id"] = df["body_id"].astype(np.int64)
        self._neurons_cache = df
        return df

    def load_connectivity(self) -> pd.DataFrame:
        """Load connectivity from the `edgelist_simple` table.

        Returns DataFrame with standardized columns:
            pre_id: int64
            post_id: int64
            weight: int (synapse count)
        """
        if self._connectivity_cache is not None:
            return self._connectivity_cache

        con = self._connect()
        df = pd.read_sql("SELECT * FROM edgelist_simple", con)
        con.close()

        # Standardize: edgelist_simple uses (pre, post, count)
        col_map = {}
        for target, candidates in [
            ("pre_id", ["pre", "pre_id", "pre_pt_root_id", "body_pre"]),
            ("post_id", ["post", "post_id", "post_pt_root_id", "body_post"]),
            ("weight", ["count", "weight", "syn_count", "n_synapses"]),
        ]:
            for c in candidates:
                if c in df.columns and target not in col_map.values():
                    col_map[c] = target
                    break

        df = df.rename(columns=col_map)
        df["pre_id"] = df["pre_id"].astype(np.int64)
        df["post_id"] = df["post_id"].astype(np.int64)

        self._connectivity_cache = df
        return df

    def load_neck_bridge(self) -> pd.DataFrame | None:
        """Load neck_bridge table (Frankenbrain only).

        Maps BANC brain IDs ↔ MANC VNC IDs across the neck connective.
        Returns None if not available.
        """
        if not self.is_available():
            return None

        con = self._connect()
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        if "neck_bridge" not in tables["name"].values:
            con.close()
            return None

        df = pd.read_sql("SELECT * FROM neck_bridge", con)
        con.close()
        return df

    def select_brain_neurons(self) -> set[int]:
        """Select brain neurons (excluding VNC)."""
        df = self.load_neurons()
        mask = df["region"].str.lower().isin(["brain", "central_brain", "optic_lobe", ""])
        # Also try super_class for brain-specific types
        mask = mask | df["super_class"].str.lower().isin(["central", "optic", "sensory"])
        return set(df[mask]["body_id"].values)

    def select_vnc_neurons(self) -> set[int]:
        """Select VNC neurons."""
        df = self.load_neurons()
        mask = df["region"].str.lower().str.contains("vnc|t1|t2|t3|ventral", na=False)
        mask = mask | df["super_class"].str.lower().str.contains("vnc|motor", na=False)
        return set(df[mask]["body_id"].values)

    def select_dns(self) -> set[int]:
        """Select descending neurons (brain -> VNC)."""
        df = self.load_neurons()
        mask = df["super_class"].str.lower().str.contains("descend", na=False)
        # Also match cell_type starting with DN
        mask = mask | df["cell_type"].str.match(r"^DN[a-z]", na=False)
        return set(df[mask]["body_id"].values)

    def select_mns(self) -> set[int]:
        """Select motor neurons (VNC -> muscles)."""
        df = self.load_neurons()
        mask = df["super_class"].str.lower().str.contains("motor", na=False)
        return set(df[mask]["body_id"].values)

    def select_by_modality(self, modality: str) -> set[int]:
        """Select neurons by modality (e.g., 'olfactory', 'visual', 'mechanosensory')."""
        df = self.load_neurons()
        mask = df["modality"].str.lower().str.contains(modality.lower(), na=False)
        return set(df[mask]["body_id"].values)

    def select_by_type(self, type_pattern: str) -> set[int]:
        """Select neurons by cell type pattern (regex)."""
        df = self.load_neurons()
        mask = df["cell_type"].str.contains(type_pattern, na=False, regex=True)
        return set(df[mask]["body_id"].values)

    def summary(self) -> str:
        """Return summary statistics."""
        if not self.is_available():
            return f"Data not available at {self.db_path}. Run: python scripts/download_banc.py"

        db_name = "Frankenbrain" if self.use_frankenbrain else "BANC"
        df = self.load_neurons()
        con = self.load_connectivity()

        lines = [
            f"{db_name}: {len(df)} neurons, {len(con)} connection pairs",
            f"  Total synapses: {con['weight'].sum():,.0f}",
            f"  DNs: {len(self.select_dns())}",
            f"  MNs: {len(self.select_mns())}",
        ]

        neck = self.load_neck_bridge()
        if neck is not None:
            lines.append(f"  Neck bridge: {len(neck)} cross-references")

        # Modality breakdown
        modalities = df["modality"].value_counts()
        if len(modalities) > 1:
            lines.append(f"  Modalities: {dict(modalities.head(8))}")

        return "\n".join(lines)
