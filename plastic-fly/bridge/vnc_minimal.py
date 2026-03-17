"""
Minimal VNC locomotion circuit extracted from MANC connectome.

Extracts the minimal circuit needed for locomotion by tracing backward
from motor neurons to find premotor interneurons, ranked by synaptic
drive to MNs.  Only the top-N premotor neurons (by total MN drive) are
kept, yielding ~1,000-2,000 LIF neurons instead of 13,101.

Same VNCInput/VNCOutput interface as Brian2VNCRunner.

Usage:
    from bridge.vnc_minimal import MinimalVNCRunner, MinimalVNCConfig
    cfg = MinimalVNCConfig(n_premotor=500)
    runner = MinimalVNCRunner(cfg=cfg)
    output = runner.step(VNCInput(group_rates={...}), sim_ms=20.0)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from time import time

from bridge.vnc_connectome import (
    VNCConfig, VNCInput, VNCOutput,
    NT_SIGN, OUR_READOUT_DN_TYPES,
    MN_TYPE_TO_JOINT_GROUP, JOINT_GROUP_TO_DOF,
    SEGMENT_SIDE_TO_LEG, LEG_ORDER,
)


@dataclass
class MinimalVNCConfig(VNCConfig):
    """Configuration for minimal VNC locomotion circuit.

    Inherits all LIF/synapse/rhythm params from VNCConfig.
    Adds extraction parameters.
    """
    n_premotor: int = 500
    min_syn_weight: int = 1

    force_dn_types: set = field(default_factory=lambda: {
        "DNa01", "DNa02", "DNb05", "DNp44", "DNg100",
        "DNb01", "DNb08", "DNb09",
    })


class MinimalVNCRunner:
    """Minimal MANC VNC circuit: premotor interneurons + motor neurons.

    Extraction algorithm:
        1. Select all thoracic leg MNs (~500)
        2. Find all intrinsic interneurons that synapse onto MNs
        3. Rank by total synaptic drive (sum of weights to MNs)
        4. Keep top-N premotor interneurons
        5. DNs are PoissonGroup inputs (not LIF neurons)
        6. Filter connectivity to selected set only
    """

    def __init__(
        self,
        cfg: MinimalVNCConfig | None = None,
        warmup: bool = True,
        shuffle_seed: int | None = None,
    ):
        self.cfg = cfg or MinimalVNCConfig()
        self.shuffle_seed = shuffle_seed
        self._check_data_files()

        label = ("SHUFFLED (seed=%d)" % shuffle_seed
                 if shuffle_seed is not None else "real")
        print(f"MinimalVNCRunner: extracting minimal circuit ({label})...")
        t0 = time()

        self._load_data()
        self._extract_circuit()
        self._build_dn_group_mapping()
        self._build_mn_rhythm_mapping()
        self._build_network()

        build_time = time() - t0
        self._report_stats(build_time)

        if warmup and self.cfg.warmup_ms > 0:
            self._warmup()

    # ---- Data loading -------------------------------------------------------

    def _check_data_files(self):
        missing = []
        for p in [self.cfg.annotations_path, self.cfg.neurotransmitters_path,
                   self.cfg.connectivity_path]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "MANC data files not found:\n"
                + "\n".join(f"  {m}" for m in missing)
            )

    def _load_data(self):
        import pandas as pd
        import pyarrow.feather as feather

        t0 = time()
        self._ann = pd.DataFrame(
            feather.read_feather(str(self.cfg.annotations_path)))

        nt_df = pd.DataFrame(
            feather.read_feather(str(self.cfg.neurotransmitters_path)))
        nt_unique = nt_df.drop_duplicates(subset="body", keep="first")
        self._nt_map = dict(zip(
            nt_unique["body"].values, nt_unique["consensus_nt"].values))

        t1 = time()
        self._conn = pd.DataFrame(
            feather.read_feather(str(self.cfg.connectivity_path)))
        print(f"  Data loaded in {time()-t0:.1f}s "
              f"(connectivity: {len(self._conn):,} rows)")

    # ---- Circuit extraction -------------------------------------------------

    def _extract_circuit(self):
        """Extract minimal locomotion circuit by MN-drive ranking."""
        import pandas as pd

        ann = self._ann
        conn = self._conn
        cfg = self.cfg

        # --- Motor neurons ---
        mn_mask = (
            (ann["superclass"] == "vnc_motor")
            & ann["somaNeuromere"].isin(["T1", "T2", "T3"])
        )
        self._mn_df = ann[mn_mask].copy()
        mn_ids = set(self._mn_df["bodyId"].astype(int))

        # --- Intrinsic interneurons ---
        int_mask = (
            (ann["superclass"] == "vnc_intrinsic")
            & ann["somaNeuromere"].isin(["T1", "T2", "T3"])
        )
        all_int_ids = set(ann.loc[int_mask, "bodyId"].astype(int))

        # --- Descending neurons ---
        dn_mask = ann["superclass"] == "descending_neuron"
        self._dn_df = ann[dn_mask].copy()
        dn_ids = set(self._dn_df["bodyId"].astype(int))

        # --- Find premotor interneurons and rank by MN drive ---
        pre_to_mn = conn[
            conn["body_post"].isin(mn_ids)
            & conn["body_pre"].isin(all_int_ids)
        ]
        drive = (pre_to_mn.groupby("body_pre")["weight"]
                 .sum().sort_values(ascending=False))

        top_int_ids = set(drive.head(cfg.n_premotor).index.astype(int))
        self._total_mn_drive = float(drive.sum())
        self._captured_mn_drive = float(drive.head(cfg.n_premotor).sum())

        # --- LIF neuron set: premotor interneurons + MNs ---
        lif_ids = top_int_ids | mn_ids
        lif_sorted = sorted(lif_ids)
        self._bodyid_to_idx = {bid: i for i, bid in enumerate(lif_sorted)}

        self.n_mn = len(mn_ids)
        self.n_intrinsic = len(top_int_ids)
        self.n_neurons = len(lif_ids)

        # MN indices
        mn_sorted = sorted(mn_ids)
        self._mn_brian_idx = np.array(
            [self._bodyid_to_idx[bid] for bid in mn_sorted], dtype=int)
        self._mn_body_ids = np.array(mn_sorted, dtype=np.int64)
        self._mn_ids = mn_ids
        self._dn_ids = dn_ids

        # --- Filter connectivity to selected LIF neurons ---
        lif_set = set(lif_sorted)
        internal_mask = (
            conn["body_pre"].isin(lif_set)
            & conn["body_post"].isin(lif_set)
        )
        if cfg.min_syn_weight > 1:
            internal_mask &= conn["weight"] >= cfg.min_syn_weight
        self._internal_conn = conn[internal_mask].copy()

        # --- DN -> LIF connections ---
        dn_to_lif_mask = (
            conn["body_pre"].isin(dn_ids)
            & conn["body_post"].isin(lif_set)
        )
        if cfg.min_syn_weight > 1:
            dn_to_lif_mask &= conn["weight"] >= cfg.min_syn_weight
        self._dn_to_lif_conn = conn[dn_to_lif_mask].copy()

        # Active DNs: those with at least one connection to selected LIF set
        active_dn_ids = set(self._dn_to_lif_conn["body_pre"].unique())

        # Force-include DNs from specified types
        for dn_type in cfg.force_dn_types:
            for _, row in self._dn_df.iterrows():
                bid = int(row["bodyId"])
                rtype = row.get("type", "")
                if isinstance(rtype, str) and rtype.strip() == dn_type:
                    active_dn_ids.add(bid)

        self._active_dn_ids = sorted(active_dn_ids & dn_ids)
        self.n_dn = len(self._active_dn_ids)

        # DN input index mapping
        self._dn_bodyid_to_input_idx = {
            int(bid): i for i, bid in enumerate(self._active_dn_ids)
        }
        self._dn_brian_idx_for_input = []
        for bid in self._active_dn_ids:
            if bid in self._bodyid_to_idx:
                self._dn_brian_idx_for_input.append(self._bodyid_to_idx[bid])
            else:
                self._dn_brian_idx_for_input.append(-1)

        # MN metadata
        self._build_mn_metadata()

        # Coverage stats
        mapped_path = cfg.mn_joint_mapping_path
        if mapped_path.exists():
            with open(mapped_path) as f:
                mapped_mn_ids = set(int(k) for k in json.load(f).keys())
            reached = set(pre_to_mn[
                pre_to_mn["body_pre"].isin(top_int_ids)
            ]["body_post"].unique())
            self._mapped_mn_coverage = len(reached & mapped_mn_ids)
            self._total_mapped_mns = len(mapped_mn_ids)
        else:
            self._mapped_mn_coverage = 0
            self._total_mapped_mns = 0

        self.n_internal_synapses = len(self._internal_conn)
        self.n_dn_synapses = len(self._dn_to_lif_conn)
        self.n_synapses = self.n_internal_synapses + self.n_dn_synapses

        print(f"  Extracted: {self.n_neurons} LIF neurons "
              f"({self.n_intrinsic} premotor + {self.n_mn} MN), "
              f"{self.n_dn} active DNs")

    def _build_mn_metadata(self):
        import pandas as pd

        self.mn_info = []
        self.mn_body_ids = self._mn_body_ids.copy()

        for _, row in self._mn_df.iterrows():
            bid = int(row["bodyId"])
            seg = str(row["somaNeuromere"])
            side = (str(row["somaSide"])
                    if pd.notna(row.get("somaSide")) else "L")
            mn_type = (str(row["type"])
                       if pd.notna(row.get("type")) else "unknown")
            jg_entry = MN_TYPE_TO_JOINT_GROUP.get(mn_type, None)
            if jg_entry is not None:
                joint_group, sign = jg_entry
            else:
                joint_group = "unmapped"
                sign = 0
            self.mn_info.append((bid, seg, side, mn_type, joint_group, sign))

    # ---- DN group mapping ---------------------------------------------------

    def _build_dn_group_mapping(self):
        import pandas as pd

        decoder_path = self.cfg.decoder_groups_path
        if not decoder_path.exists():
            from bridge.vnc_connectome import DATA_DIR
            for alt in ["decoder_groups_v5_steering.json",
                        "decoder_groups_v4_looming.json",
                        "decoder_groups_v3.json"]:
                alt_path = DATA_DIR / alt
                if alt_path.exists():
                    decoder_path = alt_path
                    break

        # Build DN type -> MANC body ID mapping
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

        # Readout DN MANC IDs
        readout_manc_ids = set()
        for dn_type in OUR_READOUT_DN_TYPES:
            for bid in dn_type_to_manc.get(dn_type, []):
                readout_manc_ids.add(bid)
        self._readout_manc_ids = sorted(readout_manc_ids)

        # DN group types (from Brian2VNCRunner)
        self._group_dn_types = {
            "forward": {"DNa01", "DNa02", "DNg100", "DNb02", "DNp30",
                        "DNp02", "DNp10", "DNp06", "DNp05"},
            "turn_left": {"DNg11", "DNg29", "DNg33", "DNg35", "DNg47",
                          "DNpe006", "DNge039", "DNge044"},
            "turn_right": {"DNg11", "DNg29", "DNg33", "DNg35", "DNg47",
                           "DNpe006", "DNge039", "DNge044"},
            "rhythm": {"DNb01", "DNb08", "DNb09", "DNg100", "DNd02",
                       "DNd03"},
            "stance": {"DNp44", "DNp35", "DNp42", "DNp23", "DNp25",
                       "DNp19"},
        }

        # Lateralize turn groups by somaSide
        import pandas as pd
        dn_side_map = {}
        for _, row in self._dn_df.iterrows():
            bid = int(row["bodyId"])
            side = (str(row.get("somaSide", ""))
                    if pd.notna(row.get("somaSide")) else "")
            dn_side_map[bid] = side

        active_dn_set = set(self._active_dn_ids)
        self._group_to_manc_dn_ids = {}
        for group_name, types in self._group_dn_types.items():
            group_ids = []
            for dt in types:
                for bid in dn_type_to_manc.get(dt, []):
                    if bid not in active_dn_set:
                        continue
                    if (group_name == "turn_left"
                            and dn_side_map.get(bid) == "R"):
                        continue
                    if (group_name == "turn_right"
                            and dn_side_map.get(bid) == "L"):
                        continue
                    group_ids.append(bid)
            self._group_to_manc_dn_ids[group_name] = sorted(set(group_ids))

        assigned = set()
        for ids in self._group_to_manc_dn_ids.values():
            assigned.update(ids)
        self._unassigned_readout_dns = [
            bid for bid in self._readout_manc_ids
            if bid not in assigned and bid in active_dn_set
        ]

        n_assigned = len(assigned)
        print(f"  DN mapping: {n_assigned} group-assigned, "
              f"{len(self._unassigned_readout_dns)} mean-rate")

    # ---- Rhythm mapping -----------------------------------------------------

    def _build_mn_rhythm_mapping(self):
        mapping_path = self.cfg.mn_joint_mapping_path
        if not mapping_path.exists():
            self._rhythm_map = {}
            return

        with open(mapping_path) as f:
            mn_map = json.load(f)

        self._rhythm_map = {}
        for bid_str, entry in mn_map.items():
            bid = int(bid_str)
            if bid not in self._mn_ids:
                continue
            leg = entry.get("leg", "LF")
            if leg not in LEG_ORDER:
                continue
            leg_idx = LEG_ORDER.index(leg)
            direction = float(entry.get("direction", 0.0))
            if direction == 0:
                continue
            phase_idx = 0 if direction > 0 else 1
            self._rhythm_map[bid] = 2 * leg_idx + phase_idx

        print(f"  Rhythm mapping: {len(self._rhythm_map)} MNs -> "
              f"12 rhythm units")

    # ---- E/I sign -----------------------------------------------------------

    def _get_nt_sign(self, body_id: int) -> float:
        nt = self._nt_map.get(int(body_id), "unclear")
        if isinstance(nt, float) or nt is None:
            return 1.0
        return NT_SIGN.get(str(nt).lower().strip(), 1.0)

    # ---- Brian2 network build -----------------------------------------------

    def _build_network(self):
        from brian2 import (
            NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
            mV, ms, Hz, second,
        )

        cfg = self.cfg
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

        eqs = """
            dv/dt = (v_0 - v + g - w_a + I_ton) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau               : volt (unless refractory)
            dw_a/dt = -w_a / tau_a         : volt
            rfc                            : second
        """

        self.neurons = NeuronGroup(
            N=self.n_neurons, model=eqs, method="euler",
            threshold="v > v_th",
            reset="v = v_rst; g = 0 * mV; w_a += b_a",
            refractory="rfc", name="vnc_min_neurons", namespace=params,
        )
        self.neurons.v = params["v_0"]
        self.neurons.g = 0
        self.neurons.w_a = 0 * mV
        self.neurons.rfc = params["t_rfc"]

        # --- Internal synapses (premotor <-> MN) ---
        t0 = time()
        pre_idx = np.array([
            self._bodyid_to_idx[bid]
            for bid in self._internal_conn["body_pre"].values
        ], dtype=int)
        post_idx = np.array([
            self._bodyid_to_idx[bid]
            for bid in self._internal_conn["body_post"].values
        ], dtype=int)
        weights_raw = self._internal_conn["weight"].values.astype(float)

        if self.shuffle_seed is not None:
            rng = np.random.RandomState(self.shuffle_seed)
            post_idx = rng.permutation(post_idx)

        nt_signs = np.array([
            self._get_nt_sign(bid)
            for bid in self._internal_conn["body_pre"].values
        ], dtype=float)

        self.synapses = Synapses(
            self.neurons, self.neurons, "w : volt",
            on_pre="g += w", delay=params["t_dly"], name="vnc_min_syn",
        )
        self.synapses.connect(i=pre_idx, j=post_idx)
        inh_mask = (nt_signs < 0).astype(float)
        scale = np.ones_like(nt_signs) + inh_mask * (cfg.inh_scale - 1.0)
        self.synapses.w = weights_raw * nt_signs * scale * params["w_syn"]
        n_inh = int(inh_mask.sum())
        n_exc = len(nt_signs) - n_inh
        print(f"  Internal synapses: {n_exc:,} exc, {n_inh:,} inh "
              f"({time()-t0:.1f}s)")

        # --- DN input via PoissonGroup ---
        n_dn = len(self._active_dn_ids)
        self.input_group = PoissonGroup(
            n_dn, rates=np.zeros(n_dn) * Hz, name="dn_min_input"
        )

        # DN -> LIF synapses
        dn_pre_idx = []
        dn_post_idx = []
        dn_weights = []
        dn_nt_signs = []
        for _, row in self._dn_to_lif_conn.iterrows():
            pre_bid = int(row["body_pre"])
            post_bid = int(row["body_post"])
            if pre_bid not in self._dn_bodyid_to_input_idx:
                continue
            if post_bid not in self._bodyid_to_idx:
                continue
            dn_pre_idx.append(self._dn_bodyid_to_input_idx[pre_bid])
            dn_post_idx.append(self._bodyid_to_idx[post_bid])
            dn_weights.append(float(row["weight"]))
            dn_nt_signs.append(self._get_nt_sign(pre_bid))

        if dn_pre_idx:
            dn_pre_idx = np.array(dn_pre_idx, dtype=int)
            dn_post_idx = np.array(dn_post_idx, dtype=int)
            dn_weights = np.array(dn_weights, dtype=float)
            dn_nt_signs = np.array(dn_nt_signs, dtype=float)

            self.input_syn = Synapses(
                self.input_group, self.neurons, "w : volt",
                on_pre="g += w", name="dn_min_input_syn",
            )
            self.input_syn.connect(i=dn_pre_idx, j=dn_post_idx)
            self.input_syn.w = (dn_weights * dn_nt_signs
                                * params["w_syn"] * cfg.w_input_scale)
            print(f"  DN->LIF synapses: {len(dn_pre_idx):,}")
        else:
            self.input_syn = Synapses(
                self.input_group, self.neurons, "w : volt",
                on_pre="g += w", name="dn_min_input_syn",
            )
            self.input_syn.connect(i=[], j=[])
            print("  WARNING: No DN->LIF connections found")

        # --- Background noise ---
        if cfg.bg_rate_hz > 0:
            self.bg_input = PoissonGroup(
                self.n_neurons,
                rates=np.ones(self.n_neurons) * cfg.bg_rate_hz * Hz,
                name="bg_min_noise",
            )
            self.bg_syn = Synapses(
                self.bg_input, self.neurons, "w : volt",
                on_pre="g += w", name="bg_min_syn",
            )
            self.bg_syn.connect(j='i')
            self.bg_syn.w = cfg.bg_weight_mV * mV
        else:
            self.bg_input = None
            self.bg_syn = None

        # Spike monitor
        self.spike_mon = SpikeMonitor(self.neurons, record=False)

        # Assemble
        net_objects = [
            self.neurons, self.synapses, self.spike_mon,
            self.input_group, self.input_syn,
        ]
        if self.bg_input is not None:
            net_objects.extend([self.bg_input, self.bg_syn])
        self.net = Network(*net_objects)

        self._ms = ms
        self._Hz = Hz
        self._second = second

        # Free large DataFrames
        del self._conn, self._internal_conn, self._dn_to_lif_conn, self._ann

    # ---- Warmup -------------------------------------------------------------

    def _warmup(self):
        print(f"  Warmup ({self.cfg.warmup_ms:.0f}ms at "
              f"{self.cfg.warmup_rate_hz:.0f}Hz)...")
        t0 = time()
        n_dn = len(self._active_dn_ids)
        self.input_group.rates = (
            np.ones(n_dn) * self.cfg.warmup_rate_hz * self._Hz)
        self.net.run(self.cfg.warmup_ms * self._ms)
        print(f"  Warmup done in {time()-t0:.1f}s")

    # ---- Step ---------------------------------------------------------------

    def step(self, vnc_input: VNCInput, sim_ms: float = 20.0) -> VNCOutput:
        gr = vnc_input.group_rates

        n_dn = len(self._active_dn_ids)
        new_rates = np.full(n_dn, self.cfg.dn_baseline_hz, dtype=np.float64)

        for group_name, manc_ids in self._group_to_manc_dn_ids.items():
            rate = float(gr.get(group_name, 0.0))
            for bid in manc_ids:
                if bid in self._dn_bodyid_to_input_idx:
                    new_rates[self._dn_bodyid_to_input_idx[bid]] = rate

        mean_rate = float(np.mean([
            gr.get("forward", 0.0), gr.get("turn_left", 0.0),
            gr.get("turn_right", 0.0), gr.get("rhythm", 0.0),
            gr.get("stance", 0.0),
        ]))
        for bid in self._unassigned_readout_dns:
            if bid in self._dn_bodyid_to_input_idx:
                new_rates[self._dn_bodyid_to_input_idx[bid]] = mean_rate

        self.input_group.rates = new_rates * self._Hz

        counts_before = np.array(self.spike_mon.count, dtype=np.int64)
        self.net.run(sim_ms * self._ms)

        counts_after = np.array(self.spike_mon.count, dtype=np.int64)
        window_s = sim_ms / 1000.0
        tonic_rates = np.zeros(len(self._mn_brian_idx), dtype=np.float32)
        for j, brian_idx in enumerate(self._mn_brian_idx):
            tonic_rates[j] = float(counts_after[brian_idx] - counts_before[brian_idx]) / window_s if window_s > 0 else 0.0

        return VNCOutput(
            mn_body_ids=self._mn_body_ids.copy(),
            firing_rates_hz=tonic_rates,
        )

    @property
    def current_time_ms(self) -> float:
        return float(self.net.t / self._ms)

    def get_dn_type_to_body_ids(self) -> dict:
        return dict(self._dn_type_to_manc)

    # ---- Stats --------------------------------------------------------------

    def _report_stats(self, build_time: float):
        drive_pct = (self._captured_mn_drive / self._total_mn_drive * 100
                     if self._total_mn_drive > 0 else 0)
        self.extraction_stats = {
            "n_neurons": self.n_neurons,
            "n_premotor": self.n_intrinsic,
            "n_mn": self.n_mn,
            "n_dn_active": self.n_dn,
            "n_internal_syn": self.n_internal_synapses,
            "n_dn_syn": self.n_dn_synapses,
            "mn_drive_pct": drive_pct,
            "mapped_mn_coverage": self._mapped_mn_coverage,
            "total_mapped_mns": self._total_mapped_mns,
            "build_time_s": build_time,
        }
        print(f"\n  MinimalVNCRunner ready ({build_time:.1f}s):")
        print(f"    {self.n_neurons} LIF neurons "
              f"({self.n_intrinsic} premotor + {self.n_mn} MN)")
        print(f"    {self.n_dn} active DNs (PoissonGroup)")
        print(f"    {self.n_internal_synapses:,} internal + "
              f"{self.n_dn_synapses:,} DN->LIF = "
              f"{self.n_synapses:,} total synapses")
        print(f"    MN drive coverage: {drive_pct:.1f}%")
        if self._total_mapped_mns > 0:
            print(f"    Mapped MN coverage: "
                  f"{self._mapped_mn_coverage}/{self._total_mapped_mns}")
