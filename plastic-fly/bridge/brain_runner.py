"""
Brain runner: wraps the Brian2 LIF model for incremental stepping.

Two implementations:
  FakeBrainRunner — pass-through for testing the loop without Brian2
  Brian2BrainRunner — real 139k neuron LIF model with PoissonGroup input

Use create_brain_runner() factory to pick the right one.
"""

import numpy as np
from pathlib import Path
from time import time

from bridge.interfaces import BrainInput, BrainOutput


class FakeBrainRunner:
    """Pass-through brain for testing the loop without Brian2.

    Echoes mean input rate to all readout neurons with small noise.
    Use this to verify everything else works before touching Brian2.
    """

    def __init__(self, readout_neuron_ids: np.ndarray):
        self.readout_neuron_ids = np.asarray(readout_neuron_ids, dtype=np.int64)
        self._rng = np.random.RandomState(42)

    def step(self, brain_input: BrainInput, sim_ms: float = 10.0) -> BrainOutput:
        mean_in = float(np.mean(brain_input.firing_rates_hz))
        noise = self._rng.normal(0, 5.0, len(self.readout_neuron_ids))
        rates = np.clip(mean_in + noise, 0, 200).astype(np.float32)

        return BrainOutput(
            neuron_ids=self.readout_neuron_ids,
            firing_rates_hz=rates,
        )

    @property
    def current_time_ms(self) -> float:
        return 0.0


class Brian2BrainRunner:
    """Real connectome-based LIF brain running incrementally via Brian2.

    Supports:
        - FlyWire: 139k neurons (default)
        - BANC: ~130-160k neurons (brain partition)

    Creates the full connectome network once, then steps in small windows.
    Uses PoissonGroup (variable rates) for sensory input instead of
    PoissonInput (fixed rates), enabling closed-loop control.
    """

    def __init__(
        self,
        sensory_neuron_ids: np.ndarray,
        readout_neuron_ids: np.ndarray,
        path_comp: str = None,
        path_con: str = None,
        w_syn: float = 0.275,
        f_poi: float = 250,
        warmup_ms: float = 200.0,
        shuffle_seed: int | None = None,
        connectome: str = "flywire",
    ):
        from bridge.config import BridgeConfig
        cfg = BridgeConfig(connectome=connectome)

        if path_comp is None:
            path_comp = str(cfg.completeness_path)
        if path_con is None:
            path_con = str(cfg.connectivity_path)

        self.sensory_ids = np.asarray(sensory_neuron_ids, dtype=np.int64)
        self.readout_ids = np.asarray(readout_neuron_ids, dtype=np.int64)
        self.shuffle_seed = shuffle_seed
        self.connectome = connectome

        label = "SHUFFLED (seed=%d)" % shuffle_seed if shuffle_seed is not None else "real"
        src = "BANC" if connectome == "banc" else "FlyWire"
        print(f"Brian2BrainRunner: building {src} neuron network ({label})...")
        t0 = time()
        self._build_network(path_comp, path_con, w_syn, f_poi)
        print(f"  Network ready in {time() - t0:.1f}s "
              f"({self.n_neurons} neurons, {len(self.sensory_ids)} sensory, "
              f"{len(self.readout_ids)} readout)")

        if warmup_ms > 0:
            print(f"  Warming up ({warmup_ms:.0f}ms at baseline)...")
            t0 = time()
            from brian2 import Hz
            self.input_group.rates = np.ones(len(self._sensory_brian_idx)) * 50.0 * Hz
            from brian2 import ms
            self.net.run(warmup_ms * ms)
            print(f"  Warmup done in {time() - t0:.1f}s")

    def _load_connectome(self, path_comp, path_con):
        """Load connectome data, handling both FlyWire and BANC formats.

        Returns:
            neuron_ids: array of neuron IDs
            pre_idx: array of presynaptic brian2 indices
            post_idx: array of postsynaptic brian2 indices
            weights: array of synaptic weights (signed)
        """
        import pandas as pd

        if self.connectome == "banc":
            # BANC format: SQLite database from Harvard Dataverse
            # path_comp = path to SQLite DB (or parquet fallback)
            # path_con = same DB (connectivity is in edgelist_simple table)
            from bridge.banc_loader import BANCLoader

            # Detect whether path_comp is SQLite or parquet
            if str(path_comp).endswith(".sqlite"):
                banc_dir = Path(path_comp).parent
                loader = BANCLoader(banc_dir)
            elif str(path_comp).endswith(".parquet"):
                # Legacy parquet fallback
                loader = None
            else:
                banc_dir = Path(path_comp).parent
                loader = BANCLoader(banc_dir)

            if loader is not None and loader.is_available():
                df_comp = loader.load_neurons()
                df_con = loader.load_connectivity()
            else:
                # Parquet fallback
                df_comp = pd.read_parquet(path_comp)
                df_con = pd.read_parquet(path_con)

            # Standardize to get body IDs
            if "body_id" in df_comp.columns:
                id_col = "body_id"
            else:
                for c in ["id", "pt_root_id", "root_id", "bodyId", "segment_id"]:
                    if c in df_comp.columns:
                        id_col = c
                        break
                else:
                    df_comp = df_comp.reset_index()
                    id_col = df_comp.columns[0]

            neuron_ids = df_comp[id_col].values.astype(np.int64)
            id_to_idx = {int(nid): i for i, nid in enumerate(neuron_ids)}

            # Standardize connectivity columns
            for target, candidates in [("pre_id", ["pre_id", "pre", "body_pre"]),
                                       ("post_id", ["post_id", "post", "body_post"]),
                                       ("weight", ["weight", "count", "syn_count"])]:
                for c in candidates:
                    if c in df_con.columns and c != target:
                        df_con = df_con.rename(columns={c: target})
                        break

            pre_ids = df_con["pre_id"].values.astype(np.int64)
            post_ids = df_con["post_id"].values.astype(np.int64)
            raw_weights = df_con["weight"].values.astype(np.float64)

            # E/I signs: default to +1 (excitatory) if not available
            # BANC edgelist_simple typically doesn't include NT type;
            # we can join with meta table later for E/I if needed.
            signs = np.ones(len(df_con), dtype=np.float64)

            # Filter to neurons in our set (vectorized for speed)
            pre_valid = np.isin(pre_ids, neuron_ids)
            post_valid = np.isin(post_ids, neuron_ids)
            valid = pre_valid & post_valid

            valid_pre = pre_ids[valid]
            valid_post = post_ids[valid]
            pre_idx = np.array([id_to_idx[int(p)] for p in valid_pre], dtype=np.int32)
            post_idx = np.array([id_to_idx[int(q)] for q in valid_post], dtype=np.int32)
            weights = (raw_weights[valid] * signs[valid])

            return neuron_ids, pre_idx, post_idx, weights

        else:
            # FlyWire format: CSV + parquet
            for label, p in [("Completeness CSV", path_comp),
                             ("Connectivity parquet", path_con)]:
                if not Path(p).exists():
                    raise FileNotFoundError(
                        f"{label} not found: {p}\n"
                        "Expected in brain-model/ (Drosophila_brain_model repo).\n"
                        "See README.md for setup instructions."
                    )
            df_comp = pd.read_csv(path_comp, index_col=0)
            df_con = pd.read_parquet(path_con)

            neuron_ids = np.array(df_comp.index, dtype=np.int64)

            pre_idx = df_con['Presynaptic_Index'].values
            post_idx = df_con['Postsynaptic_Index'].values
            weights = df_con['Excitatory x Connectivity'].values

            return neuron_ids, pre_idx, post_idx, weights

    def _build_network(self, path_comp, path_con, w_syn, f_poi):
        import pandas as pd
        from brian2 import (
            NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
            mV, ms, Hz, second,
        )

        neuron_ids, pre_idx, post_idx, weights = self._load_connectome(path_comp, path_con)
        self.n_neurons = len(neuron_ids)

        # Neuron ID ↔ brian2 index mappings
        self._flyid_to_idx = {int(fid): i for i, fid in enumerate(neuron_ids)}
        self._idx_to_flyid = {i: int(fid) for i, fid in enumerate(neuron_ids)}

        # Convert FlyWire IDs to brian2 indices (filter to valid only)
        sensory_valid = [(fid, self._flyid_to_idx[fid]) for fid in self.sensory_ids
                         if fid in self._flyid_to_idx]
        self._sensory_brian_idx = np.array([bi for _, bi in sensory_valid], dtype=int)

        readout_valid = [(fid, self._flyid_to_idx[fid]) for fid in self.readout_ids
                         if fid in self._flyid_to_idx]
        self._readout_brian_idx = np.array([bi for _, bi in readout_valid], dtype=int)
        self._readout_flyids = np.array([fid for fid, _ in readout_valid], dtype=np.int64)

        # Precompute input mapping: FlyWire ID -> PoissonGroup index
        sensory_flyids_valid = [fid for fid, _ in sensory_valid]
        self._fid_to_input_idx = {int(fid): i for i, fid in enumerate(sensory_flyids_valid)}

        params = {
            'v_0': -52 * mV, 'v_rst': -52 * mV, 'v_th': -45 * mV,
            't_mbr': 20 * ms, 'tau': 5 * ms, 't_rfc': 2.2 * ms,
            't_dly': 1.8 * ms, 'w_syn': w_syn * mV,
        }

        eqs = '''
            dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau               : volt (unless refractory)
            rfc                            : second
        '''

        self.neurons = NeuronGroup(
            N=self.n_neurons, model=eqs, method='linear',
            threshold='v > v_th', reset='v = v_rst; g = 0 * mV',
            refractory='rfc', name='neurons', namespace=params,
        )
        self.neurons.v = params['v_0']
        self.neurons.g = 0
        self.neurons.rfc = params['t_rfc']

        self.synapses = Synapses(
            self.neurons, self.neurons, 'w : volt',
            on_pre='g += w', delay=params['t_dly'], name='synapses',
        )
        if self.shuffle_seed is not None:
            # Shuffle postsynaptic targets to destroy specific wiring
            # Preserves: out-degree per neuron, total synapse count, weight distribution
            rng = np.random.RandomState(self.shuffle_seed)
            post_idx = rng.permutation(post_idx)
        self.synapses.connect(i=pre_idx, j=post_idx)
        self.synapses.w = weights * params['w_syn']

        self.spike_mon = SpikeMonitor(self.neurons, record=False)

        n_input = len(self._sensory_brian_idx)
        self.input_group = PoissonGroup(n_input, rates=np.zeros(n_input) * Hz, name='input')
        self.input_syn = Synapses(
            self.input_group, self.neurons, 'w : volt',
            on_pre='g += w', name='input_syn',
        )
        self.input_syn.connect(i=np.arange(n_input), j=self._sensory_brian_idx)
        self.input_syn.w = params['w_syn'] * f_poi

        for idx in self._sensory_brian_idx:
            self.neurons[int(idx)].rfc = 0 * ms

        self.net = Network(
            self.neurons, self.synapses, self.spike_mon,
            self.input_group, self.input_syn,
        )
        self._ms = ms
        self._Hz = Hz
        self._second = second

    def step(self, brain_input: BrainInput, sim_ms: float = 10.0) -> BrainOutput:
        # Set input rates (using precomputed mapping)
        new_rates = np.zeros(len(self._sensory_brian_idx), dtype=np.float64)
        for fid, rate in zip(brain_input.neuron_ids, brain_input.firing_rates_hz):
            fid = int(fid)
            if fid in self._fid_to_input_idx:
                new_rates[self._fid_to_input_idx[fid]] = float(rate)

        self.input_group.rates = new_rates * self._Hz

        # Snapshot cumulative counts before stepping
        counts_before = np.array(self.spike_mon.count, dtype=np.int64)
        self.net.run(sim_ms * self._ms)

        # Compute per-neuron spike counts for this window only (O(n_readout))
        counts_after = np.array(self.spike_mon.count, dtype=np.int64)
        window_s = sim_ms / 1000.0
        readout_rates = np.zeros(len(self._readout_brian_idx), dtype=np.float32)
        for j, brian_idx in enumerate(self._readout_brian_idx):
            readout_rates[j] = float(counts_after[brian_idx] - counts_before[brian_idx]) / window_s if window_s > 0 else 0.0

        return BrainOutput(
            neuron_ids=self._readout_flyids,
            firing_rates_hz=readout_rates,
        )

    @property
    def current_time_ms(self) -> float:
        return float(self.net.t / self._ms)


def create_brain_runner(
    sensory_ids: np.ndarray,
    readout_ids: np.ndarray,
    use_fake: bool = False,
    connectome: str = "flywire",
    **kwargs,
):
    """Factory: returns FakeBrainRunner or Brian2BrainRunner."""
    if use_fake:
        return FakeBrainRunner(readout_neuron_ids=readout_ids)
    return Brian2BrainRunner(
        sensory_neuron_ids=sensory_ids,
        readout_neuron_ids=readout_ids,
        connectome=connectome,
        **kwargs,
    )
