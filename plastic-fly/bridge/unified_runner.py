"""
Unified brain+VNC runner for BANC connectome.

Instead of separate Brain (FlyWire, 139K) -> DN decoder -> VNC (MANC, 13K)
pipeline, this runs the full BANC connectome (~160K neurons) as a single
Brian2 network. Motor output is read directly from MN firing rates.

Pipeline simplification:
    Current: SensoryEncoder -> Brain -> DNDecoder -> group_rates -> VNC -> MNDecoder -> joints
    Unified: SensoryEncoder -> UnifiedBrain -> MN rates -> CPG modulation -> MNDecoder -> joints

Usage:
    from bridge.unified_runner import UnifiedBrainRunner
    runner = UnifiedBrainRunner(sensory_ids, mn_ids, readout_dn_ids)
    brain_output = runner.step(brain_input, sim_ms=20.0)
    mn_rates = runner.get_mn_rates()
"""

import sys
import numpy as np
from pathlib import Path
from time import time

from bridge.interfaces import BrainInput, BrainOutput


ROOT = Path(__file__).resolve().parent.parent


class UnifiedBrainRunner:
    """Single Brian2 network for full BANC brain+VNC.

    Loads the complete BANC connectome and simulates it as one network.
    Sensory input drives brain sensory neurons (same as Brian2BrainRunner).
    Motor output is read from MN firing rates in the VNC partition.
    DN activity is monitored for logging/ablation but not injected.

    This eliminates the artificial DN bridge between brain and VNC.
    """

    def __init__(
        self,
        sensory_neuron_ids: np.ndarray,
        mn_neuron_ids: np.ndarray,
        readout_dn_ids: np.ndarray | None = None,
        banc_dir: str | Path | None = None,
        w_syn: float = 0.275,
        f_poi: float = 250,
        warmup_ms: float = 200.0,
        min_syn_count: int = 3,
    ):
        """
        Args:
            sensory_neuron_ids: BANC IDs of sensory neurons to receive input.
            mn_neuron_ids: BANC IDs of motor neurons to read output from.
            readout_dn_ids: BANC IDs of DNs to monitor (optional, for logging).
            banc_dir: Path to BANC data directory.
            w_syn: Base synaptic weight in mV.
            f_poi: Poisson input weight scaling.
            warmup_ms: Brain warmup duration.
            min_syn_count: Minimum synapse count to include an edge (reduces memory).
        """
        if banc_dir is None:
            banc_dir = ROOT / "data" / "banc"
        self.banc_dir = Path(banc_dir)

        self.sensory_ids = np.asarray(sensory_neuron_ids, dtype=np.int64)
        self.mn_ids = np.asarray(mn_neuron_ids, dtype=np.int64)
        self.readout_dn_ids = np.asarray(readout_dn_ids, dtype=np.int64) if readout_dn_ids is not None else np.array([], dtype=np.int64)
        self.min_syn_count = min_syn_count

        print(f"UnifiedBrainRunner: building full BANC network...")
        print(f"  {len(self.sensory_ids)} sensory, {len(self.mn_ids)} MNs, "
              f"{len(self.readout_dn_ids)} monitor DNs")
        t0 = time()
        self._build_network(w_syn, f_poi)
        print(f"  Network ready in {time() - t0:.1f}s "
              f"({self.n_neurons} neurons, {self.n_synapses} synapses)")

        if warmup_ms > 0:
            print(f"  Warming up ({warmup_ms:.0f}ms at baseline)...")
            t0 = time()
            from brian2 import Hz, ms as brian_ms
            self.input_group.rates = np.ones(len(self._sensory_brian_idx)) * 50.0 * Hz
            self.net.run(warmup_ms * brian_ms)
            print(f"  Warmup done in {time() - t0:.1f}s")

    def _build_network(self, w_syn, f_poi):
        import pandas as pd
        from brian2 import (
            NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
            mV, ms, Hz, second,
        )

        # Load BANC data
        neurons_path = self.banc_dir / "banc_neurons.parquet"
        conn_path = self.banc_dir / "banc_connectivity.parquet"

        if not neurons_path.exists() or not conn_path.exists():
            raise FileNotFoundError(
                f"BANC data not found at {self.banc_dir}. "
                "Run: python scripts/download_banc.py"
            )

        from bridge.banc_loader import BANCLoader
        loader = BANCLoader(self.banc_dir)
        df_neurons = loader.load_neurons()
        df_conn = loader.load_connectivity()

        # Filter weak synapses to reduce memory
        if self.min_syn_count > 1:
            n_before = len(df_conn)
            df_conn = df_conn[df_conn["weight"] >= self.min_syn_count]
            print(f"  Filtered synapses: {n_before} -> {len(df_conn)} "
                  f"(min_count={self.min_syn_count})")

        # Build ID mappings
        all_ids = df_neurons["body_id"].values.astype(np.int64)
        self.n_neurons = len(all_ids)
        self._id_to_idx = {int(nid): i for i, nid in enumerate(all_ids)}
        self._idx_to_id = {i: int(nid) for i, nid in enumerate(all_ids)}

        # Map sensory/MN/DN IDs to brian2 indices
        self._sensory_brian_idx = np.array(
            [self._id_to_idx[int(s)] for s in self.sensory_ids if int(s) in self._id_to_idx],
            dtype=int,
        )
        self._mn_brian_idx = np.array(
            [self._id_to_idx[int(m)] for m in self.mn_ids if int(m) in self._id_to_idx],
            dtype=int,
        )
        self._mn_valid_ids = np.array(
            [int(m) for m in self.mn_ids if int(m) in self._id_to_idx],
            dtype=np.int64,
        )
        self._dn_brian_idx = np.array(
            [self._id_to_idx[int(d)] for d in self.readout_dn_ids if int(d) in self._id_to_idx],
            dtype=int,
        )
        self._dn_valid_ids = np.array(
            [int(d) for d in self.readout_dn_ids if int(d) in self._id_to_idx],
            dtype=np.int64,
        )

        # Input mapping
        sensory_valid = [int(s) for s in self.sensory_ids if int(s) in self._id_to_idx]
        self._fid_to_input_idx = {fid: i for i, fid in enumerate(sensory_valid)}

        # Build Brian2 network
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

        # Build synapses
        pre_ids = df_conn["pre_id"].values.astype(np.int64)
        post_ids = df_conn["post_id"].values.astype(np.int64)
        weights_raw = df_conn["weight"].values.astype(np.float64)
        nt_signs = df_conn["nt_sign"].values.astype(np.float64) if "nt_sign" in df_conn.columns else np.ones(len(df_conn))

        # Filter to neurons in our set (vectorized for speed on large connectomes)
        valid_ids = np.array(list(self._id_to_idx.keys()), dtype=np.int64)
        pre_valid = np.isin(pre_ids, valid_ids)
        post_valid = np.isin(post_ids, valid_ids)
        valid_mask = pre_valid & post_valid

        valid_pre = pre_ids[valid_mask]
        valid_post = post_ids[valid_mask]
        pre_idx = np.array([self._id_to_idx[int(p)] for p in valid_pre], dtype=np.int32)
        post_idx = np.array([self._id_to_idx[int(q)] for q in valid_post], dtype=np.int32)
        syn_weights = (weights_raw[valid_mask] * nt_signs[valid_mask])

        self.n_synapses = len(pre_idx)

        self.synapses = Synapses(
            self.neurons, self.neurons, 'w : volt',
            on_pre='g += w', delay=params['t_dly'], name='synapses',
        )
        self.synapses.connect(i=pre_idx, j=post_idx)
        self.synapses.w = syn_weights * params['w_syn']

        # Spike monitor
        self.spike_mon = SpikeMonitor(self.neurons, record=False)

        # PoissonGroup for sensory input
        n_input = len(self._sensory_brian_idx)
        self.input_group = PoissonGroup(n_input, rates=np.zeros(n_input) * Hz, name='input')
        self.input_syn = Synapses(
            self.input_group, self.neurons, 'w : volt',
            on_pre='g += w', name='input_syn',
        )
        self.input_syn.connect(i=np.arange(n_input), j=self._sensory_brian_idx)
        self.input_syn.w = params['w_syn'] * f_poi

        # Make sensory neurons non-refractory
        for idx in self._sensory_brian_idx:
            self.neurons[int(idx)].rfc = 0 * ms

        self.net = Network(
            self.neurons, self.synapses, self.spike_mon,
            self.input_group, self.input_syn,
        )
        self._ms = ms
        self._Hz = Hz

    def step(self, brain_input: BrainInput, sim_ms: float = 20.0) -> BrainOutput:
        """Run one simulation step. Returns DN readout (for compatibility).

        Also updates internal MN rates accessible via get_mn_rates().
        """
        # Set sensory input rates
        new_rates = np.zeros(len(self._sensory_brian_idx), dtype=np.float64)
        for fid, rate in zip(brain_input.neuron_ids, brain_input.firing_rates_hz):
            fid = int(fid)
            if fid in self._fid_to_input_idx:
                new_rates[self._fid_to_input_idx[fid]] = float(rate)

        self.input_group.rates = new_rates * self._Hz

        # Run simulation
        counts_before = np.array(self.spike_mon.count, dtype=np.int64)
        self.net.run(sim_ms * self._ms)
        counts_after = np.array(self.spike_mon.count, dtype=np.int64)

        window_s = sim_ms / 1000.0

        # Compute DN readout rates (for backward compatibility / logging)
        dn_rates = np.zeros(len(self._dn_brian_idx), dtype=np.float32)
        for j, idx in enumerate(self._dn_brian_idx):
            dn_rates[j] = float(counts_after[idx] - counts_before[idx]) / window_s

        # Cache MN rates
        self._mn_rates = np.zeros(len(self._mn_brian_idx), dtype=np.float32)
        for j, idx in enumerate(self._mn_brian_idx):
            self._mn_rates[j] = float(counts_after[idx] - counts_before[idx]) / window_s

        return BrainOutput(
            neuron_ids=self._dn_valid_ids,
            firing_rates_hz=dn_rates,
        )

    def get_mn_rates(self) -> tuple[np.ndarray, np.ndarray]:
        """Get motor neuron firing rates from last step.

        Returns:
            (mn_ids, firing_rates_hz) — BANC MN IDs and their rates.
        """
        return self._mn_valid_ids.copy(), self._mn_rates.copy()

    def silence_neurons(self, body_ids):
        """Silence specific neurons (ablation)."""
        from brian2 import second
        for bid in body_ids:
            bid = int(bid)
            if bid in self._id_to_idx:
                idx = self._id_to_idx[bid]
                self.neurons[idx].rfc = 1e9 * second
                # Zero outgoing weights
                mask = self.synapses.i == idx
                if hasattr(mask, '__len__'):
                    self.synapses.w[mask] = 0.0

    @property
    def current_time_ms(self) -> float:
        return float(self.net.t / self._ms)
