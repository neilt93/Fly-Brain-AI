"""
Brain runner: wraps the Brian2 LIF model for incremental stepping.

Two implementations:
  FakeBrainRunner — pass-through for testing the loop without Brian2
  Brian2BrainRunner — real 139k neuron LIF model with PoissonGroup input

Use create_brain_runner() factory to pick the right one.
"""

import sys
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
    """Real 139k-neuron LIF brain running incrementally via Brian2.

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
    ):
        from bridge.config import BridgeConfig
        cfg = BridgeConfig()

        if path_comp is None:
            path_comp = str(cfg.completeness_path)
        if path_con is None:
            path_con = str(cfg.connectivity_path)

        self.sensory_ids = np.asarray(sensory_neuron_ids, dtype=np.int64)
        self.readout_ids = np.asarray(readout_neuron_ids, dtype=np.int64)
        self.shuffle_seed = shuffle_seed

        label = "SHUFFLED (seed=%d)" % shuffle_seed if shuffle_seed is not None else "real"
        print(f"Brian2BrainRunner: building 139k neuron network ({label})...")
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

    def _build_network(self, path_comp, path_con, w_syn, f_poi):
        import pandas as pd
        from brian2 import (
            NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
            mV, ms, Hz, second,
        )

        df_comp = pd.read_csv(path_comp, index_col=0)
        df_con = pd.read_parquet(path_con)
        self.n_neurons = len(df_comp)

        # FlyWire ID ↔ brian2 index mappings
        self._flyid_to_idx = {fid: i for i, fid in enumerate(df_comp.index)}
        self._idx_to_flyid = {i: fid for fid, i in self._flyid_to_idx.items()}

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
        pre_idx = df_con['Presynaptic_Index'].values
        post_idx = df_con['Postsynaptic_Index'].values
        if self.shuffle_seed is not None:
            # Shuffle postsynaptic targets to destroy specific wiring
            # Preserves: out-degree per neuron, total synapse count, weight distribution
            rng = np.random.RandomState(self.shuffle_seed)
            post_idx = rng.permutation(post_idx)
        self.synapses.connect(i=pre_idx, j=post_idx)
        self.synapses.w = df_con['Excitatory x Connectivity'].values * params['w_syn']

        self.spike_mon = SpikeMonitor(self.neurons)

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

        # Record spike count before stepping to only read new spikes
        n_before = self.spike_mon.num_spikes
        self.net.run(sim_ms * self._ms)

        # Read firing rates for readout neurons from new spikes only
        readout_rates = np.zeros(len(self._readout_brian_idx), dtype=np.float32)
        n_after = self.spike_mon.num_spikes
        if n_after > n_before:
            new_i = np.array(self.spike_mon.i[n_before:])
            for j, brian_idx in enumerate(self._readout_brian_idx):
                readout_rates[j] = float(np.sum(new_i == brian_idx)) / (sim_ms / 1000.0)

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
    **kwargs,
):
    """Factory: returns FakeBrainRunner or Brian2BrainRunner."""
    if use_fake:
        return FakeBrainRunner(readout_neuron_ids=readout_ids)
    return Brian2BrainRunner(
        sensory_neuron_ids=sensory_ids,
        readout_neuron_ids=readout_ids,
        **kwargs,
    )
