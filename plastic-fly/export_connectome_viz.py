"""
export_connectome_viz.py — Bridge script for connectome brain visualization.

Runs a single trial of the LIF brain model (Brian2) with sensory neuron stimulation,
selects the most active neurons for visualization, extracts their subgraph from the
FlyWire 783 connectome, and exports connectome_activity.json for Unity.

Usage:
    cd plastic-fly
    python export_connectome_viz.py                     # default: sugar GRNs at 100Hz
    python export_connectome_viz.py --freq 150          # custom stimulation frequency
    python export_connectome_viz.py --n-neurons 200     # fewer neurons for performance
    python export_connectome_viz.py --walking-frames 4000  # match your walking data length
"""

import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import tempfile


def _write_json_atomic(path, obj):
    """Write JSON to *path* atomically via a temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise

# Add brain-model to path
BRAIN_MODEL_DIR = Path(__file__).resolve().parent.parent / "brain-model"
sys.path.insert(0, str(BRAIN_MODEL_DIR))


def run_brain_simulation(freq_hz=100, n_neurons_total=138639):
    """Run a single trial of the LIF brain model with sugar GRN stimulation.

    Returns spike_trains dict {neuron_index: array_of_spike_times_in_seconds}
    """
    from brian2 import Hz, ms
    from model import default_params, run_trial

    params = dict(default_params)
    params['t_run'] = 1000 * ms   # 1 second simulation
    params['n_run'] = 1           # single trial
    params['r_poi'] = freq_hz * Hz

    # Completeness and connectivity files (783 version — full connectome)
    path_comp = str(BRAIN_MODEL_DIR / "Completeness_783.csv")
    path_con  = str(BRAIN_MODEL_DIR / "Connectivity_783.parquet")

    # Sugar-sensing GRNs (right hemisphere) — well-characterized sensory neurons
    # These are the 21 labellar sugar gustatory receptor neurons from the paper
    neu_sugar_ids = [
        720575940624963786, 720575940630233916, 720575940637568838,
        720575940638202345, 720575940617000768, 720575940630797113,
        720575940632889389, 720575940621754367, 720575940621502051,
        720575940640649691, 720575940639332736, 720575940616885538,
        720575940639198653, 720575940620900446, 720575940617937543,
        720575940632425919, 720575940633143833, 720575940612670570,
        720575940628853239, 720575940629176663, 720575940611875570,
    ]

    # Map flywire IDs to brian indices
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
    i2flyid = {i: fid for fid, i in flyid2i.items()}

    exc_indices = [flyid2i[n] for n in neu_sugar_ids if n in flyid2i]

    print(f"Running brain simulation: {len(exc_indices)} stimulated neurons at {freq_hz} Hz...")
    print(f"  Total neurons: {len(df_comp)}")
    t0 = time()

    spike_trains = run_trial(
        exc=exc_indices,
        exc2=[],
        slnc=[],
        path_comp=path_comp,
        path_con=path_con,
        params=params
    )

    elapsed = time() - t0
    total_spikes = sum(len(v) for v in spike_trains.values())
    print(f"  Done in {elapsed:.1f}s — {len(spike_trains)} active neurons, {total_spikes} total spikes")

    return spike_trains, i2flyid, flyid2i, exc_indices, df_comp


def select_viz_neurons(spike_trains, exc_indices, n_neurons=250):
    """Select the most active neurons for visualization, ensuring stimulated neurons are included."""

    # Compute firing rates
    rates = {}
    for idx, times in spike_trains.items():
        rates[idx] = len(times)  # spikes per second (1s sim)

    # Sort by activity
    sorted_neurons = sorted(rates.items(), key=lambda x: -x[1])

    # Always include stimulated neurons first, even if they were silent.
    selected = []
    selected_set = set()
    stim_set = set(exc_indices)
    for idx in exc_indices:
        if len(selected) >= n_neurons:
            break
        if idx in selected_set:
            continue
        selected.append(idx)
        selected_set.add(idx)

    # Fill remaining slots with most active non-stimulated neurons
    for idx, rate in sorted_neurons:
        if len(selected) >= n_neurons:
            break
        if idx in selected_set:
            continue
        selected.append(idx)
        selected_set.add(idx)

    print(f"  Selected {len(selected)} neurons for visualization "
          f"({sum(1 for s in selected if s in stim_set)} stimulated)")

    return selected


def extract_subgraph(selected_indices, max_connections=300):
    """Extract connections between selected neurons from the full connectome."""

    path_con = str(BRAIN_MODEL_DIR / "Connectivity_783.parquet")

    # Only load columns we need
    df_con = pd.read_parquet(path_con, columns=[
        'Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'
    ])

    selected_set = set(selected_indices)
    idx_map = {old: new for new, old in enumerate(selected_indices)}

    # Filter to connections between selected neurons
    mask = (df_con['Presynaptic_Index'].isin(selected_set) &
            df_con['Postsynaptic_Index'].isin(selected_set))
    sub = df_con[mask].copy()

    # Sort by absolute weight and take top connections
    sub['abs_w'] = sub['Excitatory x Connectivity'].abs()
    sub = sub.nlargest(max_connections, 'abs_w')

    connections = []
    for _, row in sub.iterrows():
        from_idx = idx_map[int(row['Presynaptic_Index'])]
        to_idx = idx_map[int(row['Postsynaptic_Index'])]
        weight = float(row['Excitatory x Connectivity'])
        connections.append([from_idx, to_idx, weight])

    print(f"  Extracted {len(connections)} connections (from {len(sub)} candidates)")
    return connections


def classify_neurons(selected_indices, exc_indices, spike_trains, df_comp, connections_raw):
    """Classify neurons by their role relative to the stimulated set.

    Categories:
    - stimulated: directly stimulated sensory neurons
    - first_order: directly connected to stimulated (1 synapse away)
    - hub: high connectivity count
    - interneuron: everything else that's active
    """
    stim_set = set(exc_indices)
    idx_map = {old: new for new, old in enumerate(selected_indices)}

    # Build adjacency from connections
    path_con = str(BRAIN_MODEL_DIR / "Connectivity_783.parquet")
    df_con = pd.read_parquet(path_con, columns=['Presynaptic_Index', 'Postsynaptic_Index'])

    # Find first-order targets of stimulated neurons
    first_order = set()
    for _, row in df_con.iterrows():
        pre = int(row['Presynaptic_Index'])
        post = int(row['Postsynaptic_Index'])
        if pre in stim_set and post in set(selected_indices):
            first_order.add(post)
    first_order -= stim_set

    # Classify
    types = []
    names = []
    for i, idx in enumerate(selected_indices):
        rate = len(spike_trains.get(idx, []))
        if idx in stim_set:
            types.append("stimulated")
            names.append(f"GRN_{i}")
        elif idx in first_order:
            types.append("first_order")
            names.append(f"DN_{i}")
        elif rate > 50:
            types.append("hub")
            names.append(f"Hub_{i}")
        else:
            types.append("interneuron")
            names.append(f"IN_{i}")

    return types, names


def classify_neurons_fast(selected_indices, exc_indices, spike_trains):
    """Fast neuron classification without re-reading the full parquet.
    Uses firing rate heuristics instead of connectivity lookup.
    """
    stim_set = set(exc_indices)

    types = []
    names = []
    for i, idx in enumerate(selected_indices):
        rate = len(spike_trains.get(idx, []))
        if idx in stim_set:
            types.append("stimulated")
            names.append(f"GRN_{i}")
        elif rate > 80:
            types.append("descending")
            names.append(f"DN_{i}")
        elif rate > 40:
            types.append("hub")
            names.append(f"Hub_{i}")
        elif rate > 15:
            types.append("interneuron")
            names.append(f"IN_{i}")
        else:
            types.append("peripheral")
            names.append(f"P_{i}")

    return types, names


def generate_3d_layout(n_neurons, neuron_types):
    """Generate brain-shaped 3D positions for neurons.

    Creates an ellipsoid layout:
    - Stimulated neurons at the top (sensory input)
    - Descending/hub neurons in the middle
    - Interneurons distributed throughout
    - Overall shape: elongated ellipsoid (brain-like)
    """
    rng = np.random.RandomState(42)
    positions = []

    # Brain ellipsoid parameters (in Unity units, will sit above fly)
    rx, ry, rz = 0.6, 0.4, 0.5  # ellipsoid radii

    for i in range(n_neurons):
        ntype = neuron_types[i] if i < len(neuron_types) else "interneuron"

        if ntype == "stimulated":
            # Top of brain, tightly clustered
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0.1, 0.4)  # near top
            r = rng.uniform(0.7, 1.0)
        elif ntype == "descending":
            # Bottom of brain (descending toward body)
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(2.2, 2.8)  # near bottom
            r = rng.uniform(0.6, 1.0)
        elif ntype == "hub":
            # Central region
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0.8, 2.0)  # middle
            r = rng.uniform(0.3, 0.8)
        else:
            # Distributed throughout
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0.2, 2.8)
            r = rng.uniform(0.4, 1.0)

        x = rx * r * np.sin(phi) * np.cos(theta)
        y = ry * r * np.cos(phi)   # Y is up in Unity
        z = rz * r * np.sin(phi) * np.sin(theta)

        positions.append([round(float(x), 4), round(float(y), 4), round(float(z), 4)])

    return positions


def bin_spikes_to_rates(spike_trains, selected_indices, bin_ms=5, sim_duration_ms=1000):
    """Convert spike trains to binned firing rates.

    Returns array of shape (n_bins, n_neurons) with normalized firing rates (0-1).
    """
    n_bins = sim_duration_ms // bin_ms
    n_neurons = len(selected_indices)
    rates = np.zeros((n_bins, n_neurons), dtype=np.float32)

    for j, idx in enumerate(selected_indices):
        if idx not in spike_trains:
            continue
        times_s = np.array(spike_trains[idx])
        times_ms = times_s * 1000  # convert to ms
        for t in times_ms:
            b = int(t / bin_ms)
            if 0 <= b < n_bins:
                rates[b, j] += 1.0

    # Normalize: convert spike counts to rate (spikes per bin / max possible)
    # Then scale to 0-1 range for visualization
    max_rate = rates.max()
    if max_rate > 0:
        rates = rates / max_rate

    # Apply mild temporal smoothing (Gaussian-like, 3-bin window)
    smoothed = np.zeros_like(rates)
    for b in range(n_bins):
        w_sum = 0
        for db in [-1, 0, 1]:
            bb = b + db
            if 0 <= bb < n_bins:
                w = 1.0 if db == 0 else 0.3
                smoothed[b] += rates[bb] * w
                w_sum += w
        smoothed[b] /= w_sum

    return smoothed


def tile_to_frames(binned_rates, target_frames):
    """Tile/loop the brain data to match the walking animation frame count."""
    n_bins = binned_rates.shape[0]

    if target_frames <= 0 or n_bins == 0:
        return np.zeros((max(target_frames, 0), binned_rates.shape[1]), dtype=np.float32)

    if target_frames <= n_bins:
        return binned_rates[:target_frames]

    # Tile with fractional indexing so loop points interpolate smoothly.
    result = np.zeros((target_frames, binned_rates.shape[1]), dtype=np.float32)
    for f in range(target_frames):
        phase = (f * n_bins) / float(target_frames)
        b = int(np.floor(phase)) % n_bins
        frac = phase - np.floor(phase)
        b_next = (b + 1) % n_bins
        result[f] = binned_rates[b] * (1 - frac) + binned_rates[b_next] * frac

    return result


def export_json(output_path, positions, connections, neuron_types, neuron_names,
                firing_rates, n_total_neurons=138639):
    """Export connectome_activity.json for Unity."""

    n_neurons = len(positions)
    n_frames = firing_rates.shape[0]

    # Convert firing rates to list of lists, rounded for file size
    rates_list = []
    for f in range(n_frames):
        frame_rates = [round(float(v), 3) for v in firing_rates[f]]
        rates_list.append(frame_rates)

    data = {
        "n_neurons": n_neurons,
        "n_frames": n_frames,
        "n_total_connectome": n_total_neurons,
        "neuron_types": neuron_types,
        "neuron_names": neuron_names,
        "positions_3d": positions,
        "connections": connections,
        "firing_rates": rates_list,
        "metadata": {
            "source": "FlyWire 783 connectome",
            "model": "LIF (Brian2)",
            "neurons_in_connectome": n_total_neurons,
            "neurons_visualized": n_neurons,
            "stimulation": "Sugar GRNs (labellar, right hemisphere)",
            "simulation_duration_ms": 1000,
            "bin_size_ms": 5
        }
    }

    output_path = Path(output_path)
    _write_json_atomic(output_path, data)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Exported {output_path} ({size_mb:.1f} MB)")
    print(f"  {n_neurons} neurons, {len(connections)} connections, {n_frames} frames")


def main():
    parser = argparse.ArgumentParser(description="Export connectome brain activity for Unity visualization")
    parser.add_argument("--freq", type=int, default=100, help="Stimulation frequency in Hz (default: 100)")
    parser.add_argument("--n-neurons", type=int, default=250, help="Number of neurons to visualize (default: 250)")
    parser.add_argument("--max-connections", type=int, default=300, help="Max connections to show (default: 300)")
    parser.add_argument("--walking-frames", type=int, default=4000, help="Target frame count to match walking data (default: 4000)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto)")
    args = parser.parse_args()

    print("=" * 60)
    print("CONNECTOME BRAIN VISUALIZATION EXPORT")
    print("=" * 60)

    # Step 1: Run brain simulation
    print("\n[1/6] Running LIF brain simulation...")
    spike_trains, i2flyid, flyid2i, exc_indices, df_comp = run_brain_simulation(freq_hz=args.freq)

    # Step 2: Select visualization neurons
    print("\n[2/6] Selecting visualization neurons...")
    selected = select_viz_neurons(spike_trains, exc_indices, n_neurons=args.n_neurons)

    # Step 3: Classify neurons
    print("\n[3/6] Classifying neurons...")
    neuron_types, neuron_names = classify_neurons_fast(selected, exc_indices, spike_trains)
    for t in set(neuron_types):
        count = neuron_types.count(t)
        print(f"  {t}: {count}")

    # Step 4: Extract subgraph connections
    print("\n[4/6] Extracting connectome subgraph...")
    connections = extract_subgraph(selected, max_connections=args.max_connections)

    # Step 5: Generate layout and bin spikes
    print("\n[5/6] Generating 3D layout and binning spikes...")
    positions = generate_3d_layout(len(selected), neuron_types)
    binned = bin_spikes_to_rates(spike_trains, selected)
    firing_rates = tile_to_frames(binned, args.walking_frames)

    # Step 6: Export
    print("\n[6/6] Exporting JSON...")
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).resolve().parent.parent / "FlyBrainViz" / "Assets" / "Resources" / "connectome_activity.json"

    export_json(
        output_path, positions, connections,
        neuron_types, neuron_names, firing_rates,
        n_total_neurons=len(df_comp)
    )

    # Also copy to logs for reference
    logs_path = Path(__file__).resolve().parent / "logs" / "connectome_activity.json"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(output_path, logs_path)
    print(f"  Also copied to {logs_path}")

    print("\n" + "=" * 60)
    print("DONE. Open Unity and press Play.")
    print("=" * 60)


if __name__ == "__main__":
    main()
