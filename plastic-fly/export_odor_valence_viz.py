"""
export_odor_valence_viz.py — Export odor valence brain activity for Unity.

Runs 4 sequential brain simulations:
  1. DM1 (Or42b, attractive) left ORNs stimulated
  2. DM1 right ORNs stimulated
  3. DM5 (Or85a, aversive) left ORNs stimulated
  4. DM5 right ORNs stimulated

Exports connectome_activity.json with odor metadata so Unity can visualize
the signal propagating from olfactory input through the connectome to
descending motor output, showing opposite turning for attractive vs aversive.

Usage:
    cd plastic-fly
    python export_odor_valence_viz.py
    python export_odor_valence_viz.py --n-neurons 300 --walking-frames 4000
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import time

BRAIN_MODEL_DIR = Path(__file__).resolve().parent.parent / "brain-model"
sys.path.insert(0, str(BRAIN_MODEL_DIR))


def load_orn_ids(glomerulus: str) -> dict:
    """Load left/right ORN neuron IDs for a glomerulus."""
    ann_path = BRAIN_MODEL_DIR / "flywire_annotations_matched.csv"
    df = pd.read_csv(ann_path, low_memory=False)
    mask = df["cell_type"] == glomerulus
    g = df[mask]
    left = sorted(g[g["side"] == "left"]["root_id"].astype(np.int64).tolist())
    right = sorted(g[g["side"] == "right"]["root_id"].astype(np.int64).tolist())
    return {"left": left, "right": right}


def run_condition(label, exc_indices, path_comp, path_con, params, duration_ms=250):
    """Run one brain simulation condition."""
    from brian2 import ms
    from model import run_trial

    p = dict(params)
    p['t_run'] = duration_ms * ms

    print("  Running %s (%d stimulated, %dms)..." % (label, len(exc_indices), duration_ms))
    t0 = time()
    spike_trains = run_trial(
        exc=exc_indices, exc2=[], slnc=[],
        path_comp=path_comp, path_con=path_con,
        params=p
    )
    elapsed = time() - t0
    total_spikes = sum(len(v) for v in spike_trains.values())
    print("    Done in %.1fs -- %d active neurons, %d spikes" % (
        elapsed, len(spike_trains), total_spikes))
    return spike_trains


def select_viz_neurons(all_spike_trains, all_exc_indices, n_neurons=250):
    """Select most active neurons across all conditions."""
    # Merge firing rates across conditions
    merged_rates = {}
    for spike_trains in all_spike_trains:
        for idx, times in spike_trains.items():
            merged_rates[idx] = merged_rates.get(idx, 0) + len(times)

    sorted_neurons = sorted(merged_rates.items(), key=lambda x: -x[1])

    # Always include stimulated neurons
    selected = set()
    all_stim = set()
    for exc_list in all_exc_indices:
        all_stim.update(exc_list)
    for idx in all_stim:
        if idx in merged_rates:
            selected.add(idx)

    for idx, rate in sorted_neurons:
        if len(selected) >= n_neurons:
            break
        selected.add(idx)

    selected = list(selected)
    n_stim = sum(1 for s in selected if s in all_stim)
    print("  Selected %d neurons (%d stimulated)" % (len(selected), n_stim))
    return selected


def classify_neurons(selected_indices, dm1_indices, dm5_indices, all_spike_trains):
    """Classify neurons by role in the olfactory circuit."""
    dm1_set = set()
    for lst in dm1_indices.values():
        dm1_set.update(lst)
    dm5_set = set()
    for lst in dm5_indices.values():
        dm5_set.update(lst)

    # Compute total rates across all conditions
    total_rates = {}
    for spike_trains in all_spike_trains:
        for idx, times in spike_trains.items():
            total_rates[idx] = total_rates.get(idx, 0) + len(times)

    types = []
    names = []
    for i, idx in enumerate(selected_indices):
        rate = total_rates.get(idx, 0)
        if idx in dm1_set:
            types.append("orn_attractive")
            names.append("DM1_%d" % i)
        elif idx in dm5_set:
            types.append("orn_aversive")
            names.append("DM5_%d" % i)
        elif rate > 200:
            types.append("descending")
            names.append("DN_%d" % i)
        elif rate > 80:
            types.append("hub")
            names.append("Hub_%d" % i)
        elif rate > 30:
            types.append("interneuron")
            names.append("IN_%d" % i)
        else:
            types.append("peripheral")
            names.append("P_%d" % i)

    return types, names


def generate_3d_layout(n_neurons, neuron_types):
    """Brain-shaped 3D positions with olfactory circuit layering."""
    rng = np.random.RandomState(42)
    positions = []
    rx, ry, rz = 0.6, 0.4, 0.5

    for i in range(n_neurons):
        ntype = neuron_types[i] if i < len(neuron_types) else "interneuron"

        if ntype == "orn_attractive":
            # Top-left (left antenna region)
            theta = rng.uniform(-0.5, 0.5)
            phi = rng.uniform(0.1, 0.4)
            r = rng.uniform(0.8, 1.0)
        elif ntype == "orn_aversive":
            # Top-right (right antenna region)
            theta = rng.uniform(np.pi - 0.5, np.pi + 0.5)
            phi = rng.uniform(0.1, 0.4)
            r = rng.uniform(0.8, 1.0)
        elif ntype == "descending":
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(2.2, 2.8)
            r = rng.uniform(0.6, 1.0)
        elif ntype == "hub":
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0.8, 2.0)
            r = rng.uniform(0.3, 0.8)
        else:
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0.2, 2.8)
            r = rng.uniform(0.4, 1.0)

        x = rx * r * np.sin(phi) * np.cos(theta)
        y = ry * r * np.cos(phi)
        z = rz * r * np.sin(phi) * np.sin(theta)
        positions.append([round(float(x), 4), round(float(y), 4), round(float(z), 4)])

    return positions


def extract_subgraph(selected_indices, max_connections=300):
    """Extract connections between selected neurons."""
    path_con = str(BRAIN_MODEL_DIR / "Connectivity_783.parquet")
    df_con = pd.read_parquet(path_con, columns=[
        'Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'
    ])

    selected_set = set(selected_indices)
    idx_map = {old: new for new, old in enumerate(selected_indices)}

    mask = (df_con['Presynaptic_Index'].isin(selected_set) &
            df_con['Postsynaptic_Index'].isin(selected_set))
    sub = df_con[mask].copy()
    sub['abs_w'] = sub['Excitatory x Connectivity'].abs()
    sub = sub.nlargest(max_connections, 'abs_w')

    connections = []
    for _, row in sub.iterrows():
        from_idx = idx_map[int(row['Presynaptic_Index'])]
        to_idx = idx_map[int(row['Postsynaptic_Index'])]
        weight = float(row['Excitatory x Connectivity'])
        connections.append([from_idx, to_idx, weight])

    print("  Extracted %d connections" % len(connections))
    return connections


def bin_spikes_to_rates(spike_trains, selected_indices, bin_ms=5, sim_duration_ms=250):
    """Convert spike trains to binned firing rates (0-1 normalized)."""
    n_bins = sim_duration_ms // bin_ms
    n_neurons = len(selected_indices)
    rates = np.zeros((n_bins, n_neurons), dtype=np.float32)

    for j, idx in enumerate(selected_indices):
        if idx not in spike_trains:
            continue
        times_s = np.array(spike_trains[idx])
        times_ms = times_s * 1000
        for t in times_ms:
            b = int(t / bin_ms)
            if 0 <= b < n_bins:
                rates[b, j] += 1.0

    return rates


def main():
    parser = argparse.ArgumentParser(description="Export odor valence brain activity for Unity")
    parser.add_argument("--n-neurons", type=int, default=250)
    parser.add_argument("--max-connections", type=int, default=300)
    parser.add_argument("--walking-frames", type=int, default=4000)
    parser.add_argument("--condition-ms", type=int, default=250)
    parser.add_argument("--freq", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from brian2 import Hz, ms
    from model import default_params

    print("=" * 60)
    print("ODOR VALENCE VISUALIZATION EXPORT")
    print("=" * 60)

    path_comp = str(BRAIN_MODEL_DIR / "Completeness_783.csv")
    path_con = str(BRAIN_MODEL_DIR / "Connectivity_783.parquet")

    # Map FlyWire IDs to brian2 indices
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    # Load ORN populations
    print("\n[1/7] Loading ORN populations...")
    dm1_ids = load_orn_ids("ORN_DM1")
    dm5_ids = load_orn_ids("ORN_DM5")
    print("  DM1 (Or42b, attractive): %d left, %d right" % (len(dm1_ids["left"]), len(dm1_ids["right"])))
    print("  DM5 (Or85a, aversive):   %d left, %d right" % (len(dm5_ids["left"]), len(dm5_ids["right"])))

    # Map to brian2 indices
    dm1_brian = {
        "left": [flyid2i[fid] for fid in dm1_ids["left"] if fid in flyid2i],
        "right": [flyid2i[fid] for fid in dm1_ids["right"] if fid in flyid2i],
    }
    dm5_brian = {
        "left": [flyid2i[fid] for fid in dm5_ids["left"] if fid in flyid2i],
        "right": [flyid2i[fid] for fid in dm5_ids["right"] if fid in flyid2i],
    }

    params = dict(default_params)
    params['n_run'] = 1
    params['r_poi'] = args.freq * Hz

    # Run 4 conditions
    print("\n[2/7] Running brain simulations...")
    conditions = [
        ("DM1_left", dm1_brian["left"], "Or42b (attractive) LEFT"),
        ("DM1_right", dm1_brian["right"], "Or42b (attractive) RIGHT"),
        ("DM5_left", dm5_brian["left"], "Or85a (aversive) LEFT"),
        ("DM5_right", dm5_brian["right"], "Or85a (aversive) RIGHT"),
    ]

    all_spike_trains = []
    all_exc_indices = []
    for cond_name, exc_idx, desc in conditions:
        spike_trains = run_condition(desc, exc_idx, path_comp, path_con, params,
                                     duration_ms=args.condition_ms)
        all_spike_trains.append(spike_trains)
        all_exc_indices.append(exc_idx)

    # Select neurons
    print("\n[3/7] Selecting visualization neurons...")
    # Map brian2 indices for DM1/DM5 populations
    dm1_brian_all = set(dm1_brian["left"] + dm1_brian["right"])
    dm5_brian_all = set(dm5_brian["left"] + dm5_brian["right"])
    selected = select_viz_neurons(all_spike_trains, all_exc_indices, n_neurons=args.n_neurons)

    # Classify
    print("\n[4/7] Classifying neurons...")
    neuron_types, neuron_names = classify_neurons(
        selected,
        {k: [flyid2i[fid] for fid in v if fid in flyid2i] for k, v in dm1_ids.items()},
        {k: [flyid2i[fid] for fid in v if fid in flyid2i] for k, v in dm5_ids.items()},
        all_spike_trains,
    )
    for t in sorted(set(neuron_types)):
        print("  %s: %d" % (t, neuron_types.count(t)))

    # Subgraph
    print("\n[5/7] Extracting connectome subgraph...")
    connections = extract_subgraph(selected, max_connections=args.max_connections)

    # Layout and bin spikes
    print("\n[6/7] Generating layout and binning spikes...")
    positions = generate_3d_layout(len(selected), neuron_types)

    # Bin each condition separately, then concatenate
    condition_rates = []
    for spike_trains in all_spike_trains:
        rates = bin_spikes_to_rates(spike_trains, selected,
                                     bin_ms=5, sim_duration_ms=args.condition_ms)
        condition_rates.append(rates)

    # Concatenate: [cond1_bins, cond2_bins, ...]
    all_rates = np.concatenate(condition_rates, axis=0)
    n_bins_total = all_rates.shape[0]

    # Normalize globally
    max_rate = all_rates.max()
    if max_rate > 0:
        all_rates = all_rates / max_rate

    # Smooth
    smoothed = np.zeros_like(all_rates)
    for b in range(n_bins_total):
        w_sum = 0
        for db in [-1, 0, 1]:
            bb = b + db
            if 0 <= bb < n_bins_total:
                w = 1.0 if db == 0 else 0.3
                smoothed[b] += all_rates[bb] * w
                w_sum += w
        smoothed[b] /= w_sum

    # Tile to target frames
    target_frames = args.walking_frames
    if target_frames <= n_bins_total:
        firing_rates = smoothed[:target_frames]
    else:
        firing_rates = np.zeros((target_frames, smoothed.shape[1]), dtype=np.float32)
        for f in range(target_frames):
            phase = (f / target_frames) * n_bins_total
            b = int(phase) % n_bins_total
            frac = phase - int(phase)
            b_next = (b + 1) % n_bins_total
            firing_rates[f] = smoothed[b] * (1 - frac) + smoothed[b_next] * frac

    # Build condition timeline (frame indices)
    bins_per_cond = args.condition_ms // 5
    frames_per_cond = target_frames // 4

    # Export
    print("\n[7/7] Exporting JSON...")
    rates_list = []
    for f in range(firing_rates.shape[0]):
        rates_list.append([round(float(v), 3) for v in firing_rates[f]])

    data = {
        "n_neurons": len(selected),
        "n_frames": firing_rates.shape[0],
        "n_total_connectome": len(df_comp),
        "neuron_types": neuron_types,
        "neuron_names": neuron_names,
        "positions_3d": positions,
        "connections": connections,
        "firing_rates": rates_list,
        "odor": {
            "conditions": ["DM1_left", "DM1_right", "DM5_left", "DM5_right"],
            "condition_labels": [
                "Or42b (attractive) - Odor LEFT",
                "Or42b (attractive) - Odor RIGHT",
                "Or85a (aversive) - Odor LEFT",
                "Or85a (aversive) - Odor RIGHT",
            ],
            "valence": ["attractive", "attractive", "aversive", "aversive"],
            "odor_side": ["left", "right", "left", "right"],
            "frames_per_condition": frames_per_cond,
            "stimulus_left": [1.0, 0.0, 1.0, 0.0],
            "stimulus_right": [0.0, 1.0, 0.0, 1.0],
            "predicted_turn": ["LEFT", "RIGHT", "RIGHT", "LEFT"],
        },
        "metadata": {
            "source": "FlyWire 783 connectome",
            "model": "LIF (Brian2)",
            "neurons_in_connectome": len(df_comp),
            "neurons_visualized": len(selected),
            "stimulation": "Odor valence: DM1 (Or42b) vs DM5 (Or85a)",
            "simulation_duration_ms": args.condition_ms * 4,
            "bin_size_ms": 5,
            "n_dm1_total": len(dm1_ids["left"]) + len(dm1_ids["right"]),
            "n_dm5_total": len(dm5_ids["left"]) + len(dm5_ids["right"]),
        },
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).resolve().parent.parent / "FlyBrainViz" / "Assets" / "Resources" / "connectome_activity.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("  Exported %s (%.1f MB)" % (output_path, size_mb))
    print("  %d neurons, %d connections, %d frames" % (
        len(selected), len(connections), firing_rates.shape[0]))
    print("  4 conditions x %d frames each" % frames_per_cond)

    # Also save to logs
    logs_path = Path(__file__).resolve().parent / "logs" / "connectome_activity_odor.json"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(output_path, logs_path)
    print("  Also copied to %s" % logs_path)

    print("\n" + "=" * 60)
    print("DONE. Open Unity and press Play.")
    print("=" * 60)


if __name__ == "__main__":
    main()
