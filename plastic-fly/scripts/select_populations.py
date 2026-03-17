"""
Select sensory and readout neuron populations from the FlyWire connectome.

Generates:
  data/sensory_ids.npy     — FlyWire IDs for sensory input neurons
  data/readout_ids.npy     — FlyWire IDs for descending readout neurons
  data/channel_map.json    — sensory neuron -> channel assignments
  data/decoder_groups.json — readout IDs grouped by locomotion function

v2: Hybrid approach:
  - Sensory: sugar GRNs (gustatory) + SEZ ascending types (proprioceptive,
    mechanosensory, vestibular) classified by out/in connectivity ratio.
  - Readout: annotated SEZ motor/descending types form group cores, then
    supplemented with top direct downstream targets of sensory neurons to
    ensure reliable activity. Supplements assigned to the group whose core
    neurons they connect to most.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig


def _write_json_atomic(path, obj, **kwargs):
    """Write JSON atomically: write to tmp file then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, **kwargs)
    tmp.replace(path)


# --- Sugar GRN IDs (well-characterized gustatory neurons) ---
SUGAR_GRN_FLYIDS = np.array([
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
], dtype=np.int64)

# --- SEZ type -> channel assignments (curated from connectivity analysis) ---
# Ascending/sensory types (out/in ratio > 1.5)
PROPRIOCEPTIVE_TYPES = ["diatom", "mandala", "weaver"]       # large groups, joint-like
MECHANOSENSORY_TYPES = ["crab", "genie", "handle", "handup", "peahen"]  # contact-related
VESTIBULAR_TYPES = ["Asteroid", "eiffel", "gazebo", "trumpet"]  # body state

# Motor/descending types (out/in ratio < 0.5) -- grouped by locomotion function
FORWARD_TYPES = ["mothership", "lion", "bobber"]
RHYTHM_TYPES = ["twirl", "trogon", "ruby", "damsel"]
STANCE_TYPES = ["shark", "mute", "aSG1"]
TURN_BILATERAL_TYPES = ["nagini", "knees", "bridle"]

# How many supplement neurons to add per group
SUPPLEMENTS_PER_GROUP = 5


def collect_ids(sez, type_names, valid_set):
    """Collect valid FlyWire IDs for a list of SEZ type names."""
    ids = []
    for name in type_names:
        if name in sez:
            ids.extend(fid for fid in sez[name] if fid in valid_set)
    return np.array(ids, dtype=np.int64)


def split_bilateral(sez, type_names, valid_set):
    """Split bilateral pairs: first neuron -> left, second -> right."""
    left, right = [], []
    for name in type_names:
        if name not in sez:
            continue
        ids = [fid for fid in sez[name] if fid in valid_set]
        mid = len(ids) // 2
        left.extend(ids[:max(mid, 1)])
        right.extend(ids[max(mid, 1):])
    return np.array(left, dtype=np.int64), np.array(right, dtype=np.int64)


def find_downstream_targets(sensory_brian, df_con, flyid2i, i2flyid, exclude_set, n_targets=30):
    """Find top direct downstream targets of sensory neurons, sorted by total weight."""
    mask = df_con['Presynaptic_Index'].isin(sensory_brian)
    downstream = df_con[mask].copy()
    downstream = downstream[~downstream['Postsynaptic_Index'].isin(sensory_brian)]

    # Aggregate by postsynaptic neuron
    agg = downstream.groupby('Postsynaptic_Index')['Excitatory x Connectivity'].sum()
    agg = agg.sort_values(ascending=False)

    # Filter out already-assigned neurons
    exclude_brian = set(flyid2i[int(fid)] for fid in exclude_set if int(fid) in flyid2i)
    agg = agg[~agg.index.isin(exclude_brian)]

    targets = []
    for brian_idx in agg.head(n_targets).index:
        targets.append(int(i2flyid[int(brian_idx)]))
    return np.array(targets, dtype=np.int64)


def assign_supplements_to_groups(supplement_ids, core_groups, flyid2i, df_con, max_per_group=5):
    """Assign supplement neurons to decoder groups using round-robin.

    Each round, every group gets the best unassigned supplement by
    connectivity score. Capped at max_per_group supplements each."""
    group_names = list(core_groups.keys())

    # Get brian indices for each group's core
    group_brian = {}
    for gname, gids in core_groups.items():
        group_brian[gname] = set(flyid2i[int(fid)] for fid in gids if int(fid) in flyid2i)

    assignments = {gname: list(gids) for gname, gids in core_groups.items()}
    supp_count = {gname: 0 for gname in group_names}
    assigned = set()

    # Score all supplements against all groups
    scores = {}
    for fid in supplement_ids:
        if int(fid) not in flyid2i:
            continue
        bidx = flyid2i[int(fid)]
        out_mask = df_con['Presynaptic_Index'] == bidx
        out_targets = set(df_con[out_mask]['Postsynaptic_Index'].values)
        in_mask = df_con['Postsynaptic_Index'] == bidx
        in_sources = set(df_con[in_mask]['Presynaptic_Index'].values)

        for gname in group_names:
            core = group_brian[gname]
            score = len(out_targets & core) + len(in_sources & core)
            scores[(int(fid), gname)] = score

    # Round-robin assignment
    for _ in range(max_per_group):
        for gname in group_names:
            if supp_count[gname] >= max_per_group:
                continue
            # Find best unassigned supplement for this group
            best_fid, best_score = None, -1
            for fid in supplement_ids:
                fid = int(fid)
                if fid in assigned:
                    continue
                s = scores.get((fid, gname), 0)
                if s > best_score:
                    best_score = s
                    best_fid = fid
            if best_fid is not None:
                assignments[gname].append(best_fid)
                assigned.add(best_fid)
                supp_count[gname] += 1

    return assignments


def main():
    cfg = BridgeConfig()
    data_dir = cfg.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading connectome data...")
    df_comp = pd.read_csv(cfg.completeness_path, index_col=0)
    flyid_list = list(df_comp.index)
    valid_set = set(flyid_list)
    flyid2i = {fid: i for i, fid in enumerate(flyid_list)}
    i2flyid = {i: fid for fid, i in flyid2i.items()}
    print("  %d neurons in connectome" % len(flyid_list))

    print("Loading SEZ neuron types...")
    sez_path = cfg.brain_repo_root / "sez_neurons.pickle"
    with open(sez_path, "rb") as f:
        sez = pickle.load(f)
    print("  %d named types, %d total neurons" % (len(sez), sum(len(v) for v in sez.values())))

    df_con = pd.read_parquet(
        cfg.connectivity_path,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )

    # === SENSORY POPULATION ===
    gustatory_ids = np.array([fid for fid in SUGAR_GRN_FLYIDS if fid in valid_set], dtype=np.int64)
    proprioceptive_ids = collect_ids(sez, PROPRIOCEPTIVE_TYPES, valid_set)
    mechanosensory_ids = collect_ids(sez, MECHANOSENSORY_TYPES, valid_set)
    vestibular_ids = collect_ids(sez, VESTIBULAR_TYPES, valid_set)

    sensory_ids = np.unique(np.concatenate([
        gustatory_ids, proprioceptive_ids, mechanosensory_ids, vestibular_ids,
    ]))

    print("\n  Sensory neurons: %d total" % len(sensory_ids))
    print("    gustatory (sugar GRN):  %d" % len(gustatory_ids))
    print("    proprioceptive (SEZ):   %d" % len(proprioceptive_ids))
    print("    mechanosensory (SEZ):   %d" % len(mechanosensory_ids))
    print("    vestibular (SEZ):       %d" % len(vestibular_ids))

    # === READOUT POPULATION: annotated cores ===
    forward_core = collect_ids(sez, FORWARD_TYPES, valid_set)
    rhythm_core = collect_ids(sez, RHYTHM_TYPES, valid_set)
    stance_core = collect_ids(sez, STANCE_TYPES, valid_set)
    turn_left_core, turn_right_core = split_bilateral(sez, TURN_BILATERAL_TYPES, valid_set)

    core_groups = {
        "forward_ids": forward_core,
        "turn_left_ids": turn_left_core,
        "turn_right_ids": turn_right_core,
        "rhythm_ids": rhythm_core,
        "stance_ids": stance_core,
    }
    all_core_ids = np.concatenate([v for v in core_groups.values()])

    print("\n  Readout cores (annotated motor/descending):")
    for gname, gids in core_groups.items():
        print("    %s: %d neurons" % (gname, len(gids)))

    # === READOUT SUPPLEMENTS: top downstream targets ===
    sensory_brian = set(flyid2i[int(fid)] for fid in sensory_ids if int(fid) in flyid2i)
    n_supplements = SUPPLEMENTS_PER_GROUP * len(core_groups)
    supplement_ids = find_downstream_targets(
        sensory_brian, df_con, flyid2i, i2flyid,
        exclude_set=np.concatenate([sensory_ids, all_core_ids]),
        n_targets=n_supplements,
    )
    print("\n  Found %d supplement downstream targets" % len(supplement_ids))

    # Assign supplements to groups based on connectivity
    augmented_groups = assign_supplements_to_groups(
        supplement_ids, core_groups, flyid2i, df_con,
    )

    # Convert to numpy arrays
    decoder_groups = {
        gname: np.array(gids, dtype=np.int64).tolist()
        for gname, gids in augmented_groups.items()
    }

    readout_ids = np.unique(np.concatenate([
        np.array(gids, dtype=np.int64) for gids in augmented_groups.values()
    ]))

    print("\n  Readout neurons (augmented): %d total" % len(readout_ids))
    for gname, gids in decoder_groups.items():
        n_core = len(core_groups[gname])
        n_supp = len(gids) - n_core
        print("    %s: %d (%d core + %d supplement)" % (gname, len(gids), n_core, n_supp))

    # Verify connectivity
    readout_brian = set(flyid2i[int(fid)] for fid in readout_ids if int(fid) in flyid2i)
    direct = df_con[
        df_con['Presynaptic_Index'].isin(sensory_brian) &
        df_con['Postsynaptic_Index'].isin(readout_brian)
    ]
    print("\n  Direct connections sensory->readout: %d" % len(direct))
    print("  Readout directly reachable: %d/%d" % (
        len(set(direct['Postsynaptic_Index'].values) & readout_brian),
        len(readout_brian),
    ))

    # === CHANNEL MAP ===
    channel_map = {
        "gustatory": gustatory_ids.tolist(),
        "proprioceptive": proprioceptive_ids.tolist(),
        "mechanosensory": mechanosensory_ids.tolist(),
        "vestibular": vestibular_ids.tolist(),
    }

    # === SAVE ===
    np.save(data_dir / "sensory_ids.npy", sensory_ids)
    np.save(data_dir / "readout_ids.npy", readout_ids)

    _write_json_atomic(data_dir / "channel_map.json", channel_map, indent=2)

    _write_json_atomic(data_dir / "decoder_groups.json", decoder_groups, indent=2)

    print("\nSaved to %s/:" % data_dir)
    print("  sensory_ids.npy:     %d FlyWire IDs" % len(sensory_ids))
    print("  readout_ids.npy:     %d FlyWire IDs" % len(readout_ids))
    print("  channel_map.json:    %d channels" % len(channel_map))
    print("  decoder_groups.json: %d groups" % len(decoder_groups))


if __name__ == "__main__":
    main()
