"""
Select sensory and readout neuron populations from BANC connectome.

Uses BANC annotations to identify:
    - Sensory neurons: ORNs (olfactory), LPLC2 (visual/looming),
      mechanosensory, proprioceptive, gustatory, vestibular
    - Readout neurons: descending neurons (DNa01, DNg13, etc.)

Primary mapping: BANC↔FlyWire cross-reference table.
Fallback: cell_type name matching.

Outputs:
    data/banc/sensory_ids_banc.npy
    data/banc/readout_ids_banc.npy
    data/banc/channel_map_banc.json
    data/banc/decoder_groups_banc.json

Usage:
    python scripts/select_populations_banc.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.banc_loader import BANCLoader


# FlyWire sensory channel definitions (from existing channel_map.json)
SENSORY_TYPE_PATTERNS = {
    "olfactory": ["ORN", "Or42b", "Or85a", "Or22a", "Or47b"],
    "visual": ["LPLC2", "LC4", "LC6"],
    "gustatory": ["GRN", "Gr5a", "Gr66a", "ppk23"],
    "proprioceptive": ["feCO", "FeCO", "proprioceptive"],
    "mechanosensory": ["JO", "Johnston", "mechanosensory", "bristle"],
    "vestibular": ["halter", "haltere", "campaniform"],
}

# DN types for readout decoder groups (from existing decoder_groups.json)
DN_GROUP_PATTERNS = {
    "forward_ids": ["DNa01", "DNa02", "DNb01", "DNg11"],
    "turn_left_ids": ["DNa03", "DNb02", "DNg13"],
    "turn_right_ids": ["DNa03", "DNb02", "DNg13"],
    "rhythm_ids": ["DNg100", "DNp09"],
    "stance_ids": ["DNb05", "DNg14"],
}


def main():
    loader = BANCLoader()
    if not loader.is_available():
        print("BANC data not available. Run: python scripts/download_banc.py")
        sys.exit(1)

    print("Loading BANC neurons...")
    neurons = loader.load_neurons()
    print(f"  {len(neurons)} neurons total")

    # Load cross-references for FlyWire mapping
    refs = loader.load_cross_references()
    flywire_map = {}
    if "flywire" in refs:
        fw = refs["flywire"]
        # Build BANC -> FlyWire ID mapping
        for _, row in fw.iterrows():
            for bcol in ["banc_id", "pt_root_id", "root_id"]:
                if bcol in fw.columns:
                    banc_id = int(row[bcol])
                    break
            for fcol in ["flywire_id", "flywire_root_id", "match_id"]:
                if fcol in fw.columns:
                    flywire_id = int(row[fcol])
                    break
            flywire_map[banc_id] = flywire_id
        print(f"  FlyWire cross-reference: {len(flywire_map)} matches")

    # === Select sensory neurons ===
    print("\nSelecting sensory neurons...")
    channel_map = {}
    all_sensory_ids = []

    for channel, patterns in SENSORY_TYPE_PATTERNS.items():
        channel_ids = []
        for pattern in patterns:
            mask = neurons["cell_type"].str.contains(pattern, na=False, case=False)
            matches = neurons[mask]["body_id"].values
            channel_ids.extend(matches.tolist())

        channel_ids = list(set(channel_ids))
        channel_map[channel] = [int(x) for x in channel_ids]
        all_sensory_ids.extend(channel_ids)
        print(f"  {channel}: {len(channel_ids)} neurons")

    all_sensory_ids = list(set(all_sensory_ids))
    print(f"  Total sensory: {len(all_sensory_ids)}")

    # === Select readout (descending) neurons ===
    print("\nSelecting readout (descending) neurons...")
    dns = loader.select_dns()
    print(f"  Total DNs in BANC: {len(dns)}")

    decoder_groups = {}
    all_readout_ids = []

    for group, patterns in DN_GROUP_PATTERNS.items():
        group_ids = []
        for pattern in patterns:
            mask = neurons["cell_type"].str.contains(pattern, na=False) & \
                   neurons["body_id"].isin(dns)
            matches = neurons[mask]["body_id"].values

            # For turn groups: lateralize by soma_side
            if "left" in group:
                side_mask = neurons[mask]["soma_side"].str.upper() == "L"
                matches = neurons[mask][side_mask]["body_id"].values
            elif "right" in group:
                side_mask = neurons[mask]["soma_side"].str.upper() == "R"
                matches = neurons[mask][side_mask]["body_id"].values

            group_ids.extend(matches.tolist())

        group_ids = list(set(group_ids))
        decoder_groups[group] = [int(x) for x in group_ids]
        all_readout_ids.extend(group_ids)
        print(f"  {group}: {len(group_ids)} neurons")

    all_readout_ids = list(set(all_readout_ids))
    print(f"  Total readout: {len(all_readout_ids)}")

    # === Save outputs ===
    out_dir = ROOT / "data" / "banc"
    out_dir.mkdir(parents=True, exist_ok=True)

    sensory_path = out_dir / "sensory_ids_banc.npy"
    np.save(sensory_path, np.array(all_sensory_ids, dtype=np.int64))
    print(f"\nSaved {len(all_sensory_ids)} sensory IDs to {sensory_path}")

    readout_path = out_dir / "readout_ids_banc.npy"
    np.save(readout_path, np.array(all_readout_ids, dtype=np.int64))
    print(f"Saved {len(all_readout_ids)} readout IDs to {readout_path}")

    channel_path = out_dir / "channel_map_banc.json"
    with open(channel_path, "w") as f:
        json.dump(channel_map, f, indent=2)
    print(f"Saved channel map to {channel_path}")

    decoder_path = out_dir / "decoder_groups_banc.json"
    with open(decoder_path, "w") as f:
        json.dump(decoder_groups, f, indent=2)
    print(f"Saved decoder groups to {decoder_path}")

    # === Summary ===
    print(f"\n{'='*50}")
    print("BANC Population Summary")
    print(f"{'='*50}")
    print(f"Sensory: {len(all_sensory_ids)} neurons across {len(channel_map)} channels")
    for ch, ids in channel_map.items():
        print(f"  {ch}: {len(ids)}")
    print(f"Readout: {len(all_readout_ids)} DNs across {len(decoder_groups)} groups")
    for gr, ids in decoder_groups.items():
        print(f"  {gr}: {len(ids)}")

    # Coverage check against FlyWire
    if flywire_map:
        sensory_fw = sum(1 for s in all_sensory_ids if s in flywire_map)
        readout_fw = sum(1 for r in all_readout_ids if r in flywire_map)
        print(f"\nFlyWire cross-reference coverage:")
        print(f"  Sensory: {sensory_fw}/{len(all_sensory_ids)} ({100*sensory_fw/max(len(all_sensory_ids),1):.0f}%)")
        print(f"  Readout: {readout_fw}/{len(all_readout_ids)} ({100*readout_fw/max(len(all_readout_ids),1):.0f}%)")


if __name__ == "__main__":
    main()
