"""
Blind tests against published literature for the Connectome Fly Brain project.

Test 1: DNb05 multimodal input audit
Test 2: Steering DN overlap with Yang 2024
Test 3: DNp44 hygrosensory check (Marin 2020)
"""
import pandas as pd
import numpy as np
import json
import pickle
import os
from collections import defaultdict

BASE = "C:/Users/neilt/Connectome Fly Brain"
BRAIN_MODEL = os.path.join(BASE, "brain-model")
BRIDGE_DATA = os.path.join(BASE, "plastic-fly/data")

# ── Load shared data ────────────────────────────────────────────────────
print("Loading data...")
ann = pd.read_csv(os.path.join(BRAIN_MODEL, "flywire_annotations_matched.csv"), low_memory=False)
conn = pd.read_parquet(os.path.join(BRAIN_MODEL, "Connectivity_783.parquet"))
with open(os.path.join(BRAIN_MODEL, "sez_neurons.pickle"), "rb") as f:
    sez_neurons = pickle.load(f)

# Build annotation lookup: root_id -> row
ann_by_id = ann.set_index("root_id")

# Build cell_type -> root_id lookup
type_to_ids = defaultdict(list)
for _, row in ann[["root_id", "cell_type"]].dropna(subset=["cell_type"]).iterrows():
    type_to_ids[row["cell_type"]].append(row["root_id"])

# Build root_id -> cell_type lookup
id_to_type = dict(zip(ann["root_id"], ann["cell_type"]))
id_to_class = dict(zip(ann["root_id"], ann["cell_class"]))
id_to_superclass = dict(zip(ann["root_id"], ann["super_class"]))

# Load bridge sensory IDs (somatosensory channel)
with open(os.path.join(BRIDGE_DATA, "channel_map_v3.json")) as f:
    channel_map = json.load(f)
bridge_somatosensory_ids = set()
for ch in ["proprioceptive", "mechanosensory", "vestibular"]:
    bridge_somatosensory_ids.update(channel_map.get(ch, []))
bridge_gustatory_ids = set(channel_map.get("gustatory", []))

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: DNb05 multimodal input audit
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("TEST 1: DNb05 MULTIMODAL INPUT AUDIT")
print("=" * 72)

dnb05_ids = type_to_ids.get("DNb05", [])
print(f"\nDNb05 neurons found: {len(dnb05_ids)}")
for nid in dnb05_ids:
    row = ann_by_id.loc[nid]
    side = row.get("side", "unknown")
    print(f"  ID {nid} (side: {side})")


def classify_modality(pre_id):
    """Classify a presynaptic neuron by sensory modality."""
    ctype = id_to_type.get(pre_id)
    cclass = id_to_class.get(pre_id)
    sclass = id_to_superclass.get(pre_id)

    if pd.isna(ctype):
        ctype = None
    if pd.isna(cclass):
        cclass = None
    if pd.isna(sclass):
        sclass = None

    # Visual: LPLC, LC, R7, R8, lobula/medulla types, visual projection
    if ctype and any(
        ctype.startswith(p)
        for p in ["LPLC", "LC", "R7", "R8", "T1", "T2", "T3", "T4", "T5",
                   "Tm", "Mi", "Dm", "Pm", "L1", "L2", "L3", "L4", "L5",
                   "C2", "C3", "Lawf", "Li"]
    ):
        return "visual"
    if sclass in ["visual_projection", "visual_centrifugal"]:
        return "visual"
    if cclass == "visual":
        return "visual"

    # Olfactory: ORN*, projection neurons from antennal lobe
    if ctype and (ctype.startswith("ORN") or ctype.startswith("Or")):
        return "olfactory"
    if cclass == "olfactory":
        return "olfactory"

    # Thermosensory: TRN_VP*
    if ctype and ctype.startswith("TRN"):
        return "thermosensory"
    if cclass == "thermosensory":
        return "thermosensory"

    # Hygrosensory: HRN_VP*
    if ctype and ctype.startswith("HRN"):
        return "hygrosensory"
    if cclass == "hygrosensory":
        return "hygrosensory"

    # Auditory: JO neurons
    if ctype and ctype.startswith("JO"):
        return "auditory"
    if cclass == "mechanosensory" and sclass == "sensory":
        # Some JO subtypes or antennal mechanosensory
        return "mechanosensory"

    # Gustatory
    if ctype and "GRN" in str(ctype):
        return "gustatory"
    if cclass == "gustatory":
        return "gustatory"

    # Somatosensory (from bridge)
    if pre_id in bridge_somatosensory_ids:
        return "somatosensory_bridge"

    # Other sensory
    if sclass == "sensory":
        return "other_sensory"
    if sclass == "ascending":
        return "ascending"

    # Central / descending / optic
    if sclass in ["central", "optic"]:
        return "central"
    if sclass == "descending":
        return "descending"

    return "unknown"


# For each DNb05 neuron, get all presynaptic inputs
total_by_modality = defaultdict(int)
total_synapses = 0

for nid in dnb05_ids:
    inputs = conn[conn["Postsynaptic_ID"] == nid]
    print(f"\n  DNb05 neuron {nid}: {len(inputs)} presynaptic partners, "
          f"{inputs['Connectivity'].sum()} total input synapses")

    modality_counts = defaultdict(int)
    modality_partners = defaultdict(int)
    modality_examples = defaultdict(list)

    for _, edge in inputs.iterrows():
        pre_id = edge["Presynaptic_ID"]
        syn_count = edge["Connectivity"]
        mod = classify_modality(pre_id)
        modality_counts[mod] += syn_count
        modality_partners[mod] += 1
        ctype = id_to_type.get(pre_id)
        if ctype and len(modality_examples[mod]) < 3:
            modality_examples[mod].append(f"{ctype}({syn_count})")

    for mod in sorted(modality_counts, key=modality_counts.get, reverse=True):
        total_by_modality[mod] += modality_counts[mod]
        examples = ", ".join(modality_examples[mod]) if modality_examples[mod] else ""
        print(f"    {mod:25s}: {modality_counts[mod]:5d} synapses "
              f"({modality_partners[mod]:3d} partners) {examples}")

    total_synapses += inputs["Connectivity"].sum()

print(f"\n  COMBINED DNb05 input summary ({total_synapses} total synapses):")
for mod in sorted(total_by_modality, key=total_by_modality.get, reverse=True):
    pct = 100.0 * total_by_modality[mod] / total_synapses
    print(f"    {mod:25s}: {total_by_modality[mod]:5d} ({pct:5.1f}%)")

# Verdict
sensory_mods = {"visual", "olfactory", "thermosensory", "hygrosensory",
                "auditory", "gustatory", "mechanosensory", "somatosensory_bridge",
                "other_sensory"}
found_sensory = {m for m in total_by_modality if m in sensory_mods and total_by_modality[m] > 0}
print(f"\n  VERDICT: DNb05 receives input from {len(found_sensory)} sensory modalities: "
      f"{', '.join(sorted(found_sensory))}")
if len(found_sensory) > 2:
    print("  => DNb05 is MULTIMODAL, not exclusively thermo/hygro")
elif found_sensory <= {"thermosensory", "hygrosensory"}:
    print("  => DNb05 is thermo/hygro exclusive")
else:
    print(f"  => DNb05 has limited sensory input from: {found_sensory}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Steering DN overlap with Yang 2024
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("TEST 2: STEERING DN OVERLAP WITH YANG 2024")
print("=" * 72)

yang_dns = ["DNa01", "DNa02", "DNb05", "DNb06", "DNg13"]
print(f"\nYang 2024 identified 5 steering DNs: {', '.join(yang_dns)}")

# Load ALL decoder group versions for comparison
decoder_versions = {
    "v1 (47 neurons)": "decoder_groups.json",
    "v2 (204 neurons)": "decoder_groups_v2.json",
    "v3 (359 neurons)": "decoder_groups_v3.json",
    "v4_looming (389 neurons)": "decoder_groups_v4_looming.json",
}

for version_name, filename in decoder_versions.items():
    filepath = os.path.join(BRIDGE_DATA, filename)
    with open(filepath) as f:
        dg = json.load(f)

    # Collect ALL readout IDs from this version
    all_readout_ids = set()
    for group_ids in dg.values():
        all_readout_ids.update(group_ids)

    print(f"\n  --- {version_name} ({filename}) ---")
    print(f"  Total unique readout neurons: {len(all_readout_ids)}")

    for dn_type in yang_dns:
        dn_ids = type_to_ids.get(dn_type, [])
        in_readout = [nid for nid in dn_ids if nid in all_readout_ids]

        if not in_readout:
            print(f"    {dn_type}: NOT in readout pool (0/{len(dn_ids)} neurons)")
            continue

        # Which groups is each neuron in?
        for nid in in_readout:
            groups_in = []
            for gname, gids in dg.items():
                if nid in gids:
                    groups_in.append(gname)
            side = ann_by_id.loc[nid].get("side", "?") if nid in ann_by_id.index else "?"
            print(f"    {dn_type} ({side}, ID={nid}): IN readout -> {', '.join(groups_in)}")

        not_in = [nid for nid in dn_ids if nid not in all_readout_ids]
        if not_in:
            for nid in not_in:
                side = ann_by_id.loc[nid].get("side", "?") if nid in ann_by_id.index else "?"
                print(f"    {dn_type} ({side}, ID={nid}): NOT in readout")

# Focus verdict on v3 (latest non-looming, matches bridge v2 memory notes)
print(f"\n  YANG 2024 VERDICT (using v3 decoder groups):")
with open(os.path.join(BRIDGE_DATA, "decoder_groups_v3.json")) as f:
    dg_v3 = json.load(f)
all_v3 = set()
for gids in dg_v3.values():
    all_v3.update(gids)

found_count = 0
in_turning = 0
for dn_type in yang_dns:
    dn_ids = type_to_ids.get(dn_type, [])
    in_readout = [nid for nid in dn_ids if nid in all_v3]
    found_count += len(in_readout)
    for nid in in_readout:
        in_turn = nid in dg_v3.get("turn_left_ids", []) or nid in dg_v3.get("turn_right_ids", [])
        if in_turn:
            in_turning += 1

print(f"  {found_count}/{sum(len(type_to_ids.get(t,[])) for t in yang_dns)} "
      f"Yang steering DN neurons found in readout pool")
print(f"  {in_turning}/{found_count} of those are assigned to turning groups")

# ═══════════════════════════════════════════════════════════════════════
# TEST 3: DNp44 hygrosensory check (Marin 2020)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("TEST 3: DNp44 HYGROSENSORY CHECK (MARIN 2020)")
print("=" * 72)

dnp44_ids = type_to_ids.get("DNp44", [])
print(f"\nDNp44 neurons found: {len(dnp44_ids)}")

for nid in dnp44_ids:
    row = ann_by_id.loc[nid]
    side = row.get("side", "unknown")
    print(f"  ID {nid} (side: {side})")

# Check readout pool membership across versions
print("\n  Readout pool membership:")
for version_name, filename in decoder_versions.items():
    with open(os.path.join(BRIDGE_DATA, filename)) as f:
        dg = json.load(f)
    all_ids = set()
    for gids in dg.values():
        all_ids.update(gids)

    in_pool = [nid for nid in dnp44_ids if nid in all_ids]
    if in_pool:
        for nid in in_pool:
            groups = [g for g, ids in dg.items() if nid in ids]
            print(f"    {version_name}: YES (ID {nid}) -> {', '.join(groups)}")
    else:
        print(f"    {version_name}: NO")

# Check hygrosensory input
print("\n  Hygrosensory input analysis:")
hygro_ids = set(type_to_ids.get("HRN_VP1d", []) + type_to_ids.get("HRN_VP4", []) +
                type_to_ids.get("HRN_VP1l", []) + type_to_ids.get("HRN_VP5", []))
# Also get all hygrosensory by class
hygro_class_ids = set(ann[ann["cell_class"] == "hygrosensory"]["root_id"].tolist())
all_hygro = hygro_ids | hygro_class_ids
print(f"  Total hygrosensory neurons in connectome: {len(all_hygro)}")

# Also get all thermosensory
thermo_class_ids = set(ann[ann["cell_class"] == "thermosensory"]["root_id"].tolist())
print(f"  Total thermosensory neurons in connectome: {len(thermo_class_ids)}")

for nid in dnp44_ids:
    inputs = conn[conn["Postsynaptic_ID"] == nid]
    side = ann_by_id.loc[nid].get("side", "?")
    print(f"\n  DNp44 {nid} ({side}): {len(inputs)} presynaptic partners, "
          f"{inputs['Connectivity'].sum()} total input synapses")

    # Count hygrosensory inputs
    hygro_inputs = inputs[inputs["Presynaptic_ID"].isin(all_hygro)]
    thermo_inputs = inputs[inputs["Presynaptic_ID"].isin(thermo_class_ids)]

    hygro_syn = hygro_inputs["Connectivity"].sum()
    thermo_syn = thermo_inputs["Connectivity"].sum()
    total_syn = inputs["Connectivity"].sum()

    print(f"    Hygrosensory input: {hygro_syn} synapses from "
          f"{len(hygro_inputs)} partners ({100*hygro_syn/total_syn:.1f}%)")
    print(f"    Thermosensory input: {thermo_syn} synapses from "
          f"{len(thermo_inputs)} partners ({100*thermo_syn/total_syn:.1f}%)")

    # Show hygro partner types
    if len(hygro_inputs) > 0:
        for _, edge in hygro_inputs.iterrows():
            pre_type = id_to_type.get(edge["Presynaptic_ID"], "?")
            print(f"      <- {pre_type} (ID {edge['Presynaptic_ID']}): "
                  f"{edge['Connectivity']} synapses")

    # Check indirect path: hygro -> ? -> DNp44 (2-hop)
    print(f"\n    2-hop hygro pathway analysis:")
    # Find all direct presynaptic partners of DNp44
    direct_partners = set(inputs["Presynaptic_ID"].tolist())

    # Find which of those receive hygro input
    hygro_relay_count = 0
    hygro_relay_synapses = 0
    for partner_id in direct_partners:
        partner_inputs = conn[(conn["Postsynaptic_ID"] == partner_id) &
                              (conn["Presynaptic_ID"].isin(all_hygro))]
        if len(partner_inputs) > 0:
            hygro_relay_count += 1
            hygro_relay_synapses += partner_inputs["Connectivity"].sum()
            if hygro_relay_count <= 5:  # Show first 5
                partner_type = id_to_type.get(partner_id, "?")
                hygro_types = [id_to_type.get(pid, "?")
                               for pid in partner_inputs["Presynaptic_ID"]]
                print(f"      Relay: hygro -> {partner_type}({partner_id}) -> DNp44 "
                      f"[{partner_inputs['Connectivity'].sum()} hygro synapses from {hygro_types}]")

    print(f"    Total 2-hop hygro relays: {hygro_relay_count} partners receive "
          f"{hygro_relay_synapses} hygro synapses")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print("""
Test 1 (DNb05 multimodal): Checks whether DNb05 is truly thermo/hygro
  exclusive or receives broader multimodal input in the FlyWire connectome.

Test 2 (Yang 2024 steering DNs): Checks whether the 5 experimentally
  validated steering DNs (DNa01, DNa02, DNb05, DNb06, DNg13) are
  represented in our bridge readout population and decoder groups.

Test 3 (Marin 2020 DNp44): Checks whether DNp44 receives hygrosensory
  input as reported by Marin et al. 2020.
""")
