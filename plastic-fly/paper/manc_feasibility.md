# MANC VNC Connectome Feasibility Assessment -- Phase 1

## Verdict: Highly Feasible

The VNC model would be ~6x smaller and faster than our brain model. Main work is data parsing, DN mapping, and motor output definition.

## Scale Comparison

| Metric | Our Brain (FlyWire) | MANC VNC | MaleCNS (brain+VNC) |
|---|---|---|---|
| Neurons | 138,639 | ~23,000 | ~166,000 |
| Connections | 15M | ~1.15M | ~312M |
| Synapses | ~54.5M | ~10M pre, ~74M post | -- |

## Data Access

MANC is fully public (CC-BY 4.0):
- **Apache Arrow Feather files** from `male-cns.janelia.org/download/`
  - `connectome-weights-male-cns-v0.9-minconf-0.5.feather` (1.1 GB) -- equivalent to our Connectivity_783.parquet
  - `body-annotations-male-cns-v0.9-minconf-0.5.feather` (13 MB) -- neuron types
  - `body-neurotransmitters-male-cns-v0.9.feather` (42 MB) -- E/I predictions
- **neuPrint API** at `neuprint.janelia.org` (dataset `manc:v1.2.1`)
- Readable by pandas/pyarrow -- same pipeline as our parquet loading

## MANC Neuron Breakdown

| Class | Count | Notes |
|---|---|---|
| Intrinsic (IN) | ~13,060 | Local VNC processing |
| Descending (DN) | 1,328 | Brain -> VNC commands |
| Motor neurons (MN) | 733 total | Output to muscles |
| -- Leg MNs | 392 | 142 T1, 119 T2, 131 T3 |
| Sensory (SN) | 5,927 | Peripheral input |
| Ascending (AN) | 1,865 | VNC -> brain feedback |
| Premotor | 3,142 | Direct MN input |

## DN Mapping (Critical Path)

- Sturner et al. 2025 matched DN types across FAFB/FANC/MANC (97% stereotypy)
- We have `sturner_FAFB_DNs.tsv` with 1,315 FlyWire DN entries
- Challenge: our 47 readout neurons use SEZ names (mothership, shark), not standard DN names
- Solution: map via Sturner supplemental files, or use MaleCNS (intact neck connective)
- Key: DN->MN connections are sparse (~7% of MN input) -- must model premotor layer

## Prior Art: Pugliese et al. 2025

"Connectome simulations identify a central pattern generator circuit for fly walking" (bioRxiv Sept 2025)
- Simulated 4,604 T1 neurons (not full 23K VNC)
- Firing rate model (JAX/JIT) + LIF validation
- Discovered DNg100 as walking command neuron (confirmed with optogenetics)
- Minimal CPG: 3 neurons (E1, E2, I1) -- rhythm is a structural property
- Network structure trumps precise synapse counts

## MaleCNS v0.9 (October 2025)

Single male fly with intact neck connective: brain + VNC in one volume. 166K neurons, 312M synapses. Could eventually eliminate brain-VNC bridge entirely. Phase 3 goal.

## Implementation Plan

1. Download Feather connectivity files (~1.1 GB)
2. Parse into Brian2-compatible format (same as brain model pipeline)
3. Map brain DN outputs -> MANC DN inputs (type-based via Sturner)
4. Define motor readout: leg MN firing rates -> joint angles
5. Build Brian2 LIF (~23K neurons) -- estimated <1s build, <0.5s warmup
6. Validate: does fly walk without preprogrammed CPG?
