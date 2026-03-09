# Connectome-Driven Fly Behavior

Closed-loop sensorimotor simulation coupling the full Drosophila FlyWire connectome (139k neurons, 15M synapses) to a MuJoCo biomechanical fly body. Three adaptive behaviors emerge from wiring alone, without learning.

## Setup

```bash
# 1. Clone with brain-model submodule
git clone --recurse-submodules <repo-url>
cd plastic-fly

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Verify brain-model data exists
ls ../brain-model/Connectivity_783.parquet  # 97MB, 15M synapses
ls ../brain-model/Completeness_783.csv      # 139k neuron IDs
```

### Requirements

- Python 3.10+
- ~8GB RAM (Brian2 network build)
- No GPU required

## Reproduce Key Results

### 1. Causal Ablation (10/10 tests pass, ~8 min)

Targeted silencing of locomotion neuron groups causes predicted behavioral deficits.

```bash
python -m experiments.ablation_study --readout-version 2
```

Expected: `Causal tests: 10/10 passed` — forward ablation reduces distance 53%, turn ablation shifts heading, rhythm ablation halves step frequency.

### 2. Odor Valence Discrimination (6/6 tests pass, ~9 min)

Attractive (Or42b/DM1) and aversive (Or85a/DM5) odors produce opposite turning.

```bash
python -m experiments.odor_valence
```

Expected: `6/6 tests passed` — DM1 turns toward odor, DM5 turns away, valence contrast +0.066, abolished in shuffled connectome.

### 3. Looming Escape (5/5 tests pass, ~12 min)

LPLC2 looming detector neurons drive contralateral escape turning via descending neurons.

```bash
python -m experiments.looming
```

Expected: `5/5 tests passed` — escape index 1.11, loom left produces rightward turn (+0.36), loom right produces leftward turn (-0.75), 21x weaker in shuffled connectome.

### Quick Smoke Test (~2 min)

Run ablation with fewer body steps to verify the pipeline works:

```bash
python -m experiments.ablation_study --body-steps 2000 --readout-version 2
```

## Architecture

```
Body observation -> SensoryEncoder -> Brian2 LIF (139k neurons) -> DescendingDecoder -> CPG modulation -> Body
     ^                                                                                                    |
     +----------------------------------------------------------------------------------------------------+
```

### Bridge Components

| Module | File | Role |
|---|---|---|
| SensoryEncoder | `bridge/sensory_encoder.py` | Body state -> Poisson rates for 275-485 sensory neurons |
| BrainRunner | `bridge/brain_runner.py` | Incremental Brian2 stepping (20-100ms windows) |
| DescendingDecoder | `bridge/descending_decoder.py` | DN firing rates -> forward/turn/freq/stance drives |
| FlygymAdapter | `bridge/flygym_adapter.py` | LocomotionCommand -> CPG-modulated joint angles |

### Sensory Channels

| Channel | Neurons | Input |
|---|---|---|
| Gustatory | 20 sugar GRNs | Contact forces |
| Proprioceptive | 33 SEZ ascending | Joint angles + velocities |
| Mechanosensory | 15 SEZ ascending | Per-leg contact forces |
| Vestibular | 7 SEZ ascending | Body velocity + orientation |
| Olfactory L/R | 100 ORNs per side | Odor intensity at antennae |
| Visual L/R | ~50 R7/R8 per side | Mean eye luminance |
| LPLC2 L/R | 108L + 102R | Looming intensity |

### Readout Versions

| Version | Neurons | Used by |
|---|---|---|
| v1 | 47 | Legacy (annotated motor types only) |
| v2 | 204 | Ablation, odor valence (hybrid annotated + connectivity) |
| v3 | 359 | Visual experiments (adds DN annotations) |
| v4 | 389 | Looming (v3 + LPLC2 DN targets) |

## Experiments

| Experiment | Script | Key Result |
|---|---|---|
| Causal ablation | `experiments/ablation_study.py` | 10/10 tests, 5 command + 5 behavioral |
| Odor valence | `experiments/odor_valence.py` | 6/6 tests, opposite valence from wiring |
| Looming escape | `experiments/looming.py` | 5/5 tests, escape index 1.11 |
| Dose-response | `experiments/ablation_extended.py` | Graded ablation depth -> graded deficit |
| Sensory perturbation | `experiments/sensory_perturbation.py` | 9/10 tests, asymmetric input -> turn |
| Closed-loop walk | `experiments/closed_loop_walk.py` | Brain-driven walking at 22mm/s |
| Robustness study | `experiments/robustness_study.py` | 10-seed causal replication |

## Data Files

Bridge configuration data in `data/`:

- `sensory_ids*.npy` — FlyWire neuron IDs for sensory populations
- `readout_ids*.npy` — FlyWire neuron IDs for descending readout
- `channel_map*.json` — sensory channel -> neuron ID mapping
- `decoder_groups*.json` — readout neuron -> motor group assignment

Brain model data in `../brain-model/`:

- `Completeness_783.csv` — 138,639 neuron IDs
- `Connectivity_783.parquet` — 15M synaptic connections (97MB)
- `sez_neurons.pickle` — 106 named SEZ neuron types

## Figures

Generate proof figures from saved results:

```bash
python figures/gen_looming_proof.py          # -> figures/looming_escape_proof.png
python figures/gen_sensory_perturbation.py   # -> figures/sensory_perturbation.png
```
