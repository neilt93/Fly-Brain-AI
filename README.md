# Connectome Fly Brain

Closed-loop simulation coupling the complete Drosophila FlyWire connectome (139K neurons, 54.5M synapses) to a MuJoCo biomechanical fly body via FlyGym v1.2.1. Three adaptive behaviors emerge from wiring alone, without learning or parameter fitting.

## Components

| Component | Description | Source |
|-----------|-------------|--------|
| **plastic-fly/** | Brain-body bridge, experiments, analysis | This work |
| **brain-model/** | Brian2 LIF brain simulation (139K neurons) | [Shiu et al. 2023](https://doi.org/10.1101/2023.05.02.539144) |
| **FlyBrainViz/** | Unity 3D visualization of fly behavior | This work |
| **eigenlayer/** | Connectome topology for interpretability research | This work |

## Quick Start

```bash
git clone --recurse-submodules https://github.com/neilt93/Fly-Brain-AI.git
cd Fly-Brain-AI/plastic-fly
pip install -r requirements.txt

# Reproduce all three main results (~30 min total)
python -m experiments.ablation_study --readout-version 2   # 10/10 causal ablation
python -m experiments.odor_valence                          # 6/6 odor valence
python -m experiments.looming                               # 5/5 looming escape
```

## Key Results

| Experiment | Result | Control |
|------------|--------|---------|
| Causal ablation | 10/10 tests pass (forward drive -90%, distance -46%) | Shuffled: 3/10 trivial |
| Odor valence | 6/6 tests, n=10 seeds (opposite turning for attractive vs aversive) | Shuffled: valence disappears |
| Looming escape | 5/5 tests, escape index 1.11 | Shuffled: 0.053 (21x collapse) |
| VNC validation | 19/20 tests, full MANC VNC (13K neurons) | Forward ablation: -97% |
| DN segregation | 6 modalities, Jaccard 0.00-0.06 | Shuffled: 4.6-21x more overlap |
| DNb05 bottleneck | 2 neurons gate thermo/hygro (specificity 16.4x) | Other modalities: 0-0.5% affected |

## Architecture

```
Sensory Input -> Sensory Encoder -> FlyWire Brain (139K LIF) -> Descending Decoder -> FlyGym Body
     ^            (275-485 neurons)    (Brian2, 15M connections)    (204-389 DNs)          |
     +--------------------------------------- closed loop --------------------------------+
```

Full motor pathway with connectome-emergent rhythm (NEW):
```
Brain -> Descending Decoder -> BANC VNC (8K neurons, firing-rate ODE) -> MN Decoder (390 MNs) -> 42 joints
```

### BANC VNC: Walking rhythm from connectome wiring

A Pugliese-style firing-rate model on the BANC female VNC connectome produces flex/ext anti-phase oscillation on 4/6 legs from network dynamics alone — no imposed CPG.

```bash
# Reproduce key results (~5 min, requires BANC database)
python scripts/reproduce_banc_vnc.py

# Run 5-phase demo (forward, turn L/R, escape, recovery)
python experiments/banc_vnc_demo.py

# Run pytest suite (19 tests)
pytest tests/test_banc_vnc_pipeline.py -v
```

| Result | Value |
|--------|-------|
| Anti-phase legs | 4/6 from connectome wiring |
| Forward ablation | -91% distance (causal) |
| Walking distance | 2.9mm / 5k steps |
| Multi-seed stability | 10/10 stable, 9/10 forward |
| Turning | +74 deg left, -6 deg right (stride asymmetry) |
| Escape pathway | Lateralized via DNg02 giant fiber |

## Documentation

- **[Paper draft](plastic-fly/PAPER_DRAFT.md)** — full manuscript with methods and results
- **[Abstract](plastic-fly/ABSTRACT.md)** — 4-sentence summary
- **[Bridge README](plastic-fly/bridge/README_RELEASE.md)** — interface architecture and API
- **[Experiment guide](plastic-fly/README.md)** — reproduction instructions for all experiments

## Requirements

- Python 3.10+
- ~8 GB RAM (Brian2 network build)
- No GPU required
- Key packages: flygym 1.2.1, brian2, torch, numpy, scipy, matplotlib

## References

- Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*, 634, 124-138.
- Shiu et al. (2023). A leaky integrate-and-fire model based on the connectome of the entire adult Drosophila brain. *bioRxiv*.
- Wang et al. (2024). FlyGym: A comprehensive toolkit for biomechanical simulations of Drosophila. *Nature Methods*.
- Takemura et al. (2024). A connectome of the male Drosophila ventral nerve cord. *eLife*.
- Azevedo et al. (2024). Tools for comprehensive reconstruction and analysis of Drosophila motor circuits. *bioRxiv* (BANC).
- Pugliese et al. (2025). Walking emerges from a firing rate model of the Drosophila VNC. *bioRxiv*.

## License

MIT
