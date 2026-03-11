# Connectome Fly Brain

Two-project workspace: `plastic-fly/` (Python simulation) and `FlyBrainViz/` (Unity visualization).

## plastic-fly (Python)

FlyGym v1.2.1 + MuJoCo locomotion experiments with plastic vs fixed neural controllers.

### Run
```bash
cd plastic-fly
python run.py                         # default 20k steps
python run.py --total-steps 200000    # longer run for Unity viz
```

### Key constraints
- Use `flygym.Fly` / `flygym.SingleFlySimulation` (NOT old `flygym.mujoco.NeuroMechFly`)
- Must use `PreprogrammedSteps` for joint angles — raw sine CPG causes physics instability
- Warmup: ramp CPG magnitude from 0 over 500 steps
- Plasticity: lr=1e-5, weight_decay=1.0, cap=0.5, modulation_scale=0.15
- Structured logging lives in `structlog/` (NOT `logging/` — stdlib conflict)
- **ALWAYS checkpoint every result** — save incrementally per generation/step, never only at the end

### Dependencies
Python 3.10, flygym 1.2.1, torch, numpy 2.0.2, matplotlib, scipy

## FlyBrainViz (Unity)

Anatomical NeuroMechFly model built at runtime from MJCF XML + 71 STL meshes.

### Architecture
- `MjcfFlyBuilder` parses `neuromechfly.xml`, builds GameObject hierarchy, loads STL via `StlImporter`
- `FlyAnimator` drives playback from `timeseries_{plastic,fixed}.json`
- `OrbitCamera` provides mouse-controlled camera (left/right-drag orbit, scroll zoom, middle-drag pan)
- Coordinate conversion: MuJoCo (X=fwd,Y=left,Z=up) -> Unity (X=right,Y=up,Z=fwd)
  - Positions (true vectors): `(-my, mz, mx)`
  - Axes/quaternion imaginary (pseudovectors, det=-1): `(ay, -az, -ax)`

### Packages
- `com.unity.nuget.newtonsoft-json` 3.2.1
- `com.unity.inputsystem` 1.11.2
- Active Input Handling: Both (old + new)

### Data pipeline
`run.py` exports `timeseries_{plastic,fixed}.json` -> copy to `FlyBrainViz/Assets/Resources/`
`export_connectome_viz.py` runs LIF brain sim -> exports `connectome_activity.json` -> `FlyBrainViz/Assets/Resources/`

## brain-model (Drosophila_brain_model)

Clone of philshiu/Drosophila_brain_model — Brian2 LIF simulation of 139k FlyWire neurons.

### Run (via bridge script)
```bash
cd plastic-fly
python export_connectome_viz.py              # default: sugar GRNs at 100Hz, 250 neurons
python export_connectome_viz.py --freq 150   # custom stimulation frequency
```

### Key files
- `model.py` — Brian2 LIF model (create_model, run_trial, run_exp)
- `Completeness_783.csv` — 138,639 neuron IDs
- `Connectivity_783.parquet` — 15M synaptic connections
- `sez_neurons.pickle` — named SEZ neuron types

### Scene modes (Unity)
- **Connectome demo** (`SceneSetup.connectomeDemo = true`): single fly + ConnectomeViz brain cloud
- **Comparison mode** (`connectomeDemo = false`): two flies (plastic vs fixed) + NeuralNetworkViz

## Brain-Body Bridge

Closed-loop sensorimotor interface: FlyGym body ↔ Brian2 brain (139k FlyWire neurons).

### Run
```bash
cd plastic-fly
python experiments/closed_loop_walk.py                    # default 2k steps
python experiments/closed_loop_walk.py --body-steps 10000 # longer run
```

### Architecture
```
Body obs → SensoryEncoder → BrainRunner (Brian2 LIF) → DescendingDecoder → CPG modulation → Body
```

### Key files
- `bridge/interfaces.py` — BodyObservation, BrainInput, BrainOutput, LocomotionCommand
- `bridge/sensory_encoder.py` — body state → Poisson rates for sensory neurons
- `bridge/brain_runner.py` — incremental Brian2 stepping (PoissonGroup, not PoissonInput)
- `bridge/descending_decoder.py` — DN firing rates → forward/turn/freq/stance drives
- `bridge/flygym_adapter.py` — LocomotionCommand → CPG-modulated joint angles
- `experiments/closed_loop_walk.py` — main closed-loop experiment

### Key constraints
- Brain warmup: 200ms at baseline before loop starts (activity needs time to propagate)
- Brain window: 10ms per step, body runs 100 steps per brain step
- Readout neurons: must be direct downstream targets of sensory neurons (from connectome)
- CPG modulation (not raw torques): brain drives {forward, turn, frequency, stance}
