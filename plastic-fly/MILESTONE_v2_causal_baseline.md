# Bridge v2: Causal Baseline Milestone

**Date**: 2026-03-09
**Tag**: `bridge-v2-causal-baseline`

## Summary

Built an open brain-body bridge between a public FlyWire-based 139k-neuron Brian2
whole-brain model and an embodied fly simulator. Across 10/10 causal ablation tests,
decoder groups produced distinct and predictable effects on both internal locomotion
commands and physical behavior, including forward motion, turning, rhythm, and stance.

## Exact Reproduction

### Generate neuron populations
```bash
cd plastic-fly
python scripts/select_populations.py
```
Output: `data/sensory_ids.npy` (75), `data/readout_ids.npy` (47),
`data/channel_map.json`, `data/decoder_groups.json`

### Run sanity checks (8/8)
```bash
python experiments/sanity_checks.py
```

### Run closed loop (3/3 validation)
```bash
python experiments/closed_loop_walk.py --body-steps 5000
```

### Run ablation study (10/10 causal tests)
```bash
python experiments/ablation_study.py --body-steps 5000
```

## Configuration
- BridgeConfig defaults in `bridge/config.py`
- brain_dt_ms: 10.0, brain_warmup_ms: 200.0, body_steps_per_brain: 100
- max_rate_hz: 100.0, baseline_rate_hz: 10.0, rate_scale: 40.0
- w_syn: 0.275, f_poi: 250

## Populations
- Sensory: 75 neurons (20 gustatory + 33 proprioceptive + 15 mechanosensory + 7 vestibular)
- Readout: 47 neurons (22 annotated motor/descending cores + 25 downstream supplements)
- 370 direct sensory->readout connections, 39/47 directly reachable

## Key Results
- Forward ablation: 90% distance drop (0.21mm -> 0.02mm)
- Turn contrast: ablate_left=-0.509 < ablate_right=+0.435 turn_drive
- Rhythm ablation: step_frequency 1.45 -> 1.00, falls appeared
- Stance ablation: stance_gain 1.18 -> 1.00, contact profile changed
- Group balance: all 5 groups active 76-90%, mean rates 13-21Hz

## Dependencies
Python 3.10, flygym 1.2.1, brian2, torch, numpy 2.0.2, matplotlib, scipy, pandas
