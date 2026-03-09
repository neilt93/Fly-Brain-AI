# Rolling Summary

## Completed
- Brain-body bridge v2 built and frozen at tag bridge-v2-causal-baseline
- 10/10 causal ablation tests passing (single seed=42, real Brian2 brain)
- Fake-brain robustness study (3 seeds) — deterministic, zero variance, doesn't count
- Overnight researcher agent scaffolding set up

## Key Findings (single seed, real brain, seed=42)
- ablate_forward: fwd_drive 0.358 -> 0.000, path 8.40 -> 2.40mm
- ablate_turn_left: turn_drive -0.022 -> -0.509; ablate_turn_right: -> +0.435
- boost_turn_left: turn +0.606; boost_turn_right: turn -0.592 (clean mirror)
- ablate_rhythm: freq 1.45 -> 1.00
- ablate_stance: stance 1.18 -> 1.00
- Readout sparse: 3-16 of 47 neurons active per brain step, mean 8.1, 17.2Hz

## Current State
- A 10-seed real-brain robustness study may still be running or may have finished
- Check logs/robustness_real/robustness_results.json first
- If it exists: analyze it, report CIs, determine which effects are significant
- If it doesn't exist or is incomplete: run it (python experiments/robustness_study.py --seeds 10 --body-steps 5000 --output-dir logs/robustness_real)
- After robustness: longer timescale runs, minimal sufficient populations, plastic vs fixed comparison
