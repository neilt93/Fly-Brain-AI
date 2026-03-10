"""
Quick diagnostic: test whether Brian2 VNC produces rhythmic MN output.

Runs the VNC standalone (no brain, no FlyGym) with tonic DN input and
measures temporal structure in MN firing rates across 10 time windows.

Expected: with adaptation, flexor and extensor MN pools should alternate,
producing oscillatory patterns at ~10-15Hz.
"""

import sys
import numpy as np
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.vnc_connectome import (
    VNCInput, VNCConfig, Brian2VNCRunner,
)


def analyze_mn_oscillation(runner, n_windows=20, window_ms=20.0):
    """Run VNC with tonic input and check for oscillation in MN rates."""
    # Tonic input: constant forward drive
    vnc_input = VNCInput(group_rates={
        "forward": 30.0,
        "turn_left": 5.0,
        "turn_right": 5.0,
        "rhythm": 15.0,
        "stance": 10.0,
    })

    # Collect MN rates over time
    rate_history = []
    for i in range(n_windows):
        output = runner.step(vnc_input, sim_ms=window_ms)
        rate_history.append(output.firing_rates_hz.copy())

    rates = np.array(rate_history)  # (n_windows, n_mn)

    # --- Basic stats ---
    mean_per_window = rates.mean(axis=1)
    std_per_window = rates.std(axis=1)
    active_per_window = (rates > 0).sum(axis=1)

    print(f"\n{'='*60}")
    print(f"MN OSCILLATION DIAGNOSTIC ({n_windows} windows x {window_ms}ms)")
    print(f"{'='*60}")
    print(f"Total MNs: {rates.shape[1]}")
    print(f"Mean rate per window: min={mean_per_window.min():.1f}, max={mean_per_window.max():.1f}, "
          f"std={mean_per_window.std():.1f} Hz")
    print(f"Active MNs per window: min={active_per_window.min()}, max={active_per_window.max()}")

    # --- Check temporal variation (oscillation) ---
    # For each MN, compute coefficient of variation across time windows
    mn_means = rates.mean(axis=0)
    mn_stds = rates.std(axis=0)
    active_mns = mn_means > 1.0
    if active_mns.sum() > 0:
        cvs = mn_stds[active_mns] / mn_means[active_mns]
        print(f"\nTemporal variation (active MNs with mean > 1 Hz):")
        print(f"  N active: {active_mns.sum()}")
        print(f"  Coeff of variation: mean={cvs.mean():.3f}, median={np.median(cvs):.3f}")
        print(f"  Highly variable (CV>0.3): {(cvs > 0.3).sum()}")
        print(f"  Oscillating (CV>0.5): {(cvs > 0.5).sum()}")
    else:
        print("\nWARNING: No MNs with mean rate > 1Hz!")

    # --- Check for anti-phase between putative flexor/extensor groups ---
    # Load MN joint mapping to identify flexors vs extensors
    import json
    mapping_path = Path(__file__).resolve().parent.parent / "data" / "mn_joint_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            mn_map = json.load(f)

        # Group MN indices by leg and direction
        mn_body_ids = runner._mn_body_ids
        from bridge.vnc_connectome import LEG_ORDER
        for leg_idx, leg in enumerate(LEG_ORDER):
            ext_indices = []
            flex_indices = []
            for j, bid in enumerate(mn_body_ids):
                bid_str = str(bid)
                if bid_str in mn_map:
                    entry = mn_map[bid_str]
                    if entry.get("leg") == leg:
                        if entry.get("direction", 0) > 0:
                            ext_indices.append(j)
                        elif entry.get("direction", 0) < 0:
                            flex_indices.append(j)

            if ext_indices and flex_indices:
                ext_rates = rates[:, ext_indices].mean(axis=1)
                flex_rates = rates[:, flex_indices].mean(axis=1)
                # Anti-correlation between ext and flex = good oscillation
                if ext_rates.std() > 0.1 and flex_rates.std() > 0.1:
                    corr = np.corrcoef(ext_rates, flex_rates)[0, 1]
                    print(f"  {leg}: ext({len(ext_indices)}) vs flex({len(flex_indices)}) "
                          f"corr={corr:+.3f} "
                          f"ext_range=[{ext_rates.min():.0f},{ext_rates.max():.0f}] "
                          f"flex_range=[{flex_rates.min():.0f},{flex_rates.max():.0f}] "
                          f"{'ANTI-PHASE' if corr < -0.3 else 'NO RHYTHM'}")
                else:
                    print(f"  {leg}: ext({len(ext_indices)}) vs flex({len(flex_indices)}) "
                          f"LOW VARIANCE ext_std={ext_rates.std():.1f} flex_std={flex_rates.std():.1f}")

    # --- Print per-window detail ---
    print(f"\nPer-window detail:")
    for i in range(n_windows):
        top5 = rates[i].argsort()[-5:][::-1]
        top5_rates = [f"{rates[i, j]:.0f}" for j in top5]
        print(f"  t={i*window_ms:.0f}ms: mean={mean_per_window[i]:.1f}Hz "
              f"active={active_per_window[i]} top5=[{','.join(top5_rates)}]")

    return rates


def main():
    print("Building Brian2 VNC with adaptation...")
    t0 = time()
    cfg = VNCConfig()
    runner = Brian2VNCRunner(cfg=cfg, warmup=True)
    print(f"Build + warmup: {time()-t0:.1f}s")

    print("\nRunning oscillation test...")
    t0 = time()
    rates = analyze_mn_oscillation(runner, n_windows=20, window_ms=20.0)
    print(f"\nTest completed in {time()-t0:.1f}s")

    # Summary verdict
    mn_means = rates.mean(axis=0)
    active = mn_means > 1.0
    if active.sum() > 0:
        cvs = rates[:, active].std(axis=0) / mn_means[active]
        oscillating = (cvs > 0.5).sum()
        print(f"\nVERDICT: {oscillating}/{active.sum()} active MNs show oscillation (CV>0.5)")
        if oscillating > active.sum() * 0.1:
            print("  VNC produces rhythmic MN output! PASS")
        else:
            print("  Insufficient oscillation -- needs parameter tuning")
    else:
        print("\nVERDICT: No active MNs -- check tonic drive / w_syn")


if __name__ == "__main__":
    main()
