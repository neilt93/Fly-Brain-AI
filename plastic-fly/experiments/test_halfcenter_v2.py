#!/usr/bin/env python3
"""
Half-center boost v2: Biased-premotor cross-inhibition.

The v1 approach boosted INs that connect to BOTH flex and ext -- this just
suppressed everything because those INs inhibit both pools simultaneously.

The v2 approach instead:
1. Identifies inhibitory premotor INs that are BIASED toward one pool
   (e.g., >60% of their MN synapses go to flexors within a leg)
2. Boosts only their connections to the OPPOSITE pool
   (flexor-biased IN -> extensor MNs get boosted, and vice versa)

This creates the reciprocal inhibition needed for half-center oscillation:
   Flexor-pool INs ---| Extensor MNs
   Extensor-pool INs ---| Flexor MNs

Also tests:
- "Premotor-only boost": only boost INs that directly synapse to MNs
- Reducing base inh_scale to 1.0 (so only the half-center paths are strong)
- Different bias thresholds (0.5, 0.6, 0.7)
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.vnc_firing_rate import (
    FiringRateVNCConfig, FiringRateVNCRunner, LEG_ORDER, NT_SIGN,
)


ROOT = Path(__file__).resolve().parent.parent


def apply_biased_crossinhibition(
    runner: FiringRateVNCRunner,
    boost: float,
    bias_threshold: float = 0.6,
    only_premotor: bool = True,
):
    """Apply reciprocal inhibition boost based on pool-biased INs.

    For each inhibitory premotor IN within each leg:
    - Compute fraction of inhibitory MN synapses going to flexor vs extensor
    - If >bias_threshold to flexors: it's "flexor-biased" -> boost its ext connections
    - If >bias_threshold to extensors: it's "extensor-biased" -> boost its flex connections

    Args:
        runner: Initialized FiringRateVNCRunner
        boost: Multiplicative factor for cross-pool connections
        bias_threshold: Fraction threshold for classifying an IN as biased (0.5-0.8)
        only_premotor: If True, only consider premotor INs (directly synapsing to MNs)
    """
    from scipy.sparse import csr_matrix

    W = runner.W.copy()

    # Build per-leg flex/ext MN index sets
    leg_flex_idx = {}
    leg_ext_idx = {}
    for i, bid in enumerate(runner.mn_body_ids):
        model_idx = runner._bodyid_to_idx[int(bid)]
        leg_idx = runner._mn_leg[i]
        if runner._mn_is_flexor[i]:
            leg_flex_idx.setdefault(leg_idx, set()).add(model_idx)
        elif runner._mn_is_extensor[i]:
            leg_ext_idx.setdefault(leg_idx, set()).add(model_idx)

    # Collect stats
    n_flex_biased = 0
    n_ext_biased = 0
    n_boosted = 0
    per_leg_stats = {leg: {"flex_biased": 0, "ext_biased": 0, "boosted": 0}
                     for leg in range(6)}

    # Consider inhibitory premotor INs
    check_ids = runner._premotor_ids if only_premotor else (runner._premotor_ids | runner._mn_ids)

    for bid in check_ids:
        idx = runner._bodyid_to_idx.get(int(bid))
        if idx is None:
            continue

        # Must be inhibitory
        nt = runner._nt_map.get(int(bid), "unclear")
        sign = NT_SIGN.get(str(nt).lower().strip(), +1.0)
        if sign >= 0:
            continue

        # Check each leg
        for leg_idx in range(6):
            flex_set = leg_flex_idx.get(leg_idx, set())
            ext_set = leg_ext_idx.get(leg_idx, set())
            if not flex_set or not ext_set:
                continue

            flex_arr = np.array(list(flex_set), dtype=np.int32)
            ext_arr = np.array(list(ext_set), dtype=np.int32)

            # Count inhibitory synapses to flex vs ext
            col = W[:, idx]
            inh_to_flex = -col[flex_arr]  # Negate because W is negative for inh
            inh_to_ext = -col[ext_arr]
            inh_to_flex = np.maximum(inh_to_flex, 0)  # Only count inhibitory
            inh_to_ext = np.maximum(inh_to_ext, 0)

            total_inh_flex = inh_to_flex.sum()
            total_inh_ext = inh_to_ext.sum()
            total = total_inh_flex + total_inh_ext

            if total < 1e-8:
                continue

            frac_flex = total_inh_flex / total
            frac_ext = total_inh_ext / total

            if frac_flex >= bias_threshold:
                # Flexor-biased IN: boost its connections to EXTENSORS
                mask = col[ext_arr] < 0  # Inhibitory connections to ext
                n_entries = int(mask.sum())
                if n_entries > 0:
                    W[ext_arr[mask], idx] *= boost
                    n_boosted += n_entries
                    n_flex_biased += 1
                    per_leg_stats[leg_idx]["flex_biased"] += 1
                    per_leg_stats[leg_idx]["boosted"] += n_entries

            elif frac_ext >= bias_threshold:
                # Extensor-biased IN: boost its connections to FLEXORS
                mask = col[flex_arr] < 0
                n_entries = int(mask.sum())
                if n_entries > 0:
                    W[flex_arr[mask], idx] *= boost
                    n_boosted += n_entries
                    n_ext_biased += 1
                    per_leg_stats[leg_idx]["ext_biased"] += 1
                    per_leg_stats[leg_idx]["boosted"] += n_entries

    # Rebuild sparse matrices
    runner.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
    runner.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
    runner.W = W

    per_leg_str = ", ".join(
        f"{LEG_ORDER[k]}: {v['flex_biased']}F+{v['ext_biased']}E={v['boosted']}w"
        for k, v in sorted(per_leg_stats.items())
    )
    print(f"  Biased cross-inhibition: {n_flex_biased} flex-biased + "
          f"{n_ext_biased} ext-biased INs, {n_boosted} weights boosted at {boost}x")
    print(f"  Per-leg: {per_leg_str}")
    print(f"  W range: [{W.min():.4f}, {W.max():.4f}]")


def run_sim(name: str, runner: FiringRateVNCRunner, sim_ms: float = 2000.0,
            dn_baseline: float = 25.0, dng100_hz: float = 60.0):
    """Run simulation and analyze flex/ext alternation."""
    runner.stimulate_all_dns(rate_hz=dn_baseline)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    dt = runner.cfg.dt_ms
    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)
    n_records = int(sim_ms / 1.0)
    time_axis = np.linspace(0, sim_ms, n_records)

    flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}

    print(f"  Simulating {sim_ms:.0f} ms...")
    t0 = time()
    rec_idx = 0
    for step_i in range(n_steps):
        runner.step(dt_ms=dt)
        if (step_i + 1) % record_every == 0 and rec_idx < n_records:
            for leg_idx, leg_name in enumerate(LEG_ORDER):
                flex_rate, ext_rate = runner.get_flexor_extensor_rates(leg_idx)
                flex_traces[leg_name][rec_idx] = flex_rate
                ext_traces[leg_name][rec_idx] = ext_rate
            rec_idx += 1
    sim_time = time() - t0
    print(f"  Done in {sim_time:.1f}s ({n_steps/sim_time:.0f} steps/s)")

    # Analyze
    skip_idx = 500
    results = {}
    for leg_name in LEG_ORDER:
        f = flex_traces[leg_name][skip_idx:]
        e = ext_traces[leg_name][skip_idx:]
        f_active = f.max() > 1.0
        e_active = e.max() > 1.0
        if f_active and e_active and f.std() > 1e-6 and e.std() > 1e-6:
            corr = float(np.corrcoef(f, e)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        results[leg_name] = {
            "flex_mean": float(f.mean()), "ext_mean": float(e.mean()),
            "flex_std": float(f.std()), "ext_std": float(e.std()),
            "flex_active": bool(f_active), "ext_active": bool(e_active),
            "correlation": corr,
        }

    # Spectral
    peak_freq = 0.0
    for check_leg in LEG_ORDER:
        f = flex_traces[check_leg][skip_idx:]
        if f.max() > 1.0 and f.std() > 0.1:
            f_c = f - f.mean()
            freqs = np.fft.rfftfreq(len(f_c), d=1.0/1000.0)
            psd = np.abs(np.fft.rfft(f_c))**2
            band = (freqs >= 3.0) & (freqs <= 30.0)
            if band.any():
                peak_freq = float(freqs[band][np.argmax(psd[band])])
                break

    n_anti = sum(1 for r in results.values() if r["correlation"] < -0.1)
    mean_corr = float(np.mean([r["correlation"] for r in results.values()]))

    print(f"\n  {name}: {n_anti}/6 anti-phase, mean_corr={mean_corr:+.3f}, freq={peak_freq:.1f}Hz")
    for leg_name in LEG_ORDER:
        r = results[leg_name]
        status = "AP" if r["correlation"] < -0.1 else ("act" if r["flex_active"] or r["ext_active"] else "---")
        print(f"    {leg_name}: r={r['correlation']:+.3f} "
              f"f={r['flex_mean']:.1f}+/-{r['flex_std']:.1f} "
              f"e={r['ext_mean']:.1f}+/-{r['ext_std']:.1f} [{status}]")

    return {
        "name": name, "n_anti_phase": n_anti, "mean_corr": mean_corr,
        "peak_freq": peak_freq, "per_leg": results,
        "flex_traces": flex_traces, "ext_traces": ext_traces, "time_axis": time_axis,
    }


def main():
    all_results = {}

    # =================================================================
    # Part 1: Biased cross-inhibition sweep (bias threshold = 0.6)
    # =================================================================
    print("\n" + "="*72)
    print("PART 1: Biased cross-inhibition boost sweep (bias_threshold=0.6)")
    print("="*72)

    for boost_val in [2.0, 5.0, 10.0, 20.0, 50.0]:
        name = f"biased_b{boost_val:.0f}x"
        print(f"\n{'='*72}")
        print(f"Config: {name}")
        print(f"{'='*72}")

        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
            half_center_boost=1.0,  # Disable built-in boost
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=42,
        )
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
        apply_biased_crossinhibition(runner, boost=boost_val, bias_threshold=0.6)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 2: Vary bias threshold at the best boost level
    # =================================================================
    print("\n" + "="*72)
    print("PART 2: Vary bias threshold")
    print("="*72)

    best_boost = max(
        [n for n in all_results if n.startswith("biased_")],
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
    )
    best_boost_val = float(best_boost.split("b")[1].rstrip("x"))
    print(f"  Best boost from Part 1: {best_boost} (AP={all_results[best_boost]['n_anti_phase']})")

    for threshold in [0.5, 0.55, 0.65, 0.7, 0.8]:
        name = f"thresh_{threshold:.2f}_b{best_boost_val:.0f}x"
        print(f"\n{'='*72}")
        print(f"Config: {name}")
        print(f"{'='*72}")

        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
            half_center_boost=1.0,
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=42,
        )
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
        apply_biased_crossinhibition(runner, boost=best_boost_val, bias_threshold=threshold)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 3: Lower base inh_scale + high biased boost
    # The idea: reduce uniform inhibition so activity survives,
    # but make the cross-pool paths very strong
    # =================================================================
    print("\n" + "="*72)
    print("PART 3: Low base inh_scale + high biased boost")
    print("="*72)

    for base_inh, boost_val in [(1.0, 10.0), (1.0, 20.0), (1.0, 50.0),
                                 (1.5, 10.0), (1.5, 20.0), (0.5, 20.0)]:
        name = f"inh{base_inh:.1f}_biased{boost_val:.0f}x"
        print(f"\n{'='*72}")
        print(f"Config: {name}")
        print(f"{'='*72}")

        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=base_inh,
            half_center_boost=1.0,
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=42,
        )
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
        apply_biased_crossinhibition(runner, boost=boost_val, bias_threshold=0.6)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 4: Low inh_scale + seg T3 high + biased boost
    # =================================================================
    print("\n" + "="*72)
    print("PART 4: Combined: seg-specific inh + biased boost")
    print("="*72)

    for boost_val in [5.0, 10.0, 20.0]:
        name = f"segT3_biased{boost_val:.0f}x"
        print(f"\n{'='*72}")
        print(f"Config: {name}")
        print(f"{'='*72}")

        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
            segment_inh_scales={"T1": 2.0, "T2": 2.0, "T3": 4.0},
            half_center_boost=1.0,
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=42,
        )
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
        apply_biased_crossinhibition(runner, boost=boost_val, bias_threshold=0.6)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"{'Config':<35s} {'AP':>3s} {'mean_r':>7s} {'freq':>6s} | "
          + " ".join(f"{l:>6s}" for l in LEG_ORDER))
    print("-" * 110)

    for name in all_results:
        r = all_results[name]
        corrs = " ".join(f"{r['per_leg'][l]['correlation']:+.2f}" for l in LEG_ORDER)
        print(f"{name:<35s} {r['n_anti_phase']:>3d} {r['mean_corr']:>+7.3f} "
              f"{r['peak_freq']:>6.1f} | {corrs}")

    best_name = max(
        all_results.keys(),
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
    )
    best = all_results[best_name]
    print(f"\nBEST: {best_name} ({best['n_anti_phase']}/6 anti-phase, "
          f"mean_corr={best['mean_corr']:+.3f})")

    # Save results
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {
            "n_anti_phase": r["n_anti_phase"],
            "mean_corr": r["mean_corr"],
            "peak_freq": r["peak_freq"],
            "per_leg": r["per_leg"],
        }
    with open(log_dir / "vnc_halfcenter_v2_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {log_dir / 'vnc_halfcenter_v2_results.json'}")

    # Plot best traces + heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Traces
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Biased Cross-Inhibition: {best_name}\n"
                     f"({best['n_anti_phase']}/6 anti-phase, "
                     f"mean_corr={best['mean_corr']:+.3f})", fontsize=12)
        t = best["time_axis"]
        for i, leg_name in enumerate(LEG_ORDER):
            ax = axes[i // 2, i % 2]
            ax.plot(t, best["flex_traces"][leg_name], "b-", alpha=0.7, label="Flexor")
            ax.plot(t, best["ext_traces"][leg_name], "r-", alpha=0.7, label="Extensor")
            ax.axvline(500, color="gray", linestyle="--", alpha=0.3)
            corr = best["per_leg"][leg_name]["correlation"]
            status = "AP" if corr < -0.1 else ""
            ax.set_title(f"{leg_name} (r={corr:+.3f}) {status}")
            ax.set_ylabel("Rate (Hz)")
            if i >= 4:
                ax.set_xlabel("Time (ms)")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_ylim(bottom=0)
        plt.tight_layout()
        out1 = ROOT / "figures" / "vnc_halfcenter_v2_best.png"
        out1.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out1), dpi=150)
        print(f"Plot saved: {out1}")
        plt.close()

        # Heatmap
        config_names = list(all_results.keys())
        n_cfgs = len(config_names)
        corr_matrix = np.zeros((n_cfgs, 6))
        for ci, name in enumerate(config_names):
            for li, leg in enumerate(LEG_ORDER):
                corr_matrix[ci, li] = all_results[name]["per_leg"][leg]["correlation"]

        fig, ax = plt.subplots(figsize=(8, max(8, n_cfgs * 0.4)))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(6))
        ax.set_xticklabels(LEG_ORDER)
        ax.set_yticks(range(n_cfgs))
        ax.set_yticklabels([
            f"{n} (AP={all_results[n]['n_anti_phase']})"
            for n in config_names
        ], fontsize=7)
        for ci in range(n_cfgs):
            for li in range(6):
                v = corr_matrix[ci, li]
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(li, ci, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6, color=color)
        ax.set_title("Biased Cross-Inhibition: Flex/Ext Correlation Heatmap")
        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()
        out2 = ROOT / "figures" / "vnc_halfcenter_v2_heatmap.png"
        plt.savefig(str(out2), dpi=150)
        print(f"Plot saved: {out2}")
        plt.close()

    except ImportError:
        print("matplotlib not available -- skipping plots")

    return all_results


if __name__ == "__main__":
    main()
