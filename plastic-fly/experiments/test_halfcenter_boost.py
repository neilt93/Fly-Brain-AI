#!/usr/bin/env python3
"""
Test targeted half-center cross-inhibition boost for flex/ext alternation.

Tests the half_center_boost parameter which selectively scales inhibitory
connections forming half-center oscillators (IN -> both flex and ext MNs).

Also tests segment-specific inh_scale for handling T1/T2/T3 asymmetries.
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.vnc_firing_rate import (
    FiringRateVNCConfig, FiringRateVNCRunner, LEG_ORDER,
)


ROOT = Path(__file__).resolve().parent.parent


def run_config(name: str, cfg: FiringRateVNCConfig, sim_ms: float = 2000.0,
               dn_baseline: float = 25.0, dng100_hz: float = 60.0,
               warmup_ms: float = 100.0):
    """Run a single configuration and return per-leg flex/ext analysis."""
    print(f"\n{'='*72}")
    print(f"Config: {name}")
    print(f"  half_center_boost={cfg.half_center_boost}, inh_scale={cfg.inh_scale}")
    if cfg.segment_inh_scales:
        print(f"  segment_inh_scales={cfg.segment_inh_scales}")
    print(f"{'='*72}")

    runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=warmup_ms)
    runner.stimulate_all_dns(rate_hz=dn_baseline)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    dt = cfg.dt_ms
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

    # Analyze: skip first 500ms transient
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
            "flex_mean": float(f.mean()),
            "ext_mean": float(e.mean()),
            "flex_std": float(f.std()),
            "ext_std": float(e.std()),
            "flex_active": bool(f_active),
            "ext_active": bool(e_active),
            "correlation": corr,
        }

    # Spectral analysis on LF flexor
    peak_freq = 0.0
    for check_leg in ["LF", "RF"]:
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

    print(f"\n  Per-leg flex/ext correlations:")
    for leg_name in LEG_ORDER:
        r = results[leg_name]
        status = "AP" if r["correlation"] < -0.1 else ("act" if r["flex_active"] or r["ext_active"] else "---")
        print(f"    {leg_name}: corr={r['correlation']:+.3f} "
              f"flex={r['flex_mean']:.1f}+/-{r['flex_std']:.1f} "
              f"ext={r['ext_mean']:.1f}+/-{r['ext_std']:.1f} [{status}]")
    print(f"  Summary: {n_anti}/6 anti-phase, mean_corr={mean_corr:+.3f}, freq={peak_freq:.1f}Hz")

    return {
        "name": name,
        "n_anti_phase": n_anti,
        "mean_corr": mean_corr,
        "peak_freq": peak_freq,
        "per_leg": results,
        "flex_traces": flex_traces,
        "ext_traces": ext_traces,
        "time_axis": time_axis,
        "config": {
            "half_center_boost": cfg.half_center_boost,
            "inh_scale": cfg.inh_scale,
            "exc_mult": cfg.exc_mult,
            "segment_inh_scales": cfg.segment_inh_scales,
        },
    }


def main():
    all_results = {}

    # =====================================================================
    # Part 1: Half-center boost sweep
    # Base config: best from previous rounds (a=1, theta=7.5, exc_mult=0.01,
    # inh_scale=2, no norm, no adapt)
    # =====================================================================
    print("\n" + "="*72)
    print("PART 1: Half-center boost sweep")
    print("="*72)

    for hc_boost in [1.0, 2.0, 5.0, 10.0, 20.0]:
        name = f"hc_boost_{hc_boost:.0f}x"
        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
            half_center_boost=hc_boost,
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=42,
        )
        all_results[name] = run_config(name, cfg)

    # =====================================================================
    # Part 2: Segment-specific inh_scale
    # The hypothesis: T1, T2, T3 need different inhibition levels
    # because the MANC wiring has segment-specific asymmetries.
    # =====================================================================
    print("\n" + "="*72)
    print("PART 2: Segment-specific inh_scale")
    print("="*72)

    segment_configs = [
        ("seg_T1high", {"T1": 4.0, "T2": 2.0, "T3": 2.0}),
        ("seg_T2high", {"T1": 2.0, "T2": 4.0, "T3": 2.0}),
        ("seg_T3high", {"T1": 2.0, "T2": 2.0, "T3": 4.0}),
        ("seg_all_high", {"T1": 4.0, "T2": 4.0, "T3": 4.0}),
        ("seg_graded", {"T1": 2.0, "T2": 3.0, "T3": 4.0}),
    ]

    for name, seg_scales in segment_configs:
        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
            segment_inh_scales=seg_scales,
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=42,
        )
        all_results[name] = run_config(name, cfg)

    # =====================================================================
    # Part 3: Combined: best half-center boost + segment-specific
    # =====================================================================
    print("\n" + "="*72)
    print("PART 3: Combined half-center boost + segment-specific inh_scale")
    print("="*72)

    # Pick the best half-center boost from Part 1
    best_hc_name = max(
        [n for n in all_results if n.startswith("hc_boost_")],
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
    )
    best_hc_boost = all_results[best_hc_name]["config"]["half_center_boost"]
    print(f"  Best half-center boost from Part 1: {best_hc_name} "
          f"(boost={best_hc_boost}, AP={all_results[best_hc_name]['n_anti_phase']}/6)")

    # Pick the best segment config from Part 2
    best_seg_name = max(
        [n for n in all_results if n.startswith("seg_")],
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
    )
    best_seg_scales = all_results[best_seg_name]["config"]["segment_inh_scales"]
    print(f"  Best segment config from Part 2: {best_seg_name} "
          f"(scales={best_seg_scales}, AP={all_results[best_seg_name]['n_anti_phase']}/6)")

    # Combine
    for hc_val in [best_hc_boost, max(best_hc_boost * 2, 10.0)]:
        name = f"combined_hc{hc_val:.0f}x_seg"
        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
            half_center_boost=hc_val,
            segment_inh_scales=best_seg_scales,
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=42,
        )
        all_results[name] = run_config(name, cfg)

    # =====================================================================
    # Summary table
    # =====================================================================
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"{'Config':<30s} {'AP':>3s} {'mean_r':>7s} {'freq':>6s} | "
          + " ".join(f"{l:>6s}" for l in LEG_ORDER))
    print("-" * 100)

    for name in all_results:
        r = all_results[name]
        corrs = " ".join(f"{r['per_leg'][l]['correlation']:+.2f}" for l in LEG_ORDER)
        print(f"{name:<30s} {r['n_anti_phase']:>3d} {r['mean_corr']:>+7.3f} "
              f"{r['peak_freq']:>6.1f} | {corrs}")

    # Find the best overall
    best_name = max(
        all_results.keys(),
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
    )
    best = all_results[best_name]
    print(f"\nBEST: {best_name} ({best['n_anti_phase']}/6 anti-phase, "
          f"mean_corr={best['mean_corr']:+.3f})")

    # Save results JSON
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {
            "n_anti_phase": r["n_anti_phase"],
            "mean_corr": r["mean_corr"],
            "peak_freq": r["peak_freq"],
            "per_leg": r["per_leg"],
            "config": r["config"],
        }
    with open(log_dir / "vnc_halfcenter_boost_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {log_dir / 'vnc_halfcenter_boost_results.json'}")

    # =====================================================================
    # Plot: best config traces + comparison heatmap
    # =====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Plot 1: Best config traces ---
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Half-Center Boost: {best_name}\n"
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
        out1 = ROOT / "figures" / "vnc_halfcenter_boost_best.png"
        out1.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out1), dpi=150)
        print(f"Plot saved: {out1}")
        plt.close()

        # --- Plot 2: Correlation heatmap across all configs ---
        config_names = list(all_results.keys())
        n_cfgs = len(config_names)
        corr_matrix = np.zeros((n_cfgs, 6))
        for ci, name in enumerate(config_names):
            for li, leg in enumerate(LEG_ORDER):
                corr_matrix[ci, li] = all_results[name]["per_leg"][leg]["correlation"]

        fig, ax = plt.subplots(figsize=(8, max(6, n_cfgs * 0.5)))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(6))
        ax.set_xticklabels(LEG_ORDER)
        ax.set_yticks(range(n_cfgs))
        ax.set_yticklabels([
            f"{n} (AP={all_results[n]['n_anti_phase']})"
            for n in config_names
        ], fontsize=8)
        for ci in range(n_cfgs):
            for li in range(6):
                v = corr_matrix[ci, li]
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(li, ci, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        ax.set_title("Flex/Ext Correlation: Half-Center Boost Experiments")
        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()
        out2 = ROOT / "figures" / "vnc_halfcenter_boost_heatmap.png"
        plt.savefig(str(out2), dpi=150)
        print(f"Plot saved: {out2}")
        plt.close()

    except ImportError:
        print("matplotlib not available -- skipping plots")

    return all_results


if __name__ == "__main__":
    main()
