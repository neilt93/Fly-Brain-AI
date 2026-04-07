#!/usr/bin/env python3
"""
Robust anti-phase experiment: maximize flex/ext alternation across all 6 legs.

Three strategies:
  1. CV sweep: lower param_cv (0.0, 0.02, 0.05, 0.10) to reduce noise that
     randomly tips bilateral pairs.
  2. Multi-seed ensemble: 10 seeds at best config, count per-leg anti-phase.
  3. BANC (female VNC): segment-specific params on BANC data.

Best known MANC config: a=1.0, theta=7.5, exc_mult=0.01, inh_scale=2.0,
  no normalization, no adaptation, segment_exc_mults={T1: 0.015, T2: 0.01, T3: 0.01},
  param_cv=0.10, seed=2024 => 3/6 anti-phase.

Usage:
    python experiments/test_robust_antiphase.py
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from time import time
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.vnc_firing_rate import FiringRateVNCRunner, FiringRateVNCConfig, LEG_ORDER

# ============================================================================
# Leg-segment mapping
# ============================================================================

SEGMENT_LEGS = {
    "T1": ["LF", "RF"],
    "T2": ["LM", "RM"],
    "T3": ["LH", "RH"],
}


# ============================================================================
# Shared simulation runner
# ============================================================================

def run_sim(
    name: str,
    runner: FiringRateVNCRunner,
    sim_ms: float = 2000.0,
    dt: float = 0.5,
    dn_baseline_hz: float = 25.0,
    dng100_hz: float = 60.0,
    skip_ms: float = 500.0,
    verbose: bool = True,
) -> dict:
    """Run simulation and compute per-leg flex/ext correlation.

    Returns dict with 'correlations' (per-leg), 'n_antiphase', 'mean_corr',
    plus 'flex_traces' and 'ext_traces' for plotting.
    """
    runner.stimulate_all_dns(rate_hz=dn_baseline_hz)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)
    n_records = int(sim_ms / 1.0)
    time_axis = np.linspace(0, sim_ms, n_records)

    flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}

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

    # Analysis: correlations on post-transient window
    skip_idx = int(skip_ms / 1.0)
    correlations = {}
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
        correlations[leg_name] = corr

    n_ap = sum(1 for c in correlations.values() if c < -0.3)
    mean_corr = float(np.mean(list(correlations.values())))

    # Spectral: peak frequency
    peak_freq = 0.0
    for check_leg in LEG_ORDER:
        f_check = flex_traces[check_leg][skip_idx:]
        if f_check.max() > 1.0 and f_check.std() > 1e-6:
            f_c = f_check - f_check.mean()
            freqs = np.fft.rfftfreq(len(f_c), d=1.0 / 1000.0)
            psd = np.abs(np.fft.rfft(f_c)) ** 2
            bm = (freqs >= 3.0) & (freqs <= 25.0)
            if bm.any():
                bp = psd[bm]
                bf = freqs[bm]
                peak_freq = float(bf[np.argmax(bp)])
                break

    result = {
        "name": name,
        "correlations": correlations,
        "n_antiphase": n_ap,
        "mean_corr": mean_corr,
        "peak_freq": peak_freq,
        "sim_time": sim_time,
        "flex_traces": flex_traces,
        "ext_traces": ext_traces,
        "time_axis": time_axis,
    }

    if verbose:
        corr_str = " ".join(f"{leg}={correlations[leg]:+.3f}" for leg in LEG_ORDER)
        print(f"  [{name}] {n_ap}/6 AP, mean_r={mean_corr:+.3f}, "
              f"freq={peak_freq:.1f}Hz, {sim_time:.1f}s")
        print(f"    {corr_str}")

    return result


def make_manc_cfg(seed=42, param_cv=0.10, **overrides) -> FiringRateVNCConfig:
    """Best MANC config with segment-specific T1 boost."""
    kw = dict(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
        use_adaptation=False,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False,
        param_cv=param_cv, seed=seed,
        segments=("T1", "T2", "T3"),
        segment_exc_mults={"T1": 0.015, "T2": 0.01, "T3": 0.01},
    )
    kw.update(overrides)
    return FiringRateVNCConfig(**kw)


def build_manc_runner(cfg: FiringRateVNCConfig, warmup_ms: float = 100.0):
    """Build MANC-based runner."""
    return FiringRateVNCRunner(cfg=cfg, warmup_ms=warmup_ms)


# ============================================================================
# Strategy 1: CV sweep
# ============================================================================

def strategy_cv_sweep(seeds: list[int] = [42, 2024]) -> list[dict]:
    """Sweep param_cv from 0.0 to 0.10 at best segment-specific config.

    At cv=0 all neurons have identical ODE params, so L/R bilateral
    symmetry should produce identical dynamics => 6/6 anti-phase if
    the circuit supports it at all.
    """
    print("\n" + "=" * 72)
    print("STRATEGY 1: CV Sweep (lower param_cv to reduce noise)")
    print("=" * 72)

    cv_values = [0.0, 0.02, 0.05, 0.10]
    all_results = []

    for seed in seeds:
        for cv in cv_values:
            name = f"cv{cv:.2f}_s{seed}"
            cfg = make_manc_cfg(seed=seed, param_cv=cv)
            try:
                runner = build_manc_runner(cfg)
                result = run_sim(name, runner, verbose=True)
                result["cv"] = cv
                result["seed"] = seed
                all_results.append(result)
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")
                all_results.append({"name": name, "error": str(e),
                                    "cv": cv, "seed": seed})

    return all_results


# ============================================================================
# Strategy 2: Multi-seed ensemble
# ============================================================================

def strategy_multi_seed(n_seeds: int = 10) -> list[dict]:
    """Run 10 seeds at best config (cv=0.10), count per-leg anti-phase frequency.

    If every leg achieves anti-phase in >=3/10 seeds, the circuit supports
    it everywhere and the limitation is stochastic noise.
    """
    print("\n" + "=" * 72)
    print("STRATEGY 2: Multi-seed ensemble (10 seeds, cv=0.10)")
    print("=" * 72)

    seeds = list(range(42, 42 + n_seeds))
    all_results = []

    for seed in seeds:
        name = f"seed{seed}"
        cfg = make_manc_cfg(seed=seed, param_cv=0.10)
        try:
            runner = build_manc_runner(cfg)
            result = run_sim(name, runner, verbose=True)
            result["seed"] = seed
            all_results.append(result)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            all_results.append({"name": name, "error": str(e), "seed": seed})

    # Per-leg summary
    print("\n  Per-leg anti-phase frequency (corr < -0.3):")
    for leg in LEG_ORDER:
        n_ap = sum(1 for r in all_results
                   if "correlations" in r and r["correlations"][leg] < -0.3)
        corrs = [r["correlations"][leg] for r in all_results if "correlations" in r]
        mean_c = np.mean(corrs) if corrs else 0.0
        print(f"    {leg}: {n_ap}/{len(corrs)} seeds anti-phase, "
              f"mean corr={mean_c:+.3f}")

    return all_results


# ============================================================================
# Strategy 3: BANC (female VNC) with segment-specific params
# ============================================================================

def strategy_banc_segment_specific() -> list[dict]:
    """Try segment-specific params on BANC (female VNC).

    The female VNC (BANC) has different neuron counts and connectivity
    density; the optimal segment scaling may differ from MANC.
    """
    print("\n" + "=" * 72)
    print("STRATEGY 3: BANC (female VNC) segment-specific params")
    print("=" * 72)

    try:
        from bridge.banc_loader import load_banc_vnc, BANCVNCData
    except ImportError as e:
        print(f"  BANC loader not available: {e}")
        return []

    configs = [
        # name, segment_exc_mults, inh_scale, cv, seed
        ("banc_baseline", None, 2.0, 0.10, 42),
        ("banc_T1_boost", {"T1": 0.015, "T2": 0.01, "T3": 0.01}, 2.0, 0.10, 42),
        ("banc_T1_boost_cv0", {"T1": 0.015, "T2": 0.01, "T3": 0.01}, 2.0, 0.0, 42),
        ("banc_T1_boost_cv002", {"T1": 0.015, "T2": 0.01, "T3": 0.01}, 2.0, 0.02, 42),
        ("banc_all_boost", {"T1": 0.015, "T2": 0.012, "T3": 0.012}, 2.0, 0.0, 42),
        ("banc_hi_inh", {"T1": 0.015, "T2": 0.01, "T3": 0.01}, 3.0, 0.0, 42),
    ]

    all_results = []
    for name, seg_exc, inh_scale, cv, seed in configs:
        print(f"\n  --- {name} ---")
        try:
            # Load BANC data with the right weight params
            banc_data = load_banc_vnc(
                normalize_weights=False,
                exc_mult=0.01,
                inh_mult=0.01,
                inh_scale=inh_scale,
            )

            cfg = FiringRateVNCConfig(
                tau_ms=20.0,
                a=1.0,
                theta=7.5,
                fr_cap=200.0,
                exc_mult=0.01,
                inh_mult=0.01,
                inh_scale=inh_scale,
                use_adaptation=False,
                use_delay=True,
                delay_inh_ms=3.0,
                normalize_weights=False,
                param_cv=cv,
                seed=seed,
                segment_exc_mults=seg_exc,
            )

            runner = FiringRateVNCRunner.from_banc(banc_data, cfg=cfg, warmup_ms=100.0)
            result = run_sim(name, runner, verbose=True)
            result["dataset"] = "BANC"
            result["cv"] = cv
            result["seed"] = seed
            all_results.append(result)

        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "error": str(e), "dataset": "BANC"})

    return all_results


# ============================================================================
# Summary and figure generation
# ============================================================================

def print_summary_table(results: list[dict], title: str):
    """Print a summary table."""
    print(f"\n{'='*110}")
    print(title)
    print(f"{'='*110}")
    print(f"{'Config':<30} {'AP':>3} {'Mean r':>8} {'Freq':>6} "
          f"{'LF':>7} {'LM':>7} {'LH':>7} {'RF':>7} {'RM':>7} {'RH':>7}")
    print("-" * 110)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} ERROR: {r['error'][:50]}")
            continue
        n_ap = r["n_antiphase"]
        mc = r["mean_corr"]
        freq = r.get("peak_freq", 0)
        corrs = r["correlations"]
        c_str = " ".join(f"{corrs[l]:>+7.3f}" for l in LEG_ORDER)
        print(f"{r['name']:<30} {n_ap:>3} {mc:>+8.3f} {freq:>5.1f}Hz {c_str}")


def generate_summary_figure(
    cv_results: list[dict],
    seed_results: list[dict],
    banc_results: list[dict],
    out_path: Path,
):
    """Generate 3-panel summary figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("matplotlib not available -- skipping figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Robust Anti-Phase: CV Sweep / Multi-Seed / BANC",
                 fontsize=14, fontweight="bold")

    # ---- Panel A: CV sweep heatmap ----
    ax = axes[0]
    ax.set_title("(A) CV Sweep: correlation by leg")

    # Build heatmap: rows = CV values, cols = legs
    cv_vals = sorted(set(r.get("cv", 0) for r in cv_results if "error" not in r))
    seeds_used = sorted(set(r.get("seed", 0) for r in cv_results if "error" not in r))

    if cv_vals and seeds_used:
        # Average across seeds for each CV
        heatmap = np.zeros((len(cv_vals), 6))
        for ci, cv in enumerate(cv_vals):
            cv_runs = [r for r in cv_results
                       if "error" not in r and abs(r.get("cv", -1) - cv) < 1e-6]
            for li, leg in enumerate(LEG_ORDER):
                vals = [r["correlations"][leg] for r in cv_runs]
                heatmap[ci, li] = np.mean(vals) if vals else 0.0

        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
        im = ax.imshow(heatmap, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(6))
        ax.set_xticklabels(LEG_ORDER, fontsize=9)
        ax.set_yticks(range(len(cv_vals)))
        ax.set_yticklabels([f"{cv:.2f}" for cv in cv_vals], fontsize=9)
        ax.set_ylabel("param_cv")
        ax.set_xlabel("Leg")

        # Annotate cells
        for ci in range(len(cv_vals)):
            for li in range(6):
                val = heatmap[ci, li]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(li, ci, f"{val:+.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Flex-Ext correlation")
    else:
        ax.text(0.5, 0.5, "No CV data", transform=ax.transAxes,
                ha="center", va="center")

    # ---- Panel B: Multi-seed histogram per leg ----
    ax = axes[1]
    ax.set_title("(B) Multi-Seed: anti-phase frequency per leg")

    valid_seed_results = [r for r in seed_results if "error" not in r]
    if valid_seed_results:
        n_seeds_total = len(valid_seed_results)
        leg_ap_counts = []
        leg_mean_corrs = []
        for leg in LEG_ORDER:
            n_ap = sum(1 for r in valid_seed_results
                       if r["correlations"][leg] < -0.3)
            leg_ap_counts.append(n_ap)
            mean_c = np.mean([r["correlations"][leg] for r in valid_seed_results])
            leg_mean_corrs.append(mean_c)

        x = np.arange(6)
        bars = ax.bar(x, leg_ap_counts, color=["#2196F3" if c < -0.3 else "#FF9800"
                                                 for c in leg_mean_corrs],
                       edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LEG_ORDER, fontsize=9)
        ax.set_ylabel(f"Seeds with anti-phase (out of {n_seeds_total})")
        ax.set_ylim(0, n_seeds_total + 1)
        ax.axhline(3, color="gray", linestyle="--", alpha=0.5, label="3/10 threshold")
        ax.legend(fontsize=8)

        # Annotate with mean correlation
        for i, (cnt, mc) in enumerate(zip(leg_ap_counts, leg_mean_corrs)):
            ax.text(i, cnt + 0.3, f"r={mc:+.2f}", ha="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No seed data", transform=ax.transAxes,
                ha="center", va="center")

    # ---- Panel C: BANC segment-specific heatmap ----
    ax = axes[2]
    ax.set_title("(C) BANC (female): correlation by config")

    valid_banc = [r for r in banc_results if "error" not in r]
    if valid_banc:
        banc_names = [r["name"] for r in valid_banc]
        heatmap_b = np.zeros((len(valid_banc), 6))
        for ri, r in enumerate(valid_banc):
            for li, leg in enumerate(LEG_ORDER):
                heatmap_b[ri, li] = r["correlations"][leg]

        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
        im2 = ax.imshow(heatmap_b, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(6))
        ax.set_xticklabels(LEG_ORDER, fontsize=9)
        ax.set_yticks(range(len(banc_names)))
        ax.set_yticklabels([n.replace("banc_", "") for n in banc_names],
                           fontsize=7)
        ax.set_xlabel("Leg")

        for ri in range(len(valid_banc)):
            for li in range(6):
                val = heatmap_b[ri, li]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(li, ri, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        plt.colorbar(im2, ax=ax, shrink=0.8, label="Flex-Ext correlation")
    else:
        ax.text(0.5, 0.5, "No BANC data", transform=ax.transAxes,
                ha="center", va="center")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("ROBUST ANTI-PHASE EXPERIMENT")
    print("=" * 72)
    print(f"Best known config: T1_exc=0.015, T2/T3_exc=0.01, "
          f"inh_scale=2.0, a=1.0, theta=7.5")
    print(f"Goal: all 6 legs anti-phase from connectome wiring alone")
    print()

    t_total = time()

    # Strategy 1: CV sweep
    cv_results = strategy_cv_sweep(seeds=[42, 2024])
    print_summary_table(cv_results, "STRATEGY 1 RESULTS: CV Sweep")

    # Strategy 2: Multi-seed ensemble
    seed_results = strategy_multi_seed(n_seeds=10)
    print_summary_table(seed_results, "STRATEGY 2 RESULTS: Multi-Seed Ensemble")

    # Strategy 3: BANC segment-specific
    banc_results = strategy_banc_segment_specific()
    if banc_results:
        print_summary_table(banc_results, "STRATEGY 3 RESULTS: BANC Segment-Specific")

    # ================================================================
    # Overall summary
    # ================================================================
    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)

    # Best CV result
    valid_cv = [r for r in cv_results if "error" not in r]
    if valid_cv:
        best_cv = max(valid_cv, key=lambda r: r["n_antiphase"] * 10 - r["mean_corr"] * 5)
        print(f"\n  Best CV: {best_cv['name']} => {best_cv['n_antiphase']}/6 AP, "
              f"mean_r={best_cv['mean_corr']:+.3f}")
        for leg in LEG_ORDER:
            print(f"    {leg}: {best_cv['correlations'][leg]:+.3f}")

    # Best multi-seed
    valid_seed = [r for r in seed_results if "error" not in r]
    if valid_seed:
        best_seed = max(valid_seed, key=lambda r: r["n_antiphase"] * 10 - r["mean_corr"] * 5)
        print(f"\n  Best seed: {best_seed['name']} => {best_seed['n_antiphase']}/6 AP, "
              f"mean_r={best_seed['mean_corr']:+.3f}")

        # Per-leg support assessment
        print("\n  Per-leg anti-phase support (corr < -0.3 in >=3/10 seeds = supported):")
        all_supported = True
        for leg in LEG_ORDER:
            n_ap = sum(1 for r in valid_seed if r["correlations"][leg] < -0.3)
            supported = n_ap >= 3
            if not supported:
                all_supported = False
            print(f"    {leg}: {n_ap}/10 => {'SUPPORTED' if supported else 'WEAK'}")
        if all_supported:
            print("  => ALL legs supported: circuit wiring alone is sufficient!")
        else:
            print("  => Some legs not robustly supported at cv=0.10")

    # Best BANC
    valid_banc = [r for r in banc_results if "error" not in r]
    if valid_banc:
        best_banc = max(valid_banc, key=lambda r: r["n_antiphase"] * 10 - r["mean_corr"] * 5)
        print(f"\n  Best BANC: {best_banc['name']} => {best_banc['n_antiphase']}/6 AP, "
              f"mean_r={best_banc['mean_corr']:+.3f}")

    # Deterministic (cv=0) result
    cv0_results = [r for r in cv_results if "error" not in r and r.get("cv", 1) == 0.0]
    if cv0_results:
        best_cv0 = max(cv0_results, key=lambda r: r["n_antiphase"])
        print(f"\n  Deterministic (cv=0): {best_cv0['n_antiphase']}/6 AP, "
              f"mean_r={best_cv0['mean_corr']:+.3f}")
        if best_cv0["n_antiphase"] == 6:
            print("  => WITH DETERMINISTIC PARAMS AND SEGMENT-SPECIFIC SCALING,")
            print("     ALL 6 LEGS ACHIEVE ANTI-PHASE FLEX/EXT ALTERNATION")
            print("     FROM CONNECTOME WIRING ALONE.")
        elif best_cv0["n_antiphase"] >= 4:
            print(f"  => {best_cv0['n_antiphase']}/6 legs anti-phase at cv=0.")
            print("     Noise (cv>0) is a factor but not the only limitation.")

    total_time = time() - t_total
    print(f"\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Generate figure
    fig_path = ROOT / "figures" / "robust_antiphase_summary.png"
    generate_summary_figure(cv_results, seed_results, banc_results, fig_path)

    # Save JSON results (without trace data for size)
    json_results = {
        "cv_sweep": [{k: v for k, v in r.items()
                      if k not in ("flex_traces", "ext_traces", "time_axis")}
                     for r in cv_results],
        "multi_seed": [{k: v for k, v in r.items()
                        if k not in ("flex_traces", "ext_traces", "time_axis")}
                       for r in seed_results],
        "banc": [{k: v for k, v in r.items()
                  if k not in ("flex_traces", "ext_traces", "time_axis")}
                 for r in banc_results],
    }
    json_path = ROOT / "figures" / "robust_antiphase_results.json"
    with open(str(json_path), "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
