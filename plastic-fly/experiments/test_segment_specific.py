"""
Segment-specific parameter sweep for CPG VNC firing rate model.

Hypothesis: T1, T2, T3 segments have different local circuit densities and
need different operating points for anti-phase flex/ext oscillation.

Evidence:
  - baseline_inh2 (exc=0.01): RM=-0.67, RH=-0.95 (right middle/hind = T2R, T3R)
  - exc_008 (exc=0.008): LM=-0.79, LH=-0.92 (left middle/hind = T2L, T3L)
  => T2/T3 respond to different excitation levels; different legs go anti-phase

Strategy (avoid 27-config combinatorial explosion):
  Phase 1: Optimize each segment independently (hold others at global 0.01)
    - T1 exc_mult in [0.006, 0.008, 0.01, 0.012, 0.015]
    - T2 exc_mult in same
    - T3 exc_mult in same
    (15 configs total, 5 per segment)
  Phase 2: Combine the per-segment optima (1 config)
  Phase 3: Fine-tune around the combined optimum (small grid)
"""

import sys
import numpy as np
from pathlib import Path
from time import time
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.vnc_firing_rate import FiringRateVNCRunner, FiringRateVNCConfig, LEG_ORDER

# Legs belonging to each segment
SEGMENT_LEGS = {
    "T1": ["LF", "RF"],
    "T2": ["LM", "RM"],
    "T3": ["LH", "RH"],
}


def run_config(name: str, cfg: FiringRateVNCConfig, sim_ms: float = 2000.0,
               dn_baseline_hz: float = 25.0, dng100_hz: float = 60.0,
               warmup_ms: float = 100.0) -> dict:
    """Run a single configuration and return flex/ext correlations per leg."""
    print(f"\n{'='*72}")
    print(f"CONFIG: {name}")
    print(f"  a={cfg.a}, theta={cfg.theta}, exc={cfg.exc_mult}, inh={cfg.inh_mult}, "
          f"inh_sc={cfg.inh_scale}, norm={cfg.normalize_weights}, adapt={cfg.use_adaptation}")
    if cfg.segment_exc_mults:
        print(f"  segment_exc_mults={cfg.segment_exc_mults}")
    if cfg.segment_inh_scales:
        print(f"  segment_inh_scales={cfg.segment_inh_scales}")
    print(f"{'='*72}")

    try:
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=warmup_ms)
    except Exception as e:
        print(f"  FAILED: {e}")
        return {"name": name, "error": str(e)}

    runner.stimulate_all_dns(rate_hz=dn_baseline_hz)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    dt = 0.5
    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)  # record every 1ms
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
    rates_all = runner.get_all_rates()
    mn_rates = runner.get_mn_rates()
    n_sat = (rates_all > 0.9 * cfg.fr_cap).sum()
    print(f"  {sim_time:.1f}s, MN mean={mn_rates.mean():.1f}Hz, "
          f"sat={100*n_sat/len(rates_all):.1f}%")

    # Skip first 500ms transient
    skip_idx = int(500.0 / 1.0)
    results = {"name": name, "correlations": {}}

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
        results["correlations"][leg_name] = corr

    corrs = list(results["correlations"].values())
    n_ap = sum(1 for c in corrs if c < -0.3)
    mc = float(np.mean(corrs))
    results["n_antiphase"] = n_ap
    results["mean_corr"] = mc
    results["mn_mean"] = float(mn_rates.mean())
    results["sat_pct"] = float(100 * n_sat / len(rates_all))

    # Per-segment anti-phase count
    for seg_name, seg_legs in SEGMENT_LEGS.items():
        seg_corrs = [results["correlations"][l] for l in seg_legs]
        seg_ap = sum(1 for c in seg_corrs if c < -0.3)
        results[f"{seg_name}_ap"] = seg_ap
        results[f"{seg_name}_mean_corr"] = float(np.mean(seg_corrs))

    # Spectral analysis
    for check_leg in ["RM", "RH", "LF", "LM"]:
        f_check = flex_traces[check_leg][skip_idx:]
        if f_check.max() > 1.0 and f_check.std() > 1e-6:
            f_c = f_check - f_check.mean()
            freqs = np.fft.rfftfreq(len(f_c), d=1.0 / 1000.0)
            psd = np.abs(np.fft.rfft(f_c)) ** 2
            bm = (freqs >= 3.0) & (freqs <= 25.0)
            if bm.any():
                bp = psd[bm]
                bf = freqs[bm]
                results["peak_freq"] = float(bf[np.argmax(bp)])
                break
    if "peak_freq" not in results:
        results["peak_freq"] = 0.0

    corr_str = " ".join(f"{leg}={results['correlations'][leg]:+.2f}" for leg in LEG_ORDER)
    print(f"  Corrs: {corr_str}")
    print(f"  => {n_ap}/6 AP, mean_r={mc:+.3f}, freq={results['peak_freq']:.1f}Hz")
    for seg_name in ["T1", "T2", "T3"]:
        print(f"     {seg_name}: {results[f'{seg_name}_ap']}/2 AP, "
              f"mean_r={results[f'{seg_name}_mean_corr']:+.3f}")

    results["time_axis"] = time_axis
    results["flex_traces"] = flex_traces
    results["ext_traces"] = ext_traces
    return results


def make_base_cfg(seed=42, **overrides) -> FiringRateVNCConfig:
    """Best known config: a=1, theta=7.5, no norm, no adapt."""
    kw = dict(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
        use_adaptation=False,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False, param_cv=0.10, seed=seed,
        segments=("T1", "T2", "T3"),
    )
    kw.update(overrides)
    return FiringRateVNCConfig(**kw)


def print_summary(results_list, title="SUMMARY"):
    """Print summary table."""
    print(f"\n{'='*110}")
    print(title)
    print(f"{'='*110}")
    print(f"{'Config':<30} {'AP':>3} {'Mean r':>8} {'Freq':>6} "
          f"{'T1_AP':>5} {'T2_AP':>5} {'T3_AP':>5} "
          f"{'LF':>6} {'LM':>6} {'LH':>6} {'RF':>6} {'RM':>6} {'RH':>6}")
    print("-" * 110)

    for r in results_list:
        if "error" in r:
            print(f"{r['name']:<30} ERROR: {r['error'][:40]}")
            continue
        n_ap = r["n_antiphase"]
        mc = r["mean_corr"]
        freq = r.get("peak_freq", 0)
        corrs = r["correlations"]
        c_str = " ".join(f"{corrs[l]:>+6.2f}" for l in LEG_ORDER)
        t1_ap = r.get("T1_ap", 0)
        t2_ap = r.get("T2_ap", 0)
        t3_ap = r.get("T3_ap", 0)
        print(f"{r['name']:<30} {n_ap:>3} {mc:>+8.3f} {freq:>5.1f}Hz "
              f"{t1_ap:>5} {t2_ap:>5} {t3_ap:>5} {c_str}")


def main():
    print("=" * 72)
    print("SEGMENT-SPECIFIC PARAMETER SWEEP")
    print("=" * 72)

    all_results = []
    exc_values = [0.006, 0.008, 0.01, 0.012, 0.015]

    # ================================================================
    # Phase 0: Baseline (global exc=0.01, no segment-specific scaling)
    # ================================================================
    print("\n" + "=" * 72)
    print("PHASE 0: Baseline (uniform exc=0.01)")
    print("=" * 72)
    baseline = run_config("baseline_global_0.01", make_base_cfg())
    all_results.append(baseline)

    # ================================================================
    # Phase 1: Optimize each segment independently
    # ================================================================

    # For each segment, vary its exc_mult while holding others at global 0.01
    segment_best = {}  # segment -> best exc_mult

    for target_seg in ["T1", "T2", "T3"]:
        print(f"\n{'='*72}")
        print(f"PHASE 1: Sweeping {target_seg} exc_mult (others at global 0.01)")
        print(f"{'='*72}")

        seg_results = []
        for exc_val in exc_values:
            seg_exc = {"T1": 0.01, "T2": 0.01, "T3": 0.01}
            seg_exc[target_seg] = exc_val

            name = f"{target_seg}_exc_{exc_val:.3f}"
            cfg = make_base_cfg(segment_exc_mults=seg_exc)
            r = run_config(name, cfg)
            seg_results.append(r)
            all_results.append(r)

        # Find best exc_mult for this segment (maximize segment anti-phase)
        best_score = -999
        best_exc = 0.01
        for r, exc_val in zip(seg_results, exc_values):
            if "error" in r:
                continue
            seg_ap = r.get(f"{target_seg}_ap", 0)
            seg_mc = r.get(f"{target_seg}_mean_corr", 0)
            total_ap = r["n_antiphase"]
            # Score: segment AP * 10 + total AP * 5 - segment mean_corr * 3
            score = seg_ap * 10 + total_ap * 5 - seg_mc * 3
            if score > best_score:
                best_score = score
                best_exc = exc_val

        segment_best[target_seg] = best_exc
        print(f"\n  => Best {target_seg} exc_mult: {best_exc} (score={best_score:.1f})")

    print_summary(all_results, "PHASE 1 SUMMARY")

    # ================================================================
    # Phase 2: Combine per-segment optima
    # ================================================================
    print(f"\n{'='*72}")
    print(f"PHASE 2: Combined per-segment optima")
    print(f"  T1={segment_best['T1']}, T2={segment_best['T2']}, T3={segment_best['T3']}")
    print(f"{'='*72}")

    combined_cfg = make_base_cfg(segment_exc_mults=segment_best.copy())
    combined = run_config("combined_optima", combined_cfg)
    all_results.append(combined)

    # Also try with per-segment inh_scale sweeps
    print(f"\n{'='*72}")
    print(f"PHASE 2b: Combined optima + inh_scale variations")
    print(f"{'='*72}")

    # Try different inh_scales per segment too
    for inh_t1 in [1.5, 2.0, 2.5]:
        for inh_t3 in [1.5, 2.0, 2.5]:
            if inh_t1 == 2.0 and inh_t3 == 2.0:
                continue  # skip the baseline
            inh_scales = {"T1": inh_t1, "T2": 2.0, "T3": inh_t3}
            name = f"comb_inh_T1={inh_t1}_T3={inh_t3}"
            cfg = make_base_cfg(
                segment_exc_mults=segment_best.copy(),
                segment_inh_scales=inh_scales,
            )
            r = run_config(name, cfg)
            all_results.append(r)

    # ================================================================
    # Phase 3: Fine-tune around best combined
    # ================================================================
    # Find the best from Phase 2
    phase2_results = [r for r in all_results
                      if r["name"].startswith("comb") or r["name"] == "combined_optima"]
    best_p2 = max(
        (r for r in phase2_results if "error" not in r),
        key=lambda r: r["n_antiphase"] * 10 - r["mean_corr"] * 5,
        default=None,
    )

    if best_p2:
        print(f"\n{'='*72}")
        print(f"PHASE 3: Fine-tuning around best: {best_p2['name']} "
              f"({best_p2['n_antiphase']}/6 AP)")
        print(f"{'='*72}")

        # Try nudging each segment +/- from current optimum
        for seg in ["T1", "T2", "T3"]:
            base_val = segment_best[seg]
            for delta in [-0.002, +0.002]:
                new_val = base_val + delta
                if new_val < 0.004 or new_val > 0.02:
                    continue
                nudged = segment_best.copy()
                nudged[seg] = new_val
                name = f"finetune_{seg}_{new_val:.3f}"
                cfg = make_base_cfg(segment_exc_mults=nudged)
                r = run_config(name, cfg)
                all_results.append(r)

        # Try different seeds with the best config
        for seed in [123, 7, 2024]:
            name = f"best_seed_{seed}"
            cfg = make_base_cfg(seed=seed, segment_exc_mults=segment_best.copy())
            r = run_config(name, cfg)
            all_results.append(r)

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print_summary(all_results, "GRAND SUMMARY - SEGMENT-SPECIFIC SWEEP")

    # Find overall best
    valid = [r for r in all_results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["n_antiphase"] * 10 - r["mean_corr"] * 5)
        print(f"\nBEST CONFIG: {best['name']}")
        print(f"  {best['n_antiphase']}/6 anti-phase, mean_corr={best['mean_corr']:+.3f}")
        for seg in ["T1", "T2", "T3"]:
            print(f"  {seg}: {best.get(f'{seg}_ap', '?')}/2 AP, "
                  f"mean_r={best.get(f'{seg}_mean_corr', 0):+.3f}")

        # Save best config
        save_best_config(best, segment_best)

        # Plot traces for the best config
        plot_best(best, all_results)


def save_best_config(best: dict, segment_exc: dict):
    """Save the best configuration to a JSON file."""
    import json
    out = ROOT / "data" / "best_segment_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "name": best["name"],
        "n_antiphase": best["n_antiphase"],
        "mean_corr": best["mean_corr"],
        "correlations": best["correlations"],
        "segment_exc_mults": segment_exc,
        "base_config": {
            "a": 1.0, "theta": 7.5, "fr_cap": 200.0,
            "exc_mult": 0.01, "inh_mult": 0.01, "inh_scale": 2.0,
            "use_adaptation": False, "normalize_weights": False,
            "delay_inh_ms": 3.0, "param_cv": 0.10,
        },
    }
    with open(str(out), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved best config: {out}")


def plot_best(best: dict, all_results: list):
    """Plot trace comparison and correlation heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Trace plot for best config ---
    if "flex_traces" in best:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Best: {best['name']} ({best['n_antiphase']}/6 AP, "
                     f"mean_r={best['mean_corr']:+.3f})", fontsize=13)

        for i, leg_name in enumerate(LEG_ORDER):
            ax = axes[i // 2, i % 2]
            ax.plot(best["time_axis"], best["flex_traces"][leg_name],
                    "b-", alpha=0.7, label="Flex", linewidth=0.8)
            ax.plot(best["time_axis"], best["ext_traces"][leg_name],
                    "r-", alpha=0.7, label="Ext", linewidth=0.8)
            ax.axvline(500, color="gray", linestyle="--", alpha=0.3)
            corr = best["correlations"][leg_name]
            marker = " *AP*" if corr < -0.3 else ""
            ax.set_title(f"{leg_name} (r={corr:+.3f}){marker}")
            ax.set_ylabel("Hz")
            if i >= 4:
                ax.set_xlabel("Time (ms)")
            ax.legend(fontsize=7)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        out = fig_dir / "vnc_segment_specific_best.png"
        plt.savefig(str(out), dpi=150)
        print(f"Plot: {out}")
        plt.close()

    # --- Heatmap: all configs x legs ---
    valid = [r for r in all_results if "error" not in r and "correlations" in r]
    if len(valid) > 1:
        names = [r["name"] for r in valid]
        corr_matrix = np.array([[r["correlations"][l] for l in LEG_ORDER] for r in valid])

        fig, ax = plt.subplots(figsize=(10, max(6, len(valid) * 0.4)))
        im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(6))
        ax.set_xticklabels(LEG_ORDER)
        ax.set_yticks(range(len(valid)))
        ax.set_yticklabels(names, fontsize=7)
        for i in range(len(valid)):
            for j in range(6):
                val = corr_matrix[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6, color=color)
        plt.colorbar(im, label="Flex-Ext Correlation")
        ax.set_title("Segment-Specific Parameter Sweep: Flex-Ext Correlations")
        plt.tight_layout()
        out = fig_dir / "vnc_segment_specific_heatmap.png"
        plt.savefig(str(out), dpi=150)
        print(f"Plot: {out}")
        plt.close()


if __name__ == "__main__":
    main()
