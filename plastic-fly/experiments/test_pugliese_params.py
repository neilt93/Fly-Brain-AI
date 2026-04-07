"""
Test Pugliese et al. 2025 exact parameters on the firing rate VNC model.

Hypothesis: Row-sum normalization destroys the half-center inhibition that
creates anti-phase flex/ext alternation. Pugliese uses raw synapse counts * 0.03
with NO normalization.

Runs multiple configurations:
  1. pugliese_exact: a=1, theta=7.5, exc/inh_mult=0.03, NO norm, NO adapt
  2. pugliese_no_norm: our params (a=10, theta=3) but NO normalization
  3. pugliese_strong_inh: Pugliese exact + inh_scale=3.0
  4. pugliese_exact_adapt: Pugliese exact + adaptation (hybrid)

For each config, records flex/ext correlations per leg.
"""

import sys
import os
import numpy as np
from pathlib import Path
from time import time

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.vnc_firing_rate import FiringRateVNCRunner, FiringRateVNCConfig, LEG_ORDER


def run_config(name: str, cfg: FiringRateVNCConfig, sim_ms: float = 2000.0,
               dn_baseline_hz: float = 25.0, dng100_hz: float = 60.0,
               warmup_ms: float = 100.0) -> dict:
    """Run a single configuration and return flex/ext correlations."""
    print(f"\n{'='*72}")
    print(f"CONFIG: {name}")
    print(f"  a={cfg.a}, theta={cfg.theta}, exc_mult={cfg.exc_mult}, "
          f"inh_mult={cfg.inh_mult}, inh_scale={cfg.inh_scale}")
    print(f"  normalize={cfg.normalize_weights}, adaptation={cfg.use_adaptation}, "
          f"delay={cfg.use_delay}")
    print(f"{'='*72}")

    try:
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=warmup_ms)
    except Exception as e:
        print(f"  FAILED to build network: {e}")
        return {"name": name, "error": str(e)}

    # Apply stimulation
    runner.stimulate_all_dns(rate_hz=dn_baseline_hz)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    # Run simulation
    dt = 0.5  # ms
    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)  # record every 1ms
    n_records = int(sim_ms / 1.0)
    time_axis = np.linspace(0, sim_ms, n_records)

    flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    all_mn_traces = np.zeros((n_records, runner.n_mn), dtype=np.float32)

    print(f"\n  Running {sim_ms:.0f}ms simulation...")
    t0 = time()
    rec_idx = 0
    for step_i in range(n_steps):
        runner.step(dt_ms=dt)
        if (step_i + 1) % record_every == 0 and rec_idx < n_records:
            mn_rates = runner.get_mn_rates()
            all_mn_traces[rec_idx] = mn_rates
            for leg_idx, leg_name in enumerate(LEG_ORDER):
                flex_rate, ext_rate = runner.get_flexor_extensor_rates(leg_idx)
                flex_traces[leg_name][rec_idx] = flex_rate
                ext_traces[leg_name][rec_idx] = ext_rate
            rec_idx += 1

    sim_time = time() - t0
    print(f"  Done in {sim_time:.1f}s ({n_steps / sim_time:.0f} steps/s)")

    # Analysis: skip first 500ms transient
    skip_idx = int(500.0 / 1.0)
    results = {"name": name, "correlations": {}, "flex_mean": {}, "ext_mean": {},
               "flex_max": {}, "ext_max": {}, "flex_std": {}, "ext_std": {}}

    print(f"\n  Flex/Ext Alternation Analysis (last {sim_ms - 500:.0f}ms):")
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
        results["flex_mean"][leg_name] = float(f.mean())
        results["ext_mean"][leg_name] = float(e.mean())
        results["flex_max"][leg_name] = float(f.max())
        results["ext_max"][leg_name] = float(e.max())
        results["flex_std"][leg_name] = float(f.std())
        results["ext_std"][leg_name] = float(e.std())

        status = "ANTI-PHASE" if corr < -0.3 else ("decorrelated" if abs(corr) < 0.2 else "CO-ACTIVE")
        print(f"    {leg_name}: flex={f.mean():.1f}+/-{f.std():.1f}Hz, "
              f"ext={e.mean():.1f}+/-{e.std():.1f}Hz, corr={corr:+.3f} -> {status}")

    # Spectral analysis on LF flexors
    f_lf = flex_traces["LF"][skip_idx:]
    if f_lf.max() > 1.0 and f_lf.std() > 1e-6:
        f_centered = f_lf - f_lf.mean()
        freqs = np.fft.rfftfreq(len(f_centered), d=1.0 / 1000.0)
        psd = np.abs(np.fft.rfft(f_centered)) ** 2
        band_mask = (freqs >= 3.0) & (freqs <= 25.0)
        if band_mask.any():
            band_psd = psd[band_mask]
            band_freqs = freqs[band_mask]
            peak_idx = np.argmax(band_psd)
            peak_freq = band_freqs[peak_idx]
            total_power = psd[freqs > 1.0].sum()
            frac = band_psd[peak_idx] / total_power if total_power > 0 else 0
            results["peak_freq"] = float(peak_freq)
            results["peak_power_frac"] = float(frac)
            print(f"  LF spectral peak: {peak_freq:.1f} Hz (power frac: {frac:.1%})")

    n_antiphase = sum(1 for c in results["correlations"].values() if c < -0.3)
    n_decorr = sum(1 for c in results["correlations"].values() if abs(c) < 0.2)
    n_active = sum(1 for leg in LEG_ORDER
                   if results["flex_max"].get(leg, 0) > 1.0 or results["ext_max"].get(leg, 0) > 1.0)
    results["n_antiphase"] = n_antiphase
    results["n_decorrelated"] = n_decorr
    results["n_active"] = n_active

    mean_corr = np.mean(list(results["correlations"].values()))
    results["mean_corr"] = float(mean_corr)

    print(f"\n  SUMMARY: {n_antiphase}/6 anti-phase, {n_decorr}/6 decorrelated, "
          f"{n_active}/6 active, mean_corr={mean_corr:+.3f}")

    # Store traces for plotting
    results["time_axis"] = time_axis
    results["flex_traces"] = flex_traces
    results["ext_traces"] = ext_traces

    return results


def main():
    print("=" * 72)
    print("PUGLIESE PARAMETER SWEEP: Finding Anti-Phase Flex/Ext Alternation")
    print("=" * 72)

    all_results = []

    # ---- Config 1: Pugliese exact ----
    cfg1 = FiringRateVNCConfig.pugliese_exact()
    r1 = run_config("pugliese_exact", cfg1)
    all_results.append(r1)

    # ---- Config 2: Our params but no normalization ----
    cfg2 = FiringRateVNCConfig.pugliese_no_norm()
    r2 = run_config("our_params_no_norm", cfg2)
    all_results.append(r2)

    # ---- Config 3: Pugliese exact + stronger inhibition ----
    cfg3 = FiringRateVNCConfig.pugliese_strong_inh()
    r3 = run_config("pugliese_strong_inh", cfg3)
    all_results.append(r3)

    # ---- Config 4: Pugliese exact + adaptation (hybrid) ----
    cfg4 = FiringRateVNCConfig.pugliese_exact()
    cfg4.use_adaptation = True
    cfg4.tau_adapt_ms = 30.0
    cfg4.b_adapt = 0.15
    r4 = run_config("pugliese_exact_with_adapt", cfg4)
    all_results.append(r4)

    # ---- Config 5: Pugliese exact + inh_scale=4.0 ----
    cfg5 = FiringRateVNCConfig.pugliese_exact()
    cfg5.inh_scale = 4.0
    r5 = run_config("pugliese_inh_scale_4", cfg5)
    all_results.append(r5)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("GRAND SUMMARY")
    print("=" * 72)
    print(f"{'Config':<30} {'Anti-phase':>10} {'Decorr':>8} {'Active':>8} {'Mean corr':>10}")
    print("-" * 72)

    best = None
    best_score = -999
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>10}")
            continue
        n_ap = r.get("n_antiphase", 0)
        n_dc = r.get("n_decorrelated", 0)
        n_act = r.get("n_active", 0)
        mc = r.get("mean_corr", 0)
        freq = r.get("peak_freq", 0)
        print(f"{r['name']:<30} {n_ap:>10}/6 {n_dc:>8}/6 {n_act:>8}/6 {mc:>+10.3f}"
              f"  peak={freq:.1f}Hz")
        # Score: maximize anti-phase, minimize mean correlation
        score = n_ap * 10 - mc * 5 + n_act
        if score > best_score:
            best_score = score
            best = r

    if best and "error" not in best:
        print(f"\nBest config: {best['name']} (score={best_score:.1f})")
        if best.get("n_antiphase", 0) >= 3:
            print("*** BREAKTHROUGH: Anti-phase alternation achieved! ***")
        elif best.get("mean_corr", 1) < 0.1:
            print("Progress: mean correlation below 0.1")
        else:
            print("Anti-phase not yet achieved. Next steps: see progress file.")

    # ---- Plot best config ----
    if best and "error" not in best and "flex_traces" in best:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Firing Rate VNC: {best['name']}\n"
                         f"(mean_corr={best['mean_corr']:+.3f}, "
                         f"{best['n_antiphase']}/6 anti-phase, "
                         f"{best['n_active']}/6 active)")

            time_axis = best["time_axis"]
            for i, leg_name in enumerate(LEG_ORDER):
                ax = axes[i // 2, i % 2]
                ax.plot(time_axis, best["flex_traces"][leg_name], "b-", alpha=0.7, label="Flexor")
                ax.plot(time_axis, best["ext_traces"][leg_name], "r-", alpha=0.7, label="Extensor")
                ax.axvline(500, color="gray", linestyle="--", alpha=0.3)
                corr = best["correlations"][leg_name]
                ax.set_title(f"{leg_name} (corr={corr:+.3f})")
                ax.set_ylabel("Rate (Hz)")
                if i >= 4:
                    ax.set_xlabel("Time (ms)")
                ax.legend(loc="upper right", fontsize=7)
                ax.set_ylim(bottom=0)

            plt.tight_layout()
            out_path = ROOT / "figures" / "vnc_pugliese_params_best.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_path), dpi=150)
            print(f"\nPlot saved to {out_path}")
            plt.close()

            # Also plot all configs' correlation profiles
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            x = np.arange(6)
            width = 0.15
            for ci, r in enumerate(all_results):
                if "error" in r:
                    continue
                corrs = [r["correlations"].get(leg, 0) for leg in LEG_ORDER]
                ax2.bar(x + ci * width, corrs, width, label=r["name"], alpha=0.8)

            ax2.set_xticks(x + width * len(all_results) / 2)
            ax2.set_xticklabels(LEG_ORDER)
            ax2.set_ylabel("Flex-Ext Correlation")
            ax2.set_title("Flex/Ext Correlation by Configuration")
            ax2.axhline(0, color="black", linewidth=0.5)
            ax2.axhline(-0.3, color="green", linewidth=0.5, linestyle="--", label="Anti-phase threshold")
            ax2.legend(fontsize=7, loc="upper left")
            ax2.set_ylim(-1.1, 1.1)

            out_path2 = ROOT / "figures" / "vnc_pugliese_params_comparison.png"
            plt.savefig(str(out_path2), dpi=150)
            print(f"Comparison plot saved to {out_path2}")
            plt.close()

        except ImportError:
            print("\nmatplotlib not available -- skipping plots")

    print("\nDone.")


if __name__ == "__main__":
    main()
