"""
Pugliese parameter sweep round 2: address saturation problem.

Key finding from round 1: raw exc_mult=0.03 creates weights in [-25.9, 24.6],
which saturates most neurons at fr_cap=200 Hz. Anti-phase only appears on 2/6
legs where inhibition happens to be strong enough to compete with saturation.

Hypothesis: The problem is SATURATION, not normalization per se. We need weights
that place the network in a balanced regime where both E and I populations
can modulate each other's activity without hitting the ceiling.

New approach: scale weights so mean input at moderate population rates (~30Hz)
sits just above threshold. This is what normalization was TRYING to do, but
we'll try it with Pugliese's activation function (a=1, theta=7.5).

Configs to test:
  1. pugliese_a1_scaled: a=1, theta=7.5, lower exc_mult to avoid saturation
  2. pugliese_a1_inh2: same + inh_scale=2 for stronger push-pull
  3. pugliese_a1_adapt: same + adaptation for robust rhythm
  4. pugliese_a1_inh2_adapt: the full combination
  5. pugliese_a1_inh3_adapt: even stronger inhibition + adaptation
"""

import sys
import numpy as np
from pathlib import Path
from time import time

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
    print(f"  normalize={cfg.normalize_weights}, adaptation={cfg.use_adaptation}")
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
    record_every = int(1.0 / dt)
    n_records = int(sim_ms / 1.0)
    time_axis = np.linspace(0, sim_ms, n_records)

    flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}

    print(f"\n  Running {sim_ms:.0f}ms...")
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
    print(f"  Done in {sim_time:.1f}s ({n_steps / sim_time:.0f} steps/s)")

    # Quick check: what's the mean firing rate?
    all_rates = runner.get_all_rates()
    mn_rates = runner.get_mn_rates()
    print(f"  Final rates: all mean={all_rates.mean():.1f}Hz (max={all_rates.max():.1f}), "
          f"MN mean={mn_rates.mean():.1f}Hz (max={mn_rates.max():.1f})")
    n_saturated = (all_rates > 0.9 * cfg.fr_cap).sum()
    print(f"  Saturated (>90% cap): {n_saturated}/{len(all_rates)} neurons "
          f"({100*n_saturated/len(all_rates):.1f}%)")

    # Analysis
    skip_idx = int(500.0 / 1.0)
    results = {"name": name, "correlations": {}, "flex_mean": {}, "ext_mean": {},
               "flex_std": {}, "ext_std": {}, "flex_max": {}, "ext_max": {}}

    print(f"\n  Flex/Ext Analysis (last {sim_ms - 500:.0f}ms):")
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
        results["flex_std"][leg_name] = float(f.std())
        results["ext_std"][leg_name] = float(e.std())
        results["flex_max"][leg_name] = float(f.max())
        results["ext_max"][leg_name] = float(e.max())

        status = "ANTI-PHASE" if corr < -0.3 else ("decorr" if abs(corr) < 0.2 else "co-act")
        print(f"    {leg_name}: F={f.mean():.1f}+/-{f.std():.1f}, "
              f"E={e.mean():.1f}+/-{e.std():.1f}, r={corr:+.3f} {status}")

    # Spectral
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
            results["peak_freq"] = float(band_freqs[peak_idx])
            print(f"  Peak freq: {results['peak_freq']:.1f} Hz")

    n_ap = sum(1 for c in results["correlations"].values() if c < -0.3)
    n_dc = sum(1 for c in results["correlations"].values() if abs(c) < 0.2)
    n_act = sum(1 for leg in LEG_ORDER
                if results["flex_max"].get(leg, 0) > 1.0 or results["ext_max"].get(leg, 0) > 1.0)
    mc = float(np.mean(list(results["correlations"].values())))
    results["n_antiphase"] = n_ap
    results["n_decorrelated"] = n_dc
    results["n_active"] = n_act
    results["mean_corr"] = mc

    print(f"  => {n_ap}/6 anti-phase, {n_dc}/6 decorr, mean_corr={mc:+.3f}")

    results["time_axis"] = time_axis
    results["flex_traces"] = flex_traces
    results["ext_traces"] = ext_traces

    return results


def main():
    print("=" * 72)
    print("PUGLIESE ROUND 2: Avoiding Saturation, Finding Anti-Phase")
    print("=" * 72)

    all_results = []

    # The key insight: with 7539 neurons averaging ~135 exc connections each,
    # mean exc row sum = 0.22 at exc_mult=0.03. At population rate 50Hz,
    # mean exc input = 0.22 * 50 = 11.0, which is above theta=7.5.
    # But max rate neuron gets 24.6 * 50 = 1230 input, which drives to saturation.
    # We need exc_mult small enough that heavy-hitter connections don't saturate.
    #
    # Target: at 50Hz pop rate, mean input ~= theta (7.5).
    # mean_exc_row_sum(at 0.03) = 0.22 * N_exc, want 0.22*50 ~ 11 (close to theta).
    # Actually that's fine for mean. Problem is HIGH fan-in neurons saturating.
    #
    # Better approach: use exc_mult=0.01 (3x less than Pugliese),
    # which gives mean_exc_row_sum ~ 0.073, mean input at 50Hz = 3.7 < theta=7.5.
    # Only highly connected neurons will cross threshold.

    # Config 1: Pugliese activation (a=1, theta=7.5) with 0.01 multiplier, no norm
    cfg1 = FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
        use_adaptation=False, use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False, param_cv=0.10, seed=42,
    )
    all_results.append(run_config("a1_mult01_inh2_nonorm", cfg1))

    # Config 2: Same + adaptation for rhythm generation
    cfg2 = FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
        use_adaptation=True, tau_adapt_ms=30.0, b_adapt=0.15,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False, param_cv=0.10, seed=42,
    )
    all_results.append(run_config("a1_mult01_inh2_adapt", cfg2))

    # Config 3: Higher inhibition (inh_scale=3) + adaptation
    cfg3 = FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=3.0,
        use_adaptation=True, tau_adapt_ms=30.0, b_adapt=0.15,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False, param_cv=0.10, seed=42,
    )
    all_results.append(run_config("a1_mult01_inh3_adapt", cfg3))

    # Config 4: Even lower threshold (theta=5) to allow more neurons to participate
    # while keeping raw weights
    cfg4 = FiringRateVNCConfig(
        a=1.0, theta=5.0, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.5,
        use_adaptation=True, tau_adapt_ms=25.0, b_adapt=0.20,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False, param_cv=0.10, seed=42,
    )
    all_results.append(run_config("a1_theta5_inh25_adapt", cfg4))

    # Config 5: Intermediate gain a=3 with lower theta
    # This gives effective gain a/fr_cap = 3/200 = 0.015
    cfg5 = FiringRateVNCConfig(
        a=3.0, theta=5.0, fr_cap=200.0,
        exc_mult=0.008, inh_mult=0.008, inh_scale=2.5,
        use_adaptation=True, tau_adapt_ms=25.0, b_adapt=0.20,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False, param_cv=0.10, seed=42,
    )
    all_results.append(run_config("a3_theta5_mult008_inh25", cfg5))

    # Config 6: Original parameters but with normalization AND higher inh_scale
    # to see if normalization + strong inhibition can produce anti-phase
    cfg6 = FiringRateVNCConfig(
        a=10.0, theta=3.0, fr_cap=200.0,
        exc_mult=0.005, inh_mult=0.005, inh_scale=4.0,
        use_adaptation=True, tau_adapt_ms=30.0, b_adapt=0.25,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=True, target_exc_sum=0.6,
        param_cv=0.10, seed=42,
    )
    all_results.append(run_config("original_inh4_norm", cfg6))

    # Config 7: Pugliese activation with normalization that targets theta
    # normalize so mean exc row sum * 30Hz = theta (7.5)
    # target_exc_sum = 7.5 / 30 = 0.25
    cfg7 = FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.03, inh_mult=0.03, inh_scale=2.0,
        use_adaptation=True, tau_adapt_ms=25.0, b_adapt=0.15,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=True, target_exc_sum=0.25,
        param_cv=0.10, seed=42,
    )
    all_results.append(run_config("a1_norm025_inh2_adapt", cfg7))

    # Summary
    print("\n" + "=" * 72)
    print("GRAND SUMMARY ROUND 2")
    print("=" * 72)
    print(f"{'Config':<32} {'AP':>3} {'DC':>3} {'Act':>3} {'Mean r':>8} {'Freq':>6}")
    print("-" * 60)

    best = None
    best_score = -999
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<32} ERROR: {r['error'][:30]}")
            continue
        n_ap = r.get("n_antiphase", 0)
        n_dc = r.get("n_decorrelated", 0)
        n_act = r.get("n_active", 0)
        mc = r.get("mean_corr", 0)
        freq = r.get("peak_freq", 0)
        print(f"{r['name']:<32} {n_ap:>3} {n_dc:>3} {n_act:>3} {mc:>+8.3f} {freq:>5.1f}Hz")
        score = n_ap * 10 - mc * 5 + n_act * 0.5
        if score > best_score:
            best_score = score
            best = r

    if best and "error" not in best:
        print(f"\nBest: {best['name']}")
        if best.get("n_antiphase", 0) >= 3:
            print("*** BREAKTHROUGH ***")

    # Plot best
    if best and "error" not in best and "flex_traces" in best:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Best: {best['name']} (mean_corr={best['mean_corr']:+.3f}, "
                         f"{best['n_antiphase']}/6 AP)")

            for i, leg_name in enumerate(LEG_ORDER):
                ax = axes[i // 2, i % 2]
                ax.plot(best["time_axis"], best["flex_traces"][leg_name], "b-", alpha=0.7, label="Flex")
                ax.plot(best["time_axis"], best["ext_traces"][leg_name], "r-", alpha=0.7, label="Ext")
                ax.axvline(500, color="gray", linestyle="--", alpha=0.3)
                corr = best["correlations"][leg_name]
                ax.set_title(f"{leg_name} (r={corr:+.3f})")
                ax.set_ylabel("Hz")
                if i >= 4:
                    ax.set_xlabel("Time (ms)")
                ax.legend(fontsize=7)
                ax.set_ylim(bottom=0)

            plt.tight_layout()
            out = ROOT / "figures" / "vnc_pugliese_round2_best.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out), dpi=150)
            print(f"\nPlot: {out}")
            plt.close()
        except ImportError:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
