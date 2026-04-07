"""
Pugliese parameter sweep round 3: fine-tuning the best regime.

Best from round 2: a=1, theta=7.5, exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
NO normalization, NO adaptation -> 2/6 anti-phase (RM, RH), mean_corr=-0.011.

Key observations:
1. Anti-phase consistently appears on RIGHT side (RM, RH), not left
2. This could be MANC wiring asymmetry (male CNS) or seed-dependent
3. Without adaptation, rhythm comes from network dynamics alone (~14.7Hz)
4. Adding adaptation destroys the anti-phase (pushes to co-active)

This round:
- Fine-tune around the best config (inh_scale 1.5-3.0, exc_mult 0.008-0.015)
- Also analyze: which specific premotor interneurons create the half-center?
- Try different seeds to test robustness
- Try T1-only (Pugliese focused on T1)
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
    print(f"  a={cfg.a}, theta={cfg.theta}, exc={cfg.exc_mult}, inh={cfg.inh_mult}, "
          f"inh_sc={cfg.inh_scale}, norm={cfg.normalize_weights}, adapt={cfg.use_adaptation}, "
          f"segs={cfg.segments}")
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

    # Spectral analysis on a working leg
    for check_leg in ["RM", "RH", "LF"]:
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

    results["time_axis"] = time_axis
    results["flex_traces"] = flex_traces
    results["ext_traces"] = ext_traces
    return results


def main():
    print("=" * 72)
    print("PUGLIESE ROUND 3: Fine-tuning Anti-Phase Regime")
    print("=" * 72)

    all_results = []

    # Baseline: best from round 2
    def make_cfg(inh_scale=2.0, exc_mult=0.01, inh_mult=0.01, adapt=False,
                 b_adapt=0.15, tau_adapt=30.0, segs=("T1", "T2", "T3"),
                 seed=42, delay_ms=3.0, theta=7.5, a=1.0):
        return FiringRateVNCConfig(
            a=a, theta=theta, fr_cap=200.0,
            exc_mult=exc_mult, inh_mult=inh_mult, inh_scale=inh_scale,
            use_adaptation=adapt, tau_adapt_ms=tau_adapt, b_adapt=b_adapt,
            use_delay=True, delay_inh_ms=delay_ms,
            normalize_weights=False, param_cv=0.10, seed=seed,
            segments=segs,
        )

    # 1. Baseline (best from round 2)
    all_results.append(run_config("baseline_inh2",
        make_cfg(inh_scale=2.0)))

    # 2. inh_scale sweep: 1.5, 2.5, 3.0
    all_results.append(run_config("inh_1.5",
        make_cfg(inh_scale=1.5)))
    all_results.append(run_config("inh_2.5",
        make_cfg(inh_scale=2.5)))
    all_results.append(run_config("inh_3.0",
        make_cfg(inh_scale=3.0)))

    # 3. exc_mult sweep: 0.008, 0.012, 0.015
    all_results.append(run_config("exc_008",
        make_cfg(exc_mult=0.008, inh_mult=0.008)))
    all_results.append(run_config("exc_012",
        make_cfg(exc_mult=0.012, inh_mult=0.012)))
    all_results.append(run_config("exc_015",
        make_cfg(exc_mult=0.015, inh_mult=0.015)))

    # 4. Longer delay (5ms, 7ms) -- more temporal separation
    all_results.append(run_config("delay_5ms",
        make_cfg(delay_ms=5.0)))
    all_results.append(run_config("delay_7ms",
        make_cfg(delay_ms=7.0)))

    # 5. Different seeds to check robustness
    all_results.append(run_config("seed_123",
        make_cfg(seed=123)))
    all_results.append(run_config("seed_7",
        make_cfg(seed=7)))

    # 6. T1-only (Pugliese's original scope)
    all_results.append(run_config("T1_only",
        make_cfg(segs=("T1",))))

    # 7. Try moderate adaptation with short tau (might enhance rhythm
    # without destroying anti-phase)
    all_results.append(run_config("mild_adapt",
        make_cfg(adapt=True, b_adapt=0.05, tau_adapt=15.0)))

    # Summary
    print("\n" + "=" * 72)
    print("GRAND SUMMARY ROUND 3")
    print("=" * 72)
    print(f"{'Config':<22} {'AP':>3} {'Mean r':>8} {'Freq':>6} {'MN':>6} {'Sat%':>5} "
          f"{'LF':>6} {'LM':>6} {'LH':>6} {'RF':>6} {'RM':>6} {'RH':>6}")
    print("-" * 100)

    best = None
    best_score = -999
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<22} ERROR")
            continue
        n_ap = r["n_antiphase"]
        mc = r["mean_corr"]
        freq = r.get("peak_freq", 0)
        mn = r.get("mn_mean", 0)
        sat = r.get("sat_pct", 0)
        corrs = r["correlations"]
        c_str = " ".join(f"{corrs[l]:>+6.2f}" for l in LEG_ORDER)
        print(f"{r['name']:<22} {n_ap:>3} {mc:>+8.3f} {freq:>5.1f}Hz {mn:>5.1f} {sat:>4.1f}% {c_str}")
        score = n_ap * 10 - mc * 5
        if score > best_score:
            best_score = score
            best = r

    if best:
        print(f"\nBest: {best['name']} (score={best_score:.1f})")

    # Plot comparison of best vs baseline
    if best and "flex_traces" in best:
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
            out = ROOT / "figures" / "vnc_pugliese_round3_best.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out), dpi=150)
            print(f"\nPlot: {out}")
            plt.close()

            # Correlation heatmap: all configs x legs
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            configs_ok = [r for r in all_results if "error" not in r]
            n_cfgs = len(configs_ok)
            corr_matrix = np.zeros((n_cfgs, 6))
            names = []
            for ci, r in enumerate(configs_ok):
                names.append(r["name"])
                for li, leg in enumerate(LEG_ORDER):
                    corr_matrix[ci, li] = r["correlations"][leg]

            im = ax2.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            ax2.set_xticks(range(6))
            ax2.set_xticklabels(LEG_ORDER)
            ax2.set_yticks(range(n_cfgs))
            ax2.set_yticklabels(names, fontsize=8)
            for ci in range(n_cfgs):
                for li in range(6):
                    v = corr_matrix[ci, li]
                    color = "white" if abs(v) > 0.5 else "black"
                    ax2.text(li, ci, f"{v:+.2f}", ha="center", va="center",
                             fontsize=7, color=color)
            plt.colorbar(im, label="Flex-Ext Correlation")
            ax2.set_title("Flex/Ext Correlation: All Configs x Legs")
            plt.tight_layout()
            out2 = ROOT / "figures" / "vnc_pugliese_round3_heatmap.png"
            plt.savefig(str(out2), dpi=150)
            print(f"Heatmap: {out2}")
            plt.close()

        except ImportError:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
