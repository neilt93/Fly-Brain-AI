"""
BANC VNC firing-rate model test: female VNC vs male MANC comparison.

Loads BANC connectome data (adult female Drosophila, Harvard Dataverse),
builds a Pugliese-style firing rate VNC model, stimulates DNg100 at 60 Hz,
and compares flexor/extensor anti-phase dynamics with the MANC (male) results.

Key question: does the FEMALE BANC VNC produce better anti-phase than
the MALE MANC VNC?

Usage:
    python experiments/test_banc_vnc.py
    python experiments/test_banc_vnc.py --sim-ms 3000  # longer run
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from time import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.banc_loader import load_banc_vnc, LEG_ORDER
from bridge.vnc_firing_rate import FiringRateVNCRunner, FiringRateVNCConfig

# ============================================================================
# Configuration
# ============================================================================

# Best config from MANC experiments (Pass 2, Round 2):
# a=1.0, theta=7.5, exc_mult=0.01, inh_scale=2.0, no norm, no adapt
BANC_CFG = FiringRateVNCConfig(
    tau_ms=20.0,
    a=1.0,
    theta=7.5,
    fr_cap=200.0,
    exc_mult=0.01,
    inh_mult=0.01,
    inh_scale=2.0,
    use_adaptation=False,
    use_delay=True,
    delay_inh_ms=3.0,
    normalize_weights=False,
    param_cv=0.10,
    seed=42,
)


# ============================================================================
# Run simulation
# ============================================================================

def run_banc_simulation(
    sim_ms: float = 2000.0,
    dt: float = 0.5,
    dn_baseline_hz: float = 25.0,
    dng100_hz: float = 60.0,
    warmup_ms: float = 100.0,
    seed: int = 42,
) -> dict:
    """Run firing-rate VNC model on BANC data and analyze results.

    Returns dict with per-leg metrics and trace data.
    """
    print("=" * 72)
    print("BANC VNC Firing Rate Model -- DNg100 Stimulation Test")
    print("=" * 72)

    # Load BANC data
    print("\n--- Loading BANC data ---")
    banc_data = load_banc_vnc(
        normalize_weights=False,
        exc_mult=BANC_CFG.exc_mult,
        inh_mult=BANC_CFG.inh_mult,
        inh_scale=BANC_CFG.inh_scale,
    )

    # Print network comparison
    print(f"\n--- BANC vs MANC network sizes ---")
    print(f"  BANC:  {banc_data.n_neurons} neurons "
          f"({banc_data.n_dn} DN, {banc_data.n_mn} MN, "
          f"{banc_data.n_premotor} premotor), "
          f"{banc_data.n_synapses:,} connections")
    print(f"  MANC:  7,539 neurons (1,314 DN, 381 MN, 5,844 premotor), "
          f"1,022,190 connections")

    # Build runner
    cfg = FiringRateVNCConfig(
        tau_ms=BANC_CFG.tau_ms,
        a=BANC_CFG.a,
        theta=BANC_CFG.theta,
        fr_cap=BANC_CFG.fr_cap,
        exc_mult=BANC_CFG.exc_mult,
        inh_mult=BANC_CFG.inh_mult,
        inh_scale=BANC_CFG.inh_scale,
        use_adaptation=BANC_CFG.use_adaptation,
        use_delay=BANC_CFG.use_delay,
        delay_inh_ms=BANC_CFG.delay_inh_ms,
        normalize_weights=BANC_CFG.normalize_weights,
        param_cv=BANC_CFG.param_cv,
        seed=seed,
    )
    runner = FiringRateVNCRunner.from_banc(banc_data, cfg=cfg, warmup_ms=warmup_ms)

    # Stimulate
    runner.stimulate_all_dns(rate_hz=dn_baseline_hz)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    # Run simulation
    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)  # 1 kHz recording
    n_records = int(sim_ms / 1.0)
    time_axis = np.linspace(0, sim_ms, n_records)

    flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    all_mn_traces = np.zeros((n_records, runner.n_mn), dtype=np.float32)

    print(f"\nRunning simulation: {sim_ms:.0f} ms, dt={dt} ms, {n_steps} steps...")
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
    print(f"Simulation done in {sim_time:.1f}s ({n_steps / sim_time:.0f} steps/s)")

    # ---- Analysis ----
    print("\n" + "=" * 72)
    print("Flexor-Extensor Alternation Analysis (BANC)")
    print("=" * 72)

    skip_ms = 500
    skip_idx = int(skip_ms / 1.0)

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
            "flex_max": float(f.max()),
            "ext_max": float(e.max()),
            "flex_active": bool(f_active),
            "ext_active": bool(e_active),
            "correlation": corr,
        }

        status = ("ANTI-PHASE" if corr < -0.3
                  else "DECOR" if corr < -0.1
                  else "ACTIVE" if (f_active or e_active)
                  else "SILENT")
        print(f"  {leg_name}: flex={f.mean():.1f}Hz (max {f.max():.1f}), "
              f"ext={e.mean():.1f}Hz (max {e.max():.1f}), "
              f"corr={corr:+.3f} -> {status}")

    n_antiphase = sum(1 for r in results.values() if r["correlation"] < -0.3)
    n_decor = sum(1 for r in results.values() if r["correlation"] < -0.1)
    n_active = sum(1 for r in results.values() if r["flex_active"] or r["ext_active"])
    mean_corr = np.mean([r["correlation"] for r in results.values()])

    print(f"\nSummary: {n_antiphase}/6 anti-phase (r<-0.3), "
          f"{n_decor}/6 decorrelated (r<-0.1), "
          f"{n_active}/6 active, mean_corr={mean_corr:+.3f}")

    # ---- Spectral analysis ----
    print("\n" + "=" * 72)
    print("Spectral Analysis (locomotor band 5-20 Hz)")
    print("=" * 72)

    spectral_results = {}
    for leg_name in LEG_ORDER:
        f = flex_traces[leg_name][skip_idx:]
        if f.max() < 1.0:
            spectral_results[leg_name] = {"peak_freq": 0.0, "band_frac": 0.0}
            print(f"  {leg_name}: too quiet")
            continue

        f_centered = f - f.mean()
        n_samples = len(f_centered)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / 1000.0)
        psd = np.abs(np.fft.rfft(f_centered)) ** 2

        band_mask = (freqs >= 5.0) & (freqs <= 20.0)
        if band_mask.any():
            band_psd = psd[band_mask]
            band_freqs = freqs[band_mask]
            peak_idx = np.argmax(band_psd)
            peak_freq = float(band_freqs[peak_idx])
            total_power = float(psd[freqs > 1.0].sum())
            band_frac = float(band_psd.sum() / total_power) if total_power > 0 else 0.0
            spectral_results[leg_name] = {"peak_freq": peak_freq, "band_frac": band_frac}
            print(f"  {leg_name}: peak at {peak_freq:.1f} Hz, "
                  f"5-20Hz band: {band_frac:.1%} of power")
        else:
            spectral_results[leg_name] = {"peak_freq": 0.0, "band_frac": 0.0}

    # ---- Plot ----
    print("\n--- Generating figures ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(
            f"BANC (female) VNC Firing Rate Model -- DNg100 @ {dng100_hz} Hz\n"
            f"Config: a={cfg.a}, theta={cfg.theta}, exc={cfg.exc_mult}, "
            f"inh_scale={cfg.inh_scale}, no_norm, no_adapt, seed={seed}",
            fontsize=12,
        )

        t_plot = time_axis / 1000.0  # convert to seconds

        for i, leg_name in enumerate(LEG_ORDER):
            ax = axes[i]
            ax.plot(t_plot, flex_traces[leg_name], "b-", alpha=0.7, lw=0.8, label="Flexor")
            ax.plot(t_plot, ext_traces[leg_name], "r-", alpha=0.7, lw=0.8, label="Extensor")
            corr = results[leg_name]["correlation"]
            tag = "AP" if corr < -0.3 else ""
            ax.set_ylabel(f"{leg_name}\n(r={corr:+.2f}) {tag}", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(bottom=0)

        axes[-1].set_xlabel("Time (s)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = ROOT / "figures" / "banc_vnc_firing_rate.png"
        fig_path.parent.mkdir(exist_ok=True)
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_path}")

        # Correlation comparison bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        manc_corrs = {
            "LF": 0.0, "LM": +0.71, "LH": +0.85,
            "RF": 0.0, "RM": -0.67, "RH": -0.95,
        }

        x = np.arange(6)
        width = 0.35
        banc_corrs = [results[leg]["correlation"] for leg in LEG_ORDER]
        manc_vals = [manc_corrs[leg] for leg in LEG_ORDER]

        bars_banc = ax2.bar(x - width / 2, banc_corrs, width, label="BANC (female)",
                            color="coral", edgecolor="black", linewidth=0.5)
        bars_manc = ax2.bar(x + width / 2, manc_vals, width, label="MANC (male)",
                            color="steelblue", edgecolor="black", linewidth=0.5)

        ax2.axhline(-0.3, color="green", ls="--", alpha=0.5, label="Anti-phase threshold")
        ax2.axhline(0, color="gray", ls="-", alpha=0.3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(LEG_ORDER)
        ax2.set_ylabel("Flex-Ext Correlation")
        ax2.set_title("BANC (female) vs MANC (male) -- Flex/Ext Correlation per Leg")
        ax2.legend()
        ax2.set_ylim(-1.1, 1.1)

        fig2_path = ROOT / "figures" / "banc_vs_manc_correlation.png"
        fig2.savefig(str(fig2_path), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {fig2_path}")

    except ImportError:
        print("  matplotlib not available, skipping plots")

    # ---- Save results JSON ----
    output = {
        "dataset": "BANC (female)",
        "config": {
            "a": cfg.a, "theta": cfg.theta, "exc_mult": cfg.exc_mult,
            "inh_scale": cfg.inh_scale, "normalize": cfg.normalize_weights,
            "adapt": cfg.use_adaptation, "delay_ms": cfg.delay_inh_ms,
            "seed": seed,
        },
        "network": {
            "n_neurons": banc_data.n_neurons, "n_dn": banc_data.n_dn,
            "n_mn": banc_data.n_mn, "n_premotor": banc_data.n_premotor,
            "n_synapses": banc_data.n_synapses,
        },
        "per_leg": results,
        "spectral": spectral_results,
        "summary": {
            "n_antiphase": n_antiphase,
            "n_decorrelated": n_decor,
            "n_active": n_active,
            "mean_corr": float(mean_corr),
        },
        "manc_comparison": {
            "manc_best_correlations": manc_corrs,
            "manc_best_n_antiphase": 2,
            "manc_best_mean_corr": -0.011,
        },
    }

    json_path = ROOT / "logs" / "banc_vnc_results.json"
    json_path.parent.mkdir(exist_ok=True)
    with open(str(json_path), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {json_path}")

    return output


# ============================================================================
# Multi-seed robustness test
# ============================================================================

def run_multi_seed(seeds: list[int] = [42, 123, 2024, 7, 99], sim_ms: float = 2000.0):
    """Run BANC model across multiple seeds and report aggregate results."""
    print("\n" + "=" * 72)
    print(f"Multi-Seed Robustness Test ({len(seeds)} seeds)")
    print("=" * 72)

    all_corrs = {leg: [] for leg in LEG_ORDER}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        banc_data = load_banc_vnc(
            normalize_weights=False,
            exc_mult=BANC_CFG.exc_mult,
            inh_mult=BANC_CFG.inh_mult,
            inh_scale=BANC_CFG.inh_scale,
            verbose=False,
        )

        cfg = FiringRateVNCConfig(
            tau_ms=BANC_CFG.tau_ms, a=BANC_CFG.a, theta=BANC_CFG.theta,
            fr_cap=BANC_CFG.fr_cap, exc_mult=BANC_CFG.exc_mult,
            inh_mult=BANC_CFG.inh_mult, inh_scale=BANC_CFG.inh_scale,
            use_adaptation=False, use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False, param_cv=0.10, seed=seed,
        )
        runner = FiringRateVNCRunner.from_banc(banc_data, cfg=cfg, warmup_ms=100.0)
        runner.stimulate_all_dns(rate_hz=25.0)
        runner.stimulate_dn_type("DNg100", rate_hz=60.0)

        dt = 0.5
        n_steps = int(sim_ms / dt)
        record_every = int(1.0 / dt)
        n_records = int(sim_ms / 1.0)

        flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
        ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}

        rec_idx = 0
        for step_i in range(n_steps):
            runner.step(dt_ms=dt)
            if (step_i + 1) % record_every == 0 and rec_idx < n_records:
                for leg_idx, leg_name in enumerate(LEG_ORDER):
                    flex_rate, ext_rate = runner.get_flexor_extensor_rates(leg_idx)
                    flex_traces[leg_name][rec_idx] = flex_rate
                    ext_traces[leg_name][rec_idx] = ext_rate
                rec_idx += 1

        skip_idx = int(500 / 1.0)
        corrs = {}
        for leg_name in LEG_ORDER:
            f = flex_traces[leg_name][skip_idx:]
            e = ext_traces[leg_name][skip_idx:]
            if f.max() > 1.0 and e.max() > 1.0 and f.std() > 1e-6 and e.std() > 1e-6:
                corr = float(np.corrcoef(f, e)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            corrs[leg_name] = corr
            all_corrs[leg_name].append(corr)

        n_ap = sum(1 for c in corrs.values() if c < -0.3)
        mean_r = np.mean(list(corrs.values()))
        corr_str = " ".join(f"{leg}={corrs[leg]:+.2f}" for leg in LEG_ORDER)
        print(f"  Seed {seed}: {n_ap}/6 AP, mean_r={mean_r:+.3f} | {corr_str}")

    # Aggregate
    print("\n" + "=" * 72)
    print("Aggregate Results (BANC, multi-seed)")
    print("=" * 72)

    for leg_name in LEG_ORDER:
        vals = all_corrs[leg_name]
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        n_ap = sum(1 for v in vals if v < -0.3)
        print(f"  {leg_name}: mean_r={mean_r:+.3f} +/- {std_r:.3f}, "
              f"AP in {n_ap}/{len(seeds)} seeds")

    overall_mean = np.mean([np.mean(v) for v in all_corrs.values()])
    print(f"\n  Overall mean correlation: {overall_mean:+.3f}")
    print(f"  MANC comparison (best single seed): mean_corr=-0.011, 2/6 AP")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BANC VNC firing-rate model test")
    parser.add_argument("--sim-ms", type=float, default=2000.0, help="Simulation duration (ms)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--multi-seed", action="store_true", help="Run multi-seed robustness test")
    args = parser.parse_args()

    # Single-seed primary experiment
    result = run_banc_simulation(sim_ms=args.sim_ms, seed=args.seed)

    # Print comparison verdict
    banc_ap = result["summary"]["n_antiphase"]
    manc_ap = result["manc_comparison"]["manc_best_n_antiphase"]
    banc_mean = result["summary"]["mean_corr"]
    manc_mean = result["manc_comparison"]["manc_best_mean_corr"]

    print("\n" + "=" * 72)
    print("VERDICT: BANC (female) vs MANC (male)")
    print("=" * 72)
    print(f"  BANC: {banc_ap}/6 anti-phase, mean_corr={banc_mean:+.3f}")
    print(f"  MANC: {manc_ap}/6 anti-phase, mean_corr={manc_mean:+.3f}")

    if banc_ap > manc_ap:
        print(f"  -> BANC (female) produces BETTER anti-phase ({banc_ap} vs {manc_ap} legs)")
    elif banc_ap == manc_ap:
        if banc_mean < manc_mean:
            print(f"  -> Same AP count, but BANC has lower mean correlation ({banc_mean:+.3f} vs {manc_mean:+.3f})")
        else:
            print(f"  -> Same AP count, MANC has lower mean correlation ({manc_mean:+.3f} vs {banc_mean:+.3f})")
    else:
        print(f"  -> MANC (male) produces better anti-phase ({manc_ap} vs {banc_ap} legs)")

    # Optional multi-seed
    if args.multi_seed:
        run_multi_seed(sim_ms=args.sim_ms)
