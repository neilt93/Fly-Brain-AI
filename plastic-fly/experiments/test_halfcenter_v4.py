#!/usr/bin/env python3
"""
Half-center v4: Fine-tuning around the 3/6 AP sweet spot.

The v3 finding: inh_scale=1.0, rebal=1.0, cross-inh 5x achieves 3/6 anti-phase.
This script does a dense sweep around that configuration to find the optimal point.

Key parameters to sweep:
- inh_scale: [0.5, 0.75, 1.0, 1.25]
- cross-inh boost: [3, 4, 5, 6, 7, 8]
- bias_threshold: [0.5, 0.6, 0.7]
- seeds: try multiple seeds to check robustness
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.vnc_firing_rate import (
    FiringRateVNCConfig, FiringRateVNCRunner, LEG_ORDER, NT_SIGN,
)

ROOT = Path(__file__).resolve().parent.parent


def apply_exc_rebalance(runner, target_ratio=1.0):
    from scipy.sparse import csr_matrix
    W = runner.W.copy()

    leg_flex_idx = {}
    leg_ext_idx = {}
    for i, bid in enumerate(runner.mn_body_ids):
        model_idx = runner._bodyid_to_idx[int(bid)]
        leg_idx = runner._mn_leg[i]
        if runner._mn_is_flexor[i]:
            leg_flex_idx.setdefault(leg_idx, set()).add(model_idx)
        elif runner._mn_is_extensor[i]:
            leg_ext_idx.setdefault(leg_idx, set()).add(model_idx)

    for leg_idx in range(6):
        flex_set = leg_flex_idx.get(leg_idx, set())
        ext_set = leg_ext_idx.get(leg_idx, set())
        if not flex_set or not ext_set:
            continue
        flex_arr = np.array(list(flex_set), dtype=np.int32)
        ext_arr = np.array(list(ext_set), dtype=np.int32)
        flex_exc = np.mean([W[idx][W[idx] > 0].sum() for idx in flex_arr])
        ext_exc = np.mean([W[idx][W[idx] > 0].sum() for idx in ext_arr])
        if ext_exc < 1e-8 or flex_exc < 1e-8:
            continue
        scale = (flex_exc * target_ratio) / ext_exc
        if scale <= 1.0:
            continue
        for idx in ext_arr:
            exc_mask = W[idx] > 0
            W[idx][exc_mask] *= scale

    runner.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
    runner.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
    runner.W = W


def apply_cross_inh_boost(runner, boost, bias_threshold=0.6):
    from scipy.sparse import csr_matrix
    W = runner.W.copy()

    leg_flex_idx = {}
    leg_ext_idx = {}
    for i, bid in enumerate(runner.mn_body_ids):
        model_idx = runner._bodyid_to_idx[int(bid)]
        leg_idx = runner._mn_leg[i]
        if runner._mn_is_flexor[i]:
            leg_flex_idx.setdefault(leg_idx, set()).add(model_idx)
        elif runner._mn_is_extensor[i]:
            leg_ext_idx.setdefault(leg_idx, set()).add(model_idx)

    n_boosted = 0
    for bid in runner._premotor_ids:
        idx = runner._bodyid_to_idx.get(int(bid))
        if idx is None:
            continue
        nt = runner._nt_map.get(int(bid), "unclear")
        sign = NT_SIGN.get(str(nt).lower().strip(), +1.0)
        if sign >= 0:
            continue

        col = W[:, idx]
        for leg_idx in range(6):
            flex_set = leg_flex_idx.get(leg_idx, set())
            ext_set = leg_ext_idx.get(leg_idx, set())
            if not flex_set or not ext_set:
                continue
            flex_arr = np.array(list(flex_set), dtype=np.int32)
            ext_arr = np.array(list(ext_set), dtype=np.int32)
            inh_to_flex = np.maximum(-col[flex_arr], 0).sum()
            inh_to_ext = np.maximum(-col[ext_arr], 0).sum()
            total = inh_to_flex + inh_to_ext
            if total < 1e-8:
                continue
            frac_flex = inh_to_flex / total
            frac_ext = inh_to_ext / total
            if frac_flex >= bias_threshold:
                mask = col[ext_arr] < 0
                n = int(mask.sum())
                if n > 0:
                    W[ext_arr[mask], idx] *= boost
                    n_boosted += n
            elif frac_ext >= bias_threshold:
                mask = col[flex_arr] < 0
                n = int(mask.sum())
                if n > 0:
                    W[flex_arr[mask], idx] *= boost
                    n_boosted += n

    runner.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
    runner.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
    runner.W = W
    return n_boosted


def run_sim(name, runner, sim_ms=2000.0, dn_baseline=25.0, dng100_hz=60.0, quiet=True):
    runner.stimulate_all_dns(rate_hz=dn_baseline)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    dt = runner.cfg.dt_ms
    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)
    n_records = int(sim_ms / 1.0)
    time_axis = np.linspace(0, sim_ms, n_records)

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
    skip_idx = 500
    results = {}
    for leg_name in LEG_ORDER:
        f = flex_traces[leg_name][skip_idx:]
        e = ext_traces[leg_name][skip_idx:]
        f_active = f.max() > 1.0
        e_active = e.max() > 1.0
        both_active = f_active and e_active
        if both_active and f.std() > 1e-6 and e.std() > 1e-6:
            corr = float(np.corrcoef(f, e)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        both_osc = f.std() > 0.5 and e.std() > 0.5
        results[leg_name] = {
            "flex_mean": float(f.mean()), "ext_mean": float(e.mean()),
            "flex_std": float(f.std()), "ext_std": float(e.std()),
            "flex_active": bool(f_active), "ext_active": bool(e_active),
            "both_active": bool(both_active),
            "both_oscillating": bool(both_osc),
            "correlation": corr,
        }

    peak_freq = 0.0
    for check_leg in LEG_ORDER:
        f = flex_traces[check_leg][skip_idx:]
        if f.max() > 1.0 and f.std() > 0.5:
            f_c = f - f.mean()
            freqs = np.fft.rfftfreq(len(f_c), d=1.0/1000.0)
            psd = np.abs(np.fft.rfft(f_c))**2
            band = (freqs >= 3.0) & (freqs <= 30.0)
            if band.any():
                peak_freq = float(freqs[band][np.argmax(psd[band])])
                break

    n_anti = sum(1 for r in results.values() if r["correlation"] < -0.1)
    n_both_active = sum(1 for r in results.values() if r["both_active"])
    mean_corr = float(np.mean([r["correlation"] for r in results.values()]))

    if not quiet:
        corr_str = " ".join(f"{results[l]['correlation']:+.2f}" for l in LEG_ORDER)
        print(f"  {name}: {n_anti}/6 AP, {n_both_active}/6 both-active, "
              f"mean_r={mean_corr:+.3f}, freq={peak_freq:.1f}Hz -- {corr_str}")

    return {
        "name": name, "n_anti_phase": n_anti, "n_both_active": n_both_active,
        "mean_corr": mean_corr, "peak_freq": peak_freq, "per_leg": results,
        "flex_traces": flex_traces, "ext_traces": ext_traces, "time_axis": time_axis,
    }


def main():
    all_results = {}

    # =================================================================
    # Dense sweep around the sweet spot
    # =================================================================
    print("="*72)
    print("Dense sweep: inh_scale x cross_inh_boost x bias_threshold")
    print("="*72)

    configs = []
    # Primary sweep: inh_scale vs cross-inh boost
    for inh_scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for boost in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            name = f"inh{inh_scale:.2f}_xb{boost:.0f}"
            configs.append((name, inh_scale, boost, 0.6, 42))

    # Bias threshold sweep at best from v3 (inh=1.0)
    for threshold in [0.5, 0.55, 0.65, 0.7]:
        for boost in [4.0, 5.0, 6.0]:
            name = f"inh1.00_xb{boost:.0f}_thr{threshold:.2f}"
            configs.append((name, 1.0, boost, threshold, 42))

    # Seed sweep for the best config from v3
    for seed in [43, 44, 45, 46, 47]:
        name = f"inh1.00_xb5_s{seed}"
        configs.append((name, 1.0, 5.0, 0.6, seed))

    print(f"Total configs: {len(configs)}")

    for ci, (name, inh_scale, boost, threshold, seed) in enumerate(configs):
        print(f"\n[{ci+1}/{len(configs)}] {name} "
              f"(inh={inh_scale}, boost={boost}, thr={threshold}, seed={seed})")

        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=inh_scale,
            half_center_boost=1.0,
            use_adaptation=False,
            use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False,
            param_cv=0.10, seed=seed,
        )
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
        apply_exc_rebalance(runner, target_ratio=1.0)
        n_boosted = apply_cross_inh_boost(runner, boost=boost, bias_threshold=threshold)
        result = run_sim(name, runner, quiet=False)
        result["config"] = {
            "inh_scale": inh_scale, "boost": boost,
            "threshold": threshold, "seed": seed,
            "n_boosted": n_boosted,
        }
        all_results[name] = result

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*72)
    print("SUMMARY (sorted by anti-phase count, then mean_corr)")
    print("="*72)
    print(f"{'Config':<30s} {'AP':>3s} {'BA':>3s} {'mean_r':>7s} {'freq':>6s} | "
          + " ".join(f"{l:>6s}" for l in LEG_ORDER))
    print("-" * 105)

    sorted_names = sorted(
        all_results.keys(),
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
        reverse=True,
    )

    for name in sorted_names:
        r = all_results[name]
        corrs = " ".join(f"{r['per_leg'][l]['correlation']:+.2f}" for l in LEG_ORDER)
        print(f"{name:<30s} {r['n_anti_phase']:>3d} {r.get('n_both_active', 0):>3d} "
              f"{r['mean_corr']:>+7.3f} {r['peak_freq']:>6.1f} | {corrs}")

    best_name = sorted_names[0]
    best = all_results[best_name]
    print(f"\nBEST: {best_name} ({best['n_anti_phase']}/6 AP, "
          f"mean_corr={best['mean_corr']:+.3f}, "
          f"config={best.get('config', {})})")

    # Save
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {
            "n_anti_phase": r["n_anti_phase"],
            "n_both_active": r.get("n_both_active", 0),
            "mean_corr": r["mean_corr"],
            "peak_freq": r["peak_freq"],
            "per_leg": r["per_leg"],
            "config": r.get("config", {}),
        }
    with open(log_dir / "vnc_halfcenter_v4_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {log_dir / 'vnc_halfcenter_v4_results.json'}")

    # Plot: heatmap of top configs + best traces
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Show top 20 configs in heatmap
        top_n = min(25, len(sorted_names))
        top_names = sorted_names[:top_n]

        corr_matrix = np.zeros((top_n, 6))
        for ci, name in enumerate(top_names):
            for li, leg in enumerate(LEG_ORDER):
                corr_matrix[ci, li] = all_results[name]["per_leg"][leg]["correlation"]

        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(6))
        ax.set_xticklabels(LEG_ORDER)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([
            f"{n} (AP={all_results[n]['n_anti_phase']})"
            for n in top_names
        ], fontsize=7)
        for ci in range(top_n):
            for li in range(6):
                v = corr_matrix[ci, li]
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(li, ci, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6, color=color)
        ax.set_title("Half-Center v4: Top Configs by Anti-Phase Count")
        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()
        out1 = ROOT / "figures" / "vnc_halfcenter_v4_heatmap.png"
        out1.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out1), dpi=150)
        print(f"Plot saved: {out1}")
        plt.close()

        # Best traces
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Half-Center v4: {best_name}\n"
                     f"({best['n_anti_phase']}/6 AP, "
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
        out2 = ROOT / "figures" / "vnc_halfcenter_v4_best.png"
        plt.savefig(str(out2), dpi=150)
        print(f"Plot saved: {out2}")
        plt.close()

        # Phase portrait: inh_scale vs boost heatmap showing AP count
        # Only for primary sweep (seed=42, threshold=0.6)
        primary = {n: all_results[n] for n in all_results
                   if all_results[n].get("config", {}).get("seed") == 42
                   and all_results[n].get("config", {}).get("threshold") == 0.6
                   and n.count("_") == 1}  # format: inhX.XX_xbY
        if primary:
            inh_vals = sorted(set(primary[n]["config"]["inh_scale"] for n in primary))
            boost_vals = sorted(set(primary[n]["config"]["boost"] for n in primary))
            grid = np.zeros((len(inh_vals), len(boost_vals)))
            for n in primary:
                c = primary[n]["config"]
                ri = inh_vals.index(c["inh_scale"])
                ci = boost_vals.index(c["boost"])
                grid[ri, ci] = primary[n]["n_anti_phase"]

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=6, aspect="auto",
                          origin="lower")
            ax.set_xticks(range(len(boost_vals)))
            ax.set_xticklabels([f"{v:.0f}" for v in boost_vals])
            ax.set_yticks(range(len(inh_vals)))
            ax.set_yticklabels([f"{v:.2f}" for v in inh_vals])
            ax.set_xlabel("Cross-inhibition boost")
            ax.set_ylabel("Base inh_scale")
            for ri in range(len(inh_vals)):
                for ci_idx in range(len(boost_vals)):
                    ax.text(ci_idx, ri, f"{int(grid[ri, ci_idx])}", ha="center",
                            va="center", fontsize=10, fontweight="bold")
            ax.set_title("Anti-Phase Legs (out of 6)")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            out3 = ROOT / "figures" / "vnc_halfcenter_v4_phase_portrait.png"
            plt.savefig(str(out3), dpi=150)
            print(f"Plot saved: {out3}")
            plt.close()

    except ImportError:
        print("matplotlib not available -- skipping plots")

    return all_results


if __name__ == "__main__":
    main()
