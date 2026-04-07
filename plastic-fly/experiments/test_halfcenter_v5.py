#!/usr/bin/env python3
"""
Half-center v5: Robustness check at best config + RM/RH leg targeting.

Best from v4: inh_scale=1.0, cross-inh boost=8x, threshold=0.6, seed=42
Achieves 4/6 AP (LF=-0.92, LM=-0.80, LH=-0.96, RF=-0.99, RM=+0.00, RH=+1.00)

This script:
1. Seed sweep at inh=1.0, boost=8x (seeds 42-51) to check robustness
2. Try boost=7 and boost=9 across seeds for fine-tuning
3. Analyze WHY RM and RH resist anti-phase
4. Try leg-specific boost for RM/RH
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


def apply_exc_rebalance(runner, target_ratio=1.0):
    from scipy.sparse import csr_matrix
    W = runner.W.copy()
    leg_flex_idx, leg_ext_idx = {}, {}
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


def apply_cross_inh_boost(runner, boost, bias_threshold=0.6, leg_boosts=None):
    """Apply cross-inhibition boost. If leg_boosts dict is provided,
    use per-leg boost values instead of global boost."""
    from scipy.sparse import csr_matrix
    W = runner.W.copy()
    leg_flex_idx, leg_ext_idx = {}, {}
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
            # Get boost for this leg
            leg_boost = boost
            if leg_boosts is not None and leg_idx in leg_boosts:
                leg_boost = leg_boosts[leg_idx]
            if leg_boost <= 1.0:
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
                    W[ext_arr[mask], idx] *= leg_boost
                    n_boosted += n
            elif frac_ext >= bias_threshold:
                mask = col[flex_arr] < 0
                n = int(mask.sum())
                if n > 0:
                    W[flex_arr[mask], idx] *= leg_boost
                    n_boosted += n
    runner.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
    runner.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
    runner.W = W
    return n_boosted


def run_sim(name, runner, sim_ms=2000.0, dn_baseline=25.0, dng100_hz=60.0):
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
        results[leg_name] = {
            "flex_mean": float(f.mean()), "ext_mean": float(e.mean()),
            "flex_std": float(f.std()), "ext_std": float(e.std()),
            "flex_active": bool(f_active), "ext_active": bool(e_active),
            "both_active": bool(both_active),
            "correlation": corr,
        }
    n_anti = sum(1 for r in results.values() if r["correlation"] < -0.1)
    mean_corr = float(np.mean([r["correlation"] for r in results.values()]))
    corr_str = " ".join(f"{results[l]['correlation']:+.2f}" for l in LEG_ORDER)
    print(f"  {name}: {n_anti}/6 AP, mean_r={mean_corr:+.3f} -- {corr_str}")
    return {
        "name": name, "n_anti_phase": n_anti, "mean_corr": mean_corr,
        "per_leg": results,
        "flex_traces": flex_traces, "ext_traces": ext_traces, "time_axis": time_axis,
    }


def main():
    all_results = {}

    # =================================================================
    # Part 1: Seed sweep at best config (inh=1.0, boost=8, thr=0.6)
    # =================================================================
    print("="*72)
    print("PART 1: Seed sweep at inh=1.0, boost=8x")
    print("="*72)

    for seed in range(42, 57):
        name = f"b8_s{seed}"
        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=1.0,
            half_center_boost=1.0,
            use_adaptation=False, use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False, param_cv=0.10, seed=seed,
        )
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
        apply_exc_rebalance(runner, target_ratio=1.0)
        apply_cross_inh_boost(runner, boost=8.0, bias_threshold=0.6)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 2: boost=7 and boost=9 seed sweep (smaller range)
    # =================================================================
    print("\n" + "="*72)
    print("PART 2: boost=7 and boost=9 seed sweep")
    print("="*72)

    for boost_val in [7.0, 9.0]:
        for seed in range(42, 52):
            name = f"b{boost_val:.0f}_s{seed}"
            cfg = FiringRateVNCConfig(
                a=1.0, theta=7.5, fr_cap=200.0,
                exc_mult=0.01, inh_mult=0.01, inh_scale=1.0,
                half_center_boost=1.0,
                use_adaptation=False, use_delay=True, delay_inh_ms=3.0,
                normalize_weights=False, param_cv=0.10, seed=seed,
            )
            runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
            apply_exc_rebalance(runner, target_ratio=1.0)
            apply_cross_inh_boost(runner, boost=boost_val, bias_threshold=0.6)
            all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 3: Per-leg boost -- extra boost on RM (idx=4) and RH (idx=5)
    # =================================================================
    print("\n" + "="*72)
    print("PART 3: Per-leg boost targeting RM/RH")
    print("="*72)

    for rm_boost, rh_boost in [(12, 8), (15, 8), (20, 8),
                                (8, 12), (8, 15), (8, 20),
                                (12, 12), (15, 15), (20, 20)]:
        name = f"b8_RM{rm_boost}_RH{rh_boost}"
        cfg = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=1.0,
            half_center_boost=1.0,
            use_adaptation=False, use_delay=True, delay_inh_ms=3.0,
            normalize_weights=False, param_cv=0.10, seed=42,
        )
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)
        apply_exc_rebalance(runner, target_ratio=1.0)
        leg_boosts = {0: 8.0, 1: 8.0, 2: 8.0, 3: 8.0,
                      4: float(rm_boost), 5: float(rh_boost)}
        apply_cross_inh_boost(runner, boost=8.0, bias_threshold=0.6,
                             leg_boosts=leg_boosts)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*72)
    print("SUMMARY (sorted by AP count)")
    print("="*72)
    print(f"{'Config':<25s} {'AP':>3s} {'mean_r':>7s} | "
          + " ".join(f"{l:>6s}" for l in LEG_ORDER))
    print("-" * 90)

    sorted_names = sorted(
        all_results.keys(),
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
        reverse=True,
    )
    for name in sorted_names:
        r = all_results[name]
        corrs = " ".join(f"{r['per_leg'][l]['correlation']:+.2f}" for l in LEG_ORDER)
        print(f"{name:<25s} {r['n_anti_phase']:>3d} {r['mean_corr']:>+7.3f} | {corrs}")

    # Stats
    b8_results = [all_results[n] for n in all_results if n.startswith("b8_s")]
    if b8_results:
        mean_ap = np.mean([r["n_anti_phase"] for r in b8_results])
        max_ap = max(r["n_anti_phase"] for r in b8_results)
        print(f"\nBoost=8 seed stats: mean={mean_ap:.1f}/6, max={max_ap}/6, "
              f"n_seeds={len(b8_results)}")

    best_name = sorted_names[0]
    best = all_results[best_name]
    print(f"\nBEST: {best_name} ({best['n_anti_phase']}/6 AP, "
          f"mean_corr={best['mean_corr']:+.3f})")

    # Save
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {
            "n_anti_phase": r["n_anti_phase"],
            "mean_corr": r["mean_corr"],
            "per_leg": r["per_leg"],
        }
    with open(log_dir / "vnc_halfcenter_v5_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {log_dir / 'vnc_halfcenter_v5_results.json'}")

    # Plot best traces
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Half-Center v5 Best: {best_name}\n"
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
        out = ROOT / "figures" / "vnc_halfcenter_v5_best.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=150)
        print(f"Plot saved: {out}")
        plt.close()

        # Seed sweep histogram
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for bi, (bval, prefix) in enumerate([(7.0, "b7_s"), (8.0, "b8_s"), (9.0, "b9_s")]):
            ax = axes[bi]
            seed_results = [all_results[n] for n in all_results if n.startswith(prefix)]
            if not seed_results:
                continue
            ap_counts = [r["n_anti_phase"] for r in seed_results]
            ax.hist(ap_counts, bins=range(0, 8), align="left", alpha=0.7,
                    edgecolor="black")
            ax.set_xlabel("Anti-phase legs (out of 6)")
            ax.set_ylabel("Count (seeds)")
            ax.set_title(f"boost={bval:.0f}x (n={len(seed_results)})\n"
                        f"mean={np.mean(ap_counts):.1f}, max={max(ap_counts)}")
            ax.set_xlim(-0.5, 6.5)
        plt.suptitle("Anti-Phase Count Distribution Across Seeds", fontsize=12)
        plt.tight_layout()
        out2 = ROOT / "figures" / "vnc_halfcenter_v5_seed_hist.png"
        plt.savefig(str(out2), dpi=150)
        print(f"Plot saved: {out2}")
        plt.close()

    except ImportError:
        print("matplotlib not available -- skipping plots")

    return all_results


if __name__ == "__main__":
    main()
