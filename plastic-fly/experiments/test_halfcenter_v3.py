#!/usr/bin/env python3
"""
Half-center v3: Excitatory rebalancing + targeted cross-inhibition.

Key insight from v1 and v2: boosting cross-inhibition alone KILLS extensors
because they're already weaker than flexors. The 2/6 baseline anti-phase
(RM, RH) works because those legs happen to have balanced E/I.

New approach:
1. First REBALANCE: boost excitatory input to the weaker pool (usually extensors)
   within each leg so flex and ext have comparable excitatory drive
2. THEN apply moderate cross-inhibition boost

Also tests: per-leg flex/ext excitatory balance analysis before any manipulation.
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


def analyze_ei_balance(runner: FiringRateVNCRunner):
    """Analyze excitatory/inhibitory balance per leg per pool (flex/ext)."""
    W = runner.W

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

    print("\n  E/I Balance Analysis (mean row sum per pool):")
    print(f"  {'Leg':<4s} {'Flex exc':>9s} {'Flex inh':>9s} {'Flex net':>9s} | "
          f"{'Ext exc':>9s} {'Ext inh':>9s} {'Ext net':>9s} | "
          f"{'#Flex':>5s} {'#Ext':>5s}")
    print("  " + "-" * 85)

    balance_info = {}
    for leg_idx, leg_name in enumerate(LEG_ORDER):
        flex_set = leg_flex_idx.get(leg_idx, set())
        ext_set = leg_ext_idx.get(leg_idx, set())

        for pool_name, pool_set in [("flex", flex_set), ("ext", ext_set)]:
            if not pool_set:
                continue
            pool_arr = np.array(list(pool_set), dtype=np.int32)
            # Sum excitatory input per MN row
            exc_sums = np.array([W[idx][W[idx] > 0].sum() for idx in pool_arr])
            inh_sums = np.array([W[idx][W[idx] < 0].sum() for idx in pool_arr])
            balance_info.setdefault(leg_idx, {})[pool_name] = {
                "exc_mean": float(exc_sums.mean()),
                "inh_mean": float(inh_sums.mean()),
                "n": len(pool_set),
            }

        if leg_idx in balance_info:
            b = balance_info[leg_idx]
            flex_b = b.get("flex", {"exc_mean": 0, "inh_mean": 0, "n": 0})
            ext_b = b.get("ext", {"exc_mean": 0, "inh_mean": 0, "n": 0})
            print(f"  {leg_name:<4s} "
                  f"{flex_b['exc_mean']:>9.3f} {flex_b['inh_mean']:>9.3f} "
                  f"{flex_b['exc_mean']+flex_b['inh_mean']:>9.3f} | "
                  f"{ext_b['exc_mean']:>9.3f} {ext_b['inh_mean']:>9.3f} "
                  f"{ext_b['exc_mean']+ext_b['inh_mean']:>9.3f} | "
                  f"{flex_b['n']:>5d} {ext_b['n']:>5d}")

    return balance_info


def apply_exc_rebalance(runner: FiringRateVNCRunner, target_ratio: float = 1.0):
    """Boost excitatory input to the weaker pool to match the stronger pool.

    For each leg, compute mean excitatory row sum for flex and ext MNs.
    If ext < flex, scale exc connections TO ext by (flex / ext).
    target_ratio: desired ext_exc / flex_exc ratio. 1.0 = equal, <1 = ext weaker.
    """
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

    print(f"\n  Excitatory rebalancing (target ratio={target_ratio:.1f}):")
    for leg_idx, leg_name in enumerate(LEG_ORDER):
        flex_set = leg_flex_idx.get(leg_idx, set())
        ext_set = leg_ext_idx.get(leg_idx, set())
        if not flex_set or not ext_set:
            continue

        flex_arr = np.array(list(flex_set), dtype=np.int32)
        ext_arr = np.array(list(ext_set), dtype=np.int32)

        # Mean exc input to each pool
        flex_exc = np.mean([W[idx][W[idx] > 0].sum() for idx in flex_arr])
        ext_exc = np.mean([W[idx][W[idx] > 0].sum() for idx in ext_arr])

        if ext_exc < 1e-8 or flex_exc < 1e-8:
            continue

        # Desired: ext_exc * scale = flex_exc * target_ratio
        scale = (flex_exc * target_ratio) / ext_exc
        if scale <= 1.0:
            # Ext already equal or stronger
            print(f"    {leg_name}: ext/flex exc ratio={ext_exc/flex_exc:.2f} -- already balanced")
            continue

        # Scale up excitatory inputs to ext MNs
        for idx in ext_arr:
            exc_mask = W[idx] > 0
            W[idx][exc_mask] *= scale

        print(f"    {leg_name}: ext/flex exc ratio={ext_exc/flex_exc:.2f} -> "
              f"scaling ext exc by {scale:.2f}x")

    # Rebuild sparse
    runner.W_exc = csr_matrix(np.where(W > 0, W, 0).astype(np.float32))
    runner.W_inh = csr_matrix(np.where(W < 0, W, 0).astype(np.float32))
    runner.W = W


def apply_cross_inh_boost(runner: FiringRateVNCRunner, boost: float,
                          bias_threshold: float = 0.6):
    """Boost cross-pool inhibition from pool-biased INs (same as v2)."""
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
    print(f"  Cross-inhibition boost: {n_boosted} weights at {boost}x")


def run_sim(name: str, runner: FiringRateVNCRunner, sim_ms: float = 2000.0,
            dn_baseline: float = 25.0, dng100_hz: float = 60.0):
    """Run simulation and return flex/ext analysis."""
    runner.stimulate_all_dns(rate_hz=dn_baseline)
    runner.stimulate_dn_type("DNg100", rate_hz=dng100_hz)

    dt = runner.cfg.dt_ms
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
    print(f"  Sim done in {sim_time:.1f}s ({n_steps/sim_time:.0f} steps/s)")

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
        # Also check: both pools oscillating (std > some threshold)?
        both_osc = f.std() > 0.5 and e.std() > 0.5
        results[leg_name] = {
            "flex_mean": float(f.mean()), "ext_mean": float(e.mean()),
            "flex_std": float(f.std()), "ext_std": float(e.std()),
            "flex_active": bool(f_active), "ext_active": bool(e_active),
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
    n_both_osc = sum(1 for r in results.values() if r["both_oscillating"])
    mean_corr = float(np.mean([r["correlation"] for r in results.values()]))

    print(f"\n  {name}: {n_anti}/6 AP, {n_both_osc}/6 both-osc, "
          f"mean_r={mean_corr:+.3f}, freq={peak_freq:.1f}Hz")
    for leg_name in LEG_ORDER:
        r = results[leg_name]
        status = "AP" if r["correlation"] < -0.1 else ("osc" if r["both_oscillating"] else ("act" if r["flex_active"] or r["ext_active"] else "---"))
        print(f"    {leg_name}: r={r['correlation']:+.3f} "
              f"f={r['flex_mean']:.1f}+/-{r['flex_std']:.1f} "
              f"e={r['ext_mean']:.1f}+/-{r['ext_std']:.1f} [{status}]")

    return {
        "name": name, "n_anti_phase": n_anti, "n_both_osc": n_both_osc,
        "mean_corr": mean_corr, "peak_freq": peak_freq, "per_leg": results,
        "flex_traces": flex_traces, "ext_traces": ext_traces, "time_axis": time_axis,
    }


def build_runner(inh_scale: float = 2.0, seg_inh=None) -> FiringRateVNCRunner:
    """Build a fresh runner with the best base config."""
    cfg = FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=inh_scale,
        segment_inh_scales=seg_inh,
        half_center_boost=1.0,
        use_adaptation=False,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False,
        param_cv=0.10, seed=42,
    )
    return FiringRateVNCRunner(cfg=cfg, warmup_ms=100.0)


def main():
    all_results = {}

    # =================================================================
    # Baseline: analyze E/I balance before any manipulation
    # =================================================================
    print("\n" + "="*72)
    print("BASELINE: E/I Balance Analysis")
    print("="*72)
    runner = build_runner()
    balance = analyze_ei_balance(runner)
    all_results["baseline"] = run_sim("baseline", runner)

    # =================================================================
    # Part 1: Excitatory rebalancing only (no cross-inh boost)
    # =================================================================
    print("\n" + "="*72)
    print("PART 1: Excitatory rebalancing only")
    print("="*72)

    for ratio in [1.0, 0.8, 0.6]:
        name = f"rebal_{ratio:.1f}"
        print(f"\n{'='*72}\nConfig: {name}\n{'='*72}")
        runner = build_runner()
        apply_exc_rebalance(runner, target_ratio=ratio)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 2: Rebalance + cross-inhibition boost
    # =================================================================
    print("\n" + "="*72)
    print("PART 2: Rebalance + cross-inhibition boost")
    print("="*72)

    for ratio, boost in [(1.0, 2.0), (1.0, 3.0), (1.0, 5.0),
                          (0.8, 2.0), (0.8, 3.0), (0.8, 5.0)]:
        name = f"rebal{ratio:.1f}_xinh{boost:.0f}x"
        print(f"\n{'='*72}\nConfig: {name}\n{'='*72}")
        runner = build_runner()
        apply_exc_rebalance(runner, target_ratio=ratio)
        apply_cross_inh_boost(runner, boost=boost, bias_threshold=0.6)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 3: Rebalance + lower base inh_scale + higher cross-inh
    # Idea: reduce GLOBAL inhibition to keep activity alive,
    # then add SELECTIVE cross-pool inhibition
    # =================================================================
    print("\n" + "="*72)
    print("PART 3: Low base inh + rebalance + targeted cross-inh")
    print("="*72)

    for base_inh, ratio, boost in [
        (1.0, 1.0, 5.0), (1.0, 1.0, 10.0),
        (1.5, 1.0, 5.0), (1.5, 1.0, 10.0),
        (1.0, 0.8, 5.0), (1.0, 0.8, 10.0),
    ]:
        name = f"inh{base_inh:.1f}_rebal{ratio:.1f}_xinh{boost:.0f}x"
        print(f"\n{'='*72}\nConfig: {name}\n{'='*72}")
        runner = build_runner(inh_scale=base_inh)
        apply_exc_rebalance(runner, target_ratio=ratio)
        apply_cross_inh_boost(runner, boost=boost, bias_threshold=0.6)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Part 4: Seg T3 high + rebalance + cross-inh
    # =================================================================
    print("\n" + "="*72)
    print("PART 4: Seg T3 high + rebalance + cross-inh")
    print("="*72)

    for boost in [2.0, 3.0, 5.0]:
        name = f"segT3_rebal1.0_xinh{boost:.0f}x"
        print(f"\n{'='*72}\nConfig: {name}\n{'='*72}")
        runner = build_runner(seg_inh={"T1": 2.0, "T2": 2.0, "T3": 4.0})
        apply_exc_rebalance(runner, target_ratio=1.0)
        apply_cross_inh_boost(runner, boost=boost, bias_threshold=0.6)
        all_results[name] = run_sim(name, runner)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"{'Config':<40s} {'AP':>3s} {'Osc':>3s} {'mean_r':>7s} {'freq':>6s} | "
          + " ".join(f"{l:>6s}" for l in LEG_ORDER))
    print("-" * 115)

    for name in all_results:
        r = all_results[name]
        corrs = " ".join(f"{r['per_leg'][l]['correlation']:+.2f}" for l in LEG_ORDER)
        print(f"{name:<40s} {r['n_anti_phase']:>3d} {r.get('n_both_osc', 0):>3d} "
              f"{r['mean_corr']:>+7.3f} {r['peak_freq']:>6.1f} | {corrs}")

    best_name = max(
        all_results.keys(),
        key=lambda n: (all_results[n]["n_anti_phase"], -all_results[n]["mean_corr"]),
    )
    best = all_results[best_name]
    print(f"\nBEST: {best_name} ({best['n_anti_phase']}/6 AP, "
          f"mean_corr={best['mean_corr']:+.3f})")

    # Save results
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {
            "n_anti_phase": r["n_anti_phase"],
            "n_both_osc": r.get("n_both_osc", 0),
            "mean_corr": r["mean_corr"],
            "peak_freq": r["peak_freq"],
            "per_leg": r["per_leg"],
        }
    with open(log_dir / "vnc_halfcenter_v3_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {log_dir / 'vnc_halfcenter_v3_results.json'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Best traces
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Half-Center v3: {best_name}\n"
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
        out1 = ROOT / "figures" / "vnc_halfcenter_v3_best.png"
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
        ax.set_title("Half-Center v3: Flex/Ext Correlation Heatmap")
        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()
        out2 = ROOT / "figures" / "vnc_halfcenter_v3_heatmap.png"
        plt.savefig(str(out2), dpi=150)
        print(f"Plot saved: {out2}")
        plt.close()

    except ImportError:
        print("matplotlib not available -- skipping plots")

    return all_results


if __name__ == "__main__":
    main()
