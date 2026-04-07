#!/usr/bin/env python3
"""
Robust anti-phase v2: targeted segment-specific tuning.

Findings from v1:
  - cv=0 (deterministic) only gives 1/6 AP (LF only): T1 works, T2/T3 don't
  - cv=0.10 noise helps randomly but is not robust
  - T2 and T3 need different operating points from T1

Strategy:
  1. Sweep exc_mult for T2 and T3 independently at cv=0 (wide range)
  2. Try segment-specific inh_scale as well
  3. Try different global params (theta, a) that might shift T2/T3 into oscillation
  4. Best combined config: seed sweep to verify robustness
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from time import time
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.vnc_firing_rate import FiringRateVNCRunner, FiringRateVNCConfig, LEG_ORDER

SEGMENT_LEGS = {
    "T1": ["LF", "RF"],
    "T2": ["LM", "RM"],
    "T3": ["LH", "RH"],
}


def run_config(name: str, cfg: FiringRateVNCConfig, sim_ms: float = 2000.0,
               warmup_ms: float = 100.0, verbose: bool = True) -> dict:
    """Build runner, run sim, return per-leg correlations."""
    try:
        runner = FiringRateVNCRunner(cfg=cfg, warmup_ms=warmup_ms)
    except Exception as e:
        print(f"  [{name}] BUILD FAILED: {e}")
        return {"name": name, "error": str(e)}

    runner.stimulate_all_dns(rate_hz=25.0)
    runner.stimulate_dn_type("DNg100", rate_hz=60.0)

    dt = 0.5
    n_steps = int(sim_ms / dt)
    record_every = int(1.0 / dt)
    n_records = int(sim_ms / 1.0)

    flex_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}
    ext_traces = {leg: np.zeros(n_records) for leg in LEG_ORDER}

    t0 = time()
    rec_idx = 0
    for step_i in range(n_steps):
        runner.step(dt_ms=dt)
        if (step_i + 1) % record_every == 0 and rec_idx < n_records:
            for leg_idx, leg_name in enumerate(LEG_ORDER):
                f_r, e_r = runner.get_flexor_extensor_rates(leg_idx)
                flex_traces[leg_name][rec_idx] = f_r
                ext_traces[leg_name][rec_idx] = e_r
            rec_idx += 1
    sim_time = time() - t0

    skip_idx = 500  # skip first 500ms
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

    # Peak frequency
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

    # Per-segment AP count
    seg_ap = {}
    for seg, legs in SEGMENT_LEGS.items():
        seg_ap[seg] = sum(1 for l in legs if correlations[l] < -0.3)

    result = {
        "name": name,
        "correlations": correlations,
        "n_antiphase": n_ap,
        "mean_corr": mean_corr,
        "peak_freq": peak_freq,
        "sim_time": sim_time,
        "seg_ap": seg_ap,
    }

    if verbose:
        corr_str = " ".join(f"{leg}={correlations[leg]:+.3f}" for leg in LEG_ORDER)
        seg_str = " ".join(f"{s}={seg_ap[s]}/2" for s in ["T1", "T2", "T3"])
        print(f"  [{name}] {n_ap}/6 AP ({seg_str}), mean_r={mean_corr:+.3f}, "
              f"freq={peak_freq:.1f}Hz, {sim_time:.1f}s")
        print(f"    {corr_str}")

    return result


def make_cfg(seed=42, param_cv=0.0, exc_mult=0.01, inh_scale=2.0,
             T1_exc=None, T2_exc=None, T3_exc=None,
             T1_inh=None, T2_inh=None, T3_inh=None,
             a=1.0, theta=7.5, **overrides) -> FiringRateVNCConfig:
    """Build config with optional per-segment overrides."""
    seg_exc = None
    if T1_exc is not None or T2_exc is not None or T3_exc is not None:
        seg_exc = {
            "T1": T1_exc if T1_exc is not None else exc_mult,
            "T2": T2_exc if T2_exc is not None else exc_mult,
            "T3": T3_exc if T3_exc is not None else exc_mult,
        }

    seg_inh = None
    if T1_inh is not None or T2_inh is not None or T3_inh is not None:
        seg_inh = {
            "T1": T1_inh if T1_inh is not None else inh_scale,
            "T2": T2_inh if T2_inh is not None else inh_scale,
            "T3": T3_inh if T3_inh is not None else inh_scale,
        }

    kw = dict(
        a=a, theta=theta, fr_cap=200.0,
        exc_mult=exc_mult, inh_mult=exc_mult, inh_scale=inh_scale,
        use_adaptation=False,
        use_delay=True, delay_inh_ms=3.0,
        normalize_weights=False,
        param_cv=param_cv, seed=seed,
        segments=("T1", "T2", "T3"),
        segment_exc_mults=seg_exc,
        segment_inh_scales=seg_inh,
    )
    kw.update(overrides)
    return FiringRateVNCConfig(**kw)


def print_summary(results_list, title="SUMMARY"):
    print(f"\n{'='*120}")
    print(title)
    print(f"{'='*120}")
    print(f"{'Config':<36} {'AP':>3} {'Mean r':>8} {'T1':>4} {'T2':>4} {'T3':>4} "
          f"{'LF':>7} {'LM':>7} {'LH':>7} {'RF':>7} {'RM':>7} {'RH':>7}")
    print("-" * 120)

    for r in results_list:
        if "error" in r:
            print(f"{r['name']:<36} ERROR: {r['error'][:40]}")
            continue
        corrs = r["correlations"]
        sa = r.get("seg_ap", {})
        c_str = " ".join(f"{corrs[l]:>+7.3f}" for l in LEG_ORDER)
        print(f"{r['name']:<36} {r['n_antiphase']:>3} {r['mean_corr']:>+8.3f} "
              f"{sa.get('T1',0):>4} {sa.get('T2',0):>4} {sa.get('T3',0):>4} "
              f"{c_str}")


def main():
    print("=" * 72)
    print("ROBUST ANTI-PHASE v2: Segment-Specific Tuning")
    print("=" * 72)
    print("cv=0 result from v1: only T1 (LF) achieves anti-phase.")
    print("Goal: find per-segment params where each segment oscillates.\n")

    all_results = []
    t_total = time()

    # ================================================================
    # Phase 1: Wide sweep of T2 exc_mult at cv=0
    # ================================================================
    print("\n" + "=" * 72)
    print("PHASE 1: T2 exc_mult sweep (T1=0.015, T3=0.01, cv=0)")
    print("=" * 72)

    t2_exc_values = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03]
    for t2_exc in t2_exc_values:
        name = f"T2exc_{t2_exc:.3f}"
        cfg = make_cfg(T1_exc=0.015, T2_exc=t2_exc, T3_exc=0.01)
        r = run_config(name, cfg)
        all_results.append(r)

    print_summary(all_results, "PHASE 1: T2 exc sweep")

    # Find best T2 exc
    best_t2_exc = 0.01
    best_t2_score = -999
    for r, val in zip(all_results, t2_exc_values):
        if "error" in r:
            continue
        score = r.get("seg_ap", {}).get("T2", 0) * 10 - r["mean_corr"] * 3
        if score > best_t2_score:
            best_t2_score = score
            best_t2_exc = val
    print(f"\n  Best T2 exc: {best_t2_exc}")

    # ================================================================
    # Phase 2: Wide sweep of T3 exc_mult at cv=0
    # ================================================================
    print("\n" + "=" * 72)
    print("PHASE 2: T3 exc_mult sweep (T1=0.015, T2=best, cv=0)")
    print("=" * 72)

    phase2_results = []
    t3_exc_values = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03]
    for t3_exc in t3_exc_values:
        name = f"T3exc_{t3_exc:.3f}"
        cfg = make_cfg(T1_exc=0.015, T2_exc=best_t2_exc, T3_exc=t3_exc)
        r = run_config(name, cfg)
        phase2_results.append(r)
        all_results.append(r)

    print_summary(phase2_results, "PHASE 2: T3 exc sweep")

    best_t3_exc = 0.01
    best_t3_score = -999
    for r, val in zip(phase2_results, t3_exc_values):
        if "error" in r:
            continue
        score = r.get("seg_ap", {}).get("T3", 0) * 10 - r["mean_corr"] * 3
        if score > best_t3_score:
            best_t3_score = score
            best_t3_exc = val
    print(f"\n  Best T3 exc: {best_t3_exc}")

    # ================================================================
    # Phase 3: Segment-specific inh_scale sweep at cv=0
    # ================================================================
    print("\n" + "=" * 72)
    print(f"PHASE 3: inh_scale sweep (T1_exc=0.015, T2_exc={best_t2_exc}, "
          f"T3_exc={best_t3_exc}, cv=0)")
    print("=" * 72)

    phase3_results = []
    inh_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    for t2_inh in inh_values:
        for t3_inh in [2.0, 3.0, 4.0]:
            if t2_inh == 2.0 and t3_inh == 2.0:
                continue  # already in baseline
            name = f"inh_T2={t2_inh}_T3={t3_inh}"
            cfg = make_cfg(
                T1_exc=0.015, T2_exc=best_t2_exc, T3_exc=best_t3_exc,
                T1_inh=2.0, T2_inh=t2_inh, T3_inh=t3_inh,
            )
            r = run_config(name, cfg)
            phase3_results.append(r)
            all_results.append(r)

    print_summary(phase3_results, "PHASE 3: inh_scale sweep")

    # ================================================================
    # Phase 4: Global param variations at cv=0
    # ================================================================
    print("\n" + "=" * 72)
    print("PHASE 4: Global param variations (theta, a) at cv=0")
    print("=" * 72)

    phase4_results = []
    for a_val in [0.5, 1.0, 2.0]:
        for theta_val in [5.0, 7.5, 10.0]:
            if a_val == 1.0 and theta_val == 7.5:
                continue
            name = f"a={a_val}_th={theta_val}"
            cfg = make_cfg(
                T1_exc=0.015, T2_exc=best_t2_exc, T3_exc=best_t3_exc,
                a=a_val, theta=theta_val,
            )
            r = run_config(name, cfg)
            phase4_results.append(r)
            all_results.append(r)

    print_summary(phase4_results, "PHASE 4: Global param variations")

    # ================================================================
    # Phase 5: Best combined + seed sweep at cv=0.02
    # ================================================================
    print("\n" + "=" * 72)
    print("PHASE 5: Best combined + seed sweep")
    print("=" * 72)

    # Find overall best from all phases
    valid = [r for r in all_results if "error" not in r]
    best_overall = max(valid, key=lambda r: r["n_antiphase"] * 10 - r["mean_corr"] * 5)
    print(f"  Best overall: {best_overall['name']} => {best_overall['n_antiphase']}/6 AP")

    # Seed sweep at best + small cv
    phase5_results = []
    for seed in range(42, 52):
        for cv in [0.0, 0.02, 0.05]:
            name = f"best_s{seed}_cv{cv:.2f}"
            cfg = make_cfg(
                seed=seed, param_cv=cv,
                T1_exc=0.015, T2_exc=best_t2_exc, T3_exc=best_t3_exc,
            )
            r = run_config(name, cfg)
            phase5_results.append(r)
            all_results.append(r)

    print_summary(phase5_results, "PHASE 5: Best combined + seed sweep")

    # Per-leg support at cv=0.02
    cv002_results = [r for r in phase5_results
                     if "error" not in r and "cv0.02" in r["name"]]
    if cv002_results:
        print("\n  Per-leg support at cv=0.02:")
        for leg in LEG_ORDER:
            n_ap = sum(1 for r in cv002_results if r["correlations"][leg] < -0.3)
            mean_c = np.mean([r["correlations"][leg] for r in cv002_results])
            print(f"    {leg}: {n_ap}/{len(cv002_results)} seeds AP, "
                  f"mean corr={mean_c:+.3f}")

    # ================================================================
    # Overall summary
    # ================================================================
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    valid = [r for r in all_results if "error" not in r]
    best = max(valid, key=lambda r: r["n_antiphase"] * 10 - r["mean_corr"] * 5)
    print(f"\n  Best config: {best['name']}")
    print(f"  Anti-phase: {best['n_antiphase']}/6 legs")
    print(f"  Mean corr: {best['mean_corr']:+.3f}")
    for leg in LEG_ORDER:
        status = "AP" if best["correlations"][leg] < -0.3 else "  "
        print(f"    {leg}: {best['correlations'][leg]:+.3f} {status}")

    # Per-seed max
    seed_max = {}
    for r in valid:
        name = r["name"]
        n = r["n_antiphase"]
        if n > seed_max.get("best_n", 0):
            seed_max["best_n"] = n
            seed_max["best_name"] = name

    total_time = time() - t_total
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results
    json_path = ROOT / "figures" / "robust_antiphase_v2_results.json"
    json_results = [{k: v for k, v in r.items()
                     if k not in ("flex_traces", "ext_traces", "time_axis")}
                    for r in all_results]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(json_path), "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
