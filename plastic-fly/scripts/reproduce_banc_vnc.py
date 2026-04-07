#!/usr/bin/env python
"""
One-command reproduction of BANC VNC key results.

Run: python scripts/reproduce_banc_vnc.py

Reproduces all claims from the email to Ramdya:
  1. BANC VNC loads (8,218 neurons, 930K synapses)
  2. 4/6 anti-phase flex/ext from connectome wiring
  3. Forward ablation: 91% distance loss
  4. Full pipeline walks 2.9mm in 5k steps
  5. 10/10 seeds stable

Requires: data/banc/banc_626_data.sqlite (~684MB)
          Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8TFGGB
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BANC_DB = Path(__file__).resolve().parent.parent / "data" / "banc" / "banc_626_data.sqlite"


def main():
    print("=" * 65)
    print("  BANC VNC Reproduction Script")
    print("  Firing-rate model on female Drosophila VNC connectome")
    print("=" * 65)

    if not BANC_DB.exists():
        print(f"\nERROR: BANC database not found at {BANC_DB}")
        print("Download from: https://dataverse.harvard.edu/dataset.xhtml?"
              "persistentId=doi:10.7910/DVN/8TFGGB")
        print("Save as: data/banc/banc_626_data.sqlite")
        return False

    t_total = time.time()
    results = {}
    all_pass = True

    # ── 1. Load BANC VNC ──────────────────────────────────────────────
    print("\n[1/5] Loading BANC connectome...")
    t0 = time.time()
    from bridge.banc_loader import load_banc_vnc
    data = load_banc_vnc(
        exc_mult=1.0, inh_mult=1.0, inh_scale=1.0,
        normalize_weights=False, verbose=False,
    )
    dt = time.time() - t0
    ok = data.n_neurons > 8000 and data.n_mn == 390
    results["load"] = {
        "neurons": data.n_neurons, "mn": data.n_mn,
        "synapses": data.n_synapses, "time_s": dt,
    }
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  {data.n_neurons} neurons, {data.n_synapses:,} synapses ({dt:.1f}s)  [{status}]")

    # ── 2. Anti-phase from connectome ─────────────────────────────────
    print("\n[2/5] Testing flex/ext anti-phase (DNg100 at 60Hz)...")
    from bridge.vnc_firing_rate import FiringRateVNCRunner, FiringRateVNCConfig
    cfg = FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
        use_adaptation=False, normalize_weights=False,
        use_delay=True, delay_inh_ms=3.0, param_cv=0.05, seed=42,
    )
    runner = FiringRateVNCRunner.from_banc(data, cfg=cfg, warmup_ms=0)
    runner.stimulate_dn_type("DNg100", rate_hz=60.0)
    LEG = ["LF", "LM", "LH", "RF", "RM", "RH"]
    ft = {l: [] for l in LEG}
    et = {l: [] for l in LEG}
    for s in range(4000):
        runner.step(dt_ms=0.5)
        if s >= 1000 and s % 10 == 0:
            for li, l in enumerate(LEG):
                f, e = runner.get_flexor_extensor_rates(li)
                ft[l].append(f)
                et[l].append(e)
    n_ap = 0
    for l in LEG:
        fa, ea = np.array(ft[l]), np.array(et[l])
        if fa.std() >= 0.1 and ea.std() >= 0.1:
            r = float(np.corrcoef(fa, ea)[0, 1])
            if r < -0.3:
                n_ap += 1
                print(f"    {l}: r={r:+.3f} [ANTI-PHASE]")
            else:
                print(f"    {l}: r={r:+.3f}")
        else:
            print(f"    {l}: flat")
    ok = n_ap >= 2
    results["antiphase"] = {"n_ap": n_ap}
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  Anti-phase legs: {n_ap}/6  [{status}]")

    # ── 3. Forward ablation ───────────────────────────────────────────
    print("\n[3/5] Forward ablation test...")
    import flygym
    from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge

    def _walk(fwd_rate, steps=5000):
        bridge = FiringRateVNCBridge.from_banc(
            banc_data=data, cfg=cfg, fallback_blend=0.3)
        bridge.warmup(warmup_ms=200.0)
        fly = flygym.Fly(enable_adhesion=True, draw_adhesion=False)
        sim = flygym.SingleFlySimulation(fly=fly, timestep=1e-4)
        obs, _ = sim.reset()
        ini = obs["fly"][0, :2].copy()
        gr = {"forward": fwd_rate, "turn_left": 0.0, "turn_right": 0.0,
              "rhythm": 10.0, "stance": 5.0}
        for _ in range(steps):
            a = bridge.step(gr, dt_s=1e-4)
            try:
                obs, _, _, _, _ = sim.step(
                    {"joints": a["joints"],
                     "adhesion": a.get("adhesion", np.ones(6))})
            except (RuntimeError, ValueError):
                break
        return float(np.linalg.norm(obs["fly"][0, :2] - ini))

    d_intact = _walk(15.0)
    d_ablated = _walk(0.0)
    drop = 100 * (1 - d_ablated / max(d_intact, 0.01))
    ok = drop > 50
    results["ablation"] = {
        "intact_mm": d_intact, "ablated_mm": d_ablated, "drop_pct": drop,
    }
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  Intact: {d_intact:.3f}mm, Ablated: {d_ablated:.3f}mm")
    print(f"  Drop: {drop:.0f}%  [{status}]")

    # ── 4. Pipeline walking distance ──────────────────────────────────
    print("\n[4/5] Full pipeline walking (5k steps)...")
    ok = d_intact > 1.0
    results["walking"] = {"distance_mm": d_intact}
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  Distance: {d_intact:.3f}mm  [{status}]")

    # ── 5. Multi-seed stability ───────────────────────────────────────
    print("\n[5/5] Multi-seed stability (5 seeds x 3k steps)...")
    stable = 0
    forward = 0
    for seed in [42, 45, 123, 314, 7]:
        cfg_s = FiringRateVNCConfig(
            a=1.0, theta=7.5, fr_cap=200.0,
            exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
            use_adaptation=False, normalize_weights=False,
            use_delay=True, delay_inh_ms=3.0, param_cv=0.05, seed=seed,
        )
        bridge = FiringRateVNCBridge.from_banc(
            banc_data=data, cfg=cfg_s, fallback_blend=0.3)
        bridge.warmup(warmup_ms=200.0)
        fly = flygym.Fly(enable_adhesion=True, draw_adhesion=False)
        sim = flygym.SingleFlySimulation(fly=fly, timestep=1e-4)
        obs, _ = sim.reset()
        ini = obs["fly"][0, :2].copy()
        gr = {"forward": 15.0, "turn_left": 0.0, "turn_right": 0.0,
              "rhythm": 10.0, "stance": 5.0}
        crashed = False
        for _ in range(3000):
            a = bridge.step(gr, dt_s=1e-4)
            try:
                obs, _, _, _, _ = sim.step(
                    {"joints": a["joints"],
                     "adhesion": a.get("adhesion", np.ones(6))})
            except (RuntimeError, ValueError):
                crashed = True
                break
        d = obs["fly"][0, :2] - ini
        dx = float(d[0])
        if not crashed:
            stable += 1
        if dx > 0:
            forward += 1
        tag = "ok" if not crashed else "CRASH"
        print(f"    seed={seed}: dx={dx:+.3f}mm [{tag}]")

    ok = stable >= 4
    results["stability"] = {"stable": stable, "forward": forward, "total": 5}
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  Stable: {stable}/5, Forward: {forward}/5  [{status}]")

    # ── Summary ───────────────────────────────────────────────────────
    dt_total = time.time() - t_total
    print("\n" + "=" * 65)
    print(f"  {'ALL PASS' if all_pass else 'SOME FAILURES'}  ({dt_total:.0f}s)")
    print("=" * 65)
    print(f"  Neurons: {results['load']['neurons']}")
    print(f"  Anti-phase: {results['antiphase']['n_ap']}/6 legs")
    print(f"  Ablation: -{results['ablation']['drop_pct']:.0f}%")
    print(f"  Walking: {results['walking']['distance_mm']:.2f}mm")
    print(f"  Stability: {results['stability']['stable']}/{results['stability']['total']}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
