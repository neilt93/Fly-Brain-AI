#!/usr/bin/env python
"""
CPG speed control: sweep forward drive and measure gait frequency + speed.

Pugliese et al. predict higher drive -> higher frequency. We test whether
our tuned CPG shows this, or if frequency stays flat while amplitude increases.

Usage:
    python experiments/cpg_speed_control.py
    python experiments/cpg_speed_control.py --body-steps 5000
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def measure_contact_frequency(contact_trace, dt_s):
    """Measure stepping frequency from contact force time series.

    Uses zero-crossing detection on the mean-subtracted signal.
    """
    if len(contact_trace) < 20:
        return 0.0
    signal = np.array(contact_trace, dtype=np.float64)
    signal = signal - np.mean(signal)
    crossings = 0
    for i in range(1, len(signal)):
        if signal[i-1] * signal[i] < 0:
            crossings += 1
    freq = crossings / (2.0 * len(signal) * dt_s)
    return freq


def measure_cpg_frequency(cpg, drive, warmup=5000, measure=10000):
    """Measure CPG E1 oscillation frequency."""
    dt = 1e-4
    for _ in range(warmup):
        cpg.step(dt, forward_drive=drive)
    e1 = []
    for _ in range(measure):
        cpg.step(dt, forward_drive=drive)
        e1.append(cpg.R[0, 0])
    e1 = np.array(e1)
    mean = np.mean(e1)
    crossings = sum(1 for i in range(1, len(e1))
                    if (e1[i-1] - mean) * (e1[i] - mean) < 0)
    freq = crossings / (2.0 * measure * dt)
    amplitude = float(e1.max() - e1.min())
    return freq, amplitude


def run_drive_sweep(drive_hz, body_steps, warmup_steps, use_cpg):
    """Run one drive level and measure metrics."""
    import flygym
    from bridge.vnc_bridge import VNCBridge

    bridge = VNCBridge(use_fake_vnc=True, use_cpg=use_cpg)
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly, arena=flygym.arena.FlatTerrain(), timestep=1e-4
    )
    obs, _ = sim.reset()
    dt_s = 1e-4

    init_joints = np.array(obs["joints"][0], dtype=np.float64)
    bridge.reset(init_angles=init_joints)

    # Warmup
    for i in range(warmup_steps):
        ramp = min(1.0, i / max(warmup_steps * 0.5, 1.0))
        rates = {"forward": drive_hz * ramp, "turn_left": 0.0,
                 "turn_right": 0.0, "rhythm": 10.0 * ramp, "stance": 10.0 * ramp}
        action = bridge.step(rates, dt_s=dt_s)
        obs, _, terminated, truncated, _ = sim.step(action)
        if terminated or truncated:
            sim.close()
            return None

    start_pos = np.array(obs["fly"][0])
    contact_trace = []  # LF leg contact force over time

    rates = {"forward": drive_hz, "turn_left": 0.0, "turn_right": 0.0,
             "rhythm": 10.0, "stance": 10.0}

    for step in range(body_steps):
        action = bridge.step(rates, dt_s=dt_s)
        try:
            obs, _, terminated, truncated, _ = sim.step(action)
        except Exception:
            break
        if terminated or truncated:
            break
        # Record LF leg contact force
        raw_cf = np.array(obs.get("contact_forces", np.zeros((30, 3))), dtype=np.float32)
        lf_mag = float(np.linalg.norm(raw_cf[:5]))
        contact_trace.append(lf_mag)

    end_pos = np.array(obs["fly"][0])
    sim.close()

    diff = end_pos - start_pos
    dist = float(np.linalg.norm(diff))
    dx = float(diff[0])
    duration_s = body_steps * dt_s
    speed_mm_s = dx / duration_s if duration_s > 0 else 0.0

    contact_freq = measure_contact_frequency(contact_trace, dt_s)

    return {
        "drive_hz": drive_hz,
        "dist": dist,
        "dx": dx,
        "speed_mm_s": speed_mm_s,
        "contact_freq_hz": contact_freq,
        "n_steps": len(contact_trace),
    }


def main():
    parser = argparse.ArgumentParser(description="CPG speed control sweep")
    parser.add_argument("--body-steps", type=int, default=3000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--cpg", action="store_true", help="Use Pugliese CPG")
    args = parser.parse_args()

    drive_levels = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]

    print(f"CPG Speed Control Sweep")
    print(f"  Rhythm: {'Pugliese CPG' if args.cpg else 'sine'}")
    print(f"  Steps: {args.body_steps}, Warmup: {args.warmup_steps}")
    print(f"  Drive levels: {drive_levels}")

    # Measure CPG frequency independently (if using CPG)
    cpg_freqs = {}
    cpg_amps = {}
    if args.cpg:
        from bridge.cpg_pugliese import PuglieseCPG
        cpg_path = Path("data/cpg_weights.json")
        if cpg_path.exists():
            print("\nMeasuring CPG frequencies...")
            for drive in drive_levels:
                cpg = PuglieseCPG.from_json(cpg_path)
                freq, amp = measure_cpg_frequency(cpg, drive)
                cpg_freqs[drive] = freq
                cpg_amps[drive] = amp
                print(f"  drive={drive:.0f}Hz: CPG freq={freq:.1f}Hz, amp={amp:.1f}")

    # FlyGym sweep
    results = []
    t0 = time.time()

    for drive in drive_levels:
        print(f"\n--- Drive = {drive:.0f} Hz ---")
        result = run_drive_sweep(
            drive, args.body_steps, args.warmup_steps, use_cpg=args.cpg
        )
        if result is not None:
            if drive in cpg_freqs:
                result["cpg_freq_hz"] = cpg_freqs[drive]
                result["cpg_amplitude"] = cpg_amps[drive]
            results.append(result)
            print(f"  speed={result['speed_mm_s']:.2f} mm/s, "
                  f"contact_freq={result['contact_freq_hz']:.1f} Hz, "
                  f"dist={result['dist']:.2f} mm")

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*70}")
    print(f"SPEED CONTROL SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Drive(Hz)':>10} {'Speed(mm/s)':>12} {'ContactFreq':>12} {'CPGFreq':>10} {'Dist(mm)':>10}")
    print("-" * 60)
    for r in results:
        cpg_f = f"{r.get('cpg_freq_hz', 0):.1f}" if 'cpg_freq_hz' in r else "N/A"
        print(f"  {r['drive_hz']:>8.0f} {r['speed_mm_s']:>11.2f} "
              f"{r['contact_freq_hz']:>11.1f} {cpg_f:>9} {r['dist']:>9.2f}")

    # Save
    output_path = Path("logs/cpg_speed_control")
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "body_steps": args.body_steps,
            "warmup_steps": args.warmup_steps,
            "use_cpg": args.cpg,
            "drive_levels": drive_levels,
        },
        "results": results,
        "elapsed_s": elapsed,
    }
    out_file = output_path / "speed_control_results.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
