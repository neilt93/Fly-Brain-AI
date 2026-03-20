#!/usr/bin/env python
"""
VNC turning validation: verify turning commands propagate through real VNC.

Tests that boosting/silencing turn_left/turn_right DN rates produces
corresponding heading changes and per-leg stride asymmetry.

Usage:
    python experiments/vnc_turning_test.py                    # fake VNC (quick)
    python experiments/vnc_turning_test.py --real-vnc          # real Brian2 VNC
    python experiments/vnc_turning_test.py --body-steps 10000  # longer run
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    for _attempt in range(5):
        try:
            tmp_path.replace(path)
            break
        except PermissionError:
            if _attempt < 4:
                import time; time.sleep(0.05 * (_attempt + 1))
            else:
                import shutil; shutil.copy2(str(tmp_path), str(path))
                tmp_path.unlink(missing_ok=True)


def run_condition(name, body_steps, warmup_steps, group_rates_fn, use_fake_vnc, use_cpg):
    """Run one turning condition and return results."""
    import flygym
    from bridge.vnc_bridge import VNCBridge
    from bridge.vnc_connectome import VNCConfig

    print(f"\n{'='*60}")
    print(f"  Condition: {name}")
    print(f"{'='*60}")

    bridge = VNCBridge(use_fake_vnc=use_fake_vnc, use_cpg=use_cpg)

    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly, arena=flygym.arena.FlatTerrain(), timestep=1e-4
    )
    obs, _ = sim.reset()
    dt_s = 1e-4

    # Init decoder smoothing from FlyGym pose
    init_joints = np.array(obs["joints"][0], dtype=np.float64)
    bridge.reset(init_angles=init_joints)

    # Warmup with neutral forward drive
    print(f"  Warmup ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        ramp = min(1.0, i / max(warmup_steps * 0.5, 1.0))
        neutral = {"forward": 20.0 * ramp, "turn_left": 0.0,
                    "turn_right": 0.0, "rhythm": 10.0 * ramp, "stance": 10.0 * ramp}
        action = bridge.step(neutral, dt_s=dt_s)
        obs, _, terminated, truncated, _ = sim.step(action)
        if terminated or truncated:
            print("  Episode ended during warmup!")
            sim.close()
            return None

    # Record start position
    start_pos = np.array(obs["fly"][0])
    positions = [start_pos.tolist()]
    contact_log = []  # per-leg contact forces over time

    # Main loop
    print(f"  Running {body_steps} steps...")
    for step in range(body_steps):
        rates = group_rates_fn(step, body_steps)
        action = bridge.step(rates, dt_s=dt_s)
        try:
            obs, _, terminated, truncated, _ = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break
        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

        if step % 50 == 0:
            positions.append(np.array(obs["fly"][0]).tolist())
            # Contact forces per leg
            raw_cf = np.array(obs.get("contact_forces", np.zeros((30, 3))), dtype=np.float32)
            per_leg = []
            for leg_i in range(6):
                mag = float(np.linalg.norm(raw_cf[leg_i * 5:(leg_i + 1) * 5]))
                per_leg.append(mag)
            contact_log.append(per_leg)

    end_pos = np.array(obs["fly"][0])
    sim.close()

    # Compute metrics
    diff = end_pos - start_pos
    dist = float(np.linalg.norm(diff))
    dx, dy = float(diff[0]), float(diff[1])
    heading = float(np.degrees(np.arctan2(dy, dx)))
    fwd_eff = dx / max(dist, 0.001)

    # Per-leg stride analysis from contact forces
    contacts = np.array(contact_log)  # (n_frames, 6)
    if len(contacts) > 10:
        leg_names = ["LF", "LM", "LH", "RF", "RM", "RH"]
        left_var = float(np.mean(np.var(contacts[:, :3], axis=0)))
        right_var = float(np.mean(np.var(contacts[:, 3:], axis=0)))
        leg_vars = {leg_names[i]: float(np.var(contacts[:, i])) for i in range(6)}
    else:
        left_var = right_var = 0.0
        leg_vars = {}

    result = {
        "name": name,
        "dist": dist,
        "dx": dx,
        "dy": dy,
        "heading": heading,
        "fwd_eff": fwd_eff,
        "left_stride_var": left_var,
        "right_stride_var": right_var,
        "leg_variances": leg_vars,
        "n_steps": len(positions) - 1,
    }

    print(f"  Result: dist={dist:.2f}mm, heading={heading:.1f}°, "
          f"dx={dx:.2f}, dy={dy:.2f}, fwd_eff={fwd_eff:.2f}")
    print(f"  Stride var: L={left_var:.2f}, R={right_var:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="VNC turning validation")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--real-vnc", action="store_true", help="Use Brian2 VNC (slow)")
    parser.add_argument("--cpg", action="store_true", help="Use Pugliese CPG rhythm")
    parser.add_argument("--boost-factor", type=float, default=3.0,
                        help="Multiplier for boosted turn rate")
    args = parser.parse_args()

    use_fake = not args.real_vnc
    boost = args.boost_factor
    base_fwd = 20.0
    base_turn = 15.0  # baseline turn rate when boosted

    print(f"VNC Turning Validation")
    print(f"  VNC: {'Fake' if use_fake else 'Brian2 MANC'}")
    print(f"  CPG: {'Pugliese' if args.cpg else 'sine'}")
    print(f"  Steps: {args.body_steps}, Warmup: {args.warmup_steps}")
    print(f"  Boost factor: {boost}x")

    # Define conditions
    def neutral_rates(step, total):
        return {"forward": base_fwd, "turn_left": 0.0, "turn_right": 0.0,
                "rhythm": 10.0, "stance": 10.0}

    def boost_left(step, total):
        return {"forward": base_fwd, "turn_left": base_turn * boost,
                "turn_right": 0.0, "rhythm": 10.0, "stance": 10.0}

    def boost_right(step, total):
        return {"forward": base_fwd, "turn_left": 0.0,
                "turn_right": base_turn * boost, "rhythm": 10.0, "stance": 10.0}

    def silence_left(step, total):
        """Boost right to see if heading goes negative when left is silent."""
        return {"forward": base_fwd, "turn_left": 0.0,
                "turn_right": base_turn, "rhythm": 10.0, "stance": 10.0}

    def silence_right(step, total):
        """Boost left to see if heading goes positive when right is silent."""
        return {"forward": base_fwd, "turn_left": base_turn,
                "turn_right": 0.0, "rhythm": 10.0, "stance": 10.0}

    conditions = [
        ("intact_forward", neutral_rates),
        ("boost_turn_left", boost_left),
        ("boost_turn_right", boost_right),
        ("silence_turn_left_boost_right", silence_left),
        ("silence_turn_right_boost_left", silence_right),
    ]

    results = []
    t0 = time.time()

    for name, rate_fn in conditions:
        result = run_condition(
            name, args.body_steps, args.warmup_steps,
            rate_fn, use_fake_vnc=use_fake, use_cpg=args.cpg,
        )
        if result is not None:
            results.append(result)

    elapsed = time.time() - t0

    # Print summary table
    print(f"\n{'='*70}")
    print(f"TURNING VALIDATION SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Condition':<35} {'Heading':>8} {'Dist':>8} {'FwdEff':>8} {'L_var':>8} {'R_var':>8}")
    print("-" * 70)
    for r in results:
        print(f"  {r['name']:<33} {r['heading']:>+7.1f}° {r['dist']:>7.2f} "
              f"{r['fwd_eff']:>7.2f} {r['left_stride_var']:>8.2f} {r['right_stride_var']:>8.2f}")

    # Validate pass criteria
    print(f"\n--- Pass/Fail Criteria ---")
    passes = 0
    total = 0

    # Find results by name
    def find(name):
        for r in results:
            if r["name"] == name:
                return r
        return None

    r_intact = find("intact_forward")
    r_left = find("boost_turn_left")
    r_right = find("boost_turn_right")

    # Test 1: Boost left -> heading > +30 degrees
    if r_left:
        total += 1
        baseline_heading = r_intact["heading"] if r_intact else 0.0
        delta = r_left["heading"] - baseline_heading
        ok = delta > 30.0
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: boost_left heading delta = {delta:+.1f}° (need > +30°)")

    # Test 2: Boost right -> heading < -30 degrees
    if r_right:
        total += 1
        baseline_heading = r_intact["heading"] if r_intact else 0.0
        delta = r_right["heading"] - baseline_heading
        ok = delta < -30.0
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: boost_right heading delta = {delta:+.1f}° (need < -30°)")

    # Test 3: Symmetry
    if r_left and r_right:
        total += 1
        baseline_heading = r_intact["heading"] if r_intact else 0.0
        delta_l = r_left["heading"] - baseline_heading
        delta_r = r_right["heading"] - baseline_heading
        asym = abs(delta_l + delta_r)
        ok = asym < 30.0
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: symmetry |delta_L + delta_R| = {asym:.1f}° (need < 30°)")

    # Test 4: Per-leg asymmetry during turning
    if r_left:
        total += 1
        # When turning left, left legs should have reduced stride
        ok = r_left["left_stride_var"] < r_left["right_stride_var"]
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: turn_left L_var ({r_left['left_stride_var']:.2f}) "
              f"< R_var ({r_left['right_stride_var']:.2f})")

    if r_right:
        total += 1
        ok = r_right["right_stride_var"] < r_right["left_stride_var"]
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: turn_right R_var ({r_right['right_stride_var']:.2f}) "
              f"< L_var ({r_right['left_stride_var']:.2f})")

    print(f"\n  Total: {passes}/{total} PASS")

    # Save results
    output_path = Path("logs/vnc_turning")
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "body_steps": args.body_steps,
            "warmup_steps": args.warmup_steps,
            "use_fake_vnc": use_fake,
            "use_cpg": args.cpg,
            "boost_factor": boost,
        },
        "results": results,
        "summary": {"passes": passes, "total": total},
        "elapsed_s": elapsed,
    }
    _write_json_atomic(output_path / "turning_results.json", payload)
    print(f"\nResults saved to {output_path / 'turning_results.json'}")


if __name__ == "__main__":
    main()
