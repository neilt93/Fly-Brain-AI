#!/usr/bin/env python
"""
VNC perturbation recovery: test how quickly gait recovers after lateral push.

Applies a lateral impulse force mid-walk and measures recovery time.
Compares intact VNC vs shuffled VNC to test if connectome wiring
contributes to perturbation recovery.

Usage:
    python experiments/vnc_perturbation_test.py
    python experiments/vnc_perturbation_test.py --real-vnc
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
    tmp_path.replace(path)


def run_perturbation(
    name, body_steps, warmup_steps, perturb_step, perturb_force,
    use_fake_vnc, use_cpg, shuffle_seed=None,
):
    """Run perturbation experiment."""
    import flygym
    from bridge.vnc_bridge import VNCBridge

    print(f"\n{'='*60}")
    print(f"  Condition: {name}")
    print(f"{'='*60}")

    bridge = VNCBridge(
        use_fake_vnc=use_fake_vnc, use_cpg=use_cpg,
        shuffle_seed=shuffle_seed,
    )

    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly, arena=flygym.arena.FlatTerrain(), timestep=1e-4
    )
    obs, _ = sim.reset()
    dt_s = 1e-4

    init_joints = np.array(obs["joints"][0], dtype=np.float64)
    bridge.reset(init_angles=init_joints)

    # Warmup
    print(f"  Warmup ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        ramp = min(1.0, i / max(warmup_steps * 0.5, 1.0))
        rates = {"forward": 20.0 * ramp, "turn_left": 0.0,
                 "turn_right": 0.0, "rhythm": 10.0 * ramp, "stance": 10.0 * ramp}
        action = bridge.step(rates, dt_s=dt_s)
        obs, _, terminated, truncated, _ = sim.step(action)
        if terminated or truncated:
            sim.close()
            return None

    rates = {"forward": 20.0, "turn_left": 0.0, "turn_right": 0.0,
             "rhythm": 10.0, "stance": 10.0}

    # Tracking
    headings = []
    positions = []
    perturbed = False
    pre_perturb_headings = []

    print(f"  Running {body_steps} steps (perturbation at step {perturb_step})...")
    for step in range(body_steps):
        action = bridge.step(rates, dt_s=dt_s)

        # Apply perturbation force at the specified step
        if step == perturb_step:
            # Find thorax body (name varies: '0/Thorax', '1/Thorax', etc.)
            try:
                model = sim.physics.model
                thorax_id = None
                for i in range(model.nbody):
                    bname = model.id2name(i, 'body')
                    if 'Thorax' in bname:
                        thorax_id = i
                        break
                if thorax_id is not None:
                    sim.physics.data.xfrc_applied[thorax_id, :3] = perturb_force
                    perturbed = True
                    print(f"  Applied force {perturb_force} to body[{thorax_id}] at step {step}")
                else:
                    # Fallback: apply to body index 1
                    sim.physics.data.xfrc_applied[1, :3] = perturb_force
                    perturbed = True
                    print(f"  Applied force to body[1] (fallback) at step {step}")
            except Exception as e:
                print(f"  WARNING: Could not apply perturbation: {e}")

        # Clear perturbation force after 10 steps (impulse)
        if perturbed and step == perturb_step + 10:
            try:
                sim.physics.data.xfrc_applied[:] = 0
            except Exception:
                pass

        try:
            obs, _, terminated, truncated, _ = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break
        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

        # Record heading
        if step % 10 == 0:
            pos = np.array(obs["fly"][0])
            positions.append(pos.tolist())
            if len(positions) >= 2:
                dp = np.array(positions[-1]) - np.array(positions[-2])
                heading = float(np.degrees(np.arctan2(dp[1], dp[0])))
                headings.append(heading)
                if step < perturb_step:
                    pre_perturb_headings.append(heading)

    sim.close()

    # Compute recovery metrics
    if len(positions) >= 2:
        start_p = np.array(positions[0])
        end_p = np.array(positions[-1])
        d = end_p - start_p
        base_dist = float(np.linalg.norm(d))
        base_dx, base_dy = float(d[0]), float(d[1])
    else:
        base_dist = base_dx = base_dy = 0.0

    if len(headings) < 10 or not perturbed:
        return {"name": name, "recovered": False, "recovery_steps": -1,
                "max_heading_deviation": 0.0, "pre_heading_var": 0.0,
                "dist": base_dist, "dx": base_dx, "dy": base_dy,
                "n_steps": len(positions)}

    # Baseline heading variance (pre-perturbation)
    pre_var = float(np.var(pre_perturb_headings[-20:])) if len(pre_perturb_headings) >= 20 else 1.0
    threshold = max(pre_var * 3.0, 5.0)  # heading variance threshold for "recovered"

    # Find perturbation index in heading array
    perturb_heading_idx = perturb_step // 10
    if perturb_heading_idx >= len(headings):
        return {"name": name, "recovered": False, "recovery_steps": -1}

    # Measure recovery: sliding window variance drops below threshold
    window = 20
    recovery_steps = -1
    post_headings = headings[perturb_heading_idx:]

    for i in range(window, len(post_headings)):
        w = post_headings[i-window:i]
        var = float(np.var(w))
        if var < threshold:
            recovery_steps = i * 10  # convert back to body steps
            break

    # Overall displacement
    if len(positions) >= 2:
        start = np.array(positions[0])
        end = np.array(positions[-1])
        diff = end - start
        dist = float(np.linalg.norm(diff))
        dx, dy = float(diff[0]), float(diff[1])
    else:
        dist = dx = dy = 0.0

    # Max heading deviation after perturbation
    if len(post_headings) > 0:
        baseline_heading = float(np.mean(pre_perturb_headings[-20:])) if pre_perturb_headings else 0.0
        max_deviation = float(np.max(np.abs(np.array(post_headings[:50]) - baseline_heading)))
    else:
        max_deviation = 0.0

    result = {
        "name": name,
        "recovered": recovery_steps > 0,
        "recovery_steps": recovery_steps,
        "max_heading_deviation": max_deviation,
        "pre_heading_var": pre_var,
        "dist": dist,
        "dx": dx,
        "dy": dy,
        "n_steps": len(positions),
    }

    status = f"recovered in {recovery_steps} steps" if recovery_steps > 0 else "did not recover"
    print(f"  Result: {status}, max_dev={max_deviation:.1f}, dist={dist:.2f}mm")

    return result


def main():
    parser = argparse.ArgumentParser(description="VNC perturbation recovery")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--perturb-step", type=int, default=2000)
    parser.add_argument("--force-magnitude", type=float, default=0.5,
                        help="Lateral force magnitude (N)")
    parser.add_argument("--real-vnc", action="store_true")
    parser.add_argument("--cpg", action="store_true")
    args = parser.parse_args()

    use_fake = not args.real_vnc
    force = [0.0, args.force_magnitude, 0.0]  # lateral (Y direction)

    print(f"VNC Perturbation Recovery Test")
    print(f"  VNC: {'Fake' if use_fake else 'Brian2 MANC'}")
    print(f"  Steps: {args.body_steps}, Perturbation at step {args.perturb_step}")
    print(f"  Force: {force}")

    conditions = [
        ("intact", None),
        ("no_perturbation", "skip"),  # Control: no push
    ]

    results = []
    t0 = time.time()

    for name, shuffle in conditions:
        if shuffle == "skip":
            # No perturbation control
            result = run_perturbation(
                name, args.body_steps, args.warmup_steps,
                perturb_step=args.body_steps + 1000,  # never perturb
                perturb_force=[0, 0, 0],
                use_fake_vnc=use_fake, use_cpg=args.cpg,
            )
        else:
            result = run_perturbation(
                name, args.body_steps, args.warmup_steps,
                perturb_step=args.perturb_step,
                perturb_force=force,
                use_fake_vnc=use_fake, use_cpg=args.cpg,
                shuffle_seed=shuffle,
            )
        if result:
            results.append(result)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"PERTURBATION RECOVERY SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'Recovered':>10} {'Steps':>8} {'MaxDev':>8} {'Dist':>8}")
    print("-" * 60)
    for r in results:
        rec = "YES" if r.get("recovered") else "NO"
        steps = r.get("recovery_steps", -1)
        print(f"  {r['name']:<23} {rec:>9} {steps:>7} "
              f"{r.get('max_heading_deviation', 0):>7.1f} {r.get('dist', 0):>7.2f}")

    # Save
    output_path = Path("logs/vnc_perturbation")
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "body_steps": args.body_steps,
            "perturb_step": args.perturb_step,
            "force": force,
            "use_fake_vnc": use_fake,
            "use_cpg": args.cpg,
        },
        "results": results,
        "elapsed_s": elapsed,
    }
    _write_json_atomic(output_path / "perturbation_results.json", payload)
    print(f"\nSaved to {output_path / 'perturbation_results.json'}")


if __name__ == "__main__":
    main()
