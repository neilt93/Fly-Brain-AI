"""
Minimal VNC validation: does the 1000-neuron circuit walk in MuJoCo?

Runs the MinimalVNCRunner through VNCBridge + FlyGym and measures:
  1. Forward distance (does it walk?)
  2. Tripod gait (is adhesion alternating?)
  3. Forward ablation (does silencing forward DNs reduce distance?)

Usage:
    python experiments/vnc_minimal_validation.py
    python experiments/vnc_minimal_validation.py --body-steps 10000
    python experiments/vnc_minimal_validation.py --n-premotor 1000
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_minimal_vnc_walk(
    body_steps: int = 5000,
    n_premotor: int = 500,
    brain_window_ms: float = 20.0,
    body_steps_per_brain: int = 200,
    shuffle_seed: int | None = None,
    ablate_forward: bool = False,
):
    """Run MinimalVNC through FlyGym and return results."""
    import flygym
    from bridge.vnc_bridge import VNCBridge
    from bridge.vnc_minimal import MinimalVNCConfig
    from bridge.vnc_connectome import VNCInput

    cfg = MinimalVNCConfig(n_premotor=n_premotor)
    bridge = VNCBridge(
        use_minimal_vnc=True,
        vnc_cfg=cfg,
        shuffle_seed=shuffle_seed,
    )

    # Set up FlyGym
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    arena = flygym.arena.FlatTerrain()
    sim = flygym.SingleFlySimulation(fly=fly, arena=arena, timestep=1e-4)
    obs, _ = sim.reset()

    # Warmup: hold init pose
    init_joints = np.array(obs["joints"][0], dtype=np.float32)
    for _ in range(500):
        action = {"joints": init_joints, "adhesion": np.ones(6, dtype=np.float32)}
        obs, _, term, trunc, _ = sim.step(action)
        if term or trunc:
            break

    bridge.reset(init_angles=init_joints)
    start_pos = np.array(obs["fly"][0])

    # Group rates: standard forward walking command
    group_rates = {
        "forward": 30.0,
        "turn_left": 5.0,
        "turn_right": 5.0,
        "rhythm": 10.0,
        "stance": 10.0,
    }

    if ablate_forward:
        group_rates["forward"] = 0.0

    positions = [start_pos.tolist()]
    adhesion_history = []
    t_start = time.time()

    step = 0
    while step < body_steps:
        # Brain step: run VNC Brian2
        bridge.step_brain(group_rates, sim_ms=brain_window_ms)

        # Body steps: rhythm modulation + MN decode at body frequency
        for _ in range(body_steps_per_brain):
            if step >= body_steps:
                break

            action = bridge.step(group_rates, dt_s=1e-4)
            try:
                obs, _, term, trunc, _ = sim.step(action)
                if term or trunc:
                    break
            except Exception:
                break

            step += 1

            if step % 500 == 0:
                pos = np.array(obs["fly"][0])
                positions.append(pos.tolist())
                adhesion_history.append(action["adhesion"].tolist())

        if term or trunc:
            break

    elapsed = time.time() - t_start
    end_pos = np.array(obs["fly"][0])
    diff = end_pos - start_pos
    dist = float(np.linalg.norm(diff))
    dx = float(diff[0])
    heading = float(np.degrees(np.arctan2(diff[1], diff[0])))

    sim.close()

    return {
        "steps": step,
        "distance_mm": dist,
        "dx_mm": dx,
        "heading_deg": heading,
        "fwd_efficiency": dx / max(dist, 0.001),
        "elapsed_s": elapsed,
        "positions": positions,
        "adhesion_samples": adhesion_history,
        "n_premotor": n_premotor,
        "ablate_forward": ablate_forward,
        "shuffle_seed": shuffle_seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal VNC walking validation")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--n-premotor", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("MINIMAL VNC VALIDATION")
    print(f"  {args.n_premotor} premotor neurons, {args.body_steps} body steps")
    print("=" * 60)

    # --- Test 1: Intact walking ---
    print("\n[1/3] Intact walking...")
    intact = run_minimal_vnc_walk(
        body_steps=args.body_steps,
        n_premotor=args.n_premotor,
    )
    print(f"  Distance: {intact['distance_mm']:.2f}mm")
    print(f"  Forward dx: {intact['dx_mm']:.2f}mm")
    print(f"  Heading: {intact['heading_deg']:.1f} deg")
    print(f"  Forward efficiency: {intact['fwd_efficiency']:.1%}")
    print(f"  Time: {intact['elapsed_s']:.1f}s")
    walks = intact["distance_mm"] > 0.5
    print(f"  WALKS: {'PASS' if walks else 'FAIL'}")

    # Check tripod gait from adhesion
    if intact["adhesion_samples"]:
        adh = np.array(intact["adhesion_samples"])
        # Tripod = legs alternate: [0,1,0,1,0,1] or [1,0,1,0,1,0]
        tripod_score = 0
        for sample in adh:
            s = np.array(sample)
            # Check if alternating pattern
            if (np.allclose(s, [0, 1, 0, 1, 0, 1]) or
                    np.allclose(s, [1, 0, 1, 0, 1, 0])):
                tripod_score += 1
        tripod_pct = tripod_score / len(adh) * 100
        print(f"  Tripod gait: {tripod_pct:.0f}% of samples")
    else:
        tripod_pct = 0

    # --- Test 2: Forward ablation ---
    print("\n[2/3] Forward DN ablation...")
    ablated = run_minimal_vnc_walk(
        body_steps=args.body_steps,
        n_premotor=args.n_premotor,
        ablate_forward=True,
    )
    print(f"  Distance: {ablated['distance_mm']:.2f}mm")
    print(f"  Forward dx: {ablated['dx_mm']:.2f}mm")

    if intact["distance_mm"] > 0.01:
        reduction = 1.0 - ablated["distance_mm"] / intact["distance_mm"]
        print(f"  Reduction: {reduction:.0%}")
        ablation_pass = reduction > 0.3
    else:
        reduction = 0
        ablation_pass = False
    print(f"  CAUSAL: {'PASS' if ablation_pass else 'FAIL'}")

    # --- Test 3: Shuffled VNC ---
    print("\n[3/3] Shuffled VNC connectivity...")
    shuffled = run_minimal_vnc_walk(
        body_steps=args.body_steps,
        n_premotor=args.n_premotor,
        shuffle_seed=999,
    )
    print(f"  Distance: {shuffled['distance_mm']:.2f}mm")
    print(f"  Forward dx: {shuffled['dx_mm']:.2f}mm")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Condition':<20} {'Distance':>10} {'dx':>10} {'Heading':>10}")
    print("-" * 52)
    for name, r in [("Intact", intact), ("Fwd ablated", ablated),
                     ("Shuffled", shuffled)]:
        print(f"{name:<20} {r['distance_mm']:>9.2f}mm {r['dx_mm']:>9.2f}mm "
              f"{r['heading_deg']:>9.1f} deg")

    n_pass = sum([walks, ablation_pass])
    print(f"\nTests passed: {n_pass}/2")
    print(f"Tripod gait: {tripod_pct:.0f}%")
    print(f"Circuit: {args.n_premotor} premotor + 500 MN = "
          f"{args.n_premotor + 500} LIF neurons")


if __name__ == "__main__":
    main()
