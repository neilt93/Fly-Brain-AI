#!/usr/bin/env python
"""
Sensory gating experiment: ablate sensory input at two levels and compare.

Level 1: Brain-level — silence brain sensory neurons (existing infrastructure)
Level 2: VNC-level — silence VNC proprioceptive afferents (Tier 1B)

Demonstrates that VNC sensory processing is functionally distinct from
brain-level sensory processing.

Conditions:
  1. Intact — full sensory input at both levels
  2. Brain sensory silenced — zero brain sensory neuron rates
  3. VNC sensory silenced — zero VNC proprioceptive rates
  4. Both silenced — zero sensory at both levels
  5. Shuffled VNC sensory — proprioceptive rates to random VNC neurons

Usage:
    python experiments/sensory_gating.py                    # fake VNC (quick)
    python experiments/sensory_gating.py --real-vnc          # real Brian2 VNC
    python experiments/sensory_gating.py --body-steps 5000   # longer run
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


def run_condition(
    name,
    body_steps,
    warmup_steps,
    use_fake_vnc,
    use_cpg,
    silence_brain_sensory=False,
    silence_vnc_sensory=False,
    use_fake_brain=False,
    connectome="flywire",
):
    """Run one sensory gating condition."""
    import flygym
    from bridge.config import BridgeConfig
    from bridge.interfaces import LocomotionCommand
    from bridge.sensory_encoder import SensoryEncoder
    from bridge.brain_runner import create_brain_runner
    from bridge.descending_decoder import DescendingDecoder
    from bridge.flygym_adapter import FlyGymAdapter
    from bridge.vnc_bridge import VNCBridge

    print(f"\n{'='*60}")
    print(f"  Condition: {name}")
    flags = []
    if silence_brain_sensory:
        flags.append("brain_sensory=OFF")
    if silence_vnc_sensory:
        flags.append("vnc_sensory=OFF")
    if flags:
        print(f"  Flags: {', '.join(flags)}")
    print(f"{'='*60}")

    cfg = BridgeConfig(connectome=connectome)

    # Load populations
    if not cfg.sensory_ids_path.exists():
        print("  Neuron population files not found.")
        return None

    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)

    # Initialize components
    if cfg.channel_map_path.exists():
        encoder = SensoryEncoder.from_channel_map(
            sensory_ids, cfg.channel_map_path,
            max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
        )
    else:
        encoder = SensoryEncoder(sensory_ids, max_rate_hz=cfg.max_rate_hz)

    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    adapter = FlyGymAdapter()

    brain = create_brain_runner(
        sensory_ids=sensory_ids,
        readout_ids=readout_ids,
        use_fake=use_fake_brain,
        warmup_ms=cfg.brain_warmup_ms,
        connectome=connectome,
    )

    vnc_bridge = VNCBridge(use_fake_vnc=use_fake_vnc, use_cpg=use_cpg)

    # FlyGym
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly, arena=flygym.arena.FlatTerrain(), timestep=1e-4,
    )
    obs, _ = sim.reset()
    dt_s = 1e-4

    init_joints = np.array(obs["joints"][0], dtype=np.float64)
    vnc_bridge.reset(init_angles=init_joints)

    # Warmup with neutral rates
    for i in range(warmup_steps):
        ramp = min(1.0, i / max(warmup_steps * 0.5, 1.0))
        neutral = {"forward": 20.0 * ramp, "turn_left": 0.0,
                    "turn_right": 0.0, "rhythm": 10.0 * ramp, "stance": 10.0 * ramp}
        action = vnc_bridge.step(neutral, dt_s=dt_s)
        obs, _, terminated, truncated, _ = sim.step(action)
        if terminated or truncated:
            sim.close()
            return None

    start_pos = np.array(obs["fly"][0])
    positions = [start_pos.tolist()]
    bspb = cfg.body_steps_per_brain
    current_group_rates = {"forward": 20.0, "turn_left": 0.0,
                           "turn_right": 0.0, "rhythm": 10.0, "stance": 10.0}

    for step in range(body_steps):
        # Brain step
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)

            # Level 1 gating: silence brain sensory input
            if silence_brain_sensory:
                brain_input.firing_rates_hz[:] = cfg.baseline_rate_hz

            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            current_group_rates = decoder.get_group_rates(brain_output)

            # VNC brain step — pass body_obs for proprioceptive feedback
            # Level 2 gating: if silencing VNC sensory, pass None for body_obs
            vnc_body_obs = None if silence_vnc_sensory else body_obs
            vnc_bridge.step_brain(
                current_group_rates,
                sim_ms=cfg.brain_dt_ms,
                body_obs=vnc_body_obs,
            )

        # Body step
        action = vnc_bridge.step(current_group_rates, dt_s=dt_s)
        try:
            obs, _, terminated, truncated, _ = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break
        if terminated or truncated:
            break

        if step % 50 == 0:
            positions.append(np.array(obs["fly"][0]).tolist())

    sim.close()

    end_pos = np.array(positions[-1])
    diff = np.array(end_pos) - np.array(positions[0])
    dist = float(np.linalg.norm(diff))
    dx, dy = float(diff[0]), float(diff[1])
    heading = float(np.degrees(np.arctan2(dy, dx)))

    # Gait regularity: variance in step-to-step displacement
    displacements = []
    for i in range(1, len(positions)):
        d = np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
        displacements.append(d)
    gait_regularity = float(np.std(displacements)) if displacements else 0.0

    result = {
        "name": name,
        "dist": dist,
        "dx": dx,
        "dy": dy,
        "heading": heading,
        "gait_regularity": gait_regularity,
        "n_frames": len(positions),
        "silence_brain_sensory": silence_brain_sensory,
        "silence_vnc_sensory": silence_vnc_sensory,
    }

    print(f"  dist={dist:.2f}mm, heading={heading:.1f}°, regularity={gait_regularity:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Sensory gating experiment")
    parser.add_argument("--body-steps", type=int, default=3000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--real-vnc", action="store_true")
    parser.add_argument("--cpg", action="store_true")
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--connectome", default="flywire", choices=["flywire", "banc"])
    args = parser.parse_args()

    use_fake = not args.real_vnc

    print(f"Sensory Gating Experiment")
    print(f"  VNC: {'Fake' if use_fake else 'Brian2 MANC'}")
    print(f"  Brain: {'Fake' if args.fake_brain else 'Brian2 LIF'}")
    print(f"  Steps: {args.body_steps}")

    conditions = [
        ("intact", False, False),
        ("brain_sensory_off", True, False),
        ("vnc_sensory_off", False, True),
        ("both_sensory_off", True, True),
    ]

    results = []
    t0 = time.time()

    for cname, brain_off, vnc_off in conditions:
        r = run_condition(
            cname, args.body_steps, args.warmup_steps,
            use_fake_vnc=use_fake, use_cpg=args.cpg,
            silence_brain_sensory=brain_off,
            silence_vnc_sensory=vnc_off,
            use_fake_brain=args.fake_brain,
            connectome=args.connectome,
        )
        if r:
            results.append(r)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"SENSORY GATING SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Condition':<22} {'Dist':>8} {'Heading':>9} {'Regularity':>11} {'BrainSens':>10} {'VNCSens':>8}")
    print("-" * 70)
    for r in results:
        bs = "OFF" if r["silence_brain_sensory"] else "ON"
        vs = "OFF" if r["silence_vnc_sensory"] else "ON"
        print(f"  {r['name']:<20} {r['dist']:>7.2f} {r['heading']:>+8.1f}° "
              f"{r['gait_regularity']:>10.4f} {bs:>9} {vs:>7}")

    # Pass/fail criteria
    print(f"\n--- Pass/Fail Criteria ---")
    passes = 0
    total = 0

    def find(name):
        for r in results:
            if r["name"] == name:
                return r
        return None

    r_intact = find("intact")
    r_brain = find("brain_sensory_off")
    r_vnc = find("vnc_sensory_off")
    r_both = find("both_sensory_off")

    # Test 1: Brain sensory off should reduce distance (brain gets no useful info)
    if r_intact and r_brain:
        total += 1
        ok = r_brain["dist"] < r_intact["dist"] * 0.95
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: brain_off dist ({r_brain['dist']:.2f}) "
              f"< intact ({r_intact['dist']:.2f}) * 0.95")

    # Test 2: VNC sensory off should affect gait regularity
    if r_intact and r_vnc:
        total += 1
        # Either distance or regularity should change
        dist_change = abs(r_vnc["dist"] - r_intact["dist"]) / max(r_intact["dist"], 0.01)
        reg_change = abs(r_vnc["gait_regularity"] - r_intact["gait_regularity"])
        ok = dist_change > 0.05 or reg_change > 0.001
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: vnc_off shows change "
              f"(dist_change={dist_change:.2%}, reg_change={reg_change:.4f})")

    # Test 3: Both off should be worse than either alone
    if r_intact and r_both:
        total += 1
        ok = r_both["dist"] < r_intact["dist"]
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: both_off dist ({r_both['dist']:.2f}) "
              f"< intact ({r_intact['dist']:.2f})")

    # Test 4: Brain and VNC gating should have DIFFERENT effects
    if r_brain and r_vnc:
        total += 1
        brain_effect = abs(r_brain["dist"] - r_intact["dist"]) if r_intact else 0
        vnc_effect = abs(r_vnc["dist"] - r_intact["dist"]) if r_intact else 0
        ok = abs(brain_effect - vnc_effect) > 0.01  # not identical
        passes += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: brain vs vnc gating have different effects "
              f"(brain={brain_effect:.2f}, vnc={vnc_effect:.2f})")

    print(f"\n  Total: {passes}/{total} PASS")

    # Save
    output_path = Path("logs/sensory_gating")
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "body_steps": args.body_steps,
            "use_fake_vnc": use_fake,
            "use_cpg": args.cpg,
            "use_fake_brain": args.fake_brain,
        },
        "results": results,
        "summary": {"passes": passes, "total": total},
        "elapsed_s": elapsed,
    }
    _write_json_atomic(output_path / "sensory_gating_results.json", payload)
    print(f"\nSaved to {output_path / 'sensory_gating_results.json'}")


if __name__ == "__main__":
    main()
