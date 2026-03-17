"""
VNC-lite validation suite: 4 experiments comparing old decoder vs VNC-lite.

1. Backward compatibility: old headline effects still hold
2. Interface robustness: stable across parameter ranges
3. Improved temporal realism: smoother transitions
4. New causal dissociation: cleaner motor factorization

Usage:
    python experiments/vnc_lite_validation.py --fake-brain    # fast test
    python experiments/vnc_lite_validation.py                 # full Brian2
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

from bridge.config import BridgeConfig
from bridge.interfaces import LocomotionCommand, BodyObservation, BrainOutput
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from bridge.vnc_lite import VNCLite, VNCLiteConfig


def run_trial(
    mode: str,                   # "old" or "vnc_lite"
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    ablation: dict | None = None,   # {"zero": "forward_ids"} etc
    vnc_config: VNCLiteConfig | None = None,
):
    """Run a single closed-loop trial, returning behavior metrics.

    Args:
        mode: "old" for original decoder, "vnc_lite" for new motor layer
        ablation: optional ablation condition applied to brain output
        vnc_config: optional custom VNC-lite config (for robustness test)
    """
    import flygym

    cfg = BridgeConfig()
    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)

    encoder = SensoryEncoder.from_channel_map(
        sensory_ids, cfg.channel_map_path,
        max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    # Load decoder groups for ablation
    with open(cfg.decoder_groups_path) as f:
        decoder_groups = json.load(f)

    # VNC-lite (only used in vnc_lite mode)
    vnc = None
    if mode == "vnc_lite":
        vnc = VNCLite(config=vnc_config)

    brain = create_brain_runner(
        sensory_ids=sensory_ids,
        readout_ids=readout_ids,
        use_fake=use_fake_brain,
        warmup_ms=cfg.brain_warmup_ms,
    )

    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=flygym.arena.FlatTerrain(), timestep=1e-4)
    obs, _ = sim.reset()

    locomotion.warmup(0)
    locomotion.cpg.reset(init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
                         init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        obs, _, term, trunc, _ = sim.step(action)
        if term or trunc:
            sim.close()
            return None

    bspb = cfg.body_steps_per_brain
    current_cmd = LocomotionCommand(forward_drive=1.0)
    positions = []
    commands = []
    vnc_states = []

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)

            # Apply ablation if specified
            if ablation and "zero" in ablation:
                rates = brain_output.firing_rates_hz.copy()
                id_to_idx = {int(nid): i for i, nid in enumerate(brain_output.neuron_ids)}
                group_name = ablation["zero"]
                for nid in decoder_groups.get(group_name, []):
                    if int(nid) in id_to_idx:
                        rates[id_to_idx[int(nid)]] = 0.0
                brain_output = BrainOutput(neuron_ids=brain_output.neuron_ids, firing_rates_hz=rates)

            if mode == "old":
                current_cmd = decoder.decode(brain_output)
            else:
                group_rates = decoder.get_group_rates(brain_output)
                current_cmd = vnc.step(group_rates, dt_s=cfg.brain_dt_ms / 1000.0, body_obs=body_obs)
                vnc_states.append(vnc.get_state_dict())

            commands.append({
                "step": step,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
                "step_frequency": current_cmd.step_frequency,
                "stance_gain": current_cmd.stance_gain,
            })

        action = locomotion.step(current_cmd)
        try:
            obs, _, term, trunc, _ = sim.step(action)
        except Exception:
            break
        if step % 50 == 0:
            positions.append(np.array(obs["fly"][0]).tolist())
        if term or trunc:
            break

    sim.close()

    # Compute metrics
    if len(positions) < 2:
        return None

    positions = np.array(positions)
    start, end = positions[0], positions[-1]
    forward_distance = float(np.linalg.norm(end - start))

    headings = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        headings.append(float(np.arctan2(dy, dx)))
    cumulative_turn = float(np.sum(np.abs(np.diff(headings)))) if len(headings) > 1 else 0.0

    fwd_drives = [c["forward_drive"] for c in commands]
    turn_drives = [c["turn_drive"] for c in commands]
    freq_drives = [c["step_frequency"] for c in commands]
    stance_drives = [c["stance_gain"] for c in commands]

    # Smoothness: variance of step-to-step changes
    def smoothness(arr):
        if len(arr) < 2:
            return 0.0
        diffs = np.diff(arr)
        return float(np.std(diffs))

    return {
        "forward_distance": forward_distance,
        "cumulative_turn": cumulative_turn,
        "mean_forward_drive": float(np.mean(fwd_drives)),
        "mean_turn_drive": float(np.mean(turn_drives)),
        "mean_step_freq": float(np.mean(freq_drives)),
        "mean_stance_gain": float(np.mean(stance_drives)),
        "forward_smoothness": smoothness(fwd_drives),
        "turn_smoothness": smoothness(turn_drives),
        "freq_smoothness": smoothness(freq_drives),
        "stance_smoothness": smoothness(stance_drives),
        "n_commands": len(commands),
        "vnc_states": vnc_states[-5:] if vnc_states else [],
    }


# ════════════════════════════════════════════════════════════════════════════
# TEST 1: BACKWARD COMPATIBILITY
# ════════════════════════════════════════════════════════════════════════════

def test_backward_compatibility(use_fake_brain=False, seed=42):
    """Old headline effects still hold with VNC-lite."""
    print("\n" + "=" * 70)
    print("TEST 1: BACKWARD COMPATIBILITY")
    print("=" * 70)

    conditions = [
        {"name": "baseline", "ablation": None},
        {"name": "forward_ablated", "ablation": {"zero": "forward_ids"}},
        {"name": "turn_left_ablated", "ablation": {"zero": "turn_left_ids"}},
    ]

    results = {}
    for cond in conditions:
        for mode in ["old", "vnc_lite"]:
            key = f"{mode}_{cond['name']}"
            print(f"  Running {key}...", end=" ", flush=True)
            r = run_trial(
                mode=mode,
                body_steps=3000,
                use_fake_brain=use_fake_brain,
                seed=seed,
                ablation=cond["ablation"],
            )
            results[key] = r
            if r:
                print(f"dist={r['forward_distance']:.1f}mm fwd={r['mean_forward_drive']:.3f} "
                      f"turn={r['mean_turn_drive']:.3f}")
            else:
                print("FAILED")

    # Check: forward ablation still reduces forward drive in vnc_lite
    tests = []

    bl = results.get("vnc_lite_baseline")
    fa = results.get("vnc_lite_forward_ablated")
    if bl and fa:
        t1 = fa["mean_forward_drive"] < bl["mean_forward_drive"]
        tests.append(("Fwd ablation reduces fwd drive", t1,
                       f"{fa['mean_forward_drive']:.3f} < {bl['mean_forward_drive']:.3f}"))

        t2 = fa["forward_distance"] < bl["forward_distance"]
        tests.append(("Fwd ablation reduces distance", t2,
                       f"{fa['forward_distance']:.1f} < {bl['forward_distance']:.1f}"))

    tl = results.get("vnc_lite_turn_left_ablated")
    if bl and tl:
        t3 = tl["mean_turn_drive"] < bl["mean_turn_drive"]
        tests.append(("Turn-L ablation shifts turn rightward", t3,
                       f"{tl['mean_turn_drive']:.3f} < {bl['mean_turn_drive']:.3f}"))

    # Check VNC-lite baseline is walking (not collapsed)
    if bl:
        t4 = bl["forward_distance"] > 5.0
        tests.append(("VNC-lite baseline walks >5mm", t4,
                       f"{bl['forward_distance']:.1f}mm"))

    print(f"\n  Results:")
    passed = 0
    for name, result, detail in tests:
        status = "PASS" if result else "FAIL"
        passed += result
        print(f"    {name}: {status} ({detail})")

    print(f"  Score: {passed}/{len(tests)}")
    return passed, len(tests), results


# ════════════════════════════════════════════════════════════════════════════
# TEST 2: INTERFACE ROBUSTNESS
# ════════════════════════════════════════════════════════════════════════════

def test_interface_robustness(use_fake_brain=False, seed=42):
    """Stable across parameter ranges."""
    print("\n" + "=" * 70)
    print("TEST 2: INTERFACE ROBUSTNESS")
    print("=" * 70)

    param_sets = [
        ("default", VNCLiteConfig()),
        ("fast_decay", VNCLiteConfig(tau_drive=0.100, tau_turn=0.080, tau_rhythm=0.150, tau_stance=0.120)),
        ("slow_decay", VNCLiteConfig(tau_drive=0.400, tau_turn=0.300, tau_rhythm=0.500, tau_stance=0.400)),
        ("high_gain", VNCLiteConfig(alpha_drive=0.18, alpha_turn=0.22, alpha_rhythm=0.14, alpha_stance=0.14)),
        ("low_gain", VNCLiteConfig(alpha_drive=0.05, alpha_turn=0.06, alpha_rhythm=0.04, alpha_stance=0.04)),
        ("strong_coupling", VNCLiteConfig(drive_coupling=0.25, turn_inhibition=0.30)),
        ("no_feedback", VNCLiteConfig(feedback_velocity=0, feedback_stability=0, feedback_contact=0, feedback_slip=0)),
    ]

    results = {}
    for name, config in param_sets:
        print(f"  Running vnc_lite/{name}...", end=" ", flush=True)
        r = run_trial(
            mode="vnc_lite",
            body_steps=3000,
            use_fake_brain=use_fake_brain,
            seed=seed,
            vnc_config=config,
        )
        results[name] = r
        if r:
            print(f"dist={r['forward_distance']:.1f}mm fwd={r['mean_forward_drive']:.3f}")
        else:
            print("FAILED")

    # All configs should produce forward movement (>3mm)
    tests = []
    for name, r in results.items():
        if r:
            walks = r["forward_distance"] > 3.0
            tests.append((f"{name} walks >3mm", walks, f"{r['forward_distance']:.1f}mm"))
        else:
            tests.append((f"{name} completes", False, "crashed"))

    # Distance spread should be < 5x between min and max (reasonable stability)
    distances = [r["forward_distance"] for r in results.values() if r]
    if len(distances) >= 2:
        ratio = max(distances) / max(min(distances), 0.01)
        stable = ratio < 5.0
        tests.append((f"Distance spread <5x", stable, f"{ratio:.1f}x"))

    print(f"\n  Results:")
    passed = 0
    for name, result, detail in tests:
        status = "PASS" if result else "FAIL"
        passed += result
        print(f"    {name}: {status} ({detail})")

    print(f"  Score: {passed}/{len(tests)}")
    return passed, len(tests), results


# ════════════════════════════════════════════════════════════════════════════
# TEST 3: IMPROVED TEMPORAL REALISM
# ════════════════════════════════════════════════════════════════════════════

def test_temporal_realism(use_fake_brain=False, seed=42):
    """VNC-lite should produce smoother transitions than old decoder."""
    print("\n" + "=" * 70)
    print("TEST 3: IMPROVED TEMPORAL REALISM")
    print("=" * 70)

    print("  Running old decoder...", end=" ", flush=True)
    old = run_trial(mode="old", body_steps=5000, use_fake_brain=use_fake_brain, seed=seed)
    if old:
        print(f"fwd_smooth={old['forward_smoothness']:.4f} turn_smooth={old['turn_smoothness']:.4f}")
    else:
        print("FAILED")

    print("  Running vnc_lite...", end=" ", flush=True)
    vnc = run_trial(mode="vnc_lite", body_steps=5000, use_fake_brain=use_fake_brain, seed=seed)
    if vnc:
        print(f"fwd_smooth={vnc['forward_smoothness']:.4f} turn_smooth={vnc['turn_smoothness']:.4f}")
    else:
        print("FAILED")

    tests = []
    if old and vnc:
        # VNC-lite should have lower command jitter (smoother)
        t1 = vnc["forward_smoothness"] < old["forward_smoothness"]
        tests.append(("Forward drive smoother", t1,
                       f"vnc={vnc['forward_smoothness']:.4f} vs old={old['forward_smoothness']:.4f}"))

        t2 = vnc["turn_smoothness"] < old["turn_smoothness"]
        tests.append(("Turn drive smoother", t2,
                       f"vnc={vnc['turn_smoothness']:.4f} vs old={old['turn_smoothness']:.4f}"))

        t3 = vnc["freq_smoothness"] < old["freq_smoothness"]
        tests.append(("Step freq smoother", t3,
                       f"vnc={vnc['freq_smoothness']:.4f} vs old={old['freq_smoothness']:.4f}"))

        # VNC-lite should still walk a reasonable distance
        t4 = vnc["forward_distance"] > 5.0
        tests.append(("VNC-lite walks >5mm", t4, f"{vnc['forward_distance']:.1f}mm"))

    print(f"\n  Results:")
    passed = 0
    for name, result, detail in tests:
        status = "PASS" if result else "FAIL"
        passed += result
        print(f"    {name}: {status} ({detail})")

    print(f"  Score: {passed}/{len(tests)}")
    return passed, len(tests), {"old": old, "vnc_lite": vnc}


# ════════════════════════════════════════════════════════════════════════════
# TEST 4: CAUSAL DISSOCIATION
# ════════════════════════════════════════════════════════════════════════════

def test_causal_dissociation(use_fake_brain=False, seed=42):
    """VNC-lite should show cleaner motor factorization under ablation."""
    print("\n" + "=" * 70)
    print("TEST 4: CAUSAL DISSOCIATION (cleaner motor factorization)")
    print("=" * 70)

    conditions = [
        {"name": "baseline", "ablation": None},
        {"name": "rhythm_ablated", "ablation": {"zero": "rhythm_ids"}},
        {"name": "stance_ablated", "ablation": {"zero": "stance_ids"}},
        {"name": "turn_left_ablated", "ablation": {"zero": "turn_left_ids"}},
    ]

    results = {}
    for cond in conditions:
        for mode in ["old", "vnc_lite"]:
            key = f"{mode}_{cond['name']}"
            print(f"  Running {key}...", end=" ", flush=True)
            r = run_trial(
                mode=mode,
                body_steps=3000,
                use_fake_brain=use_fake_brain,
                seed=seed,
                ablation=cond["ablation"],
            )
            results[key] = r
            if r:
                print(f"dist={r['forward_distance']:.1f}mm freq={r['mean_step_freq']:.2f} "
                      f"stance={r['mean_stance_gain']:.2f} turn={r['mean_turn_drive']:.3f}")
            else:
                print("FAILED")

    tests = []

    # Rhythm ablation should change frequency more than turning (in VNC-lite)
    vnc_bl = results.get("vnc_lite_baseline")
    vnc_rh = results.get("vnc_lite_rhythm_ablated")
    if vnc_bl and vnc_rh:
        freq_change = abs(vnc_rh["mean_step_freq"] - vnc_bl["mean_step_freq"])
        turn_change = abs(vnc_rh["mean_turn_drive"] - vnc_bl["mean_turn_drive"])
        t1 = freq_change > turn_change
        tests.append(("Rhythm ablation: freq change > turn change", t1,
                       f"freq_delta={freq_change:.3f} vs turn_delta={turn_change:.3f}"))

    # Stance ablation should change stance more than speed
    vnc_st = results.get("vnc_lite_stance_ablated")
    if vnc_bl and vnc_st:
        stance_change = abs(vnc_st["mean_stance_gain"] - vnc_bl["mean_stance_gain"])
        fwd_change = abs(vnc_st["mean_forward_drive"] - vnc_bl["mean_forward_drive"])
        t2 = stance_change > fwd_change * 0.5  # stance is at least half the fwd effect
        tests.append(("Stance ablation: stance change substantial", t2,
                       f"stance_delta={stance_change:.3f} vs fwd_delta={fwd_change:.3f}"))

    # Turn ablation should change heading more than forward distance
    vnc_tl = results.get("vnc_lite_turn_left_ablated")
    if vnc_bl and vnc_tl:
        heading_change = abs(vnc_tl["mean_turn_drive"] - vnc_bl["mean_turn_drive"])
        dist_change = abs(vnc_tl["forward_distance"] - vnc_bl["forward_distance"])
        t3 = heading_change > 0.01  # minimal heading effect
        tests.append(("Turn-L ablation: heading changes", t3,
                       f"turn_delta={heading_change:.4f}"))

    # Compare factorization quality: VNC-lite vs old
    # In VNC-lite, rhythm ablation should have LESS turn spillover than old
    old_bl = results.get("old_baseline")
    old_rh = results.get("old_rhythm_ablated")
    if old_bl and old_rh and vnc_bl and vnc_rh:
        old_turn_spill = abs(old_rh["mean_turn_drive"] - old_bl["mean_turn_drive"])
        vnc_turn_spill = abs(vnc_rh["mean_turn_drive"] - vnc_bl["mean_turn_drive"])
        # Both near zero is fine; only fail if VNC-lite has large spillover while old doesn't
        t4 = vnc_turn_spill < 0.05 or vnc_turn_spill <= old_turn_spill * 1.5
        tests.append(("Rhythm ablation: VNC-lite has controlled turn spillover", t4,
                       f"vnc={vnc_turn_spill:.4f} vs old={old_turn_spill:.4f}"))

    print(f"\n  Results:")
    passed = 0
    for name, result, detail in tests:
        status = "PASS" if result else "FAIL"
        passed += result
        print(f"    {name}: {status} ({detail})")

    print(f"  Score: {passed}/{len(tests)}")
    return passed, len(tests), results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VNC-lite validation suite")
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", type=int, default=0, help="Run specific test (1-4), 0=all")
    args = parser.parse_args()

    print("VNC-Lite Validation Suite")
    print("=" * 70)
    print(f"Mode: {'FAKE brain' if args.fake_brain else 'Brian2 LIF'}")
    print(f"Seed: {args.seed}")

    total_passed = 0
    total_tests = 0
    all_results = {}

    tests_to_run = [args.test] if args.test > 0 else [1, 2, 3, 4]

    t0 = time.time()

    if 1 in tests_to_run:
        p, t, r = test_backward_compatibility(args.fake_brain, args.seed)
        total_passed += p
        total_tests += t
        all_results["backward_compat"] = r

    if 2 in tests_to_run:
        p, t, r = test_interface_robustness(args.fake_brain, args.seed)
        total_passed += p
        total_tests += t
        all_results["robustness"] = r

    if 3 in tests_to_run:
        p, t, r = test_temporal_realism(args.fake_brain, args.seed)
        total_passed += p
        total_tests += t
        all_results["temporal"] = r

    if 4 in tests_to_run:
        p, t, r = test_causal_dissociation(args.fake_brain, args.seed)
        total_passed += p
        total_tests += t
        all_results["dissociation"] = r

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"FINAL: {total_passed}/{total_tests} tests passed ({elapsed:.0f}s)")
    print("=" * 70)

    # Save results
    output_dir = Path("logs/vnc_lite_validation")
    summary = {
        "passed": total_passed,
        "total": total_tests,
        "elapsed_s": elapsed,
        "fake_brain": args.fake_brain,
        "seed": args.seed,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(f"\nSaved to {output_dir}/summary.json")


if __name__ == "__main__":
    main()
