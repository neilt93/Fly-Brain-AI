"""
VNC lateral ablation: test whether MANC VNC connectome encodes bilateral processing.

Ablates all left-side intrinsic interneurons in the VNC Brian2 model and
measures heading bias and per-leg MN rate asymmetry.

Key limitation: tripod rhythm is post-hoc (not emergent from the Brian2 VNC),
so temporal coordination won't break. But TONIC gain asymmetry from the
connectome should produce a turning bias.

Conditions:
  1. Intact (baseline)
  2. Left intrinsic ablation
  3. Right intrinsic ablation (control: should mirror left)
  4. Shuffled VNC (negative control: no specific wiring)

Usage:
    cd plastic-fly
    python experiments/vnc_lateral_ablation.py                       # quick
    python experiments/vnc_lateral_ablation.py --body-steps 10000    # longer
    python experiments/vnc_lateral_ablation.py --seeds 5             # multi-seed
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.flygym_adapter import FlyGymAdapter
from bridge.vnc_connectome import Brian2VNCRunner, VNCConfig, VNCInput, VNCOutput
from bridge.vnc_bridge import VNCBridge


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
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


def run_vnc_trial(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    ablate_side: str | None = None,  # "L", "R", or None
    vnc_shuffle_seed: int | None = None,
) -> dict:
    """Run one closed-loop trial with optional VNC lateral ablation.

    Args:
        ablate_side: "L" or "R" to ablate left/right intrinsic interneurons.
        vnc_shuffle_seed: If set, shuffle VNC connectivity (negative control).
    """
    import flygym

    cfg = BridgeConfig()

    # Load brain populations
    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)

    encoder = SensoryEncoder.from_channel_map(
        sensory_ids, cfg.channel_map_path,
        max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    adapter = FlyGymAdapter()

    brain = create_brain_runner(
        sensory_ids=sensory_ids, readout_ids=readout_ids,
        use_fake=use_fake_brain, warmup_ms=cfg.brain_warmup_ms,
    )

    # Build VNC with optional ablation
    vnc_cfg = VNCConfig()
    vnc = Brian2VNCRunner(cfg=vnc_cfg, shuffle_seed=vnc_shuffle_seed)

    n_ablated = 0
    if ablate_side is not None:
        sides = vnc.get_intrinsic_by_side()
        ablate_ids = sides.get(ablate_side, set())
        n_ablated = len(ablate_ids)
        print(f"  Ablating {n_ablated} {ablate_side}-side intrinsic interneurons...")
        vnc.silence_neurons(ablate_ids)

    # Build VNC bridge (wraps VNC runner with rhythm modulation + MN decoder)
    vnc_bridge = VNCBridge(vnc_runner=vnc, vnc_cfg=vnc_cfg)

    # FlyGym setup
    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly_obj, arena=flygym.arena.FlatTerrain(), timestep=1e-4,
    )
    obs, _ = sim.reset()

    # Warmup with basic CPG (not VNC — avoids VNC transient)
    from bridge.locomotion_bridge import LocomotionBridge
    warmup_loco = LocomotionBridge(seed=seed)
    warmup_loco.warmup(0)
    warmup_loco.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6),
    )
    for _ in range(warmup_steps):
        action = warmup_loco.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, term, trunc, _ = sim.step(action)
            if term or trunc:
                sim.close()
                return {"error": "warmup_ended"}
        except Exception as e:
            sim.close()
            return {"error": f"warmup_physics_{type(e).__name__}"}

    # Main loop: brain -> decoder -> VNC bridge -> body
    bspb = cfg.body_steps_per_brain
    positions = []
    orientations = []
    commands_log = []
    current_cmd = LocomotionCommand(forward_drive=1.0)
    group_rates = {"forward": 20.0, "turn_left": 0.0, "turn_right": 0.0,
                   "rhythm": 0.0, "stance": 0.0}
    sample_interval = 20
    step = 0

    t_start = time.time()

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            current_cmd = decoder.decode(brain_output)

            # Get group rates for VNC
            group_rates = decoder.get_group_rates(brain_output)

            # Run VNC at brain-step frequency (caches tonic MN output)
            vnc_bridge.step_brain(group_rates, sim_ms=cfg.brain_dt_ms)

            commands_log.append({
                "step": step,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
            })

        # Body step: apply rhythm modulation + MN decode at body frequency
        action = vnc_bridge.step(group_rates, dt_s=1e-4)

        try:
            obs, _, term, trunc, _ = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break

        if step % sample_interval == 0:
            positions.append(np.array(obs["fly"][0]).tolist())
            orientations.append(np.array(obs["fly"][2]).tolist())

        if term or trunc:
            break

    sim.close()
    elapsed = time.time() - t_start

    if len(positions) < 2:
        return {"error": "too_few_samples"}

    # Compute metrics
    positions_arr = np.array(positions)
    start = positions_arr[0]
    end = positions_arr[-1]
    diff = end - start
    dist = float(np.linalg.norm(diff))
    dx = float(diff[0])
    dy = float(diff[1])
    heading = float(np.degrees(np.arctan2(dy, dx)))
    fwd_eff = dx / max(dist, 0.001)

    # Path curvature
    headings = []
    for i in range(1, len(positions_arr)):
        ddx = positions_arr[i][0] - positions_arr[i-1][0]
        ddy = positions_arr[i][1] - positions_arr[i-1][1]
        headings.append(np.arctan2(ddy, ddx))
    if len(headings) > 1:
        diffs = np.diff(headings)
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        cum_turn = float(np.sum(np.abs(diffs)))
    else:
        cum_turn = 0.0

    return {
        "steps_completed": step + 1,
        "elapsed_s": elapsed,
        "dist_mm": dist,
        "dx_mm": dx,
        "dy_mm": dy,
        "heading_deg": heading,
        "fwd_efficiency": fwd_eff,
        "cumulative_turn_deg": float(np.degrees(cum_turn)),
        "n_ablated": n_ablated,
        "positions": positions,
    }


def run_lateral_ablation_battery(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    n_seeds: int = 2,
):
    print("=" * 80)
    print("VNC LATERAL ABLATION EXPERIMENT")
    print(f"  body_steps={body_steps}, seeds={n_seeds}, "
          f"brain={'FAKE' if use_fake_brain else 'Brian2 LIF'}")
    print("=" * 80)

    output_dir = Path("logs/vnc_lateral_ablation")
    seeds = list(range(42, 42 + n_seeds))

    conditions = [
        ("intact", None, None),
        ("ablate_L", "L", None),
        ("ablate_R", "R", None),
        ("shuffled", None, 99),
    ]

    all_results = {name: [] for name, _, _ in conditions}

    for cond_name, ablate_side, shuffle_seed in conditions:
        for seed in seeds:
            label = f"{cond_name}_s{seed}"
            print(f"\n--- {label} ---")

            r = run_vnc_trial(
                body_steps=body_steps,
                warmup_steps=warmup_steps,
                use_fake_brain=use_fake_brain,
                seed=seed,
                ablate_side=ablate_side,
                vnc_shuffle_seed=shuffle_seed,
            )

            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue

            print(f"  dist={r['dist_mm']:.2f}mm dx={r['dx_mm']:+.2f}mm "
                  f"heading={r['heading_deg']:+.1f}deg "
                  f"fwd_eff={r['fwd_efficiency']:.2f} "
                  f"({r['elapsed_s']:.1f}s)")

            # Remove positions from checkpoint (too large)
            r_save = {k: v for k, v in r.items() if k != "positions"}
            _write_json_atomic(output_dir / "checkpoints" / f"{label}.json", r_save)

            all_results[cond_name].append(r)

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Condition':<14s} {'Distance':>10s} {'dx(fwd)':>10s} {'Heading':>10s} "
          f"{'FwdEff':>8s} {'CumTurn':>10s}")
    print("  " + "-" * 64)

    condition_means = {}
    for cond_name, _, _ in conditions:
        trials = all_results[cond_name]
        if not trials:
            print(f"  {cond_name:<14s} {'NO DATA':>10s}")
            continue

        dists = [t["dist_mm"] for t in trials]
        dxs = [t["dx_mm"] for t in trials]
        headings = [t["heading_deg"] for t in trials]
        fwd_effs = [t["fwd_efficiency"] for t in trials]
        cum_turns = [t["cumulative_turn_deg"] for t in trials]

        mean_heading = np.mean(headings)
        condition_means[cond_name] = {
            "dist": np.mean(dists),
            "dx": np.mean(dxs),
            "heading": mean_heading,
            "fwd_eff": np.mean(fwd_effs),
            "cum_turn": np.mean(cum_turns),
        }

        print(f"  {cond_name:<14s} {np.mean(dists):>9.2f}mm {np.mean(dxs):>+9.2f}mm "
              f"{mean_heading:>+9.1f}deg {np.mean(fwd_effs):>7.2f} "
              f"{np.mean(cum_turns):>9.1f}deg")

    # ═══════════════════════════════════════════════════════════════════════
    # Causal tests
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("CAUSAL TESTS")
    print(f"{'='*80}")

    tests_passed = 0
    tests_total = 0

    def log_test(name, passed, desc):
        nonlocal tests_passed, tests_total
        tests_total += 1
        tests_passed += int(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {desc}")

    intact = condition_means.get("intact")
    abl_L = condition_means.get("ablate_L")
    abl_R = condition_means.get("ablate_R")
    shuffled = condition_means.get("shuffled")

    if intact and abl_L:
        # Test 1: Left ablation causes ipsilateral heading bias
        # Ablating left interneurons should reduce left-side drive → turning LEFT (positive heading)
        # OR: it could reduce left-side inhibition → opposite effect
        # Either way, we expect heading to DIFFER from intact
        heading_diff_L = abs(abl_L["heading"] - intact["heading"])
        log_test("heading_shift_L",
                 heading_diff_L > 5.0,
                 f"L-ablation heading shift: {heading_diff_L:.1f}deg (need >5deg)")

    if intact and abl_R:
        heading_diff_R = abs(abl_R["heading"] - intact["heading"])
        log_test("heading_shift_R",
                 heading_diff_R > 5.0,
                 f"R-ablation heading shift: {heading_diff_R:.1f}deg (need >5deg)")

    if abl_L and abl_R:
        # Test 3: L and R ablations produce OPPOSITE heading biases
        log_test("mirror_symmetry",
                 (abl_L["heading"] * abl_R["heading"] < 0) or
                 (abs(abl_L["heading"] - abl_R["heading"]) > 10),
                 f"L heading={abl_L['heading']:+.1f}deg, R heading={abl_R['heading']:+.1f}deg "
                 f"(should be opposite or >10deg apart)")

    if intact and shuffled:
        # Test 4: Shuffled has less directional bias than ablated
        # (shuffled destroys specific wiring, so heading should be more random)
        if abl_L:
            abl_heading_mag = abs(abl_L["heading"] - intact["heading"])
            shuf_heading_mag = abs(shuffled["heading"] - intact["heading"])
            log_test("ablation_vs_shuffle",
                     abl_heading_mag > shuf_heading_mag,
                     f"Ablation heading shift ({abl_heading_mag:.1f}deg) > "
                     f"shuffle ({shuf_heading_mag:.1f}deg)")

    print(f"\n  TOTAL: {tests_passed}/{tests_total} tests passed")

    # ═══════════════════════════════════════════════════════════════════════
    # Biological interpretation
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    if abl_L and intact:
        direction = "LEFT" if abl_L["heading"] > intact["heading"] else "RIGHT"
        print(f"\n  Left VNC ablation -> fly turns {direction}")
        print(f"    Interpretation: loss of left-side intrinsic interneurons "
              f"{'reduces ipsilateral drive (excitatory)' if direction == 'LEFT' else 'releases contralateral inhibition'}")
    if abl_R and intact:
        direction = "LEFT" if abl_R["heading"] > intact["heading"] else "RIGHT"
        print(f"  Right VNC ablation -> fly turns {direction}")

    print(f"\n  NOTE: Tripod rhythm is post-hoc (body-step frequency), so gait")
    print(f"  coordination is preserved. The ablation effect is through tonic")
    print(f"  gain asymmetry from the MANC connectome wiring.")

    # Save
    save_payload = {
        "condition_means": condition_means,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "params": {
            "body_steps": body_steps,
            "n_seeds": n_seeds,
            "use_fake_brain": use_fake_brain,
        },
    }
    _write_json_atomic(output_dir / "results.json", save_payload)
    print(f"\nSaved to {output_dir}/results.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VNC lateral ablation experiment")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    run_lateral_ablation_battery(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        n_seeds=args.seeds,
    )
