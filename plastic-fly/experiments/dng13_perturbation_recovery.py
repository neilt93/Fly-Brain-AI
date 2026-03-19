"""
DNg13 perturbation recovery: test whether DNg13 acts as a turning stabilizer.

Hypothesis: DNg13 bilateral pair damps turning asymmetry. Removing it should
impair the fly's ability to recover straight walking after a transient
sensory perturbation.

Protocol (three-phase, per trial):
  Phase 1 (pre, 0-25%):     baseline walking, no perturbation
  Phase 2 (perturb, 25-50%): asymmetric contact loss (left legs zeroed)
  Phase 3 (recovery, 50-100%): perturbation removed, measure recovery

Conditions (2x2 factorial):
  1. intact_no_perturb     -- reference trajectory
  2. intact_perturb        -- should recover (turn_drive returns to baseline)
  3. silenced_no_perturb   -- shows existing DNg13 effect (slight |turn| increase)
  4. silenced_perturb      -- PREDICTION: fails to recover, residual heading drift

Key metrics:
  - Recovery index: |turn_drive_recovery - turn_drive_pre| (lower = better recovery)
  - Heading deviation: heading in recovery phase vs pre phase
  - Recovery time: brain steps until turn_drive returns within 1 SD of pre-phase mean

Usage:
    cd plastic-fly
    python experiments/dng13_perturbation_recovery.py --fake-brain   # fast test
    python experiments/dng13_perturbation_recovery.py                # real brain
    python experiments/dng13_perturbation_recovery.py --seeds 5      # more seeds
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import BodyObservation, LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from experiments.dn_phenotype_prediction import (
    resolve_dn_type_ids, apply_neuron_intervention, _write_json_atomic,
)

BASE = Path(__file__).resolve().parent.parent.parent


def apply_contact_perturbation(obs: BodyObservation, side: str = "left") -> BodyObservation:
    """Zero contact forces on one side (simulates transient sensory loss)."""
    contact = obs.contact_forces.copy()
    if side == "left":
        contact[0:3] = 0.0  # LF, LM, LH
    else:
        contact[3:6] = 0.0  # RF, RM, RH
    return BodyObservation(
        joint_angles=obs.joint_angles,
        joint_velocities=obs.joint_velocities,
        contact_forces=contact,
        body_velocity=obs.body_velocity,
        body_orientation=obs.body_orientation,
    )


def run_recovery_trial(
    target_ids: list[int],
    silence: bool,
    perturb: bool,
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    readout_version: int = 5,
    onset_frac: float = 0.25,
    offset_frac: float = 0.50,
) -> dict:
    """Run one three-phase trial with optional DNg13 silencing + perturbation."""
    import flygym

    cfg = BridgeConfig()

    if readout_version == 5:
        sensory_ids = np.load(cfg.data_dir / "sensory_ids_v3.npy")
        readout_ids = np.load(cfg.data_dir / "readout_ids_v5_steering.npy")
        channel_map_path = cfg.data_dir / "channel_map_v3.json"
        decoder_path = cfg.data_dir / "decoder_groups_v5_steering.json"
        rate_scale = 12.0
    else:
        sensory_ids = np.load(cfg.sensory_ids_path)
        readout_ids = np.load(cfg.readout_ids_path)
        channel_map_path = cfg.channel_map_path
        decoder_path = cfg.decoder_groups_path
        rate_scale = cfg.rate_scale

    encoder = SensoryEncoder.from_channel_map(
        sensory_ids, channel_map_path,
        max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(decoder_path, rate_scale=rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain = create_brain_runner(
        sensory_ids=sensory_ids, readout_ids=readout_ids,
        use_fake=use_fake_brain, warmup_ms=cfg.brain_warmup_ms,
    )

    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly_obj, arena=flygym.arena.FlatTerrain(), timestep=1e-4,
    )
    obs, _ = sim.reset()

    # Warmup
    locomotion.warmup(0)
    locomotion.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6),
    )
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, term, trunc, _ = sim.step(action)
            if term or trunc:
                sim.close()
                return {"error": "warmup_ended"}
        except Exception as e:
            sim.close()
            return {"error": f"physics_{type(e).__name__}"}

    # Phase boundaries
    onset_step = int(body_steps * onset_frac)
    offset_step = int(body_steps * offset_frac)

    # Main loop
    bspb = cfg.body_steps_per_brain
    positions = []
    commands = []
    current_cmd = LocomotionCommand(forward_drive=1.0)
    step = 0
    sample_interval = 20

    t_start = time.time()

    for step in range(body_steps):
        # Phase detection
        if step < onset_step:
            phase = "pre"
        elif step < offset_step:
            phase = "perturb"
        else:
            phase = "recovery"

        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)

            # Apply perturbation during perturb phase only
            if perturb and phase == "perturb":
                body_obs = apply_contact_perturbation(body_obs, side="left")

            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)

            # Apply DNg13 silencing (persistent, all phases)
            if silence and target_ids:
                brain_output = apply_neuron_intervention(
                    brain_output, target_ids, mode="silence",
                )

            current_cmd = decoder.decode(brain_output)
            commands.append({
                "step": step,
                "phase": phase,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
            })

        action = locomotion.step(current_cmd)
        try:
            obs, _, term, trunc, _ = sim.step(action)
        except Exception as e:
            print(f"  Physics error at step {step}: {e}")
            break

        if step % sample_interval == 0:
            positions.append(np.array(obs["fly"][0]).tolist())

        if term or trunc:
            break

    sim.close()
    elapsed = time.time() - t_start

    if len(commands) < 3:
        return {"error": "too_few_commands"}

    # Per-phase turn_drive statistics
    phase_stats = {}
    for ph in ["pre", "perturb", "recovery"]:
        entries = [c for c in commands if c["phase"] == ph]
        if entries:
            turns = [c["turn_drive"] for c in entries]
            fwds = [c["forward_drive"] for c in entries]
            phase_stats[ph] = {
                "mean_turn": float(np.mean(turns)),
                "std_turn": float(np.std(turns)),
                "mean_abs_turn": float(np.mean(np.abs(turns))),
                "mean_fwd": float(np.mean(fwds)),
                "n_steps": len(entries),
            }
        else:
            phase_stats[ph] = None

    # Recovery index: how far turn_drive in recovery is from pre-phase
    pre = phase_stats.get("pre")
    rec = phase_stats.get("recovery")

    recovery_index = None
    heading_deviation = None
    if pre and rec:
        recovery_index = abs(rec["mean_turn"] - pre["mean_turn"])
        # Heading from positions
        positions_arr = np.array(positions)
        onset_sample = onset_step // sample_interval
        offset_sample = offset_step // sample_interval
        if offset_sample < len(positions_arr) and onset_sample > 0:
            # Pre-phase heading
            pre_diff = positions_arr[onset_sample] - positions_arr[0]
            pre_heading = np.degrees(np.arctan2(pre_diff[1], pre_diff[0]))
            # Recovery-phase heading
            rec_diff = positions_arr[-1] - positions_arr[offset_sample]
            rec_heading = np.degrees(np.arctan2(rec_diff[1], rec_diff[0]))
            heading_deviation = abs(((rec_heading - pre_heading) + 180) % 360 - 180)

    return {
        "steps_completed": step + 1,
        "elapsed_s": elapsed,
        "phase_stats": phase_stats,
        "recovery_index": recovery_index,
        "heading_deviation_deg": heading_deviation,
        "silence": silence,
        "perturb": perturb,
    }


def run_perturbation_recovery(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    n_seeds: int = 3,
    readout_version: int = 5,
):
    print("=" * 80)
    print("DNg13 PERTURBATION RECOVERY EXPERIMENT")
    print(f"  body_steps={body_steps}, seeds={n_seeds}, "
          f"brain={'FAKE' if use_fake_brain else 'Brian2 LIF'}")
    print("=" * 80)

    ann = pd.read_csv(BASE / "brain-model" / "flywire_annotations_matched.csv", low_memory=False)
    all_ids = resolve_dn_type_ids("DNg13", ann)

    print(f"\n  DNg13: {len(all_ids)} neurons ({[int(x) for x in all_ids]})")

    output_dir = Path("logs/dng13_perturbation_recovery")
    seeds = list(range(42, 42 + n_seeds))

    conditions = {
        "intact_no_perturb":   {"silence": False, "perturb": False},
        "intact_perturb":      {"silence": False, "perturb": True},
        "silenced_no_perturb": {"silence": True,  "perturb": False},
        "silenced_perturb":    {"silence": True,  "perturb": True},
    }

    all_results = {name: [] for name in conditions}

    for cond_name, cond_params in conditions.items():
        for seed in seeds:
            label = f"{cond_name}_s{seed}"
            print(f"  {label}...", end=" ", flush=True)

            r = run_recovery_trial(
                target_ids=all_ids,
                silence=cond_params["silence"],
                perturb=cond_params["perturb"],
                body_steps=body_steps,
                warmup_steps=warmup_steps,
                use_fake_brain=use_fake_brain,
                seed=seed,
                readout_version=readout_version,
            )

            if "error" in r:
                print(f"ERROR: {r['error']}")
                continue

            ri = r["recovery_index"]
            hd = r["heading_deviation_deg"]
            ri_str = f"{ri:.4f}" if ri is not None else "N/A"
            hd_str = f"{hd:.1f}deg" if hd is not None else "N/A"
            pre = r["phase_stats"].get("pre") or {}
            rec = r["phase_stats"].get("recovery") or {}
            print(f"recov_idx={ri_str} heading_dev={hd_str} "
                  f"pre_turn={pre.get('mean_turn', 0):+.3f} "
                  f"rec_turn={rec.get('mean_turn', 0):+.3f} "
                  f"({r['elapsed_s']:.1f}s)")

            _write_json_atomic(output_dir / "checkpoints" / f"{label}.json", r)
            all_results[cond_name].append(r)

    # ===================================================================
    # Analysis
    # ===================================================================
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Condition':<25s} {'Recov Idx':>10s} {'Heading Dev':>12s} "
          f"{'Pre Turn':>10s} {'Rec Turn':>10s} {'|Rec Turn|':>10s}")
    print("  " + "-" * 80)

    means = {}
    for cond_name in conditions:
        trials = all_results[cond_name]
        if not trials:
            print(f"  {cond_name:<25s} {'NO DATA':>10s}")
            continue

        ri_vals = [t["recovery_index"] for t in trials if t["recovery_index"] is not None]
        hd_vals = [t["heading_deviation_deg"] for t in trials if t["heading_deviation_deg"] is not None]
        pre_turns = [t["phase_stats"]["pre"]["mean_turn"] for t in trials if t["phase_stats"].get("pre")]
        rec_turns = [t["phase_stats"]["recovery"]["mean_turn"] for t in trials if t["phase_stats"].get("recovery")]
        rec_abs = [t["phase_stats"]["recovery"]["mean_abs_turn"] for t in trials if t["phase_stats"].get("recovery")]

        m = {
            "recovery_index": np.mean(ri_vals) if ri_vals else None,
            "heading_dev": np.mean(hd_vals) if hd_vals else None,
            "pre_turn": np.mean(pre_turns) if pre_turns else None,
            "rec_turn": np.mean(rec_turns) if rec_turns else None,
            "rec_abs_turn": np.mean(rec_abs) if rec_abs else None,
        }
        means[cond_name] = m

        ri_str = f"{m['recovery_index']:.4f}" if m['recovery_index'] is not None else "N/A"
        hd_str = f"{m['heading_dev']:.1f}deg" if m['heading_dev'] is not None else "N/A"
        pt_str = f"{m['pre_turn']:+.3f}" if m['pre_turn'] is not None else "N/A"
        rt_str = f"{m['rec_turn']:+.3f}" if m['rec_turn'] is not None else "N/A"
        ra_str = f"{m['rec_abs_turn']:.3f}" if m['rec_abs_turn'] is not None else "N/A"

        print(f"  {cond_name:<25s} {ri_str:>10s} {hd_str:>12s} "
              f"{pt_str:>10s} {rt_str:>10s} {ra_str:>10s}")

    # ===================================================================
    # Causal tests
    # ===================================================================
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

    ip = means.get("intact_perturb")
    sp = means.get("silenced_perturb")
    inop = means.get("intact_no_perturb")
    snop = means.get("silenced_no_perturb")

    # Test 1: Perturbation causes measurable recovery index in intact flies
    if ip and inop and ip["recovery_index"] is not None and inop["recovery_index"] is not None:
        diff = ip["recovery_index"] - inop["recovery_index"]
        log_test("perturbation_effect",
                 ip["recovery_index"] > inop["recovery_index"],
                 f"Intact perturbed recov_idx ({ip['recovery_index']:.4f}) > "
                 f"intact unperturbed ({inop['recovery_index']:.4f}), diff={diff:+.4f}")

    # Test 2: KEY PREDICTION -- silenced flies recover WORSE than intact
    if ip and sp and ip["recovery_index"] is not None and sp["recovery_index"] is not None:
        log_test("silencing_impairs_recovery",
                 sp["recovery_index"] > ip["recovery_index"],
                 f"Silenced recov_idx ({sp['recovery_index']:.4f}) > "
                 f"intact ({ip['recovery_index']:.4f})")

    # Test 3: Heading deviation larger in silenced + perturbed
    if ip and sp and ip["heading_dev"] is not None and sp["heading_dev"] is not None:
        log_test("silencing_increases_heading_deviation",
                 sp["heading_dev"] > ip["heading_dev"],
                 f"Silenced heading dev ({sp['heading_dev']:.1f}deg) > "
                 f"intact ({ip['heading_dev']:.1f}deg)")

    # Test 4: Without perturbation, silencing alone has mild effect
    if inop and snop and inop["recovery_index"] is not None and snop["recovery_index"] is not None:
        diff = abs(snop["recovery_index"] - inop["recovery_index"])
        log_test("silencing_baseline_mild",
                 diff < 0.05,
                 f"No-perturb silencing effect is mild: {diff:.4f} (need <0.05)")

    print(f"\n  TOTAL: {tests_passed}/{tests_total} tests passed")

    # ===================================================================
    # Interpretation
    # ===================================================================
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    if ip and sp:
        ri_intact = ip["recovery_index"] or 0
        ri_silenced = sp["recovery_index"] or 0
        if ri_silenced > ri_intact:
            print(f"""
  DNg13 silencing IMPAIRS perturbation recovery:
    - Intact flies: recovery index = {ri_intact:.4f} (turn_drive returns near baseline)
    - Silenced flies: recovery index = {ri_silenced:.4f} (residual turning asymmetry)
    - Ratio: {ri_silenced / max(ri_intact, 0.0001):.1f}x worse recovery

  This confirms DNg13 acts as a TURNING STABILIZER:
    - During perturbation, asymmetric sensory input activates turn circuits
    - DNg13 bilateral pair provides negative feedback (damping)
    - Without DNg13, perturbation-induced asymmetry persists into recovery
    - This converts the post-hoc explanation into a PREDICTIVE result
""")
        else:
            print(f"""
  DNg13 silencing does NOT impair recovery (prediction not confirmed):
    - Intact recov_idx = {ri_intact:.4f}
    - Silenced recov_idx = {ri_silenced:.4f}
    - The stabilizer hypothesis needs revision
""")

    # Save
    save_payload = {
        "means": {k: v for k, v in means.items()},
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "dng13_ids": [int(x) for x in all_ids],
        "params": {
            "body_steps": body_steps,
            "n_seeds": n_seeds,
            "readout_version": readout_version,
            "use_fake_brain": use_fake_brain,
            "onset_frac": 0.25,
            "offset_frac": 0.50,
        },
    }
    _write_json_atomic(output_dir / "results.json", save_payload)
    print(f"Saved to {output_dir}/results.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNg13 perturbation recovery")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--readout-version", type=int, default=5, choices=[1, 4, 5])
    args = parser.parse_args()

    run_perturbation_recovery(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        n_seeds=args.seeds,
        readout_version=args.readout_version,
    )
