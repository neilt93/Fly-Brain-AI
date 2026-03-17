"""
Extended ablation experiments:
  1. Dose-response: 25/50/75/100% of forward neurons silenced
  2. Random ablation control: silence same # of random neurons

Usage:
    python experiments/ablation_extended.py --mode dose-response
    python experiments/ablation_extended.py --mode random-control
    python experiments/ablation_extended.py --mode both
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import LocomotionCommand, BrainOutput
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from analysis.behavior_metrics import compute_behavior


def _write_json_atomic(path: Path, payload: dict):
    """Write JSON atomically so checkpoints stay readable if the run is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _record_frame(obs, positions, orientations, contact_forces_log):
    """Extract and append one frame of position, orientation, and contact data."""
    positions.append(np.array(obs["fly"][0]))
    orientations.append(np.array(obs["fly"][2]))
    cf_raw = np.asarray(obs.get("contact_forces", np.zeros((30, 3))))
    magnitudes = np.linalg.norm(cf_raw, axis=1) if cf_raw.ndim == 2 else np.zeros(30)
    per_leg = np.array([magnitudes[i*5:(i+1)*5].max() for i in range(6)])
    contact_forces_log.append(np.clip(per_leg / 10.0, 0.0, 1.0))


def _save_trial_checkpoint(
    output_path, label, steps_completed, body_steps,
    positions, orientations, contact_forces_log, episode_log, status,
):
    """Write an incremental checkpoint for one ablation trial."""
    checkpoint = {
        "label": label,
        "summary": {
            "steps_completed": steps_completed,
            "body_steps": body_steps,
            "status": status,
        },
        "positions": [np.asarray(p).tolist() for p in positions],
        "orientations": [np.asarray(o).tolist() for o in orientations],
        "contact_forces": [np.asarray(c).tolist() for c in contact_forces_log],
        "episode_log": episode_log,
    }
    _write_json_atomic(output_path / f"checkpoint_{label}.json", checkpoint)


def run_trial_with_ablation(
    label, ablate_ids, body_steps, warmup_steps, use_fake_brain, seed,
    decoder_groups, sample_interval=20, readout_version=2,
    checkpoint_dir=None,
):
    """Run one trial, zeroing specified neuron IDs in brain output."""
    import flygym

    cfg = BridgeConfig()

    # Use v2/v3 populations when available
    if readout_version == 3:
        sensory_ids = np.load(cfg.data_dir / "sensory_ids_v3.npy")
        readout_ids = np.load(cfg.data_dir / "readout_ids_v3.npy")
        channel_map_path = cfg.data_dir / "channel_map_v3.json"
        decoder_path = cfg.data_dir / "decoder_groups_v3.json"
        rate_scale = 12.0
    elif readout_version == 2:
        sensory_ids_path = cfg.data_dir / "sensory_ids_v3.npy"
        sensory_ids = np.load(sensory_ids_path) if sensory_ids_path.exists() else np.load(cfg.sensory_ids_path)
        readout_ids = np.load(cfg.data_dir / "readout_ids_v2.npy")
        channel_map_path = cfg.data_dir / "channel_map_v3.json"
        decoder_path = cfg.data_dir / "decoder_groups_v2.json"
        rate_scale = 15.0
    else:
        sensory_ids = np.load(cfg.sensory_ids_path)
        readout_ids = np.load(cfg.readout_ids_path)
        channel_map_path = cfg.channel_map_path
        decoder_path = cfg.decoder_groups_path
        rate_scale = cfg.rate_scale

    if channel_map_path.exists():
        encoder = SensoryEncoder.from_channel_map(
            sensory_ids, channel_map_path,
            max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
        )
    else:
        encoder = SensoryEncoder(sensory_ids, max_rate_hz=cfg.max_rate_hz)

    decoder = DescendingDecoder.from_json(decoder_path, rate_scale=rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain = create_brain_runner(
        sensory_ids=sensory_ids, readout_ids=readout_ids,
        use_fake=use_fake_brain, warmup_ms=cfg.brain_warmup_ms,
    )

    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=flygym.arena.FlatTerrain(), timestep=1e-4)
    obs, info = sim.reset()

    locomotion.warmup(0)
    locomotion.cpg.reset(init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
                         init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        obs, _, terminated, truncated, info = sim.step(action)
        if terminated or truncated:
            sim.close()
            return {"label": label, "error": "warmup_ended"}

    ablate_set = set(int(x) for x in ablate_ids)
    bspb = cfg.body_steps_per_brain
    current_cmd = LocomotionCommand(forward_drive=1.0)
    positions = []
    orientations = []
    contact_forces_log = []
    episode_log = []
    brain_steps = 0
    step = 0
    t0 = time.time()

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)

            # Apply ablation
            if ablate_set:
                rates = brain_output.firing_rates_hz.copy()
                for i, nid in enumerate(brain_output.neuron_ids):
                    if int(nid) in ablate_set:
                        rates[i] = 0.0
                brain_output = BrainOutput(neuron_ids=brain_output.neuron_ids,
                                           firing_rates_hz=rates)

            current_cmd = decoder.decode(brain_output)
            brain_steps += 1

            episode_log.append({
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
            })

        action = locomotion.step(current_cmd)
        try:
            obs, _, terminated, truncated, info = sim.step(action)
        except Exception:
            break

        if step % sample_interval == 0:
            _record_frame(obs, positions, orientations, contact_forces_log)
            if checkpoint_dir is not None:
                _save_trial_checkpoint(
                    Path(checkpoint_dir), label, step + 1, body_steps,
                    positions, orientations, contact_forces_log,
                    episode_log, "running",
                )

        if terminated or truncated:
            break

    sim.close()
    elapsed = time.time() - t0

    behavior = compute_behavior(
        positions=positions, orientations=orientations,
        contact_forces=contact_forces_log,
        steps_completed=step + 1, steps_intended=body_steps,
        sample_dt=sample_interval * 1e-4,
    )

    mean_fwd = float(np.mean([e["forward_drive"] for e in episode_log])) if episode_log else 0
    mean_turn = float(np.mean([e["turn_drive"] for e in episode_log])) if episode_log else 0

    return {
        "label": label,
        "n_ablated": len(ablate_set),
        "forward_distance": behavior.forward_distance,
        "total_path_length": behavior.total_path_length,
        "mean_forward_drive": mean_fwd,
        "mean_turn_drive": mean_turn,
        "elapsed_s": elapsed,
        "brain_steps": brain_steps,
        "behavior_summary": behavior.summary_line(),
    }


def run_dose_response(body_steps=5000, warmup_steps=500, use_fake_brain=False,
                       seed=42, output_dir="logs/dose_response", readout_version=2):
    """Ablate 25/50/75/100% of forward neurons, measure distance drop."""
    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if readout_version == 3:
        dec_path = cfg.data_dir / "decoder_groups_v3.json"
    elif readout_version == 2:
        dec_path = cfg.data_dir / "decoder_groups_v2.json"
    else:
        dec_path = cfg.decoder_groups_path

    with open(dec_path) as f:
        decoder_groups = json.load(f)

    forward_ids = decoder_groups["forward_ids"]
    n_fwd = len(forward_ids)
    rng = np.random.RandomState(seed)

    print("=" * 60)
    print("DOSE-RESPONSE: Forward Neuron Ablation")
    print("Forward group: %d neurons" % n_fwd)
    print("=" * 60)

    results = []

    # Baseline (0% ablation)
    print("\n--- 0%% ablation (baseline) ---")
    r = run_trial_with_ablation("baseline_0pct", [], body_steps, warmup_steps,
                                 use_fake_brain, seed, decoder_groups,
                                 readout_version=readout_version)
    r["dose_pct"] = 0
    results.append(r)
    print("  %s | dist=%.2fmm fwd=%.3f" % (r["label"], r["forward_distance"], r["mean_forward_drive"]))

    for pct in [25, 50, 75, 100]:
        n_ablate = int(n_fwd * pct / 100)
        # Deterministic selection: always ablate first N (sorted by ID)
        ablate = sorted(forward_ids)[:n_ablate]
        print("\n--- %d%% ablation (%d/%d neurons) ---" % (pct, n_ablate, n_fwd))
        r = run_trial_with_ablation("forward_%dpct" % pct, ablate, body_steps,
                                     warmup_steps, use_fake_brain, seed, decoder_groups,
                                     readout_version=readout_version)
        r["dose_pct"] = pct
        results.append(r)
        print("  %s | dist=%.2fmm fwd=%.3f" % (r["label"], r["forward_distance"], r["mean_forward_drive"]))

    # Summary
    print("\n" + "=" * 60)
    print("DOSE-RESPONSE SUMMARY")
    print("=" * 60)
    print("%-20s %10s %10s %10s" % ("Condition", "Dose%", "FwdDist", "FwdDrive"))
    print("-" * 52)
    for r in results:
        print("%-20s %9d%% %9.2fmm %9.3f" % (
            r["label"], r["dose_pct"], r["forward_distance"], r["mean_forward_drive"]))

    # Test: graded response
    dists = [r["forward_distance"] for r in results]
    monotonic = all(abs(dists[i]) >= abs(dists[i+1]) * 0.8 for i in range(len(dists)-1))
    print("\n  [%s] Graded response (monotonic distance drop)" % ("PASS" if monotonic else "FAIL"))

    _write_json_atomic(output_path / "dose_response_results.json", results)
    print("Saved to %s" % (output_path / "dose_response_results.json"))
    return results


def run_random_ablation_control(body_steps=5000, warmup_steps=500, use_fake_brain=False,
                                 seeds=None, output_dir="logs/random_ablation",
                                 readout_version=2):
    """Ablate same # of RANDOM neurons as targeted groups. Should have smaller effect."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if readout_version == 3:
        dec_path = cfg.data_dir / "decoder_groups_v3.json"
        rid_path = cfg.data_dir / "readout_ids_v3.npy"
    elif readout_version == 2:
        dec_path = cfg.data_dir / "decoder_groups_v2.json"
        rid_path = cfg.data_dir / "readout_ids_v2.npy"
    else:
        dec_path = cfg.decoder_groups_path
        rid_path = cfg.readout_ids_path

    with open(dec_path) as f:
        decoder_groups = json.load(f)

    readout_ids = np.load(rid_path)
    all_readout = set(int(x) for x in readout_ids)

    # For each targeted ablation, create a random control
    targeted_groups = {
        "forward": decoder_groups["forward_ids"],
        "turn_left": decoder_groups["turn_left_ids"],
        "turn_right": decoder_groups["turn_right_ids"],
    }

    print("=" * 60)
    print("RANDOM ABLATION CONTROL")
    print("=" * 60)

    all_results = {}

    for group_name, group_ids in targeted_groups.items():
        n_ablate = len(group_ids)
        print("\n### %s group: %d neurons ###" % (group_name, n_ablate))

        # Targeted ablation
        targeted_results = []
        random_results = []

        for seed in seeds:
            print("\n  Seed %d:" % seed)

            # Targeted
            r_targeted = run_trial_with_ablation(
                "targeted_%s_seed%d" % (group_name, seed),
                group_ids, body_steps, warmup_steps, use_fake_brain, seed, decoder_groups,
                readout_version=readout_version)
            targeted_results.append(r_targeted)
            print("    targeted: dist=%.2fmm fwd=%.3f turn=%+.4f" % (
                r_targeted["forward_distance"], r_targeted["mean_forward_drive"],
                r_targeted["mean_turn_drive"]))

            # Random (same count, different neurons each seed)
            rng = np.random.RandomState(seed + 1000)
            candidates = list(all_readout - set(int(x) for x in group_ids))
            random_ids = list(rng.choice(candidates, size=min(n_ablate, len(candidates)),
                                          replace=False))
            r_random = run_trial_with_ablation(
                "random_%s_seed%d" % (group_name, seed),
                random_ids, body_steps, warmup_steps, use_fake_brain, seed, decoder_groups,
                readout_version=readout_version)
            random_results.append(r_random)
            print("    random:   dist=%.2fmm fwd=%.3f turn=%+.4f" % (
                r_random["forward_distance"], r_random["mean_forward_drive"],
                r_random["mean_turn_drive"]))

        # Also run baseline for comparison
        baseline = run_trial_with_ablation(
            "baseline_%s" % group_name, [], body_steps, warmup_steps,
            use_fake_brain, seeds[0], decoder_groups,
            readout_version=readout_version)

        all_results[group_name] = {
            "n_ablated": n_ablate,
            "baseline": baseline,
            "targeted": targeted_results,
            "random": random_results,
        }

    # Summary
    print("\n" + "=" * 60)
    print("RANDOM vs TARGETED ABLATION SUMMARY")
    print("=" * 60)

    tests = []
    for group_name, data in all_results.items():
        base_dist = abs(data["baseline"]["forward_distance"])
        targeted_dists = [abs(r["forward_distance"]) for r in data["targeted"]]
        random_dists = [abs(r["forward_distance"]) for r in data["random"]]

        targeted_effect = base_dist - np.mean(targeted_dists)
        random_effect = base_dist - np.mean(random_dists)

        print("\n  %s (%d neurons):" % (group_name, data["n_ablated"]))
        print("    baseline dist:  %.2fmm" % base_dist)
        print("    targeted effect: %.2fmm (dist drop)" % targeted_effect)
        print("    random effect:   %.2fmm (dist drop)" % random_effect)

        specific = abs(targeted_effect) > abs(random_effect)
        tests.append({"group": group_name, "passed": specific,
                       "targeted": targeted_effect, "random": random_effect})
        print("    [%s] Targeted > random: %.2f > %.2f" % (
            "PASS" if specific else "FAIL", abs(targeted_effect), abs(random_effect)))

    n_pass = sum(1 for t in tests if t["passed"])
    print("\n  %d/%d specificity tests passed" % (n_pass, len(tests)))

    _write_json_atomic(output_path / "random_ablation_results.json", all_results)
    print("Saved to %s" % (output_path / "random_ablation_results.json"))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended ablation experiments")
    parser.add_argument("--mode", choices=["dose-response", "random-control", "both"],
                        default="both")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--readout-version", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    if args.mode in ("dose-response", "both"):
        run_dose_response(
            body_steps=args.body_steps, warmup_steps=args.warmup_steps,
            use_fake_brain=args.fake_brain, seed=args.seed,
            readout_version=args.readout_version,
        )

    if args.mode in ("random-control", "both"):
        run_random_ablation_control(
            body_steps=args.body_steps, warmup_steps=args.warmup_steps,
            use_fake_brain=args.fake_brain, seeds=args.seeds,
            readout_version=args.readout_version,
        )
