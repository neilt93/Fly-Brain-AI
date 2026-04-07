"""
Predict untested optogenetic phenotypes for DN types NOT in Cande 2018.

For each candidate DN type:
  1. Look up FlyWire root IDs and decoder group assignment
  2. Silencing: zero those neurons' rates in brain output
  3. Boosting: 3x those neurons' rates
  4. Run closed-loop walk (5000 steps, multiple seeds)
  5. Measure forward distance, heading, path straightness vs baseline
  6. Generate testable prediction for experimental validation

Candidates:
  - DNg100 (turning group, bilateral pair, "walking command" per Pugliese 2025)
    Prediction: bilateral silencing reduces turning drive; unilateral silencing
    causes contralateral turning bias.
  - DNg47 (turning group, bilateral pair, NOT in Cande 2018)
    Prediction: similar to DNg100 — steering contribution.
  - DNa01 (turning group in v5; Cande 2018 reports "locomotor increase + steering")
    Used as POSITIVE CONTROL — known phenotype.

Usage:
    cd plastic-fly
    python experiments/dn_phenotype_prediction.py --fake-brain     # fast
    python experiments/dn_phenotype_prediction.py                  # real brain
    python experiments/dn_phenotype_prediction.py --seeds 5        # more seeds
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import LocomotionCommand, BrainOutput
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from analysis.behavior_metrics import compute_behavior

BASE = Path(__file__).resolve().parent.parent.parent


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


# ═══════════════════════════════════════════════════════════════════════════
# DN type resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_dn_type_ids(dn_type, ann):
    """Resolve a DN type name to FlyWire root IDs."""
    matches = ann[ann["cell_type"] == dn_type]
    return list(matches["root_id"].astype(np.int64).values)


def find_group_assignment(root_ids, decoder_groups):
    """Find which decoder groups contain these root IDs."""
    id_to_groups = defaultdict(set)
    for gname, gids in decoder_groups.items():
        clean = gname.replace("_ids", "")
        for nid in gids:
            id_to_groups[np.int64(nid)].add(clean)

    assignments = {}
    for rid in root_ids:
        assignments[np.int64(rid)] = id_to_groups.get(np.int64(rid), set())
    return assignments


# ═══════════════════════════════════════════════════════════════════════════
# Per-neuron ablation (extends apply_ablation from ablation_study.py)
# ═══════════════════════════════════════════════════════════════════════════

def apply_neuron_intervention(
    brain_output: BrainOutput,
    target_ids: list[int],
    mode: str = "silence",  # "silence" or "boost"
    factor: float = 3.0,
) -> BrainOutput:
    """Apply per-neuron silencing or boosting to brain output.

    Args:
        brain_output: Original brain output.
        target_ids: FlyWire root IDs to intervene on.
        mode: "silence" (zero rates) or "boost" (multiply by factor).
        factor: Boost multiplier (only used if mode="boost").
    """
    rates = brain_output.firing_rates_hz.copy()
    id_to_idx = {np.int64(nid): i for i, nid in enumerate(brain_output.neuron_ids)}

    target_set = set(np.int64(x) for x in target_ids)
    for nid in target_set:
        if nid in id_to_idx:
            idx = id_to_idx[nid]
            if mode == "silence":
                rates[idx] = 0.0
            elif mode == "boost":
                rates[idx] *= factor

    return BrainOutput(neuron_ids=brain_output.neuron_ids, firing_rates_hz=rates)


# ═══════════════════════════════════════════════════════════════════════════
# Single trial runner
# ═══════════════════════════════════════════════════════════════════════════

def run_trial(
    target_ids: list[int],
    mode: str,  # "baseline", "silence", "boost"
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    readout_version: int = 5,
    boost_factor: float = 3.0,
) -> dict:
    """Run one closed-loop trial with per-neuron intervention."""
    import flygym

    cfg = BridgeConfig()

    if readout_version == 5:
        sensory_ids = np.load(cfg.data_dir / "sensory_ids_v3.npy")
        readout_ids = np.load(cfg.data_dir / "readout_ids_v5_steering.npy")
        channel_map_path = cfg.data_dir / "channel_map_v3.json"
        decoder_path = cfg.data_dir / "decoder_groups_v5_steering.json"
        rate_scale = 12.0
    elif readout_version == 4:
        sensory_ids = np.load(cfg.data_dir / "sensory_ids_v4_looming.npy")
        readout_ids = np.load(cfg.data_dir / "readout_ids_v4_looming.npy")
        channel_map_path = cfg.data_dir / "channel_map_v4_looming.json"
        decoder_path = cfg.data_dir / "decoder_groups_v4_looming.json"
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

    # Main loop
    bspb = cfg.body_steps_per_brain
    positions = []
    orientations = []
    contact_forces_log = []
    commands = []
    current_cmd = LocomotionCommand(forward_drive=1.0)
    step = 0
    sample_interval = 20

    t_start = time.time()

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)

            # Apply intervention
            if mode == "silence":
                brain_output = apply_neuron_intervention(
                    brain_output, target_ids, mode="silence",
                )
            elif mode == "boost":
                brain_output = apply_neuron_intervention(
                    brain_output, target_ids, mode="boost", factor=boost_factor,
                )

            current_cmd = decoder.decode(brain_output)
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
        except (RuntimeError, ValueError):  # MuJoCo physics instability
            break

        if step % sample_interval == 0:
            positions.append(np.array(obs["fly"][0]))
            orientations.append(np.array(obs["fly"][2]))
            cf_raw = np.asarray(obs.get("contact_forces", np.zeros((30, 3))))
            mags = np.linalg.norm(cf_raw, axis=1) if cf_raw.ndim == 2 else np.zeros(30)
            per_leg = np.array([mags[i*5:(i+1)*5].max() for i in range(6)])
            contact_forces_log.append(np.clip(per_leg / 10.0, 0.0, 1.0))

        if term or trunc:
            break

    sim.close()
    elapsed = time.time() - t_start

    if len(positions) < 2:
        return {"error": "too_few_samples"}

    # Behavior metrics
    behavior = compute_behavior(
        positions=positions,
        orientations=orientations,
        contact_forces=contact_forces_log,
        steps_completed=step + 1,
        steps_intended=body_steps,
        sample_dt=sample_interval * 1e-4,
    )

    # Command statistics
    fwd_drives = [c["forward_drive"] for c in commands]
    turn_drives = [c["turn_drive"] for c in commands]

    return {
        "steps_completed": step + 1,
        "elapsed_s": elapsed,
        "behavior": behavior.to_dict(),
        "mean_forward_drive": float(np.mean(fwd_drives)),
        "mean_turn_drive": float(np.mean(turn_drives)),
        "mean_abs_turn_drive": float(np.mean(np.abs(turn_drives))),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DN type definitions and predictions
# ═══════════════════════════════════════════════════════════════════════════

DN_CANDIDATES = {
    "DNg100": {
        "description": "Walking command neuron (Pugliese 2025), bilateral pair",
        "in_cande_2018": False,
        "prediction_silence": (
            "Bilateral silencing reduces turning drive. "
            "In our decoder, DNg100 L/R are in turn_left/turn_right groups. "
            "Silencing both should reduce |turn_drive| without affecting forward."
        ),
        "prediction_boost": (
            "Bilateral boosting increases |turn_drive| or causes circling."
        ),
    },
    "DNg47": {
        "description": "Turning DN, bilateral pair, NOT in Cande 2018",
        "in_cande_2018": False,
        "prediction_silence": (
            "Bilateral silencing reduces turning capability. "
            "Both neurons are in turn_left/turn_right groups."
        ),
        "prediction_boost": (
            "Bilateral boosting amplifies turning response."
        ),
    },
    "DNa01": {
        "description": "Locomotor increase + steering (Cande 2018 POSITIVE CONTROL)",
        "in_cande_2018": True,
        "prediction_silence": (
            "Reduces forward locomotion and steering. "
            "Known phenotype from Cande 2018: 'locomotor increase / steering'."
        ),
        "prediction_boost": (
            "Increased locomotion and/or steering."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_dn_prediction(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    n_seeds: int = 3,
    readout_version: int = 5,
):
    print("=" * 80)
    print("DN PHENOTYPE PREDICTION EXPERIMENT")
    print(f"  body_steps={body_steps}, seeds={n_seeds}, "
          f"brain={'FAKE' if use_fake_brain else 'Brian2 LIF'}")
    print("=" * 80)

    # Load annotations
    ann = pd.read_csv(BASE / "brain-model" / "flywire_annotations_matched.csv", low_memory=False)
    cfg = BridgeConfig()

    # Load decoder groups
    if readout_version == 5:
        dec_path = cfg.data_dir / "decoder_groups_v5_steering.json"
    elif readout_version == 4:
        dec_path = cfg.data_dir / "decoder_groups_v4_looming.json"
    else:
        dec_path = cfg.decoder_groups_path
    with open(dec_path) as f:
        decoder_groups = json.load(f)

    output_dir = Path("logs/dn_phenotype_prediction")

    # Resolve all DN types
    print("\n--- DN TYPE RESOLUTION ---")
    dn_info = {}
    for dn_type, meta in DN_CANDIDATES.items():
        root_ids = resolve_dn_type_ids(dn_type, ann)
        assignments = find_group_assignment(root_ids, decoder_groups)

        all_groups = set()
        for g in assignments.values():
            all_groups.update(g)

        dn_info[dn_type] = {
            "root_ids": root_ids,
            "groups": list(all_groups),
            "per_neuron": {str(k): list(v) for k, v in assignments.items()},
            "n_in_readout": sum(1 for g in assignments.values() if g),
        }

        print(f"  {dn_type}: {len(root_ids)} neurons, "
              f"groups={list(all_groups)}, "
              f"in_readout={dn_info[dn_type]['n_in_readout']}/{len(root_ids)}")
        for rid, grps in assignments.items():
            side = ann[ann["root_id"] == rid]["side"].values
            side_str = str(side[0]) if len(side) > 0 else "?"
            print(f"    {rid} ({side_str}): {list(grps)}")

    # Run experiments
    all_results = {}
    seeds = list(range(42, 42 + n_seeds))

    for dn_type, meta in DN_CANDIDATES.items():
        info = dn_info[dn_type]
        root_ids = info["root_ids"]

        if not root_ids:
            print(f"\n  {dn_type}: SKIPPED (no FlyWire neurons found)")
            continue

        in_readout = info["n_in_readout"]
        if in_readout == 0:
            print(f"\n  {dn_type}: SKIPPED (not in readout pool)")
            continue

        print(f"\n{'='*80}")
        print(f"  {dn_type}: {meta['description']}")
        print(f"  Cande 2018: {'YES (positive control)' if meta['in_cande_2018'] else 'NO (novel prediction)'}")
        print(f"  Prediction (silence): {meta['prediction_silence']}")
        print(f"{'='*80}")

        conditions = ["baseline", "silence", "boost"]
        dn_results = {cond: [] for cond in conditions}

        for seed in seeds:
            for cond in conditions:
                label = f"{dn_type}_{cond}_s{seed}"
                print(f"    Running {label}...", end=" ", flush=True)

                r = run_trial(
                    target_ids=root_ids,
                    mode=cond,
                    body_steps=body_steps,
                    warmup_steps=warmup_steps,
                    use_fake_brain=use_fake_brain,
                    seed=seed,
                    readout_version=readout_version,
                )

                if "error" in r:
                    print(f"ERROR: {r['error']}")
                    continue

                b = r["behavior"]
                print(f"fwd={b['forward_distance']:+.2f}mm "
                      f"heading={np.degrees(b['final_heading']):+.1f}deg "
                      f"turn_drive={r['mean_turn_drive']:+.3f} "
                      f"({r['elapsed_s']:.1f}s)")

                dn_results[cond].append(r)

                # Checkpoint after each trial
                _write_json_atomic(
                    output_dir / "checkpoints" / f"{label}.json",
                    r,
                )

        all_results[dn_type] = dn_results

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHENOTYPE PREDICTION RESULTS")
    print(f"{'='*80}")

    prediction_results = []

    for dn_type, dn_results in all_results.items():
        meta = DN_CANDIDATES[dn_type]
        info = dn_info[dn_type]

        print(f"\n  --- {dn_type} (groups: {info['groups']}) ---")

        for cond in ["baseline", "silence", "boost"]:
            trials = dn_results[cond]
            if not trials:
                print(f"    {cond}: no data")
                continue

            dists = [t["behavior"]["forward_distance"] for t in trials]
            headings_rad = [t["behavior"]["final_heading"] for t in trials]
            turn_drives = [t["mean_turn_drive"] for t in trials]
            abs_turns = [t["mean_abs_turn_drive"] for t in trials]

            # Circular mean for heading
            mean_heading_deg = np.degrees(np.arctan2(
                np.mean(np.sin(headings_rad)), np.mean(np.cos(headings_rad))))

            print(f"    {cond:>10s}: fwd={np.mean(dists):+.2f}+/-{np.std(dists):.2f}mm  "
                  f"heading={mean_heading_deg:+.1f}deg  "
                  f"turn_drive={np.mean(turn_drives):+.3f}  "
                  f"|turn|={np.mean(abs_turns):.3f}")

        # Compute effects
        baseline_trials = dn_results["baseline"]
        silence_trials = dn_results["silence"]
        boost_trials = dn_results["boost"]

        if baseline_trials and silence_trials:
            bl_dist = np.mean([t["behavior"]["forward_distance"] for t in baseline_trials])
            sl_dist = np.mean([t["behavior"]["forward_distance"] for t in silence_trials])
            bl_abs_turn = np.mean([t["mean_abs_turn_drive"] for t in baseline_trials])
            sl_abs_turn = np.mean([t["mean_abs_turn_drive"] for t in silence_trials])
            bl_h = [t["behavior"]["final_heading"] for t in baseline_trials]
            sl_h = [t["behavior"]["final_heading"] for t in silence_trials]
            bl_heading = np.arctan2(np.mean(np.sin(bl_h)), np.mean(np.cos(bl_h)))
            sl_heading = np.arctan2(np.mean(np.sin(sl_h)), np.mean(np.cos(sl_h)))

            dist_change = sl_dist - bl_dist
            dist_change_pct = (dist_change / abs(bl_dist) * 100) if bl_dist != 0 else 0
            turn_change = sl_abs_turn - bl_abs_turn
            heading_change = np.degrees(sl_heading - bl_heading)

            print(f"\n    SILENCING EFFECT:")
            print(f"      Forward distance: {dist_change:+.2f}mm ({dist_change_pct:+.1f}%)")
            print(f"      |Turn drive| change: {turn_change:+.4f}")
            print(f"      Heading shift: {heading_change:+.1f}deg")

            prediction_results.append({
                "dn_type": dn_type,
                "groups": info["groups"],
                "in_cande_2018": meta["in_cande_2018"],
                "prediction": meta["prediction_silence"],
                "silence_dist_change_pct": dist_change_pct,
                "silence_turn_change": turn_change,
                "silence_heading_shift_deg": heading_change,
                "n_seeds": len(silence_trials),
            })

        if baseline_trials and boost_trials:
            bl_dist = np.mean([t["behavior"]["forward_distance"] for t in baseline_trials])
            bo_dist = np.mean([t["behavior"]["forward_distance"] for t in boost_trials])
            bl_abs_turn = np.mean([t["mean_abs_turn_drive"] for t in baseline_trials])
            bo_abs_turn = np.mean([t["mean_abs_turn_drive"] for t in boost_trials])

            print(f"    BOOSTING EFFECT:")
            print(f"      Forward distance: {bo_dist - bl_dist:+.2f}mm "
                  f"({((bo_dist - bl_dist)/abs(bl_dist)*100) if bl_dist else 0:+.1f}%)")
            print(f"      |Turn drive| change: {bo_abs_turn - bl_abs_turn:+.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Testable predictions summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TESTABLE PREDICTIONS FOR EXPERIMENTAL VALIDATION")
    print(f"{'='*80}")

    for pr in prediction_results:
        novel = "NOVEL" if not pr["in_cande_2018"] else "VALIDATION"
        print(f"\n  [{novel}] {pr['dn_type']} (groups: {pr['groups']})")
        print(f"    Model prediction: {pr['prediction']}")
        print(f"    Measured: dist {pr['silence_dist_change_pct']:+.1f}%, "
              f"turn {pr['silence_turn_change']:+.4f}, "
              f"heading {pr['silence_heading_shift_deg']:+.1f}deg")

    # Save
    final_payload = {
        "dn_info": {k: {kk: vv for kk, vv in v.items() if kk != "root_ids"}
                     for k, v in dn_info.items()},
        "prediction_results": prediction_results,
        "params": {
            "body_steps": body_steps,
            "n_seeds": n_seeds,
            "readout_version": readout_version,
            "use_fake_brain": use_fake_brain,
        },
    }
    # Include root_ids as strings for JSON
    for k, v in dn_info.items():
        final_payload["dn_info"][k]["root_ids"] = [str(x) for x in v["root_ids"]]

    _write_json_atomic(output_dir / "results.json", final_payload)
    print(f"\nSaved to {output_dir}/results.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DN phenotype prediction experiment")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--readout-version", type=int, default=5, choices=[1, 4, 5])
    args = parser.parse_args()

    run_dn_prediction(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        n_seeds=args.seeds,
        readout_version=args.readout_version,
    )
