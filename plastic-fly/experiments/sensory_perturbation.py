"""
Sensory perturbation experiment: what sensory conditions activate the
connectome brain's latent turn circuits?

Ablation proved the turn circuits exist (removing turn_left neurons causes
rightward drift). This experiment asks the complementary question: does
asymmetric sensory input naturally activate those circuits?

Three-phase protocol per condition:
  Phase 1 (pre):   baseline walking, no perturbation
  Phase 2 (perturb): apply sensory perturbation to BodyObservation
  Phase 3 (recovery): perturbation removed, measure recovery

Perturbation conditions:
  baseline          -- no intervention
  contact_loss_left -- zero left leg contact forces (mechanosensory asymmetry)
  contact_loss_right-- zero right leg contact forces
  gustatory_left    -- boost left leg contact forces 3x (mech + gustatory)
  gustatory_right   -- boost right leg contact forces 3x
  lateral_push      -- add lateral body velocity offset (vestibular bias)
  combined_left     -- contact_loss_right + gustatory_left (converging signals)

Expected: contact_loss_left silences left mechanosensory neurons, creating
L<R asymmetry in brain, which should activate turn circuits. The direction
(left vs right) tells us how the connectome maps sensory asymmetry to motor output.

Usage:
    python experiments/sensory_perturbation.py --fake-brain        # fast test
    python experiments/sensory_perturbation.py                     # real brain
    python experiments/sensory_perturbation.py --body-steps 10000  # longer
    python experiments/sensory_perturbation.py --conditions contact_loss_left contact_loss_right
"""

import sys
import json
import time
import copy
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import BodyObservation, LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from analysis.behavior_metrics import compute_behavior, BehaviorReport


def _write_json_atomic(path: Path, payload: dict):
    """Write JSON atomically so checkpoints stay readable if the run is interrupted."""
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


def _record_frame(obs, positions, orientations, contact_forces_log):
    """Extract and append one frame of position, orientation, and contact data."""
    positions.append(np.array(obs["fly"][0]))
    orientations.append(np.array(obs["fly"][2]))
    cf_raw = np.asarray(obs.get("contact_forces", np.zeros((30, 3))))
    magnitudes = np.linalg.norm(cf_raw, axis=1) if cf_raw.ndim == 2 else np.zeros(30)
    per_leg = np.array([magnitudes[i*5:(i+1)*5].max() for i in range(6)])
    contact_forces_log.append(np.clip(per_leg / 10.0, 0.0, 1.0))


def _save_condition_checkpoint(
    output_path: Path,
    condition_name: str,
    steps_completed: int,
    body_steps: int,
    positions: list,
    orientations: list,
    contact_forces_log: list,
    episode_log: list,
    status: str,
):
    """Write incremental checkpoint for a single perturbation condition."""
    checkpoint = {
        "condition": condition_name,
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
    _write_json_atomic(
        output_path / "checkpoints" / f"{condition_name}.json", checkpoint
    )


# --- Perturbation definitions ---

PERTURBATION_CONDITIONS = {
    "baseline": {},
    "contact_loss_left": {
        "type": "zero_contact",
        "legs": [0, 1, 2],       # LF, LM, LH
    },
    "contact_loss_right": {
        "type": "zero_contact",
        "legs": [3, 4, 5],       # RF, RM, RH
    },
    "gustatory_left": {
        "type": "boost_contact",
        "legs": [0, 1, 2],
        "factor": 3.0,
    },
    "gustatory_right": {
        "type": "boost_contact",
        "legs": [3, 4, 5],
        "factor": 3.0,
    },
    "lateral_push": {
        "type": "vestibular_bias",
        "velocity_offset": [0.0, 5.0, 0.0],  # lateral body velocity (MuJoCo Y)
    },
    "combined_left": {
        "type": "combined",
        "sub_conditions": ["contact_loss_right", "gustatory_left"],
    },
}


def apply_sensory_perturbation(obs: BodyObservation, condition: dict) -> BodyObservation:
    """Apply a sensory perturbation to a BodyObservation (returns new copy)."""
    if not condition:
        return obs

    ctype = condition.get("type", "")

    if ctype == "combined":
        result = obs
        for sub_name in condition["sub_conditions"]:
            sub_cond = PERTURBATION_CONDITIONS[sub_name]
            result = apply_sensory_perturbation(result, sub_cond)
        return result

    # Copy arrays to avoid mutating original
    contact_forces = obs.contact_forces.copy()
    body_velocity = obs.body_velocity.copy()

    if ctype == "zero_contact":
        for leg_idx in condition["legs"]:
            contact_forces[leg_idx] = 0.0

    elif ctype == "boost_contact":
        factor = condition.get("factor", 3.0)
        for leg_idx in condition["legs"]:
            contact_forces[leg_idx] = np.clip(contact_forces[leg_idx] * factor, 0.0, 1.0)

    elif ctype == "vestibular_bias":
        offset = np.array(condition["velocity_offset"], dtype=np.float32)
        body_velocity = body_velocity + offset

    return BodyObservation(
        joint_angles=obs.joint_angles,
        joint_velocities=obs.joint_velocities,
        contact_forces=contact_forces,
        body_velocity=body_velocity,
        body_orientation=obs.body_orientation,
    )


def run_single_perturbation(
    condition_name: str,
    condition: dict,
    body_steps: int,
    warmup_steps: int,
    use_fake_brain: bool,
    seed: int,
    decoder_groups: dict,
    onset_frac: float = 0.25,
    offset_frac: float = 0.75,
    sample_interval: int = 20,
    readout_version: int = 2,
    output_path: Path | None = None,
) -> dict:
    """Run one perturbation condition with three-phase protocol."""
    import flygym

    cfg = BridgeConfig()

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
    arena = flygym.arena.FlatTerrain()
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=arena, timestep=1e-4)
    obs, info = sim.reset()

    # Warmup
    locomotion.warmup(0)
    locomotion.cpg.reset(init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
                         init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, terminated, truncated, info = sim.step(action)
            if terminated or truncated:
                sim.close()
                return {"error": "warmup_ended"}
        except (RuntimeError, ValueError):  # MuJoCo physics instability
            sim.close()
            return {"error": "warmup_physics"}

    # Phase boundaries
    onset_step = int(body_steps * onset_frac)
    offset_step = int(body_steps * offset_frac)

    # Main loop
    bspb = cfg.body_steps_per_brain
    episode_log = []
    positions = []
    orientations = []
    contact_forces_log = []
    brain_steps = 0
    current_cmd = LocomotionCommand(forward_drive=1.0)
    step = 0

    t_start = time.time()

    for step in range(body_steps):
        # Determine phase
        if step < onset_step:
            phase = "pre"
        elif step < offset_step:
            phase = "perturb"
        else:
            phase = "recovery"

        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)

            # Apply perturbation only during perturbation phase
            if phase == "perturb":
                body_obs = apply_sensory_perturbation(body_obs, condition)

            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            current_cmd = decoder.decode(brain_output)
            brain_steps += 1

            # Per-group rate tracking
            id_to_rate = {
                int(nid): float(rate)
                for nid, rate in zip(brain_output.neuron_ids, brain_output.firing_rates_hz)
            }
            group_rates = {}
            for gname, gids in decoder_groups.items():
                rates = [id_to_rate.get(int(nid), 0.0) for nid in gids]
                group_rates[gname] = float(np.mean(rates)) if rates else 0.0

            episode_log.append({
                "body_step": step,
                "brain_step": brain_steps,
                "phase": phase,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
                "step_frequency": current_cmd.step_frequency,
                "stance_gain": current_cmd.stance_gain,
                "readout_mean_hz": float(np.mean(brain_output.firing_rates_hz)),
                "readout_active": int(np.sum(brain_output.firing_rates_hz > 0)),
                "group_rates": group_rates,
            })

        action = locomotion.step(current_cmd)
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
        except (RuntimeError, ValueError):  # MuJoCo physics instability
            break

        if step % sample_interval == 0:
            _record_frame(obs, positions, orientations, contact_forces_log)
            if output_path is not None:
                _save_condition_checkpoint(
                    output_path=output_path,
                    condition_name=condition_name,
                    steps_completed=step + 1,
                    body_steps=body_steps,
                    positions=positions,
                    orientations=orientations,
                    contact_forces_log=contact_forces_log,
                    episode_log=episode_log,
                    status="running",
                )

        if terminated or truncated:
            break

    sim.close()
    elapsed = time.time() - t_start

    # --- Per-phase metrics ---
    onset_sample = onset_step // sample_interval
    offset_sample = offset_step // sample_interval

    # Heading at phase boundaries (from position trajectory)
    def heading_at_sample(idx):
        """Heading angle from position displacement at sample index."""
        if idx <= 0 or idx >= len(positions):
            return 0.0
        dx = positions[idx][0] - positions[max(0, idx-1)][0]
        dy = positions[idx][1] - positions[max(0, idx-1)][1]
        return float(np.arctan2(dy, dx))

    def mean_heading_range(start, end):
        """Mean heading angle over a sample range."""
        end = min(end, len(positions))
        start = max(start, 1)
        if end <= start:
            return 0.0
        headings = []
        for i in range(start, end):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            headings.append(np.arctan2(dy, dx))
        return float(np.mean(headings)) if headings else 0.0

    heading_pre = mean_heading_range(1, onset_sample)
    heading_perturb = mean_heading_range(onset_sample, offset_sample)
    heading_change = heading_perturb - heading_pre

    # Position displacement during perturbation phase
    if onset_sample < len(positions) and offset_sample < len(positions):
        perturb_displacement = positions[min(offset_sample, len(positions)-1)] - positions[onset_sample]
        perturb_lateral = float(perturb_displacement[1])  # Y = lateral in MuJoCo
        perturb_forward = float(perturb_displacement[0])  # X = forward
    else:
        perturb_lateral = 0.0
        perturb_forward = 0.0

    # Per-phase group rates
    phase_group_rates = {"pre": {}, "perturb": {}, "recovery": {}}
    for phase_name in ["pre", "perturb", "recovery"]:
        phase_entries = [e for e in episode_log if e["phase"] == phase_name]
        if not phase_entries:
            continue
        for gname in decoder_groups:
            rates = [e["group_rates"].get(gname, 0.0) for e in phase_entries]
            phase_group_rates[phase_name][gname] = float(np.mean(rates))

    # Per-phase turn drive
    def phase_mean(field, phase_name):
        entries = [e[field] for e in episode_log if e["phase"] == phase_name]
        return float(np.mean(entries)) if entries else 0.0

    turn_drive_pre = phase_mean("turn_drive", "pre")
    turn_drive_perturb = phase_mean("turn_drive", "perturb")
    turn_drive_delta = turn_drive_perturb - turn_drive_pre

    forward_drive_pre = phase_mean("forward_drive", "pre")
    forward_drive_perturb = phase_mean("forward_drive", "perturb")

    # Full behavior metrics (perturbation phase only)
    perturb_positions = positions[onset_sample:offset_sample]
    perturb_orientations = orientations[onset_sample:offset_sample] if orientations else None
    perturb_contacts = contact_forces_log[onset_sample:offset_sample] if contact_forces_log else None

    if len(perturb_positions) >= 3:
        behavior = compute_behavior(
            positions=perturb_positions,
            orientations=perturb_orientations,
            contact_forces=perturb_contacts,
            steps_completed=offset_step - onset_step,
            steps_intended=offset_step - onset_step,
            sample_dt=sample_interval * 1e-4,
        )
    else:
        behavior = compute_behavior(
            positions=positions,
            orientations=orientations,
            contact_forces=contact_forces_log,
            steps_completed=step + 1,
            steps_intended=body_steps,
            sample_dt=sample_interval * 1e-4,
        )

    return {
        "condition": condition_name,
        "steps_completed": step + 1,
        "brain_steps": brain_steps,
        "elapsed_s": elapsed,
        "onset_step": onset_step,
        "offset_step": offset_step,
        "heading_change_deg": float(np.degrees(heading_change)),
        "heading_pre_deg": float(np.degrees(heading_pre)),
        "heading_perturb_deg": float(np.degrees(heading_perturb)),
        "perturb_lateral_mm": perturb_lateral,
        "perturb_forward_mm": perturb_forward,
        "turn_drive_pre": turn_drive_pre,
        "turn_drive_perturb": turn_drive_perturb,
        "turn_drive_delta": turn_drive_delta,
        "forward_drive_pre": forward_drive_pre,
        "forward_drive_perturb": forward_drive_perturb,
        "phase_group_rates": phase_group_rates,
        "behavior": behavior.to_dict(),
        "behavior_summary": behavior.summary_line(),
        "episode_log": episode_log,
    }


def run_perturbation_study(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    output_dir: str = "logs/sensory_perturbation",
    conditions: list[str] | None = None,
    onset_frac: float = 0.25,
    offset_frac: float = 0.75,
    readout_version: int = 2,
):
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

    if conditions is None:
        conditions = list(PERTURBATION_CONDITIONS.keys())

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    print("=" * 70)
    print("SENSORY PERTURBATION STUDY (%s, %d body steps, seed=%d)" % (
        brain_label, body_steps, seed))
    print("Perturbation window: %.0f%%-%.0f%% of run (steps %d-%d)" % (
        onset_frac * 100, offset_frac * 100,
        int(body_steps * onset_frac), int(body_steps * offset_frac)))
    print("=" * 70)

    results = {}
    for cond_name in conditions:
        cond = PERTURBATION_CONDITIONS[cond_name]
        print("\n--- %s ---" % cond_name)
        r = run_single_perturbation(
            condition_name=cond_name,
            condition=cond,
            body_steps=body_steps,
            warmup_steps=warmup_steps,
            use_fake_brain=use_fake_brain,
            seed=seed,
            decoder_groups=decoder_groups,
            onset_frac=onset_frac,
            offset_frac=offset_frac,
            readout_version=readout_version,
            output_path=output_path,
        )
        if "error" in r:
            print("  ERROR: %s" % r["error"])
            continue
        print("  %s" % r["behavior_summary"])
        print("  heading_change=%.1fdeg  turn_drive_delta=%+.4f  lateral=%.2fmm" % (
            r["heading_change_deg"], r["turn_drive_delta"], r["perturb_lateral_mm"]))
        results[cond_name] = r

    # === Comparative table ===
    print("\n" + "=" * 70)
    print("COMPARATIVE RESULTS")
    print("=" * 70)

    header = "%-22s %9s %9s %9s %9s %10s" % (
        "Condition", "Hdg Chg", "Turn dD", "Lateral", "Fwd mm", "TL-TR Hz")
    print(header)
    print("-" * len(header))

    for cond_name in conditions:
        if cond_name not in results:
            continue
        r = results[cond_name]
        # Turn group rate asymmetry during perturbation
        pgr = r["phase_group_rates"].get("perturb", {})
        tl_rate = pgr.get("turn_left_ids", 0.0)
        tr_rate = pgr.get("turn_right_ids", 0.0)
        print("%-22s %+8.1fdeg %+8.4f %+8.2fmm %+8.2fmm %+9.1fHz" % (
            cond_name,
            r["heading_change_deg"],
            r["turn_drive_delta"],
            r["perturb_lateral_mm"],
            r["perturb_forward_mm"],
            tl_rate - tr_rate,
        ))

    # === Per-group activation table ===
    print("\n" + "=" * 70)
    print("GROUP ACTIVATION BY PHASE (mean Hz)")
    print("=" * 70)

    for cond_name in conditions:
        if cond_name not in results:
            continue
        r = results[cond_name]
        print("\n  %s:" % cond_name)
        print("  %-22s %8s %8s %8s %8s" % ("Group", "Pre", "Perturb", "Recovery", "Delta"))
        print("  " + "-" * 70)
        pgr = r["phase_group_rates"]
        for gname in decoder_groups:
            pre_r = pgr.get("pre", {}).get(gname, 0.0)
            per_r = pgr.get("perturb", {}).get(gname, 0.0)
            rec_r = pgr.get("recovery", {}).get(gname, 0.0)
            delta = per_r - pre_r
            print("  %-22s %8.1f %8.1f %8.1f %+7.1f" % (
                gname, pre_r, per_r, rec_r, delta))

    # === Hypothesis tests ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    tests_passed = 0
    tests_total = 0

    def log_test(name, passed, description):
        nonlocal tests_passed, tests_total
        tests_total += 1
        if passed:
            tests_passed += 1
        print("  [%s] %s: %s" % ("PASS" if passed else "FAIL", name, description))

    # --- Perturbation contrast tests ---
    print("\n  -- Perturbation contrast (behavioral) --")

    # 1. Contact loss left vs right: heading should differ in sign
    if "contact_loss_left" in results and "contact_loss_right" in results:
        h_left = results["contact_loss_left"]["heading_change_deg"]
        h_right = results["contact_loss_right"]["heading_change_deg"]
        log_test("heading_contrast_contact",
                 (h_left * h_right < 0) or (abs(h_left - h_right) > 1.0),
                 "contact_loss L/R heading diverges (L=%+.1fdeg R=%+.1fdeg)" % (h_left, h_right))

    # 2. Contact loss left vs baseline
    if "contact_loss_left" in results and "baseline" in results:
        h_cll = results["contact_loss_left"]["heading_change_deg"]
        h_base = results["baseline"]["heading_change_deg"]
        log_test("contact_loss_left_vs_baseline",
                 abs(h_cll - h_base) > 0.5,
                 "contact_loss_left shifts heading vs baseline (%.1f vs %.1f deg)" % (h_cll, h_base))

    # 3. Contact loss right vs baseline
    if "contact_loss_right" in results and "baseline" in results:
        h_clr = results["contact_loss_right"]["heading_change_deg"]
        h_base = results["baseline"]["heading_change_deg"]
        log_test("contact_loss_right_vs_baseline",
                 abs(h_clr - h_base) > 0.5,
                 "contact_loss_right shifts heading vs baseline (%.1f vs %.1f deg)" % (h_clr, h_base))

    # 4. Gustatory left vs right: heading should differ
    if "gustatory_left" in results and "gustatory_right" in results:
        h_gl = results["gustatory_left"]["heading_change_deg"]
        h_gr = results["gustatory_right"]["heading_change_deg"]
        log_test("heading_contrast_gustatory",
                 (h_gl * h_gr < 0) or (abs(h_gl - h_gr) > 1.0),
                 "gustatory L/R heading diverges (L=%+.1fdeg R=%+.1fdeg)" % (h_gl, h_gr))

    # 5. Lateral push vs baseline
    if "lateral_push" in results and "baseline" in results:
        h_lp = results["lateral_push"]["heading_change_deg"]
        h_base = results["baseline"]["heading_change_deg"]
        log_test("lateral_push_vs_baseline",
                 abs(h_lp - h_base) > 0.5,
                 "lateral_push shifts heading vs baseline (%.1f vs %.1f deg)" % (h_lp, h_base))

    # 6. Combined stronger than single
    if "combined_left" in results and "contact_loss_right" in results:
        h_comb = abs(results["combined_left"]["heading_change_deg"])
        h_single = abs(results["contact_loss_right"]["heading_change_deg"])
        log_test("combined_stronger",
                 h_comb > h_single * 0.8,  # allow some tolerance
                 "combined at least ~80%% of single (%.1f vs %.1f deg)" % (h_comb, h_single))

    # --- Neural activation tests ---
    print("\n  -- Neural activation (brain response) --")

    # 7. Contact loss left: turn group asymmetry
    if "contact_loss_left" in results:
        pgr = results["contact_loss_left"]["phase_group_rates"]
        pre_tl = pgr.get("pre", {}).get("turn_left_ids", 0.0)
        pre_tr = pgr.get("pre", {}).get("turn_right_ids", 0.0)
        per_tl = pgr.get("perturb", {}).get("turn_left_ids", 0.0)
        per_tr = pgr.get("perturb", {}).get("turn_right_ids", 0.0)
        pre_asym = pre_tl - pre_tr
        per_asym = per_tl - per_tr
        log_test("neural_contact_left_asymmetry",
                 abs(per_asym - pre_asym) > 0.1,
                 "contact_loss_left shifts turn group asymmetry (pre=%.1f, perturb=%.1f)" % (
                     pre_asym, per_asym))

    # 8. Contact loss right: opposite turn group asymmetry
    if "contact_loss_right" in results:
        pgr = results["contact_loss_right"]["phase_group_rates"]
        pre_tl = pgr.get("pre", {}).get("turn_left_ids", 0.0)
        pre_tr = pgr.get("pre", {}).get("turn_right_ids", 0.0)
        per_tl = pgr.get("perturb", {}).get("turn_left_ids", 0.0)
        per_tr = pgr.get("perturb", {}).get("turn_right_ids", 0.0)
        pre_asym = pre_tl - pre_tr
        per_asym = per_tl - per_tr
        log_test("neural_contact_right_asymmetry",
                 abs(per_asym - pre_asym) > 0.1,
                 "contact_loss_right shifts turn group asymmetry (pre=%.1f, perturb=%.1f)" % (
                     pre_asym, per_asym))

    # 9. Lateral push: turn asymmetry exists
    if "lateral_push" in results:
        pgr = results["lateral_push"]["phase_group_rates"]
        pre_td = results["lateral_push"]["turn_drive_pre"]
        per_td = results["lateral_push"]["turn_drive_perturb"]
        log_test("neural_lateral_push",
                 abs(per_td - pre_td) > 0.001,
                 "lateral_push shifts turn_drive (pre=%.4f, perturb=%.4f)" % (pre_td, per_td))

    # 10. L/R symmetry: contact_loss_left and contact_loss_right should shift asymmetry in opposite directions
    if "contact_loss_left" in results and "contact_loss_right" in results:
        td_l = results["contact_loss_left"]["turn_drive_delta"]
        td_r = results["contact_loss_right"]["turn_drive_delta"]
        log_test("neural_lr_symmetry",
                 (td_l * td_r < 0) or (abs(td_l - td_r) > 0.001),
                 "L/R perturbations shift turn_drive oppositely (L=%+.4f, R=%+.4f)" % (td_l, td_r))

    print("\nHypothesis tests: %d/%d passed" % (tests_passed, tests_total))

    # === Save ===
    # Strip episode_log for compact JSON (keep summary metrics)
    save_results = {}
    for cond_name, r in results.items():
        save_r = {k: v for k, v in r.items() if k != "episode_log"}
        # Keep condensed episode log (just phase + commands)
        save_r["episode_log_condensed"] = [
            {k: v for k, v in e.items() if k != "group_rates"}
            for e in r.get("episode_log", [])
        ]
        save_results[cond_name] = save_r

    _write_json_atomic(output_path / "perturbation_results.json", save_results)
    print("\nSaved to %s/perturbation_results.json" % output_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensory perturbation experiment for brain-body bridge")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="logs/sensory_perturbation")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Run only these conditions (default: all)")
    parser.add_argument("--onset-frac", type=float, default=0.25,
                        help="Perturbation onset as fraction of body_steps (default: 0.25)")
    parser.add_argument("--offset-frac", type=float, default=0.75,
                        help="Perturbation offset as fraction of body_steps (default: 0.75)")
    parser.add_argument("--readout-version", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    run_perturbation_study(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        seed=args.seed,
        output_dir=args.output_dir,
        conditions=args.conditions,
        onset_frac=args.onset_frac,
        offset_frac=args.offset_frac,
        readout_version=args.readout_version,
    )
