"""
Terrain Shift Experiment — Continuous Episode Design

The fly walks on a single composite terrain: flat → blocks → flat.
One continuous simulation per controller — no restarts, no CPG resets
between phases. The terrain change happens naturally as the fly walks
forward, testing online adaptation in real time.

Phase detection is position-based:
  - Flat:   x < blocks_start
  - Blocks: blocks_start ≤ x < blocks_end
  - Post:   x ≥ blocks_end  (forgetting test)

Logs positions, contacts, velocities, and weight drift (plastic only).
Generates recovery curves and comparison plots.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.fixed_controller import FixedController
from controllers.plastic_controller import PlasticController
from analysis.recovery_metrics import (
    compute_distance,
    compute_velocity,
    compute_gait_symmetry,
    compute_recovery_report,
    RecoveryReport,
)
from analysis.plots import (
    plot_recovery_curves,
    plot_comparison_bars,
    plot_weight_drift,
    plot_gait_symmetry_over_time,
    plot_contact_raster,
    plot_tripod_score,
)
from analysis.gait_metrics import (
    classify_stance_swing,
    compute_tripod_score,
    compute_gait_report,
)
from structlog.structured_log import (
    RunRecord,
    EventRecord,
    append_run,
    append_event,
    write_state,
)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Timing
    timestep: float = 1e-4
    total_steps: int = 20000   # enough for full flat→blocks→flat traversal
    warmup_steps: int = 500    # CPG ramp-up at start

    # Terrain geometry (mm)
    flat_length: float = 8.0
    blocks_length: float = 15.0
    block_size: float = 1.3
    height_range: tuple = (0.2, 0.4)

    # Controller (shared between fixed and plastic)
    hidden_dim: int = 64
    sparsity: float = 0.8
    cpg_freq: float = 12.0
    cpg_amplitude: float = 1.0
    modulation_scale: float = 0.15  # identical for both controllers
    seed: int = 42

    # Plastic-specific
    plastic_lr: float = 1e-5
    plastic_decay: float = 1.0

    # Recording
    log_interval: int = 50  # log every N steps
    output_dir: str = "logs/terrain_shift"


def _make_arena(config: ExperimentConfig):
    """Create the composite terrain arena."""
    from flygym_env.terrain import FlatThenBlocksTerrain
    return FlatThenBlocksTerrain(
        flat_length=config.flat_length,
        blocks_length=config.blocks_length,
        block_size=config.block_size,
        height_range=config.height_range,
        rand_seed=config.seed,
    )


def _empty_result():
    """Return empty result dict for failed episodes."""
    return {
        "positions": np.zeros((1, 3)),
        "contacts": np.zeros((1, 6)),
        "contact_forces_raw": np.zeros((1, 6)),
        "end_effectors": np.zeros((1, 6, 3)),
        "fly_orientations": np.zeros((1, 3)),
        "weight_drifts": [],
        "joint_angles": np.zeros((1, 42)),
        "joint_names": [],
        "steps_completed": 0,
        "perturbation_idx": None,
        "recovery_idx": None,
    }


def run_continuous_episode(
    controller,
    config: ExperimentConfig,
    label: str = "",
):
    """Run a single continuous episode on FlatThenBlocksTerrain.

    The fly walks flat → blocks → flat in one simulation.
    No sim restarts, no CPG resets between terrain zones.

    Returns dict with positions, contacts, weight_drifts,
    perturbation_idx (when fly enters blocks),
    recovery_idx (when fly exits blocks).
    """
    import flygym

    arena = _make_arena(config)
    fly_obj = flygym.Fly(
        enable_adhesion=True,
        init_pose="stretch",
        control="position",
    )
    sim = flygym.SingleFlySimulation(
        fly=fly_obj,
        arena=arena,
        timestep=config.timestep,
    )

    obs, info = sim.reset()

    # Reset CPG to zero and ramp up (only at episode start)
    controller.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6),
    )
    controller.net.reset_state()

    legs = ["LF", "LM", "LH", "RF", "RM", "RH"]

    # Warmup: let CPG magnitudes converge before data collection
    for _ in range(config.warmup_steps):
        controller.cpg.step()
        joints = []
        adhesion = []
        for i, leg in enumerate(legs):
            joints.append(controller.steps.get_joint_angles(
                leg, controller.cpg.curr_phases[i],
                controller.cpg.curr_magnitudes[i],
            ))
            adhesion.append(controller.steps.get_adhesion_onoff(
                leg, controller.cpg.curr_phases[i],
            ))
        action_dict = {
            "joints": np.concatenate(joints),
            "adhesion": np.array(adhesion).astype(int),
        }
        try:
            obs, _, terminated, truncated, _ = sim.step(action_dict)
            if terminated or truncated:
                print(f"  [{label}] Episode ended during warmup")
                sim.close()
                return _empty_result()
        except Exception as e:
            print(f"  [{label}] Physics error during warmup: {e}")
            sim.close()
            return _empty_result()

    # Main loop — continuous data collection
    positions = []
    contacts = []
    contact_forces_raw = []
    end_effectors_list = []
    fly_orientations = []
    weight_drifts = []
    joint_angles_log = []
    steps_done = 0

    blocks_start = arena.blocks_start
    blocks_end = arena.blocks_end

    # State tracking for structured logging
    state_path = str(Path(config.output_dir) / "logs" / "dashboard_state.json")
    events_path = str(Path(config.output_dir) / "logs" / "events.jsonl")
    entered_blocks = False
    exited_blocks = False

    for step in range(config.total_steps):
        joint_angles = controller.get_action(obs, config.timestep)
        adhesion_arr = controller.get_adhesion()

        action_dict = {
            "joints": joint_angles,
            "adhesion": adhesion_arr,
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action_dict)
        except Exception as e:
            print(f"  [{label}] Physics error at step {step}: {e}")
            break

        steps_done = step + 1

        # Log at intervals
        if step % config.log_interval == 0:
            pos = np.array(obs["fly"][0])
            positions.append(pos)

            if "contact_forces" in obs:
                cf = np.array(obs["contact_forces"])
                magnitudes = np.linalg.norm(cf, axis=1)
                # Binary contact (existing)
                per_leg = np.array([
                    float(magnitudes[i * 5:(i + 1) * 5].max() > 0.01)
                    for i in range(6)
                ])
                contacts.append(per_leg)
                # Raw per-leg force magnitudes (new)
                per_leg_force = np.array([
                    float(magnitudes[i * 5:(i + 1) * 5].max())
                    for i in range(6)
                ])
                contact_forces_raw.append(per_leg_force)
            else:
                contacts.append(np.zeros(6))
                contact_forces_raw.append(np.zeros(6))

            # End effector positions (new)
            if "end_effectors" in obs:
                end_effectors_list.append(np.array(obs["end_effectors"]))
            else:
                end_effectors_list.append(np.zeros((6, 3)))

            # Fly orientation (new)
            if "fly_orientation" in obs:
                fly_orientations.append(np.array(obs["fly_orientation"]))
            else:
                fly_orientations.append(np.zeros(3))

            # Joint angles (42 DOFs)
            joint_angles_log.append(joint_angles.tolist())

            if hasattr(controller, "get_weight_drift"):
                weight_drifts.append(controller.get_weight_drift())

            # Emit terrain transition events
            if not entered_blocks and pos[0] >= blocks_start:
                entered_blocks = True
                append_event(events_path, EventRecord(
                    run_id=label, timestamp=time.time(), step=step,
                    event_type="terrain_transition",
                    details={"from": "flat", "to": "blocks", "x": float(pos[0])},
                ))
            elif entered_blocks and not exited_blocks and pos[0] >= blocks_end:
                exited_blocks = True
                append_event(events_path, EventRecord(
                    run_id=label, timestamp=time.time(), step=step,
                    event_type="terrain_transition",
                    details={"from": "blocks", "to": "flat", "x": float(pos[0])},
                ))

        # Update dashboard state periodically
        if step % 500 == 0 and step > 0 and positions:
            phase = "flat"
            if entered_blocks and not exited_blocks:
                phase = "blocks"
            elif exited_blocks:
                phase = "post"
            last_pos = positions[-1]
            write_state(state_path, {
                "run_id": label,
                "step": step,
                "total_steps": config.total_steps,
                "phase": phase,
                "x_pos": float(last_pos[0]),
                "timestamp": time.time(),
            })

        if terminated or truncated:
            print(f"  [{label}] Episode ended at step {step}")
            break

    sim.close()

    positions = np.array(positions) if positions else np.zeros((1, 3))
    contacts = np.array(contacts) if contacts else np.zeros((1, 6))
    contact_forces_raw = np.array(contact_forces_raw) if contact_forces_raw else np.zeros((1, 6))
    end_effectors_arr = np.array(end_effectors_list) if end_effectors_list else np.zeros((1, 6, 3))
    fly_orientations = np.array(fly_orientations) if fly_orientations else np.zeros((1, 3))

    # Detect phase transitions from x-position
    perturbation_idx = None
    recovery_idx = None
    for i, pos in enumerate(positions):
        if perturbation_idx is None and pos[0] >= blocks_start:
            perturbation_idx = i
        if (perturbation_idx is not None
                and recovery_idx is None
                and pos[0] >= blocks_end):
            recovery_idx = i

    print(f"  [{label}] {steps_done} steps, "
          f"x_final={positions[-1, 0]:.2f}mm, "
          f"perturb@idx={perturbation_idx}, "
          f"recover@idx={recovery_idx}")

    joint_angles_arr = np.array(joint_angles_log) if joint_angles_log else np.zeros((1, 42))

    return {
        "positions": positions,
        "contacts": contacts,
        "contact_forces_raw": contact_forces_raw,
        "end_effectors": end_effectors_arr,
        "fly_orientations": fly_orientations,
        "weight_drifts": weight_drifts,
        "joint_angles": joint_angles_arr,
        "joint_names": list(fly_obj.actuated_joints),
        "steps_completed": steps_done,
        "perturbation_idx": perturbation_idx,
        "recovery_idx": recovery_idx,
    }


def _split_by_phase(data: dict):
    """Split continuous episode data into flat/blocks/post phases."""
    pos = data["positions"]
    con = data["contacts"]
    p_idx = data["perturbation_idx"]
    r_idx = data["recovery_idx"]

    if p_idx is None:
        # Fly never reached blocks — all flat
        return pos, np.zeros((1, 3)), np.zeros((1, 3)), \
               con, np.zeros((1, 6)), np.zeros((1, 6))

    pos_flat = pos[:p_idx] if p_idx > 0 else pos[:1]
    con_flat = con[:p_idx] if p_idx > 0 else con[:1]

    if r_idx is not None:
        pos_blocks = pos[p_idx:r_idx]
        con_blocks = con[p_idx:r_idx]
        pos_post = pos[r_idx:]
        con_post = con[r_idx:]
    else:
        pos_blocks = pos[p_idx:]
        con_blocks = con[p_idx:]
        pos_post = np.zeros((1, 3))
        con_post = np.zeros((1, 6))

    return pos_flat, pos_blocks, pos_post, con_flat, con_blocks, con_post


def run_experiment(config: Optional[ExperimentConfig] = None):
    """Run the full terrain shift experiment.

    Both controllers walk on identical FlatThenBlocksTerrain in
    separate continuous simulations. No restarts between phases.
    """
    if config is None:
        config = ExperimentConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print("=" * 60)
    print("TERRAIN SHIFT EXPERIMENT (continuous)")
    print("=" * 60)
    print(f"Terrain: {config.flat_length}mm flat -> "
          f"{config.blocks_length}mm blocks -> 10mm flat")
    print(f"Total steps: {config.total_steps} "
          f"(+ {config.warmup_steps} warmup)")
    print(f"Modulation scale: {config.modulation_scale} (both controllers)")
    print()

    # Initialize controllers with identical hyperparams
    shared_kwargs = dict(
        hidden_dim=config.hidden_dim,
        sparsity=config.sparsity,
        cpg_freq=config.cpg_freq,
        cpg_amplitude=config.cpg_amplitude,
        timestep=config.timestep,
        modulation_scale=config.modulation_scale,
        seed=config.seed,
    )

    fixed = FixedController(**shared_kwargs)
    plastic = PlasticController(
        **shared_kwargs,
        learning_rate=config.plastic_lr,
        weight_decay=config.plastic_decay,
    )

    # --- Run fixed controller ---
    print("Running fixed controller...")
    t0 = time.time()
    data_fixed = run_continuous_episode(fixed, config, label="fixed")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Run plastic controller ---
    print("\nRunning plastic controller...")
    t0 = time.time()
    data_plastic = run_continuous_episode(plastic, config, label="plastic")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Split into phases for analysis ---
    (pos_f_flat, pos_f_blocks, pos_f_post,
     con_f_flat, con_f_blocks, con_f_post) = _split_by_phase(data_fixed)

    (pos_p_flat, pos_p_blocks, pos_p_post,
     con_p_flat, con_p_blocks, con_p_post) = _split_by_phase(data_plastic)

    # --- Compute metrics per phase ---
    results = {
        "config": asdict(config),
        "baseline": {
            "fixed": {
                "distance": compute_distance(pos_f_flat),
                "symmetry": compute_gait_symmetry(con_f_flat),
            },
            "plastic": {
                "distance": compute_distance(pos_p_flat),
                "symmetry": compute_gait_symmetry(con_p_flat),
            },
        },
        "shift": {
            "fixed": {
                "distance": compute_distance(pos_f_blocks),
                "symmetry": compute_gait_symmetry(con_f_blocks),
            },
            "plastic": {
                "distance": compute_distance(pos_p_blocks),
                "symmetry": compute_gait_symmetry(con_p_blocks),
                "weight_drift": (data_plastic["weight_drifts"][-1]
                                 if data_plastic["weight_drifts"] else 0),
            },
        },
    }

    # Forgetting phase (post-blocks flat)
    if (data_fixed["recovery_idx"] is not None
            or data_plastic["recovery_idx"] is not None):
        results["forgetting"] = {
            "fixed": {
                "distance": compute_distance(pos_f_post),
                "symmetry": compute_gait_symmetry(con_f_post),
            },
            "plastic": {
                "distance": compute_distance(pos_p_post),
                "symmetry": compute_gait_symmetry(con_p_post),
                "weight_drift": (data_plastic["weight_drifts"][-1]
                                 if data_plastic["weight_drifts"] else 0),
            },
        }

    # --- Recovery reports ---
    perturbation_step_f = data_fixed["perturbation_idx"] or 0
    perturbation_step_p = data_plastic["perturbation_idx"] or 0

    report_fixed = compute_recovery_report(
        pos_f_flat, pos_f_blocks,
        con_f_flat, con_f_blocks,
        dt=config.timestep * config.log_interval,
        perturbation_step=perturbation_step_f,
    )
    report_plastic = compute_recovery_report(
        pos_p_flat, pos_p_blocks,
        con_p_flat, con_p_blocks,
        dt=config.timestep * config.log_interval,
        perturbation_step=perturbation_step_p,
        weight_drift=results["shift"]["plastic"].get("weight_drift", 0),
    )

    # --- Compute gait reports ---
    effective_dt = config.timestep * config.log_interval

    gait_reports = {}
    for ctrl_name, data in [("fixed", data_fixed), ("plastic", data_plastic)]:
        gait_reports[ctrl_name] = compute_gait_report(
            data["contact_forces_raw"],
            data["end_effectors"],
            data["fly_orientations"],
            dt=effective_dt,
        )
    results["gait"] = gait_reports

    # --- Generate plots ---
    print("\nGenerating plots...")

    plot_recovery_curves(
        data_fixed["positions"], data_plastic["positions"],
        dt=effective_dt,
        perturbation_step=data_fixed["perturbation_idx"] or 0,
        save_path=str(output_dir / "recovery_curves.png"),
        title="Velocity: Flat -> Blocks -> Flat (continuous)",
    )

    plot_comparison_bars(
        report_fixed, report_plastic,
        save_path=str(output_dir / "comparison.png"),
    )

    plot_gait_symmetry_over_time(
        data_fixed["contacts"], data_plastic["contacts"],
        dt=effective_dt,
        perturbation_step=data_fixed["perturbation_idx"],
        save_path=str(output_dir / "gait_symmetry.png"),
    )

    if data_plastic["weight_drifts"]:
        plot_weight_drift(
            data_plastic["weight_drifts"],
            dt=effective_dt,
            save_path=str(output_dir / "weight_drift.png"),
        )

    # Gait plots (new)
    for ctrl_name, data in [("fixed", data_fixed), ("plastic", data_plastic)]:
        ss = classify_stance_swing(data["contact_forces_raw"])
        plot_contact_raster(
            ss, dt=effective_dt,
            perturbation_step=data.get("perturbation_idx"),
            save_path=str(output_dir / f"contact_raster_{ctrl_name}.png"),
            title=f"Contact Raster ({ctrl_name})",
        )
        tripod = compute_tripod_score(ss)
        plot_tripod_score(
            tripod, dt=effective_dt,
            perturbation_step=data.get("perturbation_idx"),
            save_path=str(output_dir / f"tripod_score_{ctrl_name}.png"),
            title=f"Tripod Coordination ({ctrl_name})",
        )

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Fixed':>10} {'Plastic':>10}")
    print("-" * 52)

    b = results["baseline"]
    s = results["shift"]
    print(f"{'Flat distance (mm)':<30} {b['fixed']['distance']:>10.4f} "
          f"{b['plastic']['distance']:>10.4f}")
    print(f"{'Flat symmetry':<30} {b['fixed']['symmetry']:>10.4f} "
          f"{b['plastic']['symmetry']:>10.4f}")
    print(f"{'Blocks distance (mm)':<30} {s['fixed']['distance']:>10.4f} "
          f"{s['plastic']['distance']:>10.4f}")
    print(f"{'Blocks symmetry':<30} {s['fixed']['symmetry']:>10.4f} "
          f"{s['plastic']['symmetry']:>10.4f}")
    print(f"{'Performance ratio':<30} "
          f"{report_fixed.performance_ratio:>10.4f} "
          f"{report_plastic.performance_ratio:>10.4f}")
    print(f"{'Recovery time (idx)':<30} "
          f"{str(report_fixed.recovery_time_steps):>10} "
          f"{str(report_plastic.recovery_time_steps):>10}")

    if "forgetting" in results:
        fg = results["forgetting"]
        print(f"{'Post-blocks distance (mm)':<30} "
              f"{fg['fixed']['distance']:>10.4f} "
              f"{fg['plastic']['distance']:>10.4f}")
        print(f"{'Post-blocks symmetry':<30} "
              f"{fg['fixed']['symmetry']:>10.4f} "
              f"{fg['plastic']['symmetry']:>10.4f}")

    wd = results["shift"]["plastic"].get("weight_drift", 0)
    print(f"\nPlastic weight drift: {wd:.4f}")

    # Phase detection summary
    print(f"\nFixed:   entered blocks @ idx {data_fixed['perturbation_idx']}, "
          f"exited @ idx {data_fixed['recovery_idx']}")
    print(f"Plastic: entered blocks @ idx {data_plastic['perturbation_idx']}, "
          f"exited @ idx {data_plastic['recovery_idx']}")

    print(f"\nPlots saved to: {output_dir}")

    # Save results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Structured logging ---
    runs_path = str(output_dir / "logs" / "runs.jsonl")
    run_id = f"seed{config.seed}_{int(time.time())}"

    for ctrl_name, report, data in [
        ("fixed", report_fixed, data_fixed),
        ("plastic", report_plastic, data_plastic),
    ]:
        metrics = {
            "distance_before": report.distance_before,
            "distance_after": report.distance_after,
            "performance_ratio": report.performance_ratio,
            "gait_symmetry_before": report.gait_symmetry_before,
            "gait_symmetry_after": report.gait_symmetry_after,
            "recovery_time_steps": report.recovery_time_steps,
            "weight_drift": report.weight_drift,
            "num_falls": report.num_falls,
        }
        # Add gait metrics
        if ctrl_name in gait_reports:
            for k, v in gait_reports[ctrl_name].items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        metrics[f"gait_{k}_{sub_k}"] = sub_v
                else:
                    metrics[f"gait_{k}"] = v

        phases = {}
        for phase_name in ("baseline", "shift", "forgetting"):
            if phase_name in results:
                phases[phase_name] = results[phase_name].get(ctrl_name, {})

        append_run(runs_path, RunRecord(
            run_id=f"{run_id}_{ctrl_name}",
            timestamp=time.time(),
            config=asdict(config),
            controller=ctrl_name,
            metrics=metrics,
            phases=phases,
            plots_dir=str(output_dir),
            duration_s=0,  # filled by caller if desired
            status="completed",
        ))

    return {
        "results": results,
        "report_fixed": report_fixed,
        "report_plastic": report_plastic,
        "data": {
            "fixed": data_fixed,
            "plastic": data_plastic,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Terrain Shift Experiment (continuous)")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--flat-length", type=float, default=8.0)
    parser.add_argument("--blocks-length", type=float, default=15.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--modulation-scale", type=float, default=0.15)
    parser.add_argument("--plastic-lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="logs/terrain_shift")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = ExperimentConfig(
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        flat_length=args.flat_length,
        blocks_length=args.blocks_length,
        hidden_dim=args.hidden_dim,
        sparsity=args.sparsity,
        modulation_scale=args.modulation_scale,
        plastic_lr=args.plastic_lr,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    run_experiment(config)
