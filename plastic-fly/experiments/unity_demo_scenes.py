"""
Unity demo scene generator: 5 stimulus conditions with real Brian2 brain.

Generates Unity-compatible timeseries JSON files for FlyBrainViz playback.
Each scene shows baseline walking (3000 steps) then stimulus response (7000 steps).

Scenes:
    baseline    - unperturbed walking (v2 readout, 20ms brain)
    loom_left   - left LPLC2 injection (v4 readout, 50ms brain)
    loom_right  - right LPLC2 injection (v4 readout, 50ms brain)
    odor_attract - Or42b/DM1 bilateral odor (v2 readout, 20ms brain)
    odor_averse  - Or85a/DM5 bilateral odor (v2 readout, 20ms brain)

Usage:
    python experiments/unity_demo_scenes.py --scene baseline
    python experiments/unity_demo_scenes.py --scene loom_left
    python experiments/unity_demo_scenes.py --all
    python experiments/unity_demo_scenes.py --all --fake-brain   # fast test
"""

import sys
import json
import time
import shutil
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
from bridge.vnc_lite import VNCLite


def _write_json_atomic(path: Path, payload: dict):
    """Write JSON atomically so checkpoints stay readable if the run is interrupted."""
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


SCENE_NAMES = ["baseline", "loom_left", "loom_right", "odor_attract", "odor_averse"]

STIMULUS_ONSET_STEP = 3000
LOG_INTERVAL = 50


# ── Helpers ────────────────────────────────────────────────────────────


def _load_orn_ids(glomerulus: str, brain_repo: Path) -> dict:
    """Load left/right ORN neuron IDs for a glomerulus from FlyWire annotations."""
    import pandas as pd

    ann_path = brain_repo / "flywire_annotations_matched.csv"
    df = pd.read_csv(ann_path, low_memory=False)
    mask = df["cell_type"] == glomerulus
    g = df[mask]
    left = sorted(g[g["side"] == "left"]["root_id"].astype(np.int64).tolist())
    right = sorted(g[g["side"] == "right"]["root_id"].astype(np.int64).tolist())
    return {"left": left, "right": right}


def _build_odor_channel_map(base_cm: dict, orn_left: list, orn_right: list) -> dict:
    """Replace generic olfactory channels with glomerulus-specific ORN IDs."""
    cm = dict(base_cm)
    cm["olfactory_left"] = orn_left
    cm["olfactory_right"] = orn_right
    return cm


def _collect_sensory_ids(channel_map: dict) -> np.ndarray:
    """Collect all unique neuron IDs from a channel map."""
    all_ids = set()
    for ids in channel_map.values():
        all_ids.update(int(x) for x in ids)
    return np.array(sorted(all_ids), dtype=np.int64)


def _compute_tripod_score(contact_binary: list) -> float:
    """Compute tripod gait score from binary contacts [LF,LM,LH,RF,RM,RH].

    Ideal tripod: LF+LH+RM in stance and LM+RF+RH in swing, or vice versa.
    Score is 0-1 indicating how close to ideal tripod.
    """
    if len(contact_binary) != 6:
        return 0.0
    c = np.array(contact_binary)
    # Tripod A: LF, LH, RM  in stance
    tripod_a = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    # Tripod B: LM, RF, RH in stance
    tripod_b = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    match_a = 1.0 - np.mean(np.abs(c - tripod_a))
    match_b = 1.0 - np.mean(np.abs(c - tripod_b))
    return float(max(match_a, match_b))


def _record_frame(obs, positions, joint_angle_frames, contact_binary_frames,
                  contact_force_frames, end_effector_frames, tripod_scores):
    """Extract and append one frame of recording data from an observation."""
    positions.append(np.array(obs["fly"][0]).tolist())
    joint_angle_frames.append(np.array(obs["joints"][0]).tolist())
    raw_cf = np.array(obs["contact_forces"])
    per_leg_forces = []
    per_leg_binary = []
    for leg_i in range(6):
        leg_forces = raw_cf[leg_i * 5:(leg_i + 1) * 5]
        mag = float(np.linalg.norm(leg_forces))
        per_leg_forces.append(mag)
        per_leg_binary.append(1.0 if mag > 0.1 else 0.0)
    contact_force_frames.append(per_leg_forces)
    contact_binary_frames.append(per_leg_binary)
    end_effector_frames.append(np.array(obs["end_effectors"]).tolist())
    tripod_scores.append(_compute_tripod_score(per_leg_binary))


def _save_scene_checkpoint(output_path, scene_name, steps_completed, body_steps,
                           positions, joint_angle_frames, contact_binary_frames,
                           contact_force_frames, end_effector_frames, tripod_scores,
                           joint_names, status):
    """Write an incremental checkpoint for a scene run."""
    checkpoint = {
        "scene": scene_name,
        "summary": {
            "steps_completed": steps_completed,
            "body_steps": body_steps,
            "status": status,
        },
        "n_frames": len(joint_angle_frames),
        "positions": positions,
        "joint_angles": joint_angle_frames,
        "joint_names": joint_names,
        "contacts": contact_binary_frames,
        "contact_forces": contact_force_frames,
        "end_effectors": end_effector_frames,
        "tripod_score": tripod_scores,
    }
    _write_json_atomic(output_path / ("checkpoint_%s.json" % scene_name), checkpoint)


# ── Scene configuration ───────────────────────────────────────────────


def _get_scene_config(scene_name: str, cfg: BridgeConfig) -> dict:
    """Return configuration dict for a given scene.

    Returns:
        dict with keys: sensory_ids, readout_ids, channel_map, decoder_path,
                        rate_scale, brain_dt_ms, body_steps_per_brain,
                        stimulus_fn (callable(step, body_obs) -> body_obs)
    """
    data_dir = cfg.data_dir

    if scene_name in ("loom_left", "loom_right"):
        # ── Looming scenes: v4 populations, 50ms brain window ──
        sensory_ids = np.load(data_dir / "sensory_ids_v4_looming.npy")
        readout_ids = np.load(data_dir / "readout_ids_v4_looming.npy")
        with open(data_dir / "channel_map_v4_looming.json") as f:
            channel_map = json.load(f)
        decoder_path = data_dir / "decoder_groups_v4_looming.json"
        rate_scale = 12.0
        brain_dt_ms = 50.0
        body_steps_per_brain = int(brain_dt_ms / (1e-4 * 1000))  # 500

        loom_left_val = 1.0 if scene_name == "loom_left" else 0.0
        loom_right_val = 1.0 if scene_name == "loom_right" else 0.0

        def stimulus_fn(step, body_obs):
            if step >= STIMULUS_ONSET_STEP:
                body_obs.looming_intensity = np.array(
                    [loom_left_val, loom_right_val], dtype=np.float32
                )
            return body_obs

        return {
            "sensory_ids": sensory_ids,
            "readout_ids": readout_ids,
            "channel_map": channel_map,
            "decoder_path": decoder_path,
            "rate_scale": rate_scale,
            "brain_dt_ms": brain_dt_ms,
            "body_steps_per_brain": body_steps_per_brain,
            "stimulus_fn": stimulus_fn,
        }

    elif scene_name in ("odor_attract", "odor_averse"):
        # ── Odor scenes: v2 readout + glomerulus ORNs, 20ms brain ──
        readout_ids = np.load(data_dir / "readout_ids_v2.npy")
        decoder_path = data_dir / "decoder_groups_v2.json"
        rate_scale = 15.0
        brain_dt_ms = 20.0
        body_steps_per_brain = cfg.body_steps_per_brain  # 200

        # Load base channel map (non-olfactory channels)
        with open(data_dir / "channel_map_v3.json") as f:
            base_cm = json.load(f)

        # Load glomerulus-specific ORNs
        if scene_name == "odor_attract":
            orn_ids = _load_orn_ids("ORN_DM1", cfg.brain_repo_root)
        else:
            orn_ids = _load_orn_ids("ORN_DM5", cfg.brain_repo_root)

        channel_map = _build_odor_channel_map(
            base_cm, orn_ids["left"], orn_ids["right"]
        )
        sensory_ids = _collect_sensory_ids(channel_map)

        # Bilateral odor: high on left, low on right (asymmetric stimulus)
        odor_high = 1.0
        odor_low = 0.0
        synthetic_odor = np.array(
            [[odor_high, odor_low, odor_high, odor_low]], dtype=np.float32
        )

        def stimulus_fn(step, body_obs):
            if step >= STIMULUS_ONSET_STEP:
                body_obs.odor_intensity = synthetic_odor
            return body_obs

        return {
            "sensory_ids": sensory_ids,
            "readout_ids": readout_ids,
            "channel_map": channel_map,
            "decoder_path": decoder_path,
            "rate_scale": rate_scale,
            "brain_dt_ms": brain_dt_ms,
            "body_steps_per_brain": body_steps_per_brain,
            "stimulus_fn": stimulus_fn,
        }

    else:
        # ── Baseline: v2 readout, default sensory, 20ms brain, no stimulus ──
        sensory_ids = np.load(cfg.sensory_ids_path)
        readout_ids = np.load(data_dir / "readout_ids_v2.npy")
        decoder_path = data_dir / "decoder_groups_v2.json"
        rate_scale = 15.0
        brain_dt_ms = 20.0
        body_steps_per_brain = cfg.body_steps_per_brain  # 200

        if cfg.channel_map_path.exists():
            with open(cfg.channel_map_path) as f:
                channel_map = json.load(f)
        else:
            channel_map = None

        def stimulus_fn(step, body_obs):
            return body_obs  # no stimulus

        return {
            "sensory_ids": sensory_ids,
            "readout_ids": readout_ids,
            "channel_map": channel_map,
            "decoder_path": decoder_path,
            "rate_scale": rate_scale,
            "brain_dt_ms": brain_dt_ms,
            "body_steps_per_brain": body_steps_per_brain,
            "stimulus_fn": stimulus_fn,
        }


# ── Main scene runner ─────────────────────────────────────────────────


def run_scene(
    scene_name: str,
    body_steps: int = 10000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    output_dir: str = "logs/unity_scenes",
):
    """Run a single demo scene and export Unity-compatible JSON.

    Returns the path to the saved JSON file, or None on failure.
    """
    import flygym

    assert scene_name in SCENE_NAMES, (
        "Unknown scene '%s'. Choose from: %s" % (scene_name, SCENE_NAMES)
    )

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SCENE: %s" % scene_name)
    print("=" * 60)

    # ── Load scene-specific configuration ──
    sc = _get_scene_config(scene_name, cfg)

    sensory_ids = sc["sensory_ids"]
    readout_ids = sc["readout_ids"]
    channel_map = sc["channel_map"]
    decoder_path = sc["decoder_path"]
    rate_scale = sc["rate_scale"]
    brain_dt_ms = sc["brain_dt_ms"]
    bspb = sc["body_steps_per_brain"]
    stimulus_fn = sc["stimulus_fn"]

    print("  Sensory: %d neurons" % len(sensory_ids))
    print("  Readout: %d neurons" % len(readout_ids))
    print("  Brain dt: %.0f ms, body steps/brain: %d" % (brain_dt_ms, bspb))
    print("  Rate scale: %.1f" % rate_scale)
    print("  Stimulus onset: step %d" % STIMULUS_ONSET_STEP)

    # ── Initialize components ──
    if channel_map is not None:
        encoder = SensoryEncoder(
            sensory_neuron_ids=sensory_ids,
            channel_map=channel_map,
            max_rate_hz=cfg.max_rate_hz,
            baseline_rate_hz=cfg.baseline_rate_hz,
        )
    else:
        encoder = SensoryEncoder(sensory_ids, max_rate_hz=cfg.max_rate_hz)

    decoder = DescendingDecoder.from_json(decoder_path, rate_scale=rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()
    vnc = VNCLite()

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    print("  Brain: %s" % brain_label)
    brain = create_brain_runner(
        sensory_ids=sensory_ids,
        readout_ids=readout_ids,
        use_fake=use_fake_brain,
        warmup_ms=cfg.brain_warmup_ms,
    )

    # ── Initialize FlyGym ──
    print("  Initializing FlyGym...")
    fly_obj = flygym.Fly(
        enable_adhesion=True, init_pose="stretch", control="position"
    )
    arena = flygym.arena.FlatTerrain()
    sim = flygym.SingleFlySimulation(fly=fly_obj, arena=arena, timestep=1e-4)
    obs, info = sim.reset()

    # Warmup CPG
    print("  Warming up locomotion (%d steps)..." % warmup_steps)
    locomotion.warmup(0)
    locomotion.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6),
    )
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, terminated, truncated, info = sim.step(action)
            if terminated or truncated:
                print("  Episode ended during warmup!")
                sim.close()
                return None
        except Exception as e:
            print("  Physics error during warmup: %s" % e)
            sim.close()
            return None

    # ── Main closed loop ──
    print("\n  Running: %d body steps, brain every %d steps, %s" % (
        body_steps, bspb, brain_label))
    print()

    # Recording buffers
    positions = []
    joint_angle_frames = []
    contact_binary_frames = []
    contact_force_frames = []
    end_effector_frames = []
    tripod_scores = []

    brain_steps = 0
    current_cmd = LocomotionCommand(forward_drive=1.0)
    step = 0

    t_start = time.time()
    t_brain = 0.0

    for step in range(body_steps):
        # ── Brain step ──
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)

            # Apply stimulus injection (only after onset step)
            body_obs = stimulus_fn(step, body_obs)

            brain_input = encoder.encode(body_obs)

            tb0 = time.time()
            brain_output = brain.step(brain_input, sim_ms=brain_dt_ms)
            t_brain += time.time() - tb0

            # VNC-lite motor layer
            group_rates = decoder.get_group_rates(brain_output)
            current_cmd = vnc.step(
                group_rates, dt_s=brain_dt_ms / 1000.0, body_obs=body_obs
            )
            brain_steps += 1

            if brain_steps % 10 == 1:
                mean_rate = float(np.mean(brain_output.firing_rates_hz))
                active = int(np.sum(brain_output.firing_rates_hz > 0))
                stim_tag = " [STIM]" if step >= STIMULUS_ONSET_STEP else ""
                print(
                    "  brain #%3d: fwd=%+.3f turn=%+.3f freq=%.2f"
                    " | rate=%.0fHz active=%d/%d%s"
                    % (
                        brain_steps,
                        current_cmd.forward_drive,
                        current_cmd.turn_drive,
                        current_cmd.step_frequency,
                        mean_rate,
                        active,
                        len(readout_ids),
                        stim_tag,
                    )
                )

        # ── Body step ──
        action = locomotion.step(current_cmd)
        try:
            obs, _, terminated, truncated, info = sim.step(action)
        except Exception as e:
            print("  Physics error at step %d: %s" % (step, e))
            break

        # ── Record every LOG_INTERVAL steps ──
        if step % LOG_INTERVAL == 0:
            _record_frame(obs, positions, joint_angle_frames,
                          contact_binary_frames, contact_force_frames,
                          end_effector_frames, tripod_scores)
            _save_scene_checkpoint(
                output_path, scene_name, step + 1, body_steps,
                positions, joint_angle_frames, contact_binary_frames,
                contact_force_frames, end_effector_frames, tripod_scores,
                list(fly_obj.actuated_joints) if hasattr(fly_obj, "actuated_joints") else [],
                "running",
            )

        if terminated or truncated:
            print("  Episode ended at step %d" % step)
            break

    sim.close()
    elapsed = time.time() - t_start

    # ── Summary ──
    print("\n  Done: %d body steps, %d brain steps in %.1fs" % (
        step + 1, brain_steps, elapsed))
    print("  Brain time: %.1fs (%.0f%%)" % (t_brain, t_brain / max(elapsed, 0.01) * 100))

    if len(positions) >= 2:
        start_pos = np.array(positions[0])
        end_pos = np.array(positions[-1])
        dist = np.linalg.norm(end_pos - start_pos)
        print("  Distance: %.2fmm | Final: (%.2f, %.2f, %.2f)" % (
            dist, end_pos[0], end_pos[1], end_pos[2]))

    # ── Build Unity JSON ──
    n_frames = len(joint_angle_frames)
    if n_frames == 0:
        print("  ERROR: No frames recorded!")
        return None

    dt = LOG_INTERVAL * 1e-4  # timestep * log_interval

    # Get joint names from fly object
    joint_names = (
        list(fly_obj.actuated_joints)
        if hasattr(fly_obj, "actuated_joints")
        else []
    )

    unity_ts = {
        "controller": "brain_driven",
        "dt": dt,
        "n_frames": n_frames,
        "positions": positions[:n_frames],
        "contacts": contact_binary_frames[:n_frames],
        "contact_forces": contact_force_frames[:n_frames],
        "end_effectors": end_effector_frames[:n_frames],
        "joint_angles": joint_angle_frames[:n_frames],
        "joint_names": joint_names,
        "tripod_score": tripod_scores[:n_frames],
        "weight_drifts": [],
        "perturbation_idx": 0,
    }

    # Save to logs
    out_file = output_path / ("timeseries_%s.json" % scene_name)
    _write_json_atomic(out_file, unity_ts)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print("\n  Saved: %s (%d frames, %.1f MB)" % (out_file, n_frames, size_mb))

    # Copy to Unity Resources
    unity_res = (
        Path(__file__).resolve().parent.parent.parent
        / "FlyBrainViz"
        / "Assets"
        / "Resources"
    )
    if unity_res.exists():
        dst = unity_res / ("timeseries_%s.json" % scene_name)
        shutil.copy2(out_file, dst)
        print("  Copied to %s" % dst)
    else:
        print("  Unity Resources dir not found: %s" % unity_res)

    return str(out_file)


# ── Entry point ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate Unity demo scene timeseries from Brian2 brain"
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=SCENE_NAMES,
        default=None,
        help="Scene to generate (one of: %s)" % ", ".join(SCENE_NAMES),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all 5 scenes",
    )
    parser.add_argument(
        "--body-steps",
        type=int,
        default=10000,
        help="Number of body simulation steps (default: 10000)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="CPG warmup steps (default: 500)",
    )
    parser.add_argument(
        "--fake-brain",
        action="store_true",
        help="Use fake brain for fast testing (no Brian2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        default="logs/unity_scenes",
    )
    args = parser.parse_args()

    if not args.all and args.scene is None:
        parser.error("Specify --scene <name> or --all")

    scenes = SCENE_NAMES if args.all else [args.scene]

    print("#" * 60)
    print("# UNITY DEMO SCENE GENERATOR")
    print("# Scenes: %s" % ", ".join(scenes))
    print("# Body steps: %d, Warmup: %d" % (args.body_steps, args.warmup_steps))
    print("# Brain: %s" % ("FAKE" if args.fake_brain else "Brian2 LIF"))
    print("#" * 60)

    results = {}
    t0 = time.time()

    for scene_name in scenes:
        print("\n")
        out = run_scene(
            scene_name=scene_name,
            body_steps=args.body_steps,
            warmup_steps=args.warmup_steps,
            use_fake_brain=args.fake_brain,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        results[scene_name] = out

    total_time = time.time() - t0

    # ── Summary ──
    print("\n" + "=" * 60)
    print("ALL SCENES COMPLETE (%.1fs total)" % total_time)
    print("=" * 60)
    for name, path in results.items():
        status = path if path else "FAILED"
        print("  %-15s %s" % (name, status))

    # Save manifest
    output_path = Path(args.output_dir)
    manifest = {
        "scenes": list(results.keys()),
        "body_steps": args.body_steps,
        "warmup_steps": args.warmup_steps,
        "fake_brain": args.fake_brain,
        "seed": args.seed,
        "total_time_s": total_time,
        "files": {k: v for k, v in results.items() if v is not None},
    }
    manifest_file = output_path / "manifest.json"
    _write_json_atomic(manifest_file, manifest)
    print("\nManifest: %s" % manifest_file)


if __name__ == "__main__":
    main()
