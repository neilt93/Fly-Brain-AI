"""
Phototaxis experiment: does the connectome produce light-directed turning?

Tests whether FlyWire visual circuit wiring (R7/R8 photoreceptors -> descending
neurons) produces turning behavior toward a light source, using synthetic
bilateral luminance injection.

Protocol:
  Inject asymmetric light signal into visual_left / visual_right channels
  (50 R7+R8 photoreceptors per side from channel_map_v3.json).
  Measure turn_drive from descending decoder.

  Conditions:
    LIGHT_LEFT:  left eye bright (0.8), right eye dim (0.2)
    LIGHT_RIGHT: right eye bright (0.8), left eye dim (0.2)

  Prediction (positive phototaxis):
    LIGHT_LEFT  -> turn LEFT  (negative turn_drive)
    LIGHT_RIGHT -> turn RIGHT (positive turn_drive)

  Control: shuffled connectome should abolish directional turning.

Usage:
    python experiments/phototaxis.py                   # real brain
    python experiments/phototaxis.py --fake-brain      # fast test
    python experiments/phototaxis.py --no-shuffle      # skip shuffled control
    python experiments/phototaxis.py --lum-high 0.9    # brighter stimulus
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import BodyObservation, BrainInput, LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter


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


def _save_trial_checkpoint(
    checkpoint_dir: Path,
    label: str,
    steps_completed: int,
    body_steps: int,
    episode_log: list,
    status: str,
):
    """Write incremental checkpoint for a single phototaxis trial."""
    checkpoint = {
        "label": label,
        "summary": {
            "steps_completed": steps_completed,
            "body_steps": body_steps,
            "status": status,
        },
        "episode_log": episode_log,
    }
    _write_json_atomic(checkpoint_dir / f"{label}.json", checkpoint)


def build_visual_channel_map(base_channel_map: dict, visual_only: bool = False) -> dict:
    """Return channel map for phototaxis.

    Args:
        visual_only: If True, strip non-visual channels to isolate the
                     visual circuit (removes proprioceptive/gustatory/etc
                     baseline bias).
    """
    if visual_only:
        return {
            k: v for k, v in base_channel_map.items()
            if k.startswith("visual_")
        }
    return dict(base_channel_map)


def build_sensory_ids(channel_map: dict) -> np.ndarray:
    """Collect all unique neuron IDs from a channel map."""
    all_ids = set()
    for ids in channel_map.values():
        all_ids.update(int(x) for x in ids)
    return np.array(sorted(all_ids), dtype=np.int64)


def make_synthetic_vision(left_lum: float, right_lum: float) -> np.ndarray:
    """Create synthetic vision array: (2, 721, 2) matching FlyGym format.

    Each eye gets uniform luminance across all 721 ommatidia and 2 channels.
    FlyGym vision values are 0-1 (already normalized).
    """
    vision = np.zeros((2, 721, 2), dtype=np.float32)
    vision[0, :, :] = left_lum   # left eye
    vision[1, :, :] = right_lum  # right eye
    return vision


def run_phototaxis_trial(
    label: str,
    sensory_ids: np.ndarray,
    channel_map: dict,
    readout_ids: np.ndarray,
    lum_left: float,
    lum_right: float,
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    shuffle_seed: int | None = None,
    decoder_path: str | Path | None = None,
    rate_scale: float | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Run a single phototaxis trial with specified light asymmetry.

    Injects synthetic vision (bilateral luminance) into the body observation,
    letting the sensory encoder drive R7/R8 photoreceptors at different rates.

    Returns dict with turn_drive time series and summary stats.
    """
    import flygym

    cfg = BridgeConfig()

    encoder = SensoryEncoder(
        sensory_neuron_ids=sensory_ids,
        channel_map=channel_map,
        max_rate_hz=cfg.max_rate_hz,
        baseline_rate_hz=cfg.baseline_rate_hz,
    )
    dec_path = decoder_path or cfg.decoder_groups_path
    dec_scale = rate_scale or cfg.rate_scale
    decoder = DescendingDecoder.from_json(dec_path, rate_scale=dec_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain = create_brain_runner(
        sensory_ids=sensory_ids,
        readout_ids=readout_ids,
        use_fake=use_fake_brain,
        warmup_ms=cfg.brain_warmup_ms,
        shuffle_seed=shuffle_seed,
    )

    fly_obj = flygym.Fly(
        enable_adhesion=True,
        init_pose="stretch",
        control="position",
    )
    sim = flygym.SingleFlySimulation(fly=fly_obj, timestep=1e-4)
    obs, info = sim.reset()

    # Warmup
    locomotion.warmup(0)
    locomotion.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6),
    )
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        obs, _, terminated, truncated, info = sim.step(action)
        if terminated or truncated:
            sim.close()
            return {"label": label, "error": "terminated during warmup"}

    # Synthetic vision: constant bilateral luminance
    synthetic_vision = make_synthetic_vision(lum_left, lum_right)

    # Main loop
    bspb = cfg.body_steps_per_brain
    current_cmd = LocomotionCommand(forward_drive=1.0)
    episode_log = []
    brain_steps = 0
    t0 = time.time()

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            # Override vision with synthetic bilateral luminance
            body_obs.vision = synthetic_vision

            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            current_cmd = decoder.decode(brain_output)
            brain_steps += 1

            mean_rate = float(np.mean(brain_output.firing_rates_hz))
            active = int(np.sum(brain_output.firing_rates_hz > 0))

            episode_log.append({
                "body_step": step,
                "brain_step": brain_steps,
                "forward_drive": current_cmd.forward_drive,
                "turn_drive": current_cmd.turn_drive,
                "step_frequency": current_cmd.step_frequency,
                "stance_gain": current_cmd.stance_gain,
                "readout_mean_hz": mean_rate,
                "readout_active": active,
            })

            if checkpoint_dir is not None and brain_steps % 10 == 0:
                _save_trial_checkpoint(
                    checkpoint_dir, label, step + 1, body_steps,
                    episode_log, status="running",
                )

        action = locomotion.step(current_cmd)
        try:
            obs, _, terminated, truncated, info = sim.step(action)
        except Exception as e:
            print("  Physics error at step %d: %s" % (step, e))
            break
        if terminated or truncated:
            break

    sim.close()
    elapsed = time.time() - t0

    # Compute summary
    if episode_log:
        turns = [e["turn_drive"] for e in episode_log]
        fwds = [e["forward_drive"] for e in episode_log]
        # Skip first 2 brain steps (brain still settling)
        settle = min(2, len(turns))
        turns_steady = turns[settle:]
        mean_turn = float(np.mean(turns_steady)) if turns_steady else 0.0
        std_turn = float(np.std(turns_steady)) if turns_steady else 0.0
    else:
        mean_turn = std_turn = 0.0

    result = {
        "label": label,
        "lum_left": lum_left,
        "lum_right": lum_right,
        "body_steps": step + 1,
        "brain_steps": brain_steps,
        "elapsed_s": elapsed,
        "mean_turn_drive": mean_turn,
        "std_turn_drive": std_turn,
        "mean_forward_drive": float(np.mean(fwds)) if episode_log else 0.0,
        "episode_log": episode_log,
    }

    print("  %s: turn=%+.4f (std=%.4f), fwd=%.3f, %d brain steps in %.1fs" % (
        label, mean_turn, std_turn, result["mean_forward_drive"],
        brain_steps, elapsed))

    return result


def _run_condition_set(
    conditions: list,
    readout_ids: np.ndarray,
    body_steps: int,
    warmup_steps: int,
    use_fake_brain: bool,
    seeds: list[int],
    shuffle_seed: int | None = None,
    decoder_path: str | Path | None = None,
    rate_scale: float | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Run a set of conditions across seeds, returning {cond_name: [results]}."""
    tag = "SHUFFLED" if shuffle_seed is not None else "REAL"
    all_results = {}
    for cond_name, sensory, cm, ll, lr, desc in conditions:
        print("\n" + "=" * 60)
        print("[%s] CONDITION: %s" % (tag, desc))
        print("  lum_left=%.2f, lum_right=%.2f" % (ll, lr))
        print("=" * 60)

        cond_results = []
        for seed in seeds:
            print("\n  --- Seed %d ---" % seed)
            result = run_phototaxis_trial(
                label="%s_%s_seed%d" % (cond_name, tag.lower(), seed),
                sensory_ids=sensory,
                channel_map=cm,
                readout_ids=readout_ids,
                lum_left=ll,
                lum_right=lr,
                body_steps=body_steps,
                warmup_steps=warmup_steps,
                use_fake_brain=use_fake_brain,
                seed=seed,
                shuffle_seed=shuffle_seed,
                decoder_path=decoder_path,
                rate_scale=rate_scale,
                checkpoint_dir=checkpoint_dir,
            )
            cond_results.append(result)
        all_results[cond_name] = cond_results
    return all_results


def _analyze_results(all_results: dict, label: str) -> dict:
    """Compute turn contrasts from a condition set."""
    def cond_stats(results):
        turns = [r["mean_turn_drive"] for r in results if "error" not in r]
        if not turns:
            return 0.0, 0.0, 0
        return float(np.mean(turns)), float(np.std(turns)), len(turns)

    left_mean, left_std, nl = cond_stats(all_results["light_left"])
    right_mean, right_std, nr = cond_stats(all_results["light_right"])

    # Positive phototaxis: left light -> negative turn (toward left)
    #                       right light -> positive turn (toward right)
    # Turn contrast = left_mean - right_mean (expect negative)
    turn_contrast = left_mean - right_mean

    # Directional index: do left and right conditions produce opposite turns?
    directional = right_mean - left_mean  # expect positive for phototaxis

    print("\n--- %s ---" % label)
    print("Condition       Mean Turn    Std      N")
    print("-" * 50)
    print("LIGHT LEFT      %+.4f      %.4f   %d" % (left_mean, left_std, nl))
    print("LIGHT RIGHT     %+.4f      %.4f   %d" % (right_mean, right_std, nr))
    print("Turn contrast (L-R): %+.4f" % turn_contrast)
    print("Directional index:   %+.4f (expect >0 for phototaxis)" % directional)

    return {
        "light_left_turn": left_mean,
        "light_left_std": left_std,
        "light_right_turn": right_mean,
        "light_right_std": right_std,
        "turn_contrast": turn_contrast,
        "directional_index": directional,
    }


def run_phototaxis(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    lum_high: float = 0.8,
    lum_low: float = 0.2,
    seeds: list[int] = None,
    run_shuffled: bool = True,
    readout_version: int = 2,
    rate_scale: float = 15.0,
    visual_only: bool = False,
    output_dir: str = "logs/phototaxis",
):
    if seeds is None:
        seeds = [42, 43, 44]

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load channel map with visual channels
    base_channel_map_path = cfg.data_dir / "channel_map_v3.json"
    with open(base_channel_map_path) as f:
        base_cm = json.load(f)

    # Count visual neurons
    n_vis_l = len(base_cm.get("visual_left", []))
    n_vis_r = len(base_cm.get("visual_right", []))
    print("Visual channels: %d left R7/R8, %d right R7/R8" % (n_vis_l, n_vis_r))

    if readout_version == 3:
        readout_ids = np.load(cfg.data_dir / "readout_ids_v3.npy")
        decoder_path = cfg.data_dir / "decoder_groups_v3.json"
    elif readout_version == 2:
        readout_ids = np.load(cfg.data_dir / "readout_ids_v2.npy")
        decoder_path = cfg.data_dir / "decoder_groups_v2.json"
    else:
        readout_ids = np.load(cfg.readout_ids_path)
        decoder_path = cfg.decoder_groups_path
    print("Readout v%d: %d neurons, rate_scale=%.1f" % (readout_version, len(readout_ids), rate_scale))

    # Build channel map and sensory IDs
    cm = build_visual_channel_map(base_cm, visual_only=visual_only)
    sensory_ids = build_sensory_ids(cm)
    n_channels = len(cm)
    mode_str = "VISUAL-ONLY" if visual_only else "full"
    print("Sensory population: %d neurons (%d channels, %s)" % (len(sensory_ids), n_channels, mode_str))

    conditions = [
        ("light_left",  sensory_ids, cm, lum_high, lum_low,  "Light LEFT (left eye bright)"),
        ("light_right", sensory_ids, cm, lum_low,  lum_high, "Light RIGHT (right eye bright)"),
    ]

    # --- Real connectome ---
    print("\n" + "#" * 60)
    print("# REAL CONNECTOME")
    print("#" * 60)
    real_results = _run_condition_set(
        conditions, readout_ids, body_steps, warmup_steps,
        use_fake_brain, seeds, shuffle_seed=None,
        decoder_path=decoder_path, rate_scale=rate_scale,
        checkpoint_dir=output_path / "checkpoints",
    )

    # --- Shuffled connectome control ---
    shuf_results = None
    if run_shuffled:
        print("\n" + "#" * 60)
        print("# SHUFFLED CONNECTOME (control)")
        print("#" * 60)
        shuf_results = _run_condition_set(
            conditions, readout_ids, body_steps, warmup_steps,
            use_fake_brain, seeds, shuffle_seed=999,
            decoder_path=decoder_path, rate_scale=rate_scale,
            checkpoint_dir=output_path / "checkpoints",
        )

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("PHOTOTAXIS RESULTS")
    print("=" * 60)

    real_summary = _analyze_results(real_results, "REAL CONNECTOME")
    shuf_summary = None
    if shuf_results is not None:
        shuf_summary = _analyze_results(shuf_results, "SHUFFLED CONNECTOME")

    # --- Tests ---
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    tests = []

    l_turn = real_summary["light_left_turn"]
    r_turn = real_summary["light_right_turn"]
    contrast = real_summary["turn_contrast"]
    directional = real_summary["directional_index"]

    # Test 1: Left light produces leftward turning (negative turn_drive)
    left_toward = l_turn < 0
    tests.append({"name": "left_light_toward", "passed": left_toward,
                   "detail": "turn=%+.4f (expect <0)" % l_turn})
    print("  [%s] Left light -> turn left: %+.4f" % ("PASS" if left_toward else "FAIL", l_turn))

    # Test 2: Right light produces rightward turning (positive turn_drive)
    right_toward = r_turn > 0
    tests.append({"name": "right_light_toward", "passed": right_toward,
                   "detail": "turn=%+.4f (expect >0)" % r_turn})
    print("  [%s] Right light -> turn right: %+.4f" % ("PASS" if right_toward else "FAIL", r_turn))

    # Test 3: Directional turning (left and right conditions differ)
    has_direction = directional > 0
    tests.append({"name": "directional_phototaxis", "passed": has_direction,
                   "detail": "directional=%+.4f (expect >0)" % directional})
    print("  [%s] Directional phototaxis: %+.4f" % ("PASS" if has_direction else "FAIL", directional))

    # Tests 4-5: Shuffled control
    if shuf_summary is not None:
        shuf_dir = shuf_summary["directional_index"]

        # Test 4: Real directional index > shuffled
        specificity = abs(directional) > abs(shuf_dir)
        tests.append({"name": "connectome_specificity", "passed": specificity,
                       "detail": "real=%.4f, shuffled=%.4f" % (abs(directional), abs(shuf_dir))})
        print("  [%s] Connectome specificity: |real|=%.4f > |shuffled|=%.4f" % (
            "PASS" if specificity else "FAIL", abs(directional), abs(shuf_dir)))

        # Test 5: Shuffled has near-zero directionality
        shuf_weak = abs(shuf_dir) < abs(directional) * 0.5
        tests.append({"name": "shuffled_weak", "passed": shuf_weak,
                       "detail": "shuffled=%.4f < 50%% of real=%.4f" % (abs(shuf_dir), abs(directional))})
        print("  [%s] Shuffled weak: |%.4f| < 50%% of |%.4f|" % (
            "PASS" if shuf_weak else "FAIL", shuf_dir, directional))

    n_pass = sum(1 for t in tests if t["passed"])
    print("\n  %d/%d tests passed" % (n_pass, len(tests)))

    # --- Save ---
    def strip_logs(results_dict):
        out = {}
        for cond_name, results in results_dict.items():
            out[cond_name] = []
            for r in results:
                r_copy = {k: v for k, v in r.items() if k != "episode_log"}
                r_copy["episode_log_length"] = len(r.get("episode_log", []))
                out[cond_name].append(r_copy)
        return out

    output_data = {
        "config": {
            "body_steps": body_steps,
            "warmup_steps": warmup_steps,
            "lum_high": lum_high,
            "lum_low": lum_low,
            "seeds": seeds,
            "use_fake_brain": use_fake_brain,
            "run_shuffled": run_shuffled,
            "readout_version": readout_version,
            "rate_scale": rate_scale,
            "n_visual_left": n_vis_l,
            "n_visual_right": n_vis_r,
            "visual_only": visual_only,
            "n_sensory_total": len(sensory_ids),
            "n_readout": len(readout_ids),
        },
        "real": {"summary": real_summary, "trials": strip_logs(real_results)},
        "tests": tests,
    }
    if shuf_summary is not None:
        output_data["shuffled"] = {
            "summary": shuf_summary,
            "trials": strip_logs(shuf_results),
        }

    out_file = output_path / "phototaxis_results.json"
    _write_json_atomic(out_file, output_data)
    print("\nSaved to %s" % out_file)

    return output_data


def generate_proof_figure(results_path: str = "logs/phototaxis/phototaxis_results.json",
                          output_path: str = "figures/phototaxis_proof.png"):
    """Generate 3-panel proof figure from saved results."""
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        data = json.load(f)

    real = data["real"]["summary"]
    has_shuffled = "shuffled" in data

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Turn drive by condition (real connectome)
    ax = axes[0]
    conditions = ["Light LEFT", "Light RIGHT"]
    means = [real["light_left_turn"], real["light_right_turn"]]
    stds = [real["light_left_std"], real["light_right_std"]]
    colors = ["#FFD700", "#FF8C00"]  # gold, dark orange
    bars = ax.bar(conditions, means, yerr=stds, color=colors, edgecolor="black",
                  capsize=8, linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Turn Drive")
    ax.set_title("Real Connectome:\nLight-Directed Turning")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                "%+.4f" % m, ha="center", va="bottom" if m >= 0 else "top",
                fontweight="bold", fontsize=10)

    # Panel 2: Real vs Shuffled directional index
    ax = axes[1]
    labels = ["Real"]
    values = [real["directional_index"]]
    bar_colors = ["#2196F3"]
    if has_shuffled:
        shuf = data["shuffled"]["summary"]
        labels.append("Shuffled")
        values.append(shuf["directional_index"])
        bar_colors.append("#9E9E9E")
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="black", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Directional Index (R-L turn)")
    ax.set_title("Connectome Specificity:\nReal vs Shuffled Wiring")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                "%+.4f" % v, ha="center", va="bottom" if v >= 0 else "top",
                fontweight="bold", fontsize=10)

    # Panel 3: Summary stats table
    ax = axes[2]
    ax.axis("off")
    tests = data.get("tests", [])
    table_data = []
    for t in tests:
        status = "PASS" if t["passed"] else "FAIL"
        table_data.append([t["name"].replace("_", " ").title(), status, t["detail"]])

    if table_data:
        table = ax.table(cellText=table_data,
                         colLabels=["Test", "Result", "Detail"],
                         loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)

        # Color pass/fail cells
        for i, t in enumerate(tests):
            cell = table[i + 1, 1]
            if t["passed"]:
                cell.set_facecolor("#C8E6C9")
            else:
                cell.set_facecolor("#FFCDD2")

    n_pass = sum(1 for t in tests if t["passed"])
    ax.set_title("Validation: %d/%d Tests Passed" % (n_pass, len(tests)))

    fig.suptitle("Phototaxis: Connectome-Driven Light-Directed Turning\n"
                 "R7/R8 Photoreceptors -> FlyWire Visual Circuits -> Descending Neurons",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print("Saved figure to %s" % output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phototaxis experiment")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--lum-high", type=float, default=0.8)
    parser.add_argument("--lum-low", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--no-shuffle", action="store_true", help="Skip shuffled control")
    parser.add_argument("--readout-version", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--rate-scale", type=float, default=15.0)
    parser.add_argument("--visual-only", action="store_true",
                        help="Use only visual channels (isolate visual circuit)")
    parser.add_argument("--output-dir", default="logs/phototaxis")
    parser.add_argument("--figure-only", action="store_true", help="Generate figure from saved results")
    args = parser.parse_args()

    if args.figure_only:
        generate_proof_figure(
            results_path=str(Path(args.output_dir) / "phototaxis_results.json"),
            output_path="figures/phototaxis_proof.png",
        )
    else:
        run_phototaxis(
            body_steps=args.body_steps,
            warmup_steps=args.warmup_steps,
            use_fake_brain=args.fake_brain,
            lum_high=args.lum_high,
            lum_low=args.lum_low,
            seeds=args.seeds,
            run_shuffled=not args.no_shuffle,
            readout_version=args.readout_version,
            rate_scale=args.rate_scale,
            visual_only=args.visual_only,
            output_dir=args.output_dir,
        )
