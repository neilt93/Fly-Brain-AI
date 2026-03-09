"""
Odor valence experiment: Or42b (attractive) vs Or85a (aversive) olfactory circuits.

Tests whether the connectome's wiring produces opposite turning behavior for
food-attractive vs aversive odors, using glomerulus-specific ORN populations
from FlyWire annotations:

  - Or42b -> DM1 glomerulus (ethyl acetate, vinegar — attractive)
  - Or85a -> DM5 glomerulus (ethyl 3-hydroxybutyrate — aversive)

Protocol:
  For each ORN type, we build a sensory population that replaces the generic
  olfactory channel with identified glomerulus-specific neurons. We then inject
  asymmetric odor (left > right or right > left) and measure the resulting
  turn_drive from the descending decoder.

  Prediction:
    Attractive (DM1): left odor -> turn LEFT (toward), right odor -> turn RIGHT
    Aversive (DM5):   left odor -> turn RIGHT (away),  right odor -> turn LEFT

Usage:
    python experiments/odor_valence.py                   # real brain
    python experiments/odor_valence.py --fake-brain      # fast test
    python experiments/odor_valence.py --body-steps 8000 # longer run
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


def load_orn_ids(glomerulus: str, brain_repo: Path) -> dict:
    """Load left/right ORN neuron IDs for a glomerulus from FlyWire annotations.

    Returns {'left': [...], 'right': [...]}.
    """
    import pandas as pd

    ann_path = brain_repo / "flywire_annotations_matched.csv"
    df = pd.read_csv(ann_path, low_memory=False)
    mask = df["cell_type"] == glomerulus
    g = df[mask]

    left = sorted(g[g["side"] == "left"]["root_id"].astype(np.int64).tolist())
    right = sorted(g[g["side"] == "right"]["root_id"].astype(np.int64).tolist())
    return {"left": left, "right": right}


def build_valence_channel_map(
    base_channel_map: dict,
    orn_left: list[int],
    orn_right: list[int],
) -> dict:
    """Replace generic olfactory channels with glomerulus-specific ORN IDs."""
    cm = dict(base_channel_map)
    cm["olfactory_left"] = orn_left
    cm["olfactory_right"] = orn_right
    return cm


def build_sensory_ids(channel_map: dict) -> np.ndarray:
    """Collect all unique neuron IDs from a channel map."""
    all_ids = set()
    for ids in channel_map.values():
        all_ids.update(int(x) for x in ids)
    return np.array(sorted(all_ids), dtype=np.int64)


def run_valence_trial(
    label: str,
    sensory_ids: np.ndarray,
    channel_map: dict,
    readout_ids: np.ndarray,
    odor_left: float,
    odor_right: float,
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    shuffle_seed: int | None = None,
    decoder_path: str | Path | None = None,
    rate_scale: float | None = None,
) -> dict:
    """Run a single trial with specified odor asymmetry.

    Args:
        shuffle_seed: If set, shuffles connectome postsynaptic targets (control).
        decoder_path: Path to decoder groups JSON. Defaults to config default.
        rate_scale: Decoder sensitivity. Defaults to config default.

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

    # Simple flat arena (no OdorArena — we inject odor signal manually)
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

    # Main loop
    bspb = cfg.body_steps_per_brain
    current_cmd = LocomotionCommand(forward_drive=1.0)
    episode_log = []
    brain_steps = 0
    t0 = time.time()

    # We inject odor as a synthetic (1, 4) array: [L_antenna, R_antenna, L_palp, R_palp]
    synthetic_odor = np.array([[odor_left, odor_right, odor_left, odor_right]], dtype=np.float32)

    for step in range(body_steps):
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            # Override odor with our synthetic asymmetric signal
            body_obs.odor_intensity = synthetic_odor

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
        "odor_left": odor_left,
        "odor_right": odor_right,
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
) -> dict:
    """Run a set of conditions across seeds, returning {cond_name: [results]}."""
    tag = "SHUFFLED" if shuffle_seed is not None else "REAL"
    all_results = {}
    for cond_name, sensory, cm, ol, or_, desc in conditions:
        print("\n" + "=" * 60)
        print("[%s] CONDITION: %s" % (tag, desc))
        print("  odor_left=%.2f, odor_right=%.2f" % (ol, or_))
        print("=" * 60)

        cond_results = []
        for seed in seeds:
            print("\n  --- Seed %d ---" % seed)
            result = run_valence_trial(
                label="%s_%s_seed%d" % (cond_name, tag.lower(), seed),
                sensory_ids=sensory,
                channel_map=cm,
                readout_ids=readout_ids,
                odor_left=ol,
                odor_right=or_,
                body_steps=body_steps,
                warmup_steps=warmup_steps,
                use_fake_brain=use_fake_brain,
                seed=seed,
                shuffle_seed=shuffle_seed,
                decoder_path=decoder_path,
                rate_scale=rate_scale,
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

    dm1_left_mean, dm1_left_std, n1l = cond_stats(all_results["DM1_left"])
    dm1_right_mean, dm1_right_std, n1r = cond_stats(all_results["DM1_right"])
    dm5_left_mean, dm5_left_std, n5l = cond_stats(all_results["DM5_left"])
    dm5_right_mean, dm5_right_std, n5r = cond_stats(all_results["DM5_right"])

    dm1_contrast = dm1_left_mean - dm1_right_mean
    dm5_contrast = dm5_left_mean - dm5_right_mean
    valence_contrast = dm5_contrast - dm1_contrast

    print("\n--- %s ---" % label)
    print("Glomerulus    Odor Side    Mean Turn    Std     N")
    print("-" * 55)
    print("DM1 (Or42b)   LEFT        %+.4f      %.4f   %d" % (dm1_left_mean, dm1_left_std, n1l))
    print("DM1 (Or42b)   RIGHT       %+.4f      %.4f   %d" % (dm1_right_mean, dm1_right_std, n1r))
    print("DM5 (Or85a)   LEFT        %+.4f      %.4f   %d" % (dm5_left_mean, dm5_left_std, n5l))
    print("DM5 (Or85a)   RIGHT       %+.4f      %.4f   %d" % (dm5_right_mean, dm5_right_std, n5r))
    print("DM1 turn contrast: %+.4f" % dm1_contrast)
    print("DM5 turn contrast: %+.4f" % dm5_contrast)
    print("Valence contrast:  %+.4f" % valence_contrast)

    return {
        "DM1_left_turn": dm1_left_mean,
        "DM1_right_turn": dm1_right_mean,
        "DM5_left_turn": dm5_left_mean,
        "DM5_right_turn": dm5_right_mean,
        "DM1_turn_contrast": dm1_contrast,
        "DM5_turn_contrast": dm5_contrast,
        "valence_contrast": valence_contrast,
    }


def run_odor_valence(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    odor_high: float = 1.0,
    odor_low: float = 0.0,
    seeds: list[int] = None,
    run_shuffled: bool = True,
    readout_version: int = 2,
    rate_scale: float = 15.0,
    output_dir: str = "logs/odor_valence",
):
    if seeds is None:
        seeds = [42, 43, 44]

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base channel map (non-olfactory channels)
    base_channel_map_path = cfg.data_dir / "channel_map_v3.json"
    with open(base_channel_map_path) as f:
        base_cm = json.load(f)

    if readout_version == 2:
        readout_ids = np.load(cfg.data_dir / "readout_ids_v2.npy")
        decoder_path = cfg.data_dir / "decoder_groups_v2.json"
    else:
        readout_ids = np.load(cfg.readout_ids_path)
        decoder_path = cfg.decoder_groups_path
    print("Readout v%d: %d neurons, rate_scale=%.1f" % (readout_version, len(readout_ids), rate_scale))

    # Load glomerulus-specific ORN populations
    print("Loading ORN populations from FlyWire annotations...")
    dm1_ids = load_orn_ids("ORN_DM1", cfg.brain_repo_root)
    dm5_ids = load_orn_ids("ORN_DM5", cfg.brain_repo_root)
    print("  DM1 (Or42b, attractive): %d left, %d right" % (len(dm1_ids["left"]), len(dm1_ids["right"])))
    print("  DM5 (Or85a, aversive):   %d left, %d right" % (len(dm5_ids["left"]), len(dm5_ids["right"])))

    # Build channel maps for each glomerulus
    dm1_cm = build_valence_channel_map(base_cm, dm1_ids["left"], dm1_ids["right"])
    dm5_cm = build_valence_channel_map(base_cm, dm5_ids["left"], dm5_ids["right"])
    dm1_sensory = build_sensory_ids(dm1_cm)
    dm5_sensory = build_sensory_ids(dm5_cm)
    print("  DM1 sensory population: %d neurons" % len(dm1_sensory))
    print("  DM5 sensory population: %d neurons" % len(dm5_sensory))

    conditions = [
        ("DM1_left",  dm1_sensory, dm1_cm, odor_high, odor_low,  "Or42b attractive, odor LEFT"),
        ("DM1_right", dm1_sensory, dm1_cm, odor_low,  odor_high, "Or42b attractive, odor RIGHT"),
        ("DM5_left",  dm5_sensory, dm5_cm, odor_high, odor_low,  "Or85a aversive, odor LEFT"),
        ("DM5_right", dm5_sensory, dm5_cm, odor_low,  odor_high, "Or85a aversive, odor RIGHT"),
    ]

    # --- Real connectome ---
    print("\n" + "#" * 60)
    print("# REAL CONNECTOME")
    print("#" * 60)
    real_results = _run_condition_set(
        conditions, readout_ids, body_steps, warmup_steps,
        use_fake_brain, seeds, shuffle_seed=None,
        decoder_path=decoder_path, rate_scale=rate_scale,
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
        )

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("ODOR VALENCE RESULTS")
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

    dm1_c = real_summary["DM1_turn_contrast"]
    dm5_c = real_summary["DM5_turn_contrast"]
    val_c = real_summary["valence_contrast"]

    # Test 1: DM1 shows odor-directed turning (negative contrast = toward)
    dm1_toward = dm1_c < 0
    tests.append({"name": "DM1_attractive_turning", "passed": dm1_toward,
                   "detail": "contrast=%+.4f (expect <0)" % dm1_c})
    print("  [%s] DM1 attractive turning: %+.4f" % ("PASS" if dm1_toward else "FAIL", dm1_c))

    # Test 2: DM5 shows odor-aversive turning (positive contrast = away)
    dm5_away = dm5_c > 0
    tests.append({"name": "DM5_aversive_turning", "passed": dm5_away,
                   "detail": "contrast=%+.4f (expect >0)" % dm5_c})
    print("  [%s] DM5 aversive turning: %+.4f" % ("PASS" if dm5_away else "FAIL", dm5_c))

    # Test 3: Opposite valence
    opposite = val_c > 0
    tests.append({"name": "opposite_valence", "passed": opposite,
                   "detail": "valence=%+.4f (expect >0)" % val_c})
    print("  [%s] Opposite valence: %+.4f" % ("PASS" if opposite else "FAIL", val_c))

    # Test 4: Shuffled control has weaker valence contrast than real
    if shuf_summary is not None:
        shuf_val = shuf_summary["valence_contrast"]
        specificity = abs(val_c) > abs(shuf_val)
        tests.append({"name": "connectome_specificity", "passed": specificity,
                       "detail": "real=%.4f, shuffled=%.4f" % (abs(val_c), abs(shuf_val))})
        print("  [%s] Connectome specificity: |real|=%.4f > |shuffled|=%.4f" % (
            "PASS" if specificity else "FAIL", abs(val_c), abs(shuf_val)))

        # Test 5: DM1 odor contrast stronger in real than shuffled
        shuf_dm1 = shuf_summary["DM1_turn_contrast"]
        dm1_spec = abs(dm1_c) > abs(shuf_dm1)
        tests.append({"name": "DM1_connectome_specificity", "passed": dm1_spec,
                       "detail": "real=%.4f, shuffled=%.4f" % (abs(dm1_c), abs(shuf_dm1))})
        print("  [%s] DM1 specificity: |real|=%.4f > |shuffled|=%.4f" % (
            "PASS" if dm1_spec else "FAIL", abs(dm1_c), abs(shuf_dm1)))

        # Test 6: DM5 odor contrast stronger in real than shuffled
        shuf_dm5 = shuf_summary["DM5_turn_contrast"]
        dm5_spec = abs(dm5_c) > abs(shuf_dm5)
        tests.append({"name": "DM5_connectome_specificity", "passed": dm5_spec,
                       "detail": "real=%.4f, shuffled=%.4f" % (abs(dm5_c), abs(shuf_dm5))})
        print("  [%s] DM5 specificity: |real|=%.4f > |shuffled|=%.4f" % (
            "PASS" if dm5_spec else "FAIL", abs(dm5_c), abs(shuf_dm5)))

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
            "odor_high": odor_high,
            "odor_low": odor_low,
            "seeds": seeds,
            "use_fake_brain": use_fake_brain,
            "run_shuffled": run_shuffled,
            "dm1_left_count": len(dm1_ids["left"]),
            "dm1_right_count": len(dm1_ids["right"]),
            "dm5_left_count": len(dm5_ids["left"]),
            "dm5_right_count": len(dm5_ids["right"]),
            "dm1_sensory_total": len(dm1_sensory),
            "dm5_sensory_total": len(dm5_sensory),
        },
        "real": {"summary": real_summary, "trials": strip_logs(real_results)},
        "tests": tests,
    }
    if shuf_summary is not None:
        output_data["shuffled"] = {
            "summary": shuf_summary,
            "trials": strip_logs(shuf_results),
        }

    out_file = output_path / "valence_results.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print("\nSaved to %s" % out_file)

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Or42b vs Or85a odor valence experiment")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--odor-high", type=float, default=1.0)
    parser.add_argument("--odor-low", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--no-shuffle", action="store_true", help="Skip shuffled control")
    parser.add_argument("--readout-version", type=int, default=2, choices=[1, 2])
    parser.add_argument("--rate-scale", type=float, default=15.0)
    parser.add_argument("--output-dir", default="logs/odor_valence")
    args = parser.parse_args()

    run_odor_valence(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        odor_high=args.odor_high,
        odor_low=args.odor_low,
        seeds=args.seeds,
        run_shuffled=not args.no_shuffle,
        readout_version=args.readout_version,
        rate_scale=args.rate_scale,
        output_dir=args.output_dir,
    )
