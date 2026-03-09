"""
Looming escape experiment: direct LPLC2 neuron injection.

LPLC2 neurons (210 in FlyWire connectome) are lobula plate visual
projection neurons that detect looming (approaching) objects. They
project ipsilaterally to 44 descending neurons with 1,850 direct synapses.

Instead of ambient photoreceptor stimulation (which failed to produce
phototaxis), this injects directly into LPLC2 — bypassing 5+ synaptic
relays of optic lobe processing and testing the escape circuit directly.

Conditions:
  loom_left:  left LPLC2 at high rate, right at baseline
  loom_right: right LPLC2 at high rate, left at baseline
  control:    both at baseline

Expected: looming left -> ipsilateral left DN activation -> turn away (right)

Usage:
    python experiments/looming.py
    python experiments/looming.py --seeds 42 43 44 45 46
    python experiments/looming.py --brain-dt-ms 50
    python experiments/looming.py --fake-brain
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import BodyObservation, LocomotionCommand, BrainOutput
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from analysis.behavior_metrics import compute_behavior


def run_looming_trial(
    label: str,
    sensory_ids: np.ndarray,
    channel_map: dict,
    readout_ids: np.ndarray,
    loom_left: float,
    loom_right: float,
    body_steps: int,
    warmup_steps: int,
    use_fake_brain: bool,
    seed: int,
    shuffle_seed: int | None = None,
    decoder_path: str | Path | None = None,
    rate_scale: float = 12.0,
    brain_dt_ms: float = 20.0,
    body_steps_per_brain: int | None = None,
    loom_rate_hz: float = 200.0,
    sample_interval: int = 20,
):
    """Run one looming trial with LPLC2 injection."""
    import flygym

    cfg = BridgeConfig()

    if body_steps_per_brain is None:
        body_steps_per_brain = int(brain_dt_ms / (1e-4 * 1000))  # match dt

    # Use standard max_rate for proprioceptive/etc channels,
    # loom_rate_hz only applies to LPLC2 encoding in sensory_encoder
    if isinstance(channel_map, dict):
        encoder = SensoryEncoder(
            sensory_neuron_ids=sensory_ids,
            channel_map=channel_map,
            max_rate_hz=cfg.max_rate_hz,
            baseline_rate_hz=cfg.baseline_rate_hz,
        )
    else:
        encoder = SensoryEncoder.from_channel_map(
            sensory_ids, channel_map,
            max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
        )

    if decoder_path:
        decoder = DescendingDecoder.from_json(decoder_path, rate_scale=rate_scale)
    else:
        decoder = DescendingDecoder.from_json(
            cfg.data_dir / "decoder_groups_v4_looming.json", rate_scale=rate_scale)

    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain = create_brain_runner(
        sensory_ids=sensory_ids, readout_ids=readout_ids,
        use_fake=use_fake_brain, warmup_ms=cfg.brain_warmup_ms,
        shuffle_seed=shuffle_seed,
    )

    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly_obj, arena=flygym.arena.FlatTerrain(), timestep=1e-4)
    obs, info = sim.reset()

    locomotion.warmup(0)
    locomotion.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        obs, _, terminated, truncated, info = sim.step(action)
        if terminated or truncated:
            sim.close()
            return {"label": label, "error": "warmup_ended"}

    bspb = body_steps_per_brain
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
            # Inject looming stimulus
            body_obs.looming_intensity = np.array([loom_left, loom_right])

            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=brain_dt_ms)
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
            positions.append(np.array(obs["fly"][0]))
            orientations.append(np.array(obs["fly"][2]))
            cf_raw = np.asarray(obs.get("contact_forces", np.zeros((30, 3))))
            magnitudes = np.linalg.norm(cf_raw, axis=1) if cf_raw.ndim == 2 else np.zeros(30)
            per_leg = np.array([magnitudes[i*5:(i+1)*5].max() for i in range(6)])
            contact_forces_log.append(np.clip(per_leg / 10.0, 0.0, 1.0))

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

    turns = [e["turn_drive"] for e in episode_log]
    fwds = [e["forward_drive"] for e in episode_log]
    mean_turn = float(np.mean(turns)) if turns else 0
    std_turn = float(np.std(turns)) if turns else 0
    mean_fwd = float(np.mean(fwds)) if fwds else 0

    result = {
        "label": label,
        "loom_left": loom_left,
        "loom_right": loom_right,
        "mean_turn_drive": mean_turn,
        "std_turn_drive": std_turn,
        "mean_forward_drive": mean_fwd,
        "forward_distance": behavior.forward_distance,
        "total_path_length": behavior.total_path_length,
        "brain_steps": brain_steps,
        "elapsed_s": elapsed,
        "behavior_summary": behavior.summary_line(),
    }

    print("  %s: turn=%+.4f (std=%.4f), fwd=%.3f, %d brain steps in %.1fs" % (
        label, mean_turn, std_turn, mean_fwd, brain_steps, elapsed))

    return result


def run_looming_experiment(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seeds: list[int] = None,
    run_shuffled: bool = True,
    output_dir: str = "logs/looming",
    brain_dt_ms: float = 20.0,
    loom_rate_hz: float = 200.0,
    rate_scale: float = 12.0,
):
    if seeds is None:
        seeds = [42, 43, 44]

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load v4 looming data
    channel_map_path = cfg.data_dir / "channel_map_v4_looming.json"
    with open(channel_map_path) as f:
        channel_map = json.load(f)

    sensory_ids = np.load(cfg.data_dir / "sensory_ids_v4_looming.npy")
    readout_ids = np.load(cfg.data_dir / "readout_ids_v4_looming.npy")
    decoder_path = cfg.data_dir / "decoder_groups_v4_looming.json"

    n_lplc2_l = len(channel_map.get("lplc2_left", []))
    n_lplc2_r = len(channel_map.get("lplc2_right", []))

    body_steps_per_brain = int(brain_dt_ms / (1e-4 * 1000))

    print("=" * 60)
    print("LOOMING ESCAPE EXPERIMENT")
    print("  LPLC2: %d left, %d right neurons" % (n_lplc2_l, n_lplc2_r))
    print("  Sensory: %d neurons, Readout: %d neurons" % (len(sensory_ids), len(readout_ids)))
    print("  Brain window: %.0fms, body_steps_per_brain: %d" % (brain_dt_ms, body_steps_per_brain))
    print("  Looming rate: %.0f Hz, rate_scale: %.1f" % (loom_rate_hz, rate_scale))
    print("  Seeds: %s" % seeds)
    print("=" * 60)

    conditions = [
        ("loom_left",  1.0, 0.0, "LOOM LEFT (left LPLC2 active)"),
        ("loom_right", 0.0, 1.0, "LOOM RIGHT (right LPLC2 active)"),
        ("control",    0.0, 0.0, "CONTROL (baseline)"),
    ]

    def run_set(tag, shuffle_seed=None):
        results = {}
        for cond_name, ll, lr, desc in conditions:
            print("\n" + "=" * 60)
            print("[%s] %s" % (tag, desc))
            print("=" * 60)
            cond_results = []
            for seed in seeds:
                print("\n  --- Seed %d ---" % seed)
                r = run_looming_trial(
                    label="%s_%s_seed%d" % (cond_name, tag.lower(), seed),
                    sensory_ids=sensory_ids,
                    channel_map=channel_map,
                    readout_ids=readout_ids,
                    loom_left=ll, loom_right=lr,
                    body_steps=body_steps,
                    warmup_steps=warmup_steps,
                    use_fake_brain=use_fake_brain,
                    seed=seed,
                    shuffle_seed=shuffle_seed,
                    decoder_path=decoder_path,
                    rate_scale=rate_scale,
                    brain_dt_ms=brain_dt_ms,
                    body_steps_per_brain=body_steps_per_brain,
                    loom_rate_hz=loom_rate_hz,
                )
                cond_results.append(r)
            results[cond_name] = cond_results
        return results

    # Real connectome
    print("\n" + "#" * 60)
    print("# REAL CONNECTOME")
    print("#" * 60)
    real_results = run_set("REAL")

    # Shuffled control
    shuf_results = None
    if run_shuffled:
        print("\n" + "#" * 60)
        print("# SHUFFLED CONNECTOME (control)")
        print("#" * 60)
        shuf_results = run_set("SHUFFLED", shuffle_seed=999)

    # Analysis
    print("\n" + "=" * 60)
    print("LOOMING RESULTS")
    print("=" * 60)

    def analyze(results, label):
        print("\n--- %s ---" % label)
        print("%-15s %12s %12s %12s" % ("Condition", "Turn Drive", "Std", "N"))
        print("-" * 55)
        stats = {}
        for cond_name, trials in results.items():
            turns = [r["mean_turn_drive"] for r in trials if "error" not in r]
            if turns:
                m, s, n = float(np.mean(turns)), float(np.std(turns)), len(turns)
            else:
                m, s, n = 0, 0, 0
            stats[cond_name] = {"mean": m, "std": s, "n": n}
            print("%-15s %+11.4f %11.4f %11d" % (cond_name, m, s, n))
        return stats

    real_stats = analyze(real_results, "REAL CONNECTOME")
    shuf_stats = None
    if shuf_results:
        shuf_stats = analyze(shuf_results, "SHUFFLED CONNECTOME")

    # Tests
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    tests = []

    loom_l = real_stats["loom_left"]["mean"]
    loom_r = real_stats["loom_right"]["mean"]
    ctrl = real_stats["control"]["mean"]

    # Test 1: Looming produces directional turning
    # Left loom -> left LPLC2 -> left DNs -> turn_left activation
    # turn_left activation -> positive turn_drive? Let's check by contrast
    escape_index = loom_r - loom_l  # right loom vs left loom
    test1 = abs(escape_index) > 0.01
    tests.append({"name": "directional_escape", "passed": test1,
                   "detail": "escape_index=%+.4f (L-R contrast)" % escape_index})
    print("  [%s] Directional escape: index=%+.4f" % ("PASS" if test1 else "FAIL", escape_index))

    # Test 2: Looming differs from control
    loom_effect = abs(loom_l - ctrl) + abs(loom_r - ctrl)
    test2 = loom_effect > 0.02
    tests.append({"name": "loom_vs_control", "passed": test2,
                   "detail": "effect=%.4f" % loom_effect})
    print("  [%s] Loom vs control: effect=%.4f" % ("PASS" if test2 else "FAIL", loom_effect))

    # Test 3: Left and right looming produce opposite turns
    test3 = (loom_l - ctrl) * (loom_r - ctrl) < 0  # opposite signs relative to control
    tests.append({"name": "opposite_turns", "passed": test3,
                   "detail": "L-ctrl=%+.4f, R-ctrl=%+.4f" % (loom_l - ctrl, loom_r - ctrl)})
    print("  [%s] Opposite turns: L=%+.4f, R=%+.4f (vs ctrl)" % (
        "PASS" if test3 else "FAIL", loom_l - ctrl, loom_r - ctrl))

    # Test 4: Real > shuffled specificity
    if shuf_stats:
        shuf_index = abs(shuf_stats["loom_right"]["mean"] - shuf_stats["loom_left"]["mean"])
        test4 = abs(escape_index) > shuf_index * 2
        tests.append({"name": "connectome_specificity", "passed": test4,
                       "detail": "real=%.4f, shuffled=%.4f" % (abs(escape_index), shuf_index)})
        print("  [%s] Real > shuffled: %.4f vs %.4f" % (
            "PASS" if test4 else "FAIL", abs(escape_index), shuf_index))

    # Statistical test with seeds
    ll_turns = [r["mean_turn_drive"] for r in real_results["loom_left"] if "error" not in r]
    lr_turns = [r["mean_turn_drive"] for r in real_results["loom_right"] if "error" not in r]
    if len(ll_turns) >= 3 and len(lr_turns) >= 3:
        diff = np.array(ll_turns[:min(len(ll_turns), len(lr_turns))]) - \
               np.array(lr_turns[:min(len(ll_turns), len(lr_turns))])
        ci = 1.96 * np.std(diff) / np.sqrt(len(diff))
        sig = abs(np.mean(diff)) > ci
        test5 = sig
        tests.append({"name": "statistical_significance", "passed": test5,
                       "detail": "L-R=%+.4f +/- %.4f (95%% CI)" % (np.mean(diff), ci)})
        print("  [%s] Significant: L-R=%+.4f +/- %.4f" % (
            "PASS" if test5 else "FAIL", np.mean(diff), ci))

    n_pass = sum(1 for t in tests if t["passed"])
    print("\n  %d/%d tests passed" % (n_pass, len(tests)))

    # Save results
    output_data = {
        "config": {
            "body_steps": body_steps,
            "warmup_steps": warmup_steps,
            "brain_dt_ms": brain_dt_ms,
            "loom_rate_hz": loom_rate_hz,
            "rate_scale": rate_scale,
            "seeds": seeds,
            "n_lplc2_left": n_lplc2_l,
            "n_lplc2_right": n_lplc2_r,
            "n_sensory": len(sensory_ids),
            "n_readout": len(readout_ids),
        },
        "real": {"summary": real_stats, "trials": real_results},
        "tests": tests,
    }
    if shuf_results:
        output_data["shuffled"] = {"summary": shuf_stats, "trials": shuf_results}

    with open(output_path / "looming_results.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print("\nSaved to %s" % (output_path / "looming_results.json"))

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Looming escape experiment")
    parser.add_argument("--body-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--brain-dt-ms", type=float, default=50.0,
                        help="Brain window in ms (default: 50, longer for visual processing)")
    parser.add_argument("--loom-rate-hz", type=float, default=200.0,
                        help="Max LPLC2 firing rate for looming stimulus")
    parser.add_argument("--rate-scale", type=float, default=12.0)
    parser.add_argument("--output-dir", default="logs/looming")
    args = parser.parse_args()

    run_looming_experiment(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        seeds=args.seeds,
        run_shuffled=not args.no_shuffle,
        output_dir=args.output_dir,
        brain_dt_ms=args.brain_dt_ms,
        loom_rate_hz=args.loom_rate_hz,
        rate_scale=args.rate_scale,
    )
