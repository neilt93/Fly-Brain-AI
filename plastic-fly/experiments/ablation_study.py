"""
Ablation study: prove that each decoder group causally controls
a distinct locomotion variable.

Conditions:
  baseline         — no intervention
  ablate_forward   — zero forward group rates
  ablate_turn_left — zero turn_left group rates
  ablate_turn_right— zero turn_right group rates
  boost_turn_left  — 3x turn_left rates
  boost_turn_right — 3x turn_right rates
  ablate_rhythm    — zero rhythm group rates
  ablate_stance    — zero stance group rates

Expected effects:
  ablate_forward    → forward distance drops
  boost_turn_left   → fly turns left (positive final_heading)
  boost_turn_right  → fly turns right (negative final_heading)
  ablate_rhythm     → step frequency changes
  ablate_stance     → contact profile / stability changes

Usage:
    python experiments/ablation_study.py --fake-brain          # fast test
    python experiments/ablation_study.py                       # real brain
    python experiments/ablation_study.py --body-steps 10000    # longer
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
from bridge.interfaces import LocomotionCommand, BrainOutput
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from analysis.behavior_metrics import compute_behavior, BehaviorReport


# Ablation condition definitions
ABLATION_CONDITIONS = {
    "baseline": {},
    "ablate_forward": {"zero": "forward_ids"},
    "ablate_turn_left": {"zero": "turn_left_ids"},
    "ablate_turn_right": {"zero": "turn_right_ids"},
    "boost_turn_left": {"boost": "turn_left_ids", "factor": 3.0},
    "boost_turn_right": {"boost": "turn_right_ids", "factor": 3.0},
    "ablate_rhythm": {"zero": "rhythm_ids"},
    "ablate_stance": {"zero": "stance_ids"},
}


def apply_ablation(brain_output: BrainOutput, condition: dict, decoder_groups: dict) -> BrainOutput:
    """Apply ablation or boost to a BrainOutput before decoding."""
    if not condition:
        return brain_output

    rates = brain_output.firing_rates_hz.copy()
    id_to_idx = {int(nid): i for i, nid in enumerate(brain_output.neuron_ids)}

    if "zero" in condition:
        group_name = condition["zero"]
        for nid in decoder_groups.get(group_name, []):
            if int(nid) in id_to_idx:
                rates[id_to_idx[int(nid)]] = 0.0

    if "boost" in condition:
        group_name = condition["boost"]
        factor = condition.get("factor", 3.0)
        for nid in decoder_groups.get(group_name, []):
            if int(nid) in id_to_idx:
                rates[id_to_idx[int(nid)]] *= factor

    return BrainOutput(neuron_ids=brain_output.neuron_ids, firing_rates_hz=rates)


def compute_group_balance(episode_log: list, decoder_groups: dict, readout_ids: np.ndarray) -> dict:
    """Compute per-group firing rate statistics from episode log."""
    if not episode_log or "group_rates" not in episode_log[0]:
        return {}

    balance = {}
    for gname in decoder_groups:
        rates = [e["group_rates"].get(gname, 0.0) for e in episode_log]
        balance[gname] = {
            "mean_rate_hz": float(np.mean(rates)),
            "std_rate_hz": float(np.std(rates)),
            "max_rate_hz": float(np.max(rates)),
            "active_frac": float(np.mean(np.array(rates) > 0)),
        }
    return balance


def run_single_condition(
    condition_name: str,
    condition: dict,
    body_steps: int,
    warmup_steps: int,
    use_fake_brain: bool,
    seed: int,
    decoder_groups: dict,
    sample_interval: int = 20,
) -> dict:
    """Run one ablation condition and return results + behavior metrics."""
    import flygym

    cfg = BridgeConfig()

    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)

    if cfg.channel_map_path.exists():
        encoder = SensoryEncoder.from_channel_map(
            sensory_ids, cfg.channel_map_path,
            max_rate_hz=cfg.max_rate_hz, baseline_rate_hz=cfg.baseline_rate_hz,
        )
    else:
        encoder = SensoryEncoder(sensory_ids, max_rate_hz=cfg.max_rate_hz)

    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
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
    locomotion.cpg.reset(init_phases=np.zeros(6), init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, terminated, truncated, info = sim.step(action)
            if terminated or truncated:
                sim.close()
                return {"error": "warmup_ended"}
        except Exception:
            sim.close()
            return {"error": "warmup_physics"}

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
        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)

            # Apply ablation
            brain_output = apply_ablation(brain_output, condition, decoder_groups)

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
    elapsed = time.time() - t_start

    # Behavior metrics
    behavior = compute_behavior(
        positions=positions,
        orientations=orientations,
        contact_forces=contact_forces_log,
        steps_completed=step + 1,
        steps_intended=body_steps,
        sample_dt=sample_interval * 1e-4,
    )

    # Group balance
    group_balance = compute_group_balance(episode_log, decoder_groups, readout_ids)

    return {
        "condition": condition_name,
        "steps_completed": step + 1,
        "brain_steps": brain_steps,
        "elapsed_s": elapsed,
        "behavior": behavior.to_dict(),
        "behavior_summary": behavior.summary_line(),
        "group_balance": group_balance,
        "episode_log": episode_log,
    }


def run_ablation_study(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seed: int = 42,
    output_dir: str = "logs/ablation",
    conditions: list[str] | None = None,
):
    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(cfg.decoder_groups_path) as f:
        decoder_groups = json.load(f)

    if conditions is None:
        conditions = list(ABLATION_CONDITIONS.keys())

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    print("=" * 60)
    print("ABLATION STUDY (%s, %d body steps, seed=%d)" % (brain_label, body_steps, seed))
    print("=" * 60)

    results = {}
    for cond_name in conditions:
        cond = ABLATION_CONDITIONS[cond_name]
        print("\n--- %s ---" % cond_name)
        r = run_single_condition(
            condition_name=cond_name,
            condition=cond,
            body_steps=body_steps,
            warmup_steps=warmup_steps,
            use_fake_brain=use_fake_brain,
            seed=seed,
            decoder_groups=decoder_groups,
        )
        if "error" in r:
            print("  ERROR: %s" % r["error"])
            continue
        print("  %s" % r["behavior_summary"])
        results[cond_name] = r

    # --- Comparative analysis ---
    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS")
    print("=" * 60)

    if "baseline" not in results:
        print("No baseline run -- cannot compare.")
        with open(output_path / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    base = results["baseline"]["behavior"]

    header = "%-22s %8s %8s %8s %8s %8s %8s" % (
        "Condition", "FwdDist", "Turn", "Heading", "CntSym", "Falls", "DutyCyc")
    print(header)
    print("-" * len(header))

    for cond_name in conditions:
        if cond_name not in results:
            continue
        b = results[cond_name]["behavior"]
        print("%-22s %+7.2fmm %6.1fdeg %+7.1fdeg %7.2f %6d %7.2f" % (
            cond_name,
            b["forward_distance"],
            np.degrees(b["cumulative_turn"]),
            np.degrees(b["final_heading"]),
            b["contact_symmetry"],
            b["fall_count"],
            b["contact_duty_cycle"],
        ))

    # --- Causal tests ---
    print("\n" + "=" * 60)
    print("CAUSAL TESTS")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    def log_test(name, passed, description):
        nonlocal tests_passed, tests_total
        tests_total += 1
        if passed:
            tests_passed += 1
        print("  [%s] %s: %s" % ("PASS" if passed else "FAIL", name, description))

    def get_mean_cmd(cond_name, field):
        if cond_name not in results:
            return None
        log = results[cond_name].get("episode_log", [])
        if not log:
            return None
        return float(np.mean([e[field] for e in log]))

    # === COMMAND-LEVEL TESTS (deterministic decoder consequences) ===
    print("\n  -- Command-level (decoder output) --")

    # 1. Forward ablation: forward_drive drops
    base_fwd = get_mean_cmd("baseline", "forward_drive")
    abl_fwd = get_mean_cmd("ablate_forward", "forward_drive")
    if base_fwd is not None and abl_fwd is not None:
        log_test("cmd_forward_ablation",
                 abl_fwd < base_fwd,
                 "ablate forward -> lower forward_drive (%.3f -> %.3f)" % (base_fwd, abl_fwd))

    # 2. Turn ablation contrast: removing turn_left shifts turn_drive negative, removing turn_right positive
    abl_tl = get_mean_cmd("ablate_turn_left", "turn_drive")
    abl_tr = get_mean_cmd("ablate_turn_right", "turn_drive")
    if abl_tl is not None and abl_tr is not None:
        log_test("cmd_turn_ablation_contrast",
                 abl_tl < abl_tr,
                 "ablate_left turn_drive < ablate_right turn_drive (%.3f < %.3f)" % (abl_tl, abl_tr))

    # 3. Turn boost contrast: boosting turn_left shifts turn_drive positive, boosting turn_right negative
    bst_tl = get_mean_cmd("boost_turn_left", "turn_drive")
    bst_tr = get_mean_cmd("boost_turn_right", "turn_drive")
    if bst_tl is not None and bst_tr is not None:
        log_test("cmd_turn_boost_contrast",
                 bst_tl > bst_tr,
                 "boost_left turn_drive > boost_right turn_drive (%.3f > %.3f)" % (bst_tl, bst_tr))

    # 4. Rhythm ablation: step_frequency drops
    base_freq = get_mean_cmd("baseline", "step_frequency")
    abl_freq = get_mean_cmd("ablate_rhythm", "step_frequency")
    if base_freq is not None and abl_freq is not None:
        log_test("cmd_rhythm_ablation",
                 abl_freq < base_freq,
                 "ablate rhythm -> lower step_frequency (%.2f -> %.2f)" % (base_freq, abl_freq))

    # 5. Stance ablation: stance_gain drops
    base_stance = get_mean_cmd("baseline", "stance_gain")
    abl_stance = get_mean_cmd("ablate_stance", "stance_gain")
    if base_stance is not None and abl_stance is not None:
        log_test("cmd_stance_ablation",
                 abl_stance < base_stance,
                 "ablate stance -> lower stance_gain (%.2f -> %.2f)" % (base_stance, abl_stance))

    # === BEHAVIORAL TESTS (physical consequences) ===
    print("\n  -- Behavioral (physics consequences) --")

    # 6. Forward ablation reduces forward distance (use abs for negative baseline)
    if "ablate_forward" in results:
        abl_dist = results["ablate_forward"]["behavior"]["forward_distance"]
        log_test("behav_forward_distance",
                 abs(abl_dist) < abs(base["forward_distance"]),
                 "ablate forward -> less distance (|%.2f|mm -> |%.2f|mm)" % (
                     base["forward_distance"], abl_dist))

    # 7. Turn ablation contrast: behavioral heading difference
    if "ablate_turn_left" in results and "ablate_turn_right" in results:
        h_abl_l = results["ablate_turn_left"]["behavior"]["final_heading"]
        h_abl_r = results["ablate_turn_right"]["behavior"]["final_heading"]
        log_test("behav_turn_ablation_contrast",
                 h_abl_l < h_abl_r,
                 "ablate_left heading < ablate_right heading (%.1fdeg < %.1fdeg)" % (
                     np.degrees(h_abl_l), np.degrees(h_abl_r)))

    # 8. Forward ablation reduces path length
    if "ablate_forward" in results:
        log_test("behav_forward_path",
                 results["ablate_forward"]["behavior"]["total_path_length"] < base["total_path_length"],
                 "ablate forward -> shorter path (%.2fmm -> %.2fmm)" % (
                     base["total_path_length"],
                     results["ablate_forward"]["behavior"]["total_path_length"]))

    # 9. Rhythm ablation changes step frequency or duty cycle
    if "ablate_rhythm" in results:
        abl_r = results["ablate_rhythm"]["behavior"]
        freq_changed = abs(abl_r["step_frequency_hz"] - base["step_frequency_hz"]) > 0.5
        duty_changed = abs(abl_r["contact_duty_cycle"] - base["contact_duty_cycle"]) > 0.01
        log_test("behav_rhythm_effect",
                 freq_changed or duty_changed,
                 "ablate rhythm -> freq or duty cycle changes")

    # 10. Stance ablation affects contact profile
    if "ablate_stance" in results:
        abl_s = results["ablate_stance"]["behavior"]
        sym_changed = abs(abl_s["contact_symmetry"] - base["contact_symmetry"]) > 0.01
        force_changed = abs(abl_s["mean_contact_force"] - base["mean_contact_force"]) > 0.01
        fall_changed = abl_s["fall_count"] != base["fall_count"]
        log_test("behav_stance_effect",
                 sym_changed or force_changed or fall_changed,
                 "ablate stance -> contact profile changes")

    print("\nCausal tests: %d/%d passed" % (tests_passed, tests_total))

    # --- Group balance (baseline) ---
    if results["baseline"].get("group_balance"):
        print("\n" + "=" * 60)
        print("GROUP BALANCE (baseline)")
        print("=" * 60)
        gb = results["baseline"]["group_balance"]
        print("%-22s %8s %8s %8s %8s" % ("Group", "MeanHz", "StdHz", "MaxHz", "Active%"))
        print("-" * 68)
        for gname, stats in gb.items():
            print("%-22s %8.1f %8.1f %8.1f %7.0f%%" % (
                gname, stats["mean_rate_hz"], stats["std_rate_hz"],
                stats["max_rate_hz"], stats["active_frac"] * 100,
            ))

    # --- Save ---
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to %s/ablation_results.json" % output_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for brain-body bridge")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="logs/ablation")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Run only these conditions (default: all)")
    args = parser.parse_args()

    run_ablation_study(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        seed=args.seed,
        output_dir=args.output_dir,
        conditions=args.conditions,
    )
