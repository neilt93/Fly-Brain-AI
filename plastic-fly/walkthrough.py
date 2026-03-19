"""
Interactive codebase walkthrough for the Connectome Fly Brain project.

Run: python walkthrough.py
Steps through each component of the brain-body bridge, shows real data,
and runs mini-demos. Press Enter to advance.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BRAIN = BASE.parent / "brain-model"

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def section(num, title):
    print("\n" + "=" * 70)
    print(f"{BOLD}{CYAN}  STEP {num}: {title}{RESET}")
    print("=" * 70)


def note(text):
    print(f"{DIM}  {text}{RESET}")


def stat(label, value):
    print(f"  {YELLOW}{label:<35}{RESET} {value}")


def pause():
    input(f"\n{DIM}  Press Enter to continue...{RESET}")


def main():
    print(BOLD + CYAN)
    print("  +==========================================================+")
    print("  |  CONNECTOME FLY BRAIN — INTERACTIVE WALKTHROUGH         |")
    print("  |  FlyWire brain (138,639 neurons) + FlyGym body (v1.2.1) |")
    print("  +==========================================================+")
    print(RESET)
    print("  This walkthrough steps through each component of the")
    print("  closed-loop brain-body bridge, with real data and live demos.")
    print()
    print("  The loop:")
    print("    Body -> SensoryEncoder -> Brain (139k LIF) -> Decoder -> CPG -> Body")
    pause()

    # ──────────────────────────────────────────────────────────────────
    section(1, "THE CONNECTOME DATA")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  The brain model uses the FlyWire connectome (v783 snapshot).")
    print(f"  Data lives in: {BRAIN}")
    print()

    import pandas as pd
    comp = pd.read_csv(BRAIN / "Completeness_783.csv")
    stat("Neurons (Completeness_783.csv)", f"{len(comp):,}")

    con = pd.read_parquet(BRAIN / "Connectivity_783.parquet")
    total_syn = int(con["Connectivity"].sum())
    stat("Connection pairs", f"{len(con):,}")
    stat("Total synapses", f"{total_syn:,}")
    stat("Columns", ", ".join(con.columns.tolist()))

    print()
    print("  Sample connections (first 5 rows):")
    print(con.head().to_string(index=False))

    exc_frac = con["Excitatory"].mean()
    stat("Excitatory fraction", f"{exc_frac:.1%}")
    stat("Mean synapses per pair", f"{total_syn / len(con):.1f}")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(2, "SENSORY POPULATIONS")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  We select specific neuron types as sensory inputs.")
    print("  Each 'channel' maps a body signal to identified neurons.")
    print()

    with open(DATA / "channel_map_v4_looming.json") as f:
        channel_map = json.load(f)

    for ch, ids in channel_map.items():
        stat(ch, f"{len(ids)} neurons")

    sensory = np.load(DATA / "sensory_ids_v4_looming.npy")
    stat("Total unique sensory IDs (v4)", len(sensory))

    print()
    note("These are real FlyWire root IDs. Example sensory neuron IDs:")
    print(f"  {sensory[:8].tolist()} ...")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(3, "READOUT POPULATIONS (Descending Neurons)")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  DNs are the brain's output to the body.")
    print("  We group them by motor function:")
    print()

    with open(DATA / "decoder_groups_v5_steering.json") as f:
        decoder_groups = json.load(f)
    for group, ids in decoder_groups.items():
        stat(group, f"{len(ids)} DNs")

    readout = np.load(DATA / "readout_ids_v5_steering.npy")
    stat("Total readout DNs (v5)", len(readout))

    print()
    print("  The decoder computes:")
    print("    forward_drive = tanh(mean(forward_ids) / rate_scale)")
    print("    turn_drive    = tanh((mean(turn_left) - mean(turn_right)) / rate_scale)")
    print("    step_freq     = 1.0 + 1.5 * tanh(mean(rhythm) / rate_scale)")
    print("    stance_gain   = 1.0 + 0.5 * tanh(mean(stance) / rate_scale)")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(4, "THE SENSORY ENCODER (live demo)")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  SensoryEncoder maps body state -> Poisson firing rates.")
    print("  Let's encode a fake body observation:")
    print()

    from bridge.interfaces import BodyObservation, BrainInput
    from bridge.sensory_encoder import SensoryEncoder

    encoder = SensoryEncoder(
        sensory_neuron_ids=sensory,
        channel_map=channel_map,
        max_rate_hz=100.0,
        baseline_rate_hz=10.0,
    )

    obs = BodyObservation(
        joint_angles=np.random.randn(42).astype(np.float32) * 0.3,
        joint_velocities=np.random.randn(42).astype(np.float32) * 0.5,
        contact_forces=np.array([0.8, 0.9, 0.7, 0.6, 0.5, 0.4], dtype=np.float32),
        body_velocity=np.array([2.0, 0.5, 0.0], dtype=np.float32),
        body_orientation=np.array([0.0, 0.1, 0.0], dtype=np.float32),
    )

    brain_input = encoder.encode(obs)

    stat("Input neuron count", len(brain_input.neuron_ids))
    stat("Rate range", f"{brain_input.firing_rates_hz.min():.1f} - {brain_input.firing_rates_hz.max():.1f} Hz")
    stat("Mean rate", f"{brain_input.firing_rates_hz.mean():.1f} Hz")

    print()
    print("  Per-channel mean rates:")
    for ch, idx_list in encoder._channels.items():
        if len(idx_list) > 0:
            ch_rates = brain_input.firing_rates_hz[idx_list]
            print(f"    {ch:<22} {ch_rates.mean():6.1f} Hz  (n={len(idx_list)})")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(5, "THE BRAIN (FakeBrainRunner demo)")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  The brain takes Poisson rates in, simulates 139k LIF neurons,")
    print("  and reads out firing rates from descending neurons.")
    print()
    print("  Two implementations:")
    print("    FakeBrainRunner  — pass-through for testing (instant)")
    print("    Brian2BrainRunner — real 139k neuron sim (~15s to build)")
    print()
    print("  Running FakeBrainRunner now:")

    from bridge.brain_runner import create_brain_runner

    brain = create_brain_runner(
        sensory_ids=sensory, readout_ids=readout, use_fake=True)

    brain_output = brain.step(brain_input, sim_ms=20.0)

    stat("Readout neuron count", len(brain_output.neuron_ids))
    stat("Firing rate range", f"{brain_output.firing_rates_hz.min():.1f} - {brain_output.firing_rates_hz.max():.1f} Hz")
    stat("Mean rate", f"{brain_output.firing_rates_hz.mean():.1f} Hz")
    stat("Active neurons (>0 Hz)", int(np.sum(brain_output.firing_rates_hz > 0)))

    print()
    note("The real brain (Brian2BrainRunner) builds the full 139k-neuron")
    note("network from Connectivity_783.parquet. Takes ~15s, uses ~4GB RAM.")
    note("Use --fake-brain flag in experiments to skip this for testing.")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(6, "THE DESCENDING DECODER (live demo)")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  DescendingDecoder converts DN rates -> locomotion command.")
    print()

    from bridge.descending_decoder import DescendingDecoder

    decoder = DescendingDecoder.from_json(
        DATA / "decoder_groups_v5_steering.json", rate_scale=12.0)

    cmd = decoder.decode(brain_output)
    group_rates = decoder.get_group_rates(brain_output)

    print("  Raw group rates (Hz):")
    for group, rate in group_rates.items():
        print(f"    {group:<15} {rate:6.1f} Hz")

    print()
    print("  Decoded command:")
    stat("forward_drive", f"{cmd.forward_drive:.3f}")
    stat("turn_drive", f"{cmd.turn_drive:+.3f}")
    stat("step_frequency", f"{cmd.step_frequency:.3f}")
    stat("stance_gain", f"{cmd.stance_gain:.3f}")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(7, "THE LOCOMOTION BRIDGE (live demo)")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  LocomotionBridge converts LocomotionCommand -> joint angles")
    print("  using FlyGym's CPG + PreprogrammedSteps.")
    print()

    from bridge.locomotion_bridge import LocomotionBridge

    loco = LocomotionBridge(seed=42)
    loco.warmup(500)  # ramp CPG from zero

    action = loco.step(cmd)

    stat("Joint angles shape", action["joints"].shape)
    stat("Adhesion (per leg)", action["adhesion"].tolist())
    stat("Joint angle range", f"{action['joints'].min():.2f} to {action['joints'].max():.2f} rad")

    print()
    print("  CPG state:")
    stat("Phases", np.round(loco.cpg.curr_phases, 2).tolist())
    stat("Magnitudes", np.round(loco.cpg.curr_magnitudes, 2).tolist())
    note("Phases = [0, pi, 0, pi, 0, pi] = tripod gait pattern")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(8, "FULL LOOP — 100 steps with FlyGym")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  Now let's run a real closed-loop: body + fake brain.")
    print("  100 body steps, brain step every 200 body steps (= 1 brain step).")
    print()

    import flygym
    from bridge.flygym_adapter import FlyGymAdapter

    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(fly=fly, arena=flygym.arena.FlatTerrain(), timestep=1e-4)
    obs, info = sim.reset()

    adapter = FlyGymAdapter()
    loco2 = LocomotionBridge(seed=42)
    loco2.warmup(0)
    loco2.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6))

    current_cmd = cmd
    positions = []

    for step in range(100):
        if step == 0:
            body_obs = adapter.extract_body_observation(obs)
            brain_in = encoder.encode(body_obs)
            brain_out = brain.step(brain_in, sim_ms=20.0)
            current_cmd = decoder.decode(brain_out)

        action = loco2.step(current_cmd)
        obs, reward, terminated, truncated, info = sim.step(action)
        pos = np.array(obs["fly"][0])
        positions.append(pos)

    sim.close()
    positions = np.array(positions)
    displacement = positions[-1] - positions[0]

    print(f"  Ran 100 body steps (= {100 * 1e-4 * 1000:.1f} ms of simulated time)")
    stat("Start position", np.round(positions[0], 3).tolist())
    stat("End position", np.round(positions[-1], 3).tolist())
    stat("Displacement (mm)", np.round(displacement, 3).tolist())
    stat("Forward (X)", f"{displacement[0]:.4f} mm")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(9, "DN SEGREGATION — why the connectome matters")
    # ──────────────────────────────────────────────────────────────────
    print()
    print("  The connectome's wiring creates modality-specific DN channels.")
    print("  1-hop analysis: which sensory modalities reach which DNs?")
    print()

    # Quick connectome analysis
    readout_set = set(readout.tolist())
    # Get root IDs for small indices
    root_ids_array = comp.iloc[:, 0].values
    readout_rootids = set()
    for rid in readout:
        if rid < 1_000_000:
            readout_rootids.add(int(root_ids_array[rid]))
        else:
            readout_rootids.add(int(rid))

    dn_edges = con[con["Postsynaptic_ID"].isin(readout_rootids)]

    modalities = {
        "somatosensory": set(channel_map.get("proprioceptive", []) +
                             channel_map.get("mechanosensory", []) +
                             channel_map.get("vestibular", []) +
                             channel_map.get("gustatory", [])),
        "visual (LPLC2)": set(channel_map.get("lplc2_left", []) +
                              channel_map.get("lplc2_right", [])),
        "olfactory": set(channel_map.get("olfactory_left", []) +
                         channel_map.get("olfactory_right", [])),
    }

    dn_targets = {}
    for mod_name, neurons in modalities.items():
        edges = dn_edges[dn_edges["Presynaptic_ID"].isin(neurons)]
        targets = set(edges["Postsynaptic_ID"].unique())
        dn_targets[mod_name] = targets
        syn_count = int(edges["Connectivity"].sum())
        stat(f"{mod_name}", f"{len(targets)} DNs, {syn_count:,} synapses")

    print()
    print("  Pairwise Jaccard overlap (lower = more segregated):")
    names = list(dn_targets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = dn_targets[names[i]], dn_targets[names[j]]
            union = len(a | b)
            jaccard = len(a & b) / union if union > 0 else 0
            print(f"    {names[i]:<18} vs {names[j]:<18} Jaccard = {jaccard:.3f}")

    print()
    note("Jaccard << 0.1 means almost no shared DN targets between modalities.")
    note("This labeled-line architecture is the key finding of the paper.")

    pause()

    # ──────────────────────────────────────────────────────────────────
    section(10, "EXPERIMENT FILES — what you can run")
    # ──────────────────────────────────────────────────────────────────
    print()
    experiments = [
        ("ablation_study.py", "10/10", "Silence each DN group, measure causal effect"),
        ("odor_valence.py", "6/6", "DM1 attractive vs DM5 aversive odor discrimination"),
        ("looming.py", "5/5", "LPLC2 injection -> escape turning (21x over shuffled)"),
        ("sensory_perturbation.py", "—", "Asymmetric sensory input -> turning circuits"),
        ("closed_loop_walk.py", "—", "Baseline brain-body closed-loop walking"),
        ("vnc_validation.py", "19/20", "MANC VNC (13K neurons) ablation validation"),
        ("bottleneck_causal.py", "4/4", "DNb05 thermo/hygro bottleneck (16.4x)"),
        ("representational_geometry.py", "WIP", "PCA/decoding/RSA on DN population codes"),
    ]

    print(f"  {'Script':<38} {'Tests':<8} Description")
    print("  " + "-" * 66)
    for script, tests, desc in experiments:
        print(f"  {script:<38} {tests:<8} {desc}")

    print()
    print("  All experiments support:")
    print("    --fake-brain    Skip Brian2, use pass-through (fast testing)")
    print("    --body-steps N  Control simulation length")
    print("    --seeds N N N   Multiple random seeds")
    print()
    print("  Example:")
    print(f"    cd {BASE}")
    print("    python experiments/ablation_study.py --fake-brain --body-steps 2000")

    pause()

    # ──────────────────────────────────────────────────────────────────
    print()
    print(BOLD + GREEN)
    print("  +==========================================================+")
    print("  |  WALKTHROUGH COMPLETE                                    |")
    print("  +==========================================================+")
    print(RESET)
    print("  Key takeaway: the FlyWire connectome wiring alone produces")
    print("  causal locomotion, olfactory valence, and looming escape —")
    print("  no learning, no parameter fitting, fully transparent.")
    print()
    print("  Paper: plastic-fly/paper/connectome_sensorimotor_paper.pdf")
    print("  Slides: plastic-fly/paper/ramdya_slides.pdf")
    print("  Figures: plastic-fly/figures/")
    print()


if __name__ == "__main__":
    main()
