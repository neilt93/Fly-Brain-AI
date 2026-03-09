"""
Main entry point: run terrain shift experiment with curator + scientist + dashboard.

Usage:
    python run.py                        # defaults
    python run.py --total-steps 30000
    python run.py --flat-length 10 --blocks-length 20
    python run.py --plastic-lr 5e-6
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.terrain_shift import ExperimentConfig, run_experiment
from curator.curator_agent import CuratorAgent
from scientist.scientist_agent import ScientistAgent
from dashboard.generate import generate_dashboard


def export_unity_timeseries(experiment_output, output_dir):
    """Export time-series JSON for Unity visualization with joint angles."""
    from analysis.gait_metrics import classify_stance_swing, compute_tripod_score

    output_path = Path(output_dir)

    for ctrl_name in ("plastic", "fixed"):
        data = experiment_output["data"][ctrl_name]

        positions = data["positions"]
        contacts = data["contacts"]
        contact_forces = data["contact_forces_raw"]
        end_effectors = data["end_effectors"]
        weight_drifts = data["weight_drifts"]
        joint_angles = data.get("joint_angles", np.zeros((len(positions), 42)))
        joint_names = data.get("joint_names", [])

        # Compute tripod score per frame
        ss = classify_stance_swing(contact_forces)
        tripod = compute_tripod_score(ss)

        n_frames = len(positions)
        dt = 0.005  # log_interval * timestep = 50 * 1e-4

        ts = {
            "controller": ctrl_name,
            "dt": dt,
            "n_frames": n_frames,
            "positions": positions.tolist() if hasattr(positions, "tolist") else positions,
            "contacts": contacts.tolist() if hasattr(contacts, "tolist") else contacts,
            "contact_forces": contact_forces.tolist() if hasattr(contact_forces, "tolist") else contact_forces,
            "end_effectors": end_effectors.tolist() if hasattr(end_effectors, "tolist") else end_effectors,
            "joint_angles": joint_angles.tolist() if hasattr(joint_angles, "tolist") else joint_angles,
            "joint_names": list(joint_names),
            "tripod_score": tripod.tolist() if hasattr(tripod, "tolist") else list(tripod),
            "weight_drifts": [float(w) for w in weight_drifts] if weight_drifts else [],
            "perturbation_idx": data["perturbation_idx"] or 0,
        }

        out_file = output_path / f"timeseries_{ctrl_name}.json"
        with open(out_file, "w") as f:
            json.dump(ts, f)
        print(f"Unity timeseries: {out_file} ({n_frames} frames, {len(joint_names)} DOFs)")


def main():
    parser = argparse.ArgumentParser(description="Plastic Fly Controller")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--flat-length", type=float, default=8.0)
    parser.add_argument("--blocks-length", type=float, default=15.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--modulation-scale", type=float, default=0.15)
    parser.add_argument("--plastic-lr", type=float, default=1e-5)
    parser.add_argument("--cpg-freq", type=float, default=12.0)
    parser.add_argument("--cpg-amplitude", type=float, default=1.0)
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
        cpg_freq=args.cpg_freq,
        cpg_amplitude=args.cpg_amplitude,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Run experiment
    print("Starting experiment...")
    experiment_output = run_experiment(config)

    # Export Unity timeseries (with joint angles)
    print("\nExporting Unity timeseries...")
    export_unity_timeseries(experiment_output, args.output_dir)

    # Curator analysis
    print("\n" + "=" * 60)
    print("CURATOR ANALYSIS")
    print("=" * 60)

    curator = CuratorAgent(log_dir=str(Path(args.output_dir) / "curator"))
    curator.ingest_run(experiment_output, run_id="run_001")

    # Print curator summary
    print(curator.summarize())

    # Print attention items
    attention = curator.get_attention_events()
    if attention:
        print(f"\n--- {len(attention)} items need your attention ---")
        for event in attention:
            print(f"  [{event.priority.name}] {event.summary}")

    # Unity exports
    unity = curator.get_unity_exports()
    if unity:
        print(f"\n--- {len(unity)} episodes tagged for Unity export ---")
        for event in unity:
            print(f"  [{event.run_id}] {event.change_type.value}")

    # Save curator summary JSON
    curator_data = curator.to_summary_json()
    curator_summary_path = Path(args.output_dir) / "curator_summary.json"
    with open(curator_summary_path, "w") as f:
        json.dump(curator_data, f, indent=2, default=str)

    # Save unity candidates
    unity_candidates = curator_data.get("unity_candidates", [])
    if unity_candidates:
        with open(Path(args.output_dir) / "unity_candidates.json", "w") as f:
            json.dump(unity_candidates, f, indent=2, default=str)

    # Scientist recommendations
    print("\n" + "=" * 60)
    print("SCIENTIST RECOMMENDATIONS")
    print("=" * 60)

    runs_path = str(Path(args.output_dir) / "logs" / "runs.jsonl")
    scientist = ScientistAgent(log_path=runs_path)
    proposals = scientist.save_proposals(
        output_path=str(Path(args.output_dir) / "logs" / "next_experiments.json")
    )
    if proposals:
        for p in proposals:
            print(f"  [{p.strategy}] {p.name} (score={p.priority_score:.2f})")
            print(f"    {p.rationale}")
    else:
        print("  No proposals yet (need more runs for meaningful recommendations)")

    # Generate dashboard
    proposal_dicts = [{
        "name": p.name,
        "strategy": p.strategy,
        "priority_score": p.priority_score,
        "rationale": p.rationale,
    } for p in proposals] if proposals else []

    curator_event_dicts = [e.to_dict() for e in curator.events]
    dashboard_path = generate_dashboard(
        experiment_dir=args.output_dir,
        curator_summary=curator.summarize(),
        curator_events=curator_event_dicts,
        curator_summary_data=curator_data,
        scientist_proposals=proposal_dicts,
    )
    print(f"\nDashboard: {dashboard_path}")
    print("Done.")


if __name__ == "__main__":
    main()
