"""
Batch runner for plastic fly experiments.

Usage:
    python run_batch.py --sweep plastic_lr 1e-6 5e-6 1e-5
    python run_batch.py --seeds 42 43 44
    python run_batch.py --from-scientist    # run top recommendations

After each run: curator + scientist + dashboard auto-regenerate.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.terrain_shift import ExperimentConfig, run_experiment
from curator.curator_agent import CuratorAgent
from scientist.scientist_agent import ScientistAgent
from dashboard.generate import generate_dashboard


def _write_json_atomic(path: Path, payload, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, **kwargs)
    tmp.replace(path)


def _run_and_update(config: ExperimentConfig, curator: CuratorAgent, run_idx: int):
    """Run one experiment, ingest into curator, return output."""
    print(f"\n{'='*60}")
    print(f"BATCH RUN {run_idx + 1}: seed={config.seed}")
    print(f"{'='*60}")

    output = run_experiment(config)
    curator.ingest_run(output, run_id=f"batch_{run_idx:03d}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Batch runner for plastic fly")
    parser.add_argument("--seeds", type=int, nargs="+", help="Seeds to run")
    parser.add_argument("--sweep", nargs="+",
                        help="Parameter sweep: PARAM val1 val2 val3 ...")
    parser.add_argument("--from-scientist", action="store_true",
                        help="Run top scientist recommendations")
    parser.add_argument("--n-proposals", type=int, default=3,
                        help="Number of scientist proposals to run")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--output-dir", default="logs/terrain_shift")
    # Base config overrides
    parser.add_argument("--plastic-lr", type=float, default=1e-5)
    parser.add_argument("--modulation-scale", type=float, default=0.15)
    parser.add_argument("--sparsity", type=float, default=0.8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curator = CuratorAgent(log_dir=str(output_dir / "curator"))

    configs = []

    if args.seeds:
        for seed in args.seeds:
            configs.append(ExperimentConfig(
                total_steps=args.total_steps,
                output_dir=args.output_dir,
                seed=seed,
                plastic_lr=args.plastic_lr,
                modulation_scale=args.modulation_scale,
                sparsity=args.sparsity,
            ))

    elif args.sweep:
        param_name = args.sweep[0]
        values = [float(v) for v in args.sweep[1:]]
        for val in values:
            cfg_kwargs = {
                "total_steps": args.total_steps,
                "output_dir": args.output_dir,
                "plastic_lr": args.plastic_lr,
                "modulation_scale": args.modulation_scale,
                "sparsity": args.sparsity,
            }
            # Override the swept parameter
            if param_name == "plastic_lr":
                cfg_kwargs["plastic_lr"] = val
            elif param_name == "modulation_scale":
                cfg_kwargs["modulation_scale"] = val
            elif param_name == "sparsity":
                cfg_kwargs["sparsity"] = val
            else:
                print(f"Unknown sweep param: {param_name}")
                sys.exit(1)
            configs.append(ExperimentConfig(**cfg_kwargs))

    elif args.from_scientist:
        runs_path = str(output_dir / "logs" / "runs.jsonl")
        scientist = ScientistAgent(log_path=runs_path)
        proposals = scientist.recommend(n=args.n_proposals)

        if not proposals:
            print("No proposals from scientist (need existing runs first).")
            sys.exit(0)

        print(f"Scientist recommends {len(proposals)} experiments:")
        for p in proposals:
            print(f"  [{p.strategy}] {p.name} (score={p.priority_score:.2f})")
            print(f"    {p.rationale}")

        for p in proposals:
            overrides = p.config_overrides
            configs.append(ExperimentConfig(
                total_steps=overrides.get("total_steps", args.total_steps),
                output_dir=args.output_dir,
                seed=overrides.get("seed", 42),
                plastic_lr=overrides.get("plastic_lr", args.plastic_lr),
                modulation_scale=overrides.get("modulation_scale", args.modulation_scale),
                sparsity=overrides.get("sparsity", args.sparsity),
            ))
    else:
        print("Specify --seeds, --sweep, or --from-scientist")
        sys.exit(1)

    # Run all experiments
    for i, config in enumerate(configs):
        _run_and_update(config, curator, i)

    # Post-batch: curator summary, scientist, dashboard
    print(f"\n{'='*60}")
    print("POST-BATCH SUMMARY")
    print(f"{'='*60}")
    print(curator.summarize())

    # Save curator summary
    curator_data = curator.to_summary_json()
    _write_json_atomic(output_dir / "curator_summary.json", curator_data, indent=2, default=str)

    # Scientist recommendations for next time
    runs_path = str(output_dir / "logs" / "runs.jsonl")
    scientist = ScientistAgent(log_path=runs_path)
    proposals = scientist.save_proposals(
        output_path=str(output_dir / "logs" / "next_experiments.json")
    )
    if proposals:
        print(f"\nScientist proposes {len(proposals)} next experiments:")
        for p in proposals:
            print(f"  [{p.strategy}] {p.name} (score={p.priority_score:.2f})")

    # Dashboard
    dashboard_path = generate_dashboard(
        experiment_dir=args.output_dir,
        curator_summary=curator.summarize(),
        curator_events=[e.to_dict() for e in curator.events],
        curator_summary_data=curator_data,
        scientist_proposals=[{
            "name": p.name,
            "strategy": p.strategy,
            "priority_score": p.priority_score,
            "rationale": p.rationale,
        } for p in proposals],
    )
    print(f"\nDashboard: {dashboard_path}")
    print("Done.")


if __name__ == "__main__":
    main()
