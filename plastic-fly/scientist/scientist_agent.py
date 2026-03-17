"""
Scientist Agent — deterministic experiment recommender.

Analyzes completed runs from runs.jsonl and proposes next experiments
using five strategies: seed replication, sweep gaps, failure follow-up,
boundary probing, and ablation.

No LLM required — pure Python heuristics over structured logs.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from collections import defaultdict


def _write_json_atomic(path, obj, **kwargs):
    """Write JSON atomically: write to tmp file then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, **kwargs)
    tmp.replace(path)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structlog.structured_log import read_runs


@dataclass
class ExperimentProposal:
    """A recommended next experiment."""
    name: str
    config_overrides: dict
    rationale: str
    priority_score: float  # 0-1, higher = more important
    strategy: str  # which strategy generated this


# Default parameter grid for sweep gap analysis
DEFAULT_GRID = {
    "plastic_lr": [1e-6, 5e-6, 1e-5, 5e-5],
    "modulation_scale": [0.05, 0.10, 0.15, 0.20, 0.30],
    "sparsity": [0.5, 0.7, 0.8, 0.9],
}


class ScientistAgent:
    """Analyzes experiment history and recommends next experiments.

    Usage:
        scientist = ScientistAgent(log_path="logs/runs.jsonl")
        proposals = scientist.recommend(n=5)
    """

    def __init__(
        self,
        log_path: str = "logs/runs.jsonl",
        min_seeds_per_config: int = 3,
        grid: Optional[dict] = None,
    ):
        self.log_path = log_path
        self.min_seeds = min_seeds_per_config
        self.grid = grid or DEFAULT_GRID
        self._runs = None

    @property
    def runs(self) -> list[dict]:
        if self._runs is None:
            self._runs = read_runs(self.log_path)
        return self._runs

    def reload(self):
        """Force reload runs from disk."""
        self._runs = None

    def _config_key(self, run: dict) -> str:
        """Extract a hashable config key (ignoring seed)."""
        cfg = run.get("config", {})
        parts = []
        for k in sorted(cfg.keys()):
            if k in ("seed", "output_dir"):
                continue
            parts.append(f"{k}={cfg[k]}")
        return "|".join(parts)

    def _group_by_config(self) -> dict[str, list[dict]]:
        """Group runs by configuration (ignoring seed)."""
        groups = defaultdict(list)
        for run in self.runs:
            key = self._config_key(run)
            groups[key].append(run)
        return dict(groups)

    def _get_metric(self, run: dict, metric: str, default: float = 0.0) -> float:
        """Extract a metric from a run record."""
        metrics = run.get("metrics", {})
        if metric in metrics:
            val = metrics[metric]
            return float(val) if val is not None else default
        # Try nested in phases
        phases = run.get("phases", {})
        for phase in phases.values():
            if isinstance(phase, dict) and metric in phase:
                val = phase[metric]
                return float(val) if val is not None else default
        return default

    def _best_run(self) -> Optional[dict]:
        """Find the run with highest performance ratio (plastic only)."""
        plastic_runs = [r for r in self.runs if r.get("controller") == "plastic"]
        if not plastic_runs:
            return None
        return max(plastic_runs, key=lambda r: self._get_metric(r, "performance_ratio"))

    def _strategy_seed_replication(self) -> list[ExperimentProposal]:
        """Propose more seeds for under-replicated configs."""
        proposals = []
        groups = self._group_by_config()

        for config_key, runs in groups.items():
            seeds_seen = {r.get("config", {}).get("seed", 0) for r in runs}
            if len(seeds_seen) < self.min_seeds:
                needed = self.min_seeds - len(seeds_seen)
                # Pick next seeds
                max_seed = max(seeds_seen) if seeds_seen else 41
                new_seeds = list(range(max_seed + 1, max_seed + 1 + needed))

                # Get config from first run
                base_config = runs[0].get("config", {})
                for seed in new_seeds:
                    overrides = {k: v for k, v in base_config.items()
                                 if k not in ("output_dir",)}
                    overrides["seed"] = seed

                    # Score: more valuable if results are promising
                    avg_ratio = np.mean([
                        self._get_metric(r, "performance_ratio")
                        for r in runs
                    ])
                    score = 0.3 + 0.4 * min(avg_ratio, 1.0) + 0.3 * (1 - len(seeds_seen) / self.min_seeds)

                    proposals.append(ExperimentProposal(
                        name=f"replicate_seed_{seed}",
                        config_overrides=overrides,
                        rationale=f"Config has {len(seeds_seen)}/{self.min_seeds} seeds. "
                                  f"Avg perf_ratio={avg_ratio:.3f}. Need {needed} more for significance.",
                        priority_score=min(score, 1.0),
                        strategy="seed_replication",
                    ))

        return proposals

    def _strategy_sweep_gap(self) -> list[ExperimentProposal]:
        """Find untested parameter combinations in the grid."""
        proposals = []

        # Collect tested combos
        tested = set()
        for run in self.runs:
            cfg = run.get("config", {})
            combo = tuple(cfg.get(k, None) for k in sorted(self.grid.keys()))
            tested.add(combo)

        # Generate all grid combos
        keys = sorted(self.grid.keys())
        all_combos = self._cartesian_product([self.grid[k] for k in keys])

        for combo in all_combos:
            if combo not in tested:
                overrides = {k: v for k, v in zip(keys, combo)}
                # Score: prefer combos near known-good regions
                proximity = self._proximity_to_best(overrides)
                coverage = 1.0 - len(tested) / max(len(all_combos), 1)
                score = 0.3 * proximity + 0.3 * coverage + 0.4 * 0.5

                proposals.append(ExperimentProposal(
                    name=f"sweep_{'_'.join(f'{k}={v}' for k, v in overrides.items())}",
                    config_overrides=overrides,
                    rationale=f"Untested grid point. Proximity to best: {proximity:.2f}.",
                    priority_score=min(score, 1.0),
                    strategy="sweep_gap",
                ))

        return proposals

    def _strategy_failure_followup(self) -> list[ExperimentProposal]:
        """Propose fixes for runs that showed forgetting or degradation."""
        proposals = []

        for run in self.runs:
            if run.get("controller") != "plastic":
                continue

            ratio = self._get_metric(run, "performance_ratio")
            drift = self._get_metric(run, "weight_drift")
            cfg = run.get("config", {})

            # High drift + low performance -> try lower lr or higher decay
            if drift > 0.1 and ratio < 0.5:
                current_lr = cfg.get("plastic_lr", 1e-5)
                current_decay = cfg.get("plastic_decay", 1.0)

                # Propose lower lr
                new_lr = current_lr * 0.5
                proposals.append(ExperimentProposal(
                    name=f"reduce_lr_{new_lr:.1e}",
                    config_overrides={**cfg, "plastic_lr": new_lr},
                    rationale=f"Run {run.get('run_id', '?')} had high drift ({drift:.3f}) "
                              f"and low perf ({ratio:.3f}). Halving lr.",
                    priority_score=0.7,
                    strategy="failure_followup",
                ))

                # Propose higher decay
                new_decay = min(current_decay * 1.5, 5.0)
                proposals.append(ExperimentProposal(
                    name=f"increase_decay_{new_decay:.1f}",
                    config_overrides={**cfg, "plastic_decay": new_decay},
                    rationale=f"Run {run.get('run_id', '?')} had high drift ({drift:.3f}). "
                              f"Increasing decay to {new_decay:.1f}.",
                    priority_score=0.65,
                    strategy="failure_followup",
                ))

        return proposals

    def _strategy_boundary_probe(self) -> list[ExperimentProposal]:
        """Push best config further in promising directions."""
        proposals = []
        best = self._best_run()
        if not best:
            return proposals

        cfg = best.get("config", {})
        ratio = self._get_metric(best, "performance_ratio")

        if ratio < 0.3:  # Not good enough to push
            return proposals

        # Try pushing each grid parameter further
        for param, values in self.grid.items():
            current = cfg.get(param)
            if current is None:
                continue

            sorted_vals = sorted(values)
            idx = None
            for i, v in enumerate(sorted_vals):
                if abs(v - current) < 1e-10:
                    idx = i
                    break

            if idx is not None:
                # Push up if at or near top
                if idx >= len(sorted_vals) - 1:
                    new_val = current * 1.5
                    proposals.append(ExperimentProposal(
                        name=f"probe_{param}_{new_val}",
                        config_overrides={**cfg, param: new_val},
                        rationale=f"Best config has {param}={current} at grid edge. "
                                  f"Probing {new_val} to find true optimum.",
                        priority_score=0.5 + 0.3 * ratio,
                        strategy="boundary_probe",
                    ))
                # Push down if at or near bottom
                if idx <= 0:
                    new_val = current * 0.5
                    proposals.append(ExperimentProposal(
                        name=f"probe_{param}_{new_val}",
                        config_overrides={**cfg, param: new_val},
                        rationale=f"Best config has {param}={current} at grid edge. "
                                  f"Probing {new_val} downward.",
                        priority_score=0.5 + 0.2 * ratio,
                        strategy="boundary_probe",
                    ))

        return proposals

    def _strategy_ablation(self) -> list[ExperimentProposal]:
        """Ablation: run best plastic config with lr=0 to confirm plasticity is causal."""
        proposals = []
        best = self._best_run()
        if not best:
            return proposals

        cfg = best.get("config", {})

        # Check if ablation already exists
        for run in self.runs:
            if (run.get("controller") == "plastic" and
                    run.get("config", {}).get("plastic_lr", 1) == 0):
                return proposals  # Already done

        proposals.append(ExperimentProposal(
            name="ablation_lr0",
            config_overrides={**cfg, "plastic_lr": 0.0},
            rationale="Ablation: same architecture with lr=0 confirms plasticity "
                      "is the cause of improved performance, not architecture.",
            priority_score=0.85,
            strategy="ablation",
        ))

        return proposals

    def _proximity_to_best(self, overrides: dict) -> float:
        """How close a config is to the best-known config (normalized)."""
        best = self._best_run()
        if not best:
            return 0.5

        best_cfg = best.get("config", {})
        distances = []
        for param, values in self.grid.items():
            v_new = overrides.get(param)
            v_best = best_cfg.get(param)
            if v_new is not None and v_best is not None:
                val_range = max(values) - min(values)
                if val_range > 0:
                    distances.append(abs(v_new - v_best) / val_range)

        if not distances:
            return 0.5
        return float(1.0 - min(1.0, sum(distances) / len(distances)))

    @staticmethod
    def _cartesian_product(lists: list[list]) -> list[tuple]:
        """Generate cartesian product of multiple lists."""
        if not lists:
            return [()]
        result = [()]
        for lst in lists:
            result = [prev + (val,) for prev in result for val in lst]
        return result

    def recommend(self, n: int = 5) -> list[ExperimentProposal]:
        """Generate top-N experiment proposals across all strategies.

        Returns proposals sorted by priority_score descending.
        """
        all_proposals = []
        all_proposals.extend(self._strategy_seed_replication())
        all_proposals.extend(self._strategy_sweep_gap())
        all_proposals.extend(self._strategy_failure_followup())
        all_proposals.extend(self._strategy_boundary_probe())
        all_proposals.extend(self._strategy_ablation())

        # Sort by score, take top N
        all_proposals.sort(key=lambda p: p.priority_score, reverse=True)
        return all_proposals[:n]

    def save_proposals(self, output_path: str = "logs/next_experiments.json"):
        """Generate recommendations and save to JSON."""
        proposals = self.recommend()
        data = [asdict(p) for p in proposals]
        path = Path(output_path)
        _write_json_atomic(path, data, indent=2, default=str)
        return proposals
