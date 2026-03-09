"""
Curator Agent

Watches experiment results and decides what is worth attention.

This agent does NOT control the fly. It answers:
- What changed?
- Was it meaningful?
- Was it better or worse than before?
- Is this worth Neil seeing right now?
- Is this worth exporting to Unity?

Architecture:
- Monitors experiment logs and metrics in real time
- Maintains a running history of "interesting" events
- Applies significance filters (not just any change — meaningful change)
- Generates attention-worthy notifications
- Tags episodes for Unity export
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class Priority(Enum):
    """How urgently Neil should see this."""
    IGNORE = 0        # Noise, not worth mentioning
    LOG = 1           # Record for completeness, don't alert
    NOTABLE = 2       # Worth seeing in the next summary
    ATTENTION = 3     # Look at this soon
    BREAKTHROUGH = 4  # Stop what you're doing and look


class ChangeType(Enum):
    """What kind of change was detected."""
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RECOVERY_DETECTED = "recovery_detected"
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"
    NOVEL_GAIT = "novel_gait"
    WEIGHT_CONVERGENCE = "weight_convergence"
    WEIGHT_DIVERGENCE = "weight_divergence"
    SYMMETRY_BREAK = "symmetry_break"
    SYMMETRY_RESTORE = "symmetry_restore"
    FIRST_SUCCESSFUL_RUN = "first_successful_run"
    NEW_BEST = "new_best"


@dataclass
class CuratorEvent:
    """A single noteworthy observation from the curator."""
    timestamp: float
    change_type: ChangeType
    priority: Priority
    summary: str
    details: dict = field(default_factory=dict)
    export_to_unity: bool = False
    run_id: str = ""

    def to_dict(self):
        d = asdict(self)
        d["change_type"] = self.change_type.value
        d["priority"] = self.priority.name
        return d


@dataclass
class RunSnapshot:
    """Summary of a single experiment run for comparison."""
    run_id: str
    timestamp: float
    controller_type: str  # "fixed" or "plastic"
    terrain: str
    distance_before: float
    distance_after: float
    performance_ratio: float
    gait_symmetry_before: float
    gait_symmetry_after: float
    recovery_time: Optional[int]
    weight_drift: float = 0.0
    num_falls: int = 0


class CuratorAgent:
    """Watches experiments and curates what matters.

    Usage:
        curator = CuratorAgent(log_dir="logs/curator")
        curator.ingest_run(results_dict, run_id="run_001")
        events = curator.get_attention_events()
        summary = curator.summarize()
    """

    def __init__(
        self,
        log_dir: str = "logs/curator",
        # Significance thresholds
        improvement_threshold: float = 0.1,   # 10% improvement = notable
        degradation_threshold: float = 0.15,  # 15% degradation = attention
        symmetry_change_threshold: float = 0.1,
        weight_drift_alert: float = 1.0,
        forgetting_threshold: float = 0.5,    # 50% drop = catastrophic
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.improvement_threshold = improvement_threshold
        self.degradation_threshold = degradation_threshold
        self.symmetry_change_threshold = symmetry_change_threshold
        self.weight_drift_alert = weight_drift_alert
        self.forgetting_threshold = forgetting_threshold

        # State
        self.history: list[RunSnapshot] = []
        self.events: list[CuratorEvent] = []
        self.best_performance: dict[str, float] = {}  # controller -> best ratio

    def ingest_run(self, experiment_results: dict, run_id: str = ""):
        """Process results from a terrain_shift experiment.

        Args:
            experiment_results: dict from terrain_shift.run_experiment()
            run_id: identifier for this run
        """
        if not run_id:
            run_id = f"run_{len(self.history):04d}"

        results = experiment_results.get("results", experiment_results)
        report_fixed = experiment_results.get("report_fixed")
        report_plastic = experiment_results.get("report_plastic")

        now = time.time()

        # Create snapshots
        for ctrl_type in ("fixed", "plastic"):
            baseline = results.get("baseline", {}).get(ctrl_type, {})
            shift = results.get("shift", {}).get(ctrl_type, {})
            report = report_fixed if ctrl_type == "fixed" else report_plastic

            dist_before = baseline.get("distance", 0)
            dist_after = shift.get("distance", 0)
            ratio = dist_after / abs(dist_before) if abs(dist_before) > 1e-6 else 0

            snapshot = RunSnapshot(
                run_id=f"{run_id}_{ctrl_type}",
                timestamp=now,
                controller_type=ctrl_type,
                terrain=results.get("config", {}).get("shift_terrain", "unknown"),
                distance_before=dist_before,
                distance_after=dist_after,
                performance_ratio=ratio,
                gait_symmetry_before=baseline.get("symmetry", 0),
                gait_symmetry_after=shift.get("symmetry", 0),
                recovery_time=report.recovery_time_steps if report else None,
                weight_drift=shift.get("weight_drift", 0),
            )
            self.history.append(snapshot)

            # Analyze this snapshot
            self._analyze_snapshot(snapshot)

        # Compare fixed vs plastic
        self._compare_controllers(run_id, results, report_fixed, report_plastic)

        # Check for forgetting
        if "forgetting" in results:
            self._check_forgetting(run_id, results)

        # Persist events
        self._save_events()

    def _analyze_snapshot(self, snap: RunSnapshot):
        """Generate events from a single run snapshot."""
        key = snap.controller_type

        # Check for new best
        prev_best = self.best_performance.get(key, 0)
        if snap.performance_ratio > prev_best:
            self.best_performance[key] = snap.performance_ratio
            if prev_best > 0:  # not the first run
                improvement = (snap.performance_ratio - prev_best) / max(prev_best, 0.01)
                if improvement > self.improvement_threshold:
                    self.events.append(CuratorEvent(
                        timestamp=snap.timestamp,
                        change_type=ChangeType.NEW_BEST,
                        priority=Priority.ATTENTION,
                        summary=f"New best for {key}: {snap.performance_ratio:.3f} "
                                f"(+{improvement*100:.1f}% over previous best)",
                        details={"ratio": snap.performance_ratio, "prev": prev_best},
                        export_to_unity=True,
                        run_id=snap.run_id,
                    ))

        # Symmetry changes
        sym_delta = snap.gait_symmetry_after - snap.gait_symmetry_before
        if abs(sym_delta) > self.symmetry_change_threshold:
            if sym_delta < 0:
                self.events.append(CuratorEvent(
                    timestamp=snap.timestamp,
                    change_type=ChangeType.SYMMETRY_BREAK,
                    priority=Priority.NOTABLE,
                    summary=f"{key}: gait symmetry dropped {sym_delta:.3f} "
                            f"({snap.gait_symmetry_before:.3f} -> {snap.gait_symmetry_after:.3f})",
                    details={"before": snap.gait_symmetry_before,
                             "after": snap.gait_symmetry_after},
                    run_id=snap.run_id,
                ))
            else:
                self.events.append(CuratorEvent(
                    timestamp=snap.timestamp,
                    change_type=ChangeType.SYMMETRY_RESTORE,
                    priority=Priority.NOTABLE,
                    summary=f"{key}: gait symmetry recovered {sym_delta:.3f}",
                    details={"before": snap.gait_symmetry_before,
                             "after": snap.gait_symmetry_after},
                    run_id=snap.run_id,
                ))

        # Weight drift alerts (plastic only)
        if snap.weight_drift > self.weight_drift_alert:
            self.events.append(CuratorEvent(
                timestamp=snap.timestamp,
                change_type=ChangeType.WEIGHT_DIVERGENCE,
                priority=Priority.ATTENTION,
                summary=f"Plastic weight drift is high: {snap.weight_drift:.3f}",
                details={"drift": snap.weight_drift},
                run_id=snap.run_id,
            ))

    def _compare_controllers(self, run_id, results, report_fixed, report_plastic):
        """Compare fixed vs plastic performance."""
        if not report_fixed or not report_plastic:
            return

        ratio_diff = report_plastic.performance_ratio - report_fixed.performance_ratio

        if ratio_diff > self.improvement_threshold:
            priority = Priority.ATTENTION if ratio_diff > 0.2 else Priority.NOTABLE
            self.events.append(CuratorEvent(
                timestamp=time.time(),
                change_type=ChangeType.RECOVERY_DETECTED,
                priority=priority,
                summary=f"Plastic outperforms fixed by {ratio_diff:.3f} "
                        f"(plastic: {report_plastic.performance_ratio:.3f}, "
                        f"fixed: {report_fixed.performance_ratio:.3f})",
                details={
                    "plastic_ratio": report_plastic.performance_ratio,
                    "fixed_ratio": report_fixed.performance_ratio,
                    "difference": ratio_diff,
                },
                export_to_unity=ratio_diff > 0.2,
                run_id=run_id,
            ))
        elif ratio_diff < -self.degradation_threshold:
            self.events.append(CuratorEvent(
                timestamp=time.time(),
                change_type=ChangeType.PERFORMANCE_DEGRADATION,
                priority=Priority.ATTENTION,
                summary=f"Plastic underperforms fixed by {abs(ratio_diff):.3f} — "
                        f"plasticity may be hurting",
                details={
                    "plastic_ratio": report_plastic.performance_ratio,
                    "fixed_ratio": report_fixed.performance_ratio,
                },
                run_id=run_id,
            ))

    def _check_forgetting(self, run_id, results):
        """Check if plastic controller forgot how to walk on original terrain."""
        baseline = results.get("baseline", {}).get("plastic", {})
        forget = results.get("forgetting", {}).get("plastic", {})

        if not baseline or not forget:
            return

        dist_baseline = baseline.get("distance", 0)
        dist_forget = forget.get("distance", 0)

        if dist_baseline > 0:
            retention = dist_forget / dist_baseline
            if retention < (1.0 - self.forgetting_threshold):
                self.events.append(CuratorEvent(
                    timestamp=time.time(),
                    change_type=ChangeType.CATASTROPHIC_FORGETTING,
                    priority=Priority.BREAKTHROUGH,  # This is a key finding either way
                    summary=f"Catastrophic forgetting detected! Plastic retained only "
                            f"{retention*100:.1f}% of baseline performance after adaptation",
                    details={
                        "baseline_distance": dist_baseline,
                        "forget_distance": dist_forget,
                        "retention": retention,
                    },
                    export_to_unity=True,
                    run_id=run_id,
                ))
            elif retention > 0.9:
                self.events.append(CuratorEvent(
                    timestamp=time.time(),
                    change_type=ChangeType.RECOVERY_DETECTED,
                    priority=Priority.ATTENTION,
                    summary=f"No catastrophic forgetting: plastic retains "
                            f"{retention*100:.1f}% of baseline after adaptation",
                    details={"retention": retention},
                    run_id=run_id,
                ))

    def rank_runs(self, top_k: int = 5) -> list[dict]:
        """Rank all runs by composite score.

        Score = 0.4*perf_ratio + 0.2*symmetry + 0.2*(1-drift_norm) + 0.2*recovery_bonus
        """
        if not self.history:
            return []

        scored = []
        max_drift = max((s.weight_drift for s in self.history), default=1.0) or 1.0

        for snap in self.history:
            drift_norm = snap.weight_drift / max_drift if max_drift > 0 else 0
            recovery_bonus = 1.0 if snap.recovery_time is not None else 0.0

            score = (
                0.4 * min(snap.performance_ratio, 2.0) / 2.0
                + 0.2 * snap.gait_symmetry_after
                + 0.2 * (1.0 - drift_norm)
                + 0.2 * recovery_bonus
            )

            scored.append({
                "run_id": snap.run_id,
                "controller": snap.controller_type,
                "score": round(score, 4),
                "performance_ratio": round(snap.performance_ratio, 4),
                "symmetry": round(snap.gait_symmetry_after, 4),
                "weight_drift": round(snap.weight_drift, 4),
                "recovery_time": snap.recovery_time,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_summary_cards(self) -> dict:
        """Generate summary cards: best, most_surprising, biggest_failure, most_demo_worthy.

        Returns dict with card names as keys, each containing run info and explanation.
        """
        cards = {}

        if not self.history:
            return cards

        # Best run: highest composite score
        ranked = self.rank_runs(top_k=1)
        if ranked:
            best = ranked[0]
            cards["best_run"] = {
                **best,
                "explanation": f"Highest composite score ({best['score']:.3f}). "
                               f"Perf ratio {best['performance_ratio']:.3f}, "
                               f"symmetry {best['symmetry']:.3f}.",
            }

        # Most surprising: largest positive delta vs mean of same controller type
        by_type = {}
        for snap in self.history:
            by_type.setdefault(snap.controller_type, []).append(snap)

        best_surprise = None
        best_surprise_delta = -999
        for ctrl_type, snaps in by_type.items():
            if len(snaps) < 2:
                continue
            mean_ratio = np.mean([s.performance_ratio for s in snaps])
            for snap in snaps:
                delta = snap.performance_ratio - mean_ratio
                if delta > best_surprise_delta:
                    best_surprise_delta = delta
                    best_surprise = snap

        if best_surprise and best_surprise_delta > 0:
            cards["most_surprising"] = {
                "run_id": best_surprise.run_id,
                "controller": best_surprise.controller_type,
                "performance_ratio": round(best_surprise.performance_ratio, 4),
                "delta_vs_mean": round(best_surprise_delta, 4),
                "explanation": f"Outperformed {best_surprise.controller_type} mean by "
                               f"{best_surprise_delta:.3f} in performance ratio.",
            }

        # Biggest failure: lowest perf_ratio or has CATASTROPHIC_FORGETTING event
        catastrophic_runs = {
            e.run_id for e in self.events
            if e.change_type == ChangeType.CATASTROPHIC_FORGETTING
        }
        worst = min(self.history, key=lambda s: s.performance_ratio)
        if catastrophic_runs:
            # Prefer catastrophic forgetting run
            for snap in self.history:
                if snap.run_id in catastrophic_runs:
                    worst = snap
                    break

        cards["biggest_failure"] = {
            "run_id": worst.run_id,
            "controller": worst.controller_type,
            "performance_ratio": round(worst.performance_ratio, 4),
            "is_catastrophic": worst.run_id in catastrophic_runs,
            "explanation": f"Lowest performance ({worst.performance_ratio:.3f}). "
                          + ("Catastrophic forgetting detected." if worst.run_id in catastrophic_runs
                             else "May need parameter tuning."),
        }

        # Most demo-worthy: unity-tagged + highest performance
        unity_run_ids = {e.run_id for e in self.events if e.export_to_unity}
        demo_candidates = [s for s in self.history if s.run_id in unity_run_ids]
        if demo_candidates:
            demo_best = max(demo_candidates, key=lambda s: s.performance_ratio)
            cards["most_demo_worthy"] = {
                "run_id": demo_best.run_id,
                "controller": demo_best.controller_type,
                "performance_ratio": round(demo_best.performance_ratio, 4),
                "explanation": f"Unity-tagged with perf ratio {demo_best.performance_ratio:.3f}. "
                               f"Good candidate for visualization.",
            }
        else:
            # Fall back to best overall
            if ranked:
                cards["most_demo_worthy"] = {
                    **ranked[0],
                    "explanation": "No Unity-tagged runs yet. Using best overall.",
                }

        return cards

    def to_summary_json(self) -> dict:
        """Produce a serializable summary dict for curator_summary.json."""
        return {
            "total_runs": len(self.history),
            "total_events": len(self.events),
            "best_performance": {k: round(v, 4) for k, v in self.best_performance.items()},
            "ranked_runs": self.rank_runs(top_k=10),
            "summary_cards": self.get_summary_cards(),
            "events": [e.to_dict() for e in self.events],
            "unity_candidates": [e.to_dict() for e in self.get_unity_exports()],
        }

    def get_attention_events(
        self, min_priority: Priority = Priority.NOTABLE
    ) -> list[CuratorEvent]:
        """Get events worth Neil's attention."""
        return [
            e for e in self.events
            if e.priority.value >= min_priority.value
        ]

    def get_unity_exports(self) -> list[CuratorEvent]:
        """Get events tagged for Unity visualization."""
        return [e for e in self.events if e.export_to_unity]

    def summarize(self) -> str:
        """Generate a human-readable summary of all findings."""
        if not self.events:
            return "No experiments ingested yet."

        lines = []
        lines.append("=" * 50)
        lines.append("CURATOR SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Total runs analyzed: {len(self.history)}")
        lines.append(f"Total events: {len(self.events)}")
        lines.append("")

        # Group by priority
        for priority in reversed(Priority):
            events = [e for e in self.events if e.priority == priority]
            if not events:
                continue
            lines.append(f"[{priority.name}] ({len(events)} events)")
            for e in events:
                lines.append(f"  - {e.summary}")
            lines.append("")

        # Best performances
        if self.best_performance:
            lines.append("Best performance ratios:")
            for ctrl, ratio in self.best_performance.items():
                lines.append(f"  {ctrl}: {ratio:.4f}")

        # Unity export count
        unity = self.get_unity_exports()
        if unity:
            lines.append(f"\nEpisodes tagged for Unity export: {len(unity)}")
            for e in unity:
                lines.append(f"  - [{e.run_id}] {e.summary[:60]}...")

        return "\n".join(lines)

    def _save_events(self):
        """Persist events to disk."""
        events_file = self.log_dir / "events.json"
        data = [e.to_dict() for e in self.events]
        with open(events_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_events(self):
        """Load events from disk."""
        events_file = self.log_dir / "events.json"
        if events_file.exists():
            with open(events_file) as f:
                data = json.load(f)
            # We don't reconstruct full CuratorEvent objects here,
            # just keep the raw data accessible
            return data
        return []
