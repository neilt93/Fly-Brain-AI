"""
Structured logging for experiment runs and events.

Writes JSONL (one JSON object per line) for append-friendly, human-readable logs.
Files:
  logs/runs.jsonl           — one entry per controller per experiment
  logs/events.jsonl         — terrain transitions, stumbles, recoveries
  logs/dashboard_state.json — current run status (updated during runs)
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RunRecord:
    """One row in runs.jsonl — a completed controller episode."""
    run_id: str
    timestamp: float
    config: dict
    controller: str  # "fixed" or "plastic"
    metrics: dict    # flat dict of all computed metrics
    phases: dict     # per-phase metrics (baseline, shift, forgetting)
    plots_dir: str
    duration_s: float
    status: str = "completed"  # "completed", "failed", "terminated"


@dataclass
class EventRecord:
    """One row in events.jsonl — a discrete event during an experiment."""
    run_id: str
    timestamp: float
    step: int
    event_type: str   # "terrain_transition", "stumble", "recovery", etc.
    details: dict = field(default_factory=dict)


def append_run(log_path: str, record: RunRecord):
    """Append a RunRecord as one JSON line to runs.jsonl."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_event(log_path: str, event: EventRecord):
    """Append an EventRecord as one JSON line to events.jsonl."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(event), default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_runs(log_path: str) -> list[dict]:
    """Read all run records from a JSONL file."""
    path = Path(log_path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_events(log_path: str) -> list[dict]:
    """Read all event records from a JSONL file."""
    return read_runs(log_path)  # Same format


def write_state(path: str, state: dict):
    """Atomically write dashboard_state.json (write tmp, rename).

    On Windows, os.replace handles atomic overwrite.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(p) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp_path, str(p))


def read_state(path: str) -> Optional[dict]:
    """Read dashboard_state.json if it exists."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
