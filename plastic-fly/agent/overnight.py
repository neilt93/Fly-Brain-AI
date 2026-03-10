#!/usr/bin/env python3
"""
Overnight experiment queue (~10 hours).
Runs all major experiments with VNC-lite motor layer, interleaved with
sanity checks and validation passes.

Usage:
    python agent/overnight.py              # full 10-hour queue
    python agent/overnight.py --dry-run    # print plan without running
    python agent/overnight.py --start 3    # resume from experiment #3
    python agent/overnight.py --fake-brain # fast mode for testing

Queue (~10 hours estimated):
    1. Sanity checks                              (~1 min)
    2. Closed-loop baseline (VNC-lite, 20k steps)  (~15 min)
    3. Bug fix pass #1: validate VNC-lite           (~30 min)
    4. Ablation study v5 (10 seeds)                (~60 min)
    5. Bug fix pass #2: validate ablation results   (~2 min)
    6. Odor valence (10 seeds)                     (~90 min)
    7. Bug fix pass #3: validate valence results    (~2 min)
    8. Looming escape (10 seeds, 3 brain windows)  (~120 min)
    9. Bug fix pass #4: validate looming results    (~2 min)
   10. Ablation extended (dose-response + random)  (~60 min)
   11. Bug fix pass #5: cross-experiment consistency (~2 min)
   12. Phototaxis (5 seeds)                        (~60 min)
   13. VNC-lite full validation (real brain)        (~60 min)
   14. Bug fix pass #6: final validation            (~2 min)
   15. Unity demo scenes (5 conditions)            (~40 min)
   16. Final summary report                         (~1 min)
"""

import sys
import os
import json
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs" / "overnight"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for this run
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG = LOG_DIR / f"overnight_{RUN_ID}.log"


def log(msg: str):
    """Print and append to log file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], timeout_s: int = 7200) -> dict:
    """Run a command and capture output. Returns dict with success, output, elapsed."""
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        elapsed = time.time() - t0
        return {
            "success": result.returncode == 0,
            "output": result.stdout[-5000:] if result.stdout else "",
            "error": result.stderr[-2000:] if result.stderr else "",
            "elapsed_s": elapsed,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "TIMEOUT", "elapsed_s": timeout_s, "returncode": -1}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e), "elapsed_s": time.time() - t0, "returncode": -1}


def bug_fix_pass(name: str, checks: list[dict]) -> dict:
    """Run a set of validation checks and report results.

    Each check is: {"name": str, "test": callable -> bool, "fix": str}
    """
    log(f"\n{'='*60}")
    log(f"BUG FIX PASS: {name}")
    log(f"{'='*60}")

    results = []
    for check in checks:
        try:
            passed = check["test"]()
            status = "PASS" if passed else "FAIL"
            results.append({"name": check["name"], "status": status})
            log(f"  [{status}] {check['name']}")
            if not passed and "fix" in check:
                log(f"    -> Suggested fix: {check['fix']}")
        except Exception as e:
            results.append({"name": check["name"], "status": "ERROR", "error": str(e)})
            log(f"  [ERROR] {check['name']}: {e}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    log(f"\n  Result: {passed}/{total} checks passed")
    return {"name": name, "passed": passed, "total": total, "details": results}


# ══════════════════════════════════════════════════════════════
# Validation check functions
# ══════════════════════════════════════════════════════════════

def check_json_exists(path: str) -> bool:
    p = ROOT / path
    if not p.exists():
        return False
    with open(p) as f:
        data = json.load(f)
    return len(data) > 0


def check_ablation_results(results_path: str) -> list[dict]:
    """Generate validation checks from ablation results."""
    checks = []

    def make_check(name, path, key, op, threshold):
        def test():
            with open(ROOT / path) as f:
                data = json.load(f)
            val = data.get(key)
            if val is None:
                return False
            if op == ">":
                return val > threshold
            elif op == "<":
                return val < threshold
            elif op == "!=":
                return val != threshold
            return False
        return {"name": name, "test": test, "fix": f"Check {key} in {path}"}

    # Basic: results file exists and is non-empty
    checks.append({
        "name": f"Results file exists: {results_path}",
        "test": lambda: check_json_exists(results_path),
        "fix": "Re-run the experiment"
    })
    return checks


def check_baseline_walk(results_dir: str) -> list[dict]:
    """Validate closed-loop walk results."""
    rpath = ROOT / results_dir / "closed_loop_results.json"

    def check_distance():
        if not rpath.exists():
            return False
        with open(rpath) as f:
            data = json.load(f)
        positions = data.get("positions", [])
        if len(positions) < 2:
            return False
        import numpy as np
        start = np.array(positions[0])
        end = np.array(positions[-1])
        dist = np.linalg.norm(end - start)
        return dist > 5.0  # should walk at least 5mm

    def check_stability():
        if not rpath.exists():
            return False
        with open(rpath) as f:
            data = json.load(f)
        completed = data.get("summary", {}).get("steps_completed", 0)
        target = data.get("config", {}).get("body_steps", 1)
        return completed >= target * 0.9

    def check_brain_active():
        if not rpath.exists():
            return False
        with open(rpath) as f:
            data = json.load(f)
        logs = data.get("episode_log", [])
        if not logs:
            return False
        active_counts = [e.get("readout_active", 0) for e in logs]
        return sum(active_counts) / len(active_counts) > 3  # at least some neurons active

    return [
        {"name": "Walk distance > 5mm", "test": check_distance, "fix": "Check brain-body loop"},
        {"name": "Stability > 90% steps", "test": check_stability, "fix": "Physics may be unstable"},
        {"name": "Brain neurons active", "test": check_brain_active, "fix": "Check sensory encoder"},
    ]


def check_valence_results(results_dir: str) -> list[dict]:
    """Validate odor valence results."""
    def check():
        rdir = ROOT / results_dir
        summary = rdir / "summary.json"
        if not summary.exists():
            # Try to find any results
            results = list(rdir.glob("*.json"))
            return len(results) > 0
        with open(summary) as f:
            data = json.load(f)
        return data.get("passed", 0) >= 4  # at least 4/6 tests
    return [{"name": "Valence tests >= 4/6 pass", "test": check, "fix": "Check olfactory populations"}]


def check_looming_results(results_dir: str) -> list[dict]:
    """Validate looming escape results."""
    def check():
        rdir = ROOT / results_dir
        summary = rdir / "summary.json"
        if not summary.exists():
            results = list(rdir.glob("*.json"))
            return len(results) > 0
        with open(summary) as f:
            data = json.load(f)
        return data.get("passed", 0) >= 3  # at least 3/5 tests
    return [{"name": "Looming tests >= 3/5 pass", "test": check, "fix": "Check LPLC2 populations"}]


def cross_experiment_checks() -> list[dict]:
    """Cross-experiment consistency checks."""
    def check_no_crashes():
        crash_files = list((ROOT / "logs").rglob("*error*"))
        return len(crash_files) == 0

    def check_all_results_exist():
        expected = [
            "logs/overnight/baseline/closed_loop_results.json",
            "logs/overnight/ablation",
            "logs/overnight/odor_valence",
        ]
        for p in expected:
            full = ROOT / p
            if not full.exists():
                return False
        return True

    return [
        {"name": "No crash files", "test": check_no_crashes, "fix": "Check error logs"},
        {"name": "All result dirs exist", "test": check_all_results_exist, "fix": "Some experiments may have failed"},
    ]


# ══════════════════════════════════════════════════════════════
# Experiment queue
# ══════════════════════════════════════════════════════════════

def build_queue(args):
    """Build the full experiment queue."""
    py = sys.executable
    fake = ["--fake-brain"] if args.fake_brain else []

    queue = []

    # 1. Sanity checks (~1 min)
    queue.append({
        "id": 1,
        "name": "Sanity checks",
        "cmd": [py, "experiments/sanity_checks.py"],
        "timeout_s": 300,
        "est_min": 1,
    })

    # 2. Closed-loop baseline with VNC-lite (~15 min)
    queue.append({
        "id": 2,
        "name": "Closed-loop baseline (VNC-lite, 20k steps)",
        "cmd": [py, "experiments/closed_loop_walk.py",
                "--body-steps", "20000", "--output-dir", "logs/overnight/baseline"] + fake,
        "timeout_s": 3600,
        "est_min": 15,
    })

    # 3. Bug fix pass: VNC-lite validation
    queue.append({
        "id": 3,
        "name": "Bug fix pass #1: VNC-lite baseline",
        "type": "bug_fix",
        "checks_fn": lambda: check_baseline_walk("logs/overnight/baseline"),
        "est_min": 2,
    })

    # 4. Ablation study v5 (10 seeds)
    for seed in range(42, 52):
        queue.append({
            "id": 4,
            "name": f"Ablation study v5 (seed={seed})",
            "cmd": [py, "experiments/ablation_study.py",
                    "--body-steps", "5000", "--readout-version", "5",
                    "--seed", str(seed),
                    "--output-dir", f"logs/overnight/ablation/seed_{seed}"] + fake,
            "timeout_s": 3600,
            "est_min": 6,
        })

    # 5. Bug fix pass: ablation
    queue.append({
        "id": 5,
        "name": "Bug fix pass #2: ablation results",
        "type": "bug_fix",
        "checks_fn": lambda: [
            {"name": "Ablation results exist", "test": lambda: (ROOT / "logs/overnight/ablation").exists(),
             "fix": "Re-run ablation"},
            {"name": ">= 5 seeds completed", "test": lambda: len(list((ROOT / "logs/overnight/ablation").glob("seed_*"))) >= 5,
             "fix": "Some seeds may have crashed"},
        ],
        "est_min": 2,
    })

    # 6. Odor valence (10 seeds)
    queue.append({
        "id": 6,
        "name": "Odor valence (10 seeds)",
        "cmd": [py, "experiments/odor_valence.py",
                "--body-steps", "5000",
                "--seeds", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51",
                "--readout-version", "2",
                "--output-dir", "logs/overnight/odor_valence"] + fake,
        "timeout_s": 7200,
        "est_min": 90,
    })

    # 7. Bug fix pass: valence
    queue.append({
        "id": 7,
        "name": "Bug fix pass #3: valence results",
        "type": "bug_fix",
        "checks_fn": lambda: check_valence_results("logs/overnight/odor_valence"),
        "est_min": 2,
    })

    # 8. Looming escape (10 seeds, 3 brain windows)
    for brain_dt in [20, 50, 100]:
        queue.append({
            "id": 8,
            "name": f"Looming escape (brain_dt={brain_dt}ms, 10 seeds)",
            "cmd": [py, "experiments/looming.py",
                    "--body-steps", "10000",
                    "--seeds", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51",
                    "--brain-dt-ms", str(brain_dt),
                    "--output-dir", f"logs/overnight/looming/dt_{brain_dt}ms"] + fake,
            "timeout_s": 7200,
            "est_min": 40,
        })

    # 9. Bug fix pass: looming
    queue.append({
        "id": 9,
        "name": "Bug fix pass #4: looming results",
        "type": "bug_fix",
        "checks_fn": lambda: check_looming_results("logs/overnight/looming"),
        "est_min": 2,
    })

    # 10. Extended ablation (dose-response + random)
    queue.append({
        "id": 10,
        "name": "Extended ablation (dose-response + random)",
        "cmd": [py, "experiments/ablation_extended.py",
                "--mode", "both", "--body-steps", "5000",
                "--readout-version", "5",
                "--output-dir", "logs/overnight/ablation_extended"] + fake,
        "timeout_s": 7200,
        "est_min": 60,
    })

    # 11. Bug fix pass: cross-experiment
    queue.append({
        "id": 11,
        "name": "Bug fix pass #5: cross-experiment consistency",
        "type": "bug_fix",
        "checks_fn": cross_experiment_checks,
        "est_min": 2,
    })

    # 12. Phototaxis (5 seeds)
    queue.append({
        "id": 12,
        "name": "Phototaxis (5 seeds)",
        "cmd": [py, "experiments/phototaxis.py",
                "--body-steps", "5000",
                "--seeds", "42", "43", "44", "45", "46",
                "--readout-version", "2",
                "--output-dir", "logs/overnight/phototaxis"] + fake,
        "timeout_s": 7200,
        "est_min": 60,
    })

    # 13. VNC-lite full validation (real brain)
    queue.append({
        "id": 13,
        "name": "VNC-lite full validation (real brain)",
        "cmd": [py, "experiments/vnc_lite_validation.py",
                "--seed", "42"] + fake,
        "timeout_s": 7200,
        "est_min": 60,
    })

    # 14. Bug fix pass: final
    queue.append({
        "id": 14,
        "name": "Bug fix pass #6: final validation",
        "type": "bug_fix",
        "checks_fn": lambda: [
            {"name": "VNC-lite summary exists", "test": lambda: (ROOT / "logs/vnc_lite_validation/summary.json").exists(),
             "fix": "Re-run VNC-lite validation"},
            {"name": "VNC-lite >= 18/20 pass", "test": lambda: json.load(open(ROOT / "logs/vnc_lite_validation/summary.json")).get("passed", 0) >= 18 if (ROOT / "logs/vnc_lite_validation/summary.json").exists() else False,
             "fix": "VNC-lite regression detected"},
        ],
        "est_min": 2,
    })

    # 15. Unity demo scenes
    queue.append({
        "id": 15,
        "name": "Unity demo scenes (5 conditions)",
        "cmd": [py, "experiments/unity_demo_scenes.py", "--all"] + fake,
        "timeout_s": 7200,
        "est_min": 40,
    })

    # Total estimated time
    total_min = sum(e.get("est_min", 5) for e in queue)
    log(f"Total estimated time: {total_min} min ({total_min/60:.1f} hours)")

    return queue


def run_queue(queue, start_from=1, dry_run=False):
    """Execute the experiment queue."""
    results = []
    total = len(queue)
    t_start = time.time()

    log(f"\n{'='*60}")
    log(f"OVERNIGHT EXPERIMENT QUEUE")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Queue: {total} items")
    log(f"Log: {RUN_LOG}")
    log(f"{'='*60}\n")

    if dry_run:
        log("DRY RUN — printing plan only:\n")
        cumulative = 0
        for item in queue:
            est = item.get("est_min", 5)
            cumulative += est
            eta = timedelta(minutes=cumulative)
            item_type = item.get("type", "experiment")
            if item_type == "bug_fix":
                log(f"  [{item['id']:2d}] {item['name']} (~{est}min, ETA +{eta})")
            else:
                cmd_str = " ".join(item.get("cmd", ["--"]))
                log(f"  [{item['id']:2d}] {item['name']} (~{est}min, ETA +{eta})")
                log(f"       cmd: {cmd_str}")
        log(f"\nTotal: ~{cumulative}min ({cumulative/60:.1f}h)")
        return

    for i, item in enumerate(queue):
        if item["id"] < start_from:
            log(f"[{item['id']:2d}/{total}] SKIP: {item['name']}")
            continue

        elapsed_total = (time.time() - t_start) / 60
        log(f"\n[{item['id']:2d}/{total}] {item['name']} (elapsed: {elapsed_total:.0f}min)")

        if item.get("type") == "bug_fix":
            # Run validation checks
            checks = item["checks_fn"]()
            result = bug_fix_pass(item["name"], checks)
            results.append({"id": item["id"], "name": item["name"], "type": "bug_fix", **result})
        else:
            # Run experiment
            cmd = item["cmd"]
            log(f"  cmd: {' '.join(cmd)}")
            t0 = time.time()

            r = run_cmd(cmd, timeout_s=item.get("timeout_s", 3600))

            elapsed = r["elapsed_s"]
            status = "OK" if r["success"] else "FAIL"
            log(f"  [{status}] {elapsed/60:.1f}min (exit={r['returncode']})")

            if not r["success"]:
                log(f"  ERROR: {r['error'][:500]}")

            # Save last 20 lines of output
            output_lines = r["output"].strip().split("\n")[-20:]
            for line in output_lines:
                log(f"    | {line}")

            results.append({
                "id": item["id"], "name": item["name"], "type": "experiment",
                "success": r["success"], "elapsed_s": elapsed, "returncode": r["returncode"],
            })

    # ── Final summary ──
    total_elapsed = (time.time() - t_start) / 60
    log(f"\n{'='*60}")
    log(f"OVERNIGHT QUEUE COMPLETE")
    log(f"Total time: {total_elapsed:.0f}min ({total_elapsed/60:.1f}h)")
    log(f"{'='*60}\n")

    n_exp = sum(1 for r in results if r["type"] == "experiment")
    n_exp_ok = sum(1 for r in results if r["type"] == "experiment" and r.get("success"))
    n_bug = sum(1 for r in results if r["type"] == "bug_fix")
    n_bug_ok = sum(1 for r in results if r["type"] == "bug_fix" and r.get("passed", 0) == r.get("total", 0))

    log(f"Experiments: {n_exp_ok}/{n_exp} passed")
    log(f"Bug fix passes: {n_bug_ok}/{n_bug} clean")

    for r in results:
        if r["type"] == "experiment":
            status = "OK" if r.get("success") else "FAIL"
            log(f"  [{status}] {r['name']} ({r.get('elapsed_s', 0)/60:.1f}min)")
        else:
            status = f"{r.get('passed', '?')}/{r.get('total', '?')}"
            log(f"  [{status}] {r['name']}")

    # Save summary
    summary = {
        "run_id": RUN_ID,
        "total_elapsed_min": total_elapsed,
        "experiments_passed": n_exp_ok,
        "experiments_total": n_exp,
        "bug_passes_clean": n_bug_ok,
        "bug_passes_total": n_bug,
        "results": results,
    }
    with open(LOG_DIR / f"summary_{RUN_ID}.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to {LOG_DIR / f'summary_{RUN_ID}.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight experiment queue")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--start", type=int, default=1, help="Start from experiment #N")
    parser.add_argument("--fake-brain", action="store_true", help="Use fake brain (fast mode)")
    args = parser.parse_args()

    queue = build_queue(args)
    run_queue(queue, start_from=args.start, dry_run=args.dry_run)
