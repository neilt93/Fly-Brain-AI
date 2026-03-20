"""Run all Eigenlayer demos and generate all figures."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(label, script):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(HERE),
    )
    return result.returncode == 0


def main():
    results = {}

    results["demo.py"] = run(
        "Abstract Bottleneck Demo (5-node)", HERE / "demo.py")

    results["connectome_demo.py"] = run(
        "Connectome-Grounded Demo (6-modality)", HERE / "connectome_demo.py")

    results["bridge_figure"] = run(
        "Bridge Figure (connectome -> eigenlayer)",
        HERE / "figures" / "gen_bridge_figure.py")

    print(f"\n{'=' * 60}")
    print("EIGENLAYER SUMMARY")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_ok = all(results.values())
    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILURES'}")
    print(f"\nFigures in: {HERE / 'figures'}")
    for f in sorted((HERE / "figures").glob("*.png")):
        print(f"  {f.name}")

    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
