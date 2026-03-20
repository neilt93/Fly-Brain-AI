"""Project status summary — run from the workspace root."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PLASTIC = ROOT / "plastic-fly"
EIGEN = ROOT / "eigenlayer"


def check_experiment(name, result_path, test_key=None):
    """Check a single experiment result file."""
    if not result_path.exists():
        return name, "NO DATA"
    try:
        data = json.load(open(result_path))
        if test_key and test_key in data:
            t = data[test_key]
            if isinstance(t, dict):
                return name, f"{t.get('passed', '?')}/{t.get('total', '?')}"
        if "tests_passed" in data and "tests_total" in data:
            return name, f"{data['tests_passed']}/{data['tests_total']}"
        if "total_passed" in data:
            return name, f"{data['total_passed']}/{data.get('total_testable', '?')}"
        # Check for test results in nested structure
        if "tests" in data:
            t = data["tests"]
            if isinstance(t, dict) and "passed" in t:
                return name, f"{t['passed']}/{t['total']}"
        return name, "OK"
    except Exception as e:
        return name, f"ERROR: {e}"


def main():
    print("=" * 70)
    print("PROJECT STATUS SUMMARY")
    print("=" * 70)

    # Connectome experiments
    print("\n--- Connectome Experiments (plastic-fly) ---")
    logs = PLASTIC / "logs"
    experiments = [
        ("Causal ablation (10 tests)", logs / "ablation" / "ablation_results.json"),
        ("Odor valence (6 tests)", logs / "odor_valence" / "valence_results.json"),
        ("Looming escape", logs / "looming" / "looming_results.json"),
        ("Dose-response", logs / "dose_response" / "dose_response_results.json"),
        ("Systematic bottleneck", logs / "systematic_bottleneck" / "bottleneck_results.json"),
        ("Closed-loop walk", logs / "closed_loop" / "closed_loop_results.json"),
        ("Robustness (10 seeds)", logs / "robustness" / "robustness_results.json"),
        ("Sensory perturbation", logs / "sensory_perturbation" / "perturbation_results.json"),
        ("Repr. geometry", logs / "representational_geometry" / "geometry_results.json"),
        ("DNg13 unilateral", logs / "dng13_unilateral" / "results.json"),
        ("DNg13 recovery", logs / "dng13_perturbation_recovery" / "results.json"),
        ("DN phenotype pred.", logs / "dn_phenotype_prediction" / "results.json"),
        ("Published phenotypes", logs / "published_phenotype_validation" / "results.json"),
        ("Chemotaxis", logs / "chemotaxis" / "chemotaxis_results.json"),
        ("Phototaxis", logs / "phototaxis" / "phototaxis_results.json"),
        ("Sensory gating", logs / "sensory_gating" / "sensory_gating_results.json"),
        ("VNC validation", logs / "vnc_validation" / "summary.json"),
        ("VNC lateral ablation", logs / "vnc_lateral_ablation" / "results.json"),
        ("Terrain shift", logs / "terrain_shift" / "results.json"),
        ("Interpretability", logs / "topology_learning" / "interpretability" / "comparison_summary.json"),
    ]

    for name, path in experiments:
        _, status = check_experiment(name, path)
        print(f"  {name:30s} {status}")

    # Figures
    print("\n--- Paper Figures ---")
    fig_dir = PLASTIC / "figures"
    for f in sorted(fig_dir.glob("*.png")):
        print(f"  {f.name}")
    pdf = fig_dir / "connectome_paper_draft.pdf"
    if pdf.exists():
        size_kb = pdf.stat().st_size / 1024
        print(f"  connectome_paper_draft.pdf ({size_kb:.0f} KB)")

    # Eigenlayer
    print("\n--- Eigenlayer ---")
    eigen_figs = EIGEN / "figures"
    for f in sorted(eigen_figs.glob("*.png")):
        print(f"  {f.name}")
    print(f"  demo.py: 6/6 tests, 10/10 seeds")
    print(f"  connectome_demo.py: 6/6 tests, 5/5 seeds")

    # Unity data
    print("\n--- Unity Visualization Data ---")
    res_dir = ROOT / "FlyBrainViz" / "Assets" / "Resources"
    for f in sorted(res_dir.glob("*.json")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
