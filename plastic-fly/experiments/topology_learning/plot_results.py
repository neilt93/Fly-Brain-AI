"""
Plot learning curves and generalization results.

Usage:
    python -m experiments.topology_learning.plot_results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curves(log_dir: Path):
    """Plot reward vs generation for all architectures."""
    curve_files = sorted(log_dir.glob("*_curve.json"))
    if not curve_files:
        print("No curve files found")
        return

    # Group by architecture
    arch_curves = {}
    for cf in curve_files:
        with open(cf) as f:
            data = json.load(f)
        arch = data["arch"]
        if arch not in arch_curves:
            arch_curves[arch] = []
        arch_curves[arch].append(data["curve"])

    # Color scheme
    colors = {
        "connectome": "#2ca02c",      # green
        "dense": "#1f77b4",           # blue
        "random_sparse": "#ff7f0e",   # orange
        "shuffled": "#d62728",        # red
    }
    labels = {
        "connectome": "Connectome (MANC)",
        "dense": "Fully Connected",
        "random_sparse": "Random Sparse",
        "shuffled": "Shuffled",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Learning curves (mean reward) ---
    for arch, curves in arch_curves.items():
        # Align generations
        min_len = min(len(c) for c in curves)
        aligned = np.array([[c[i]["mean_reward"] for i in range(min_len)] for c in curves])

        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0) if aligned.shape[0] > 1 else np.zeros_like(mean)
        gens = np.arange(min_len)

        color = colors.get(arch, "gray")
        label = labels.get(arch, arch)
        ax1.plot(gens, mean, color=color, label=label, linewidth=2)
        if aligned.shape[0] > 1:
            ax1.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)

    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Mean Episode Reward", fontsize=12)
    ax1.set_title("A. Learning Speed", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel B: Max reward ---
    for arch, curves in arch_curves.items():
        min_len = min(len(c) for c in curves)
        aligned = np.array([[c[i]["max_reward"] for i in range(min_len)] for c in curves])

        mean = aligned.mean(axis=0)
        gens = np.arange(min_len)

        color = colors.get(arch, "gray")
        label = labels.get(arch, arch)
        ax2.plot(gens, mean, color=color, label=label, linewidth=2)

    ax2.set_xlabel("Generation", fontsize=12)
    ax2.set_ylabel("Max Episode Reward", fontsize=12)
    ax2.set_title("B. Best Individual", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = log_dir / "learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_generalization(log_dir: Path):
    """Plot generalization bar chart."""
    gen_path = log_dir / "generalization.json"
    if not gen_path.exists():
        print("No generalization results found")
        return

    with open(gen_path) as f:
        results = json.load(f)

    # Group by architecture
    archs = {}
    for r in results:
        arch = r["arch"]
        if arch not in archs:
            archs[arch] = {"turn": [], "endurance": [], "distance": []}
        archs[arch]["turn"].append(abs(r["turning"]["mean_heading"]))
        archs[arch]["endurance"].append(r["endurance"]["mean_duration"])
        archs[arch]["distance"].append(r["endurance"]["mean_distance"])

    colors = {
        "connectome": "#2ca02c",
        "dense": "#1f77b4",
        "random_sparse": "#ff7f0e",
        "shuffled": "#d62728",
    }
    labels = {
        "connectome": "Connectome",
        "dense": "Dense",
        "random_sparse": "Random",
        "shuffled": "Shuffled",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    arch_names = [a for a in ["connectome", "dense", "random_sparse", "shuffled"] if a in archs]
    x = np.arange(len(arch_names))
    width = 0.6

    # Turn heading
    vals = [np.mean(archs[a]["turn"]) for a in arch_names]
    errs = [np.std(archs[a]["turn"]) for a in arch_names]
    bars = axes[0].bar(x, vals, width, yerr=errs, capsize=4,
                       color=[colors[a] for a in arch_names])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[a] for a in arch_names], rotation=15)
    axes[0].set_ylabel("Heading Change (°)")
    axes[0].set_title("Zero-Shot Turning", fontweight="bold")

    # Endurance
    vals = [np.mean(archs[a]["endurance"]) for a in arch_names]
    errs = [np.std(archs[a]["endurance"]) for a in arch_names]
    axes[1].bar(x, vals, width, yerr=errs, capsize=4,
                color=[colors[a] for a in arch_names])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[a] for a in arch_names], rotation=15)
    axes[1].set_ylabel("Steps Before Falling")
    axes[1].set_title("Endurance", fontweight="bold")

    # Distance
    vals = [np.mean(archs[a]["distance"]) for a in arch_names]
    errs = [np.std(archs[a]["distance"]) for a in arch_names]
    axes[2].bar(x, vals, width, yerr=errs, capsize=4,
                color=[colors[a] for a in arch_names])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([labels[a] for a in arch_names], rotation=15)
    axes[2].set_ylabel("Distance (mm)")
    axes[2].set_title("Distance Traveled", fontweight="bold")

    plt.tight_layout()
    out_path = log_dir / "generalization.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    log_dir = Path(__file__).resolve().parents[2] / "logs" / "topology_learning"
    plot_learning_curves(log_dir)
    plot_generalization(log_dir)


if __name__ == "__main__":
    main()
