"""
Visualization for plastic fly experiments.

Generates:
- Recovery curves (velocity over time)
- Comparison bar charts
- Gait symmetry plots
- Weight drift visualization
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from .recovery_metrics import (
    compute_velocity,
    compute_smoothed_velocity,
    RecoveryReport,
)


def plot_recovery_curves(
    positions_fixed: np.ndarray,
    positions_plastic: np.ndarray,
    dt: float,
    perturbation_step: int,
    save_path: Optional[str] = None,
    smooth_window: int = 200,
    title: str = "Recovery After Terrain Shift",
):
    """Plot velocity over time for both controllers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Clamp smooth window to at most 1/4 of the data length
    n_min = min(len(positions_fixed), len(positions_plastic))
    effective_window = min(smooth_window, max(n_min // 4, 2))

    vel_fixed = compute_smoothed_velocity(positions_fixed, dt, effective_window)
    vel_plastic = compute_smoothed_velocity(positions_plastic, dt, effective_window)

    t_fixed = np.arange(len(vel_fixed)) * dt
    t_plastic = np.arange(len(vel_plastic)) * dt

    ax.plot(t_fixed, vel_fixed, label="Fixed", color="#2196F3", alpha=0.8)
    ax.plot(t_plastic, vel_plastic, label="Plastic", color="#FF5722", alpha=0.8)

    # Mark perturbation
    perturb_time = perturbation_step * dt
    ax.axvline(perturb_time, color="gray", linestyle="--", alpha=0.5,
               label="Terrain shift")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward velocity (m/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_comparison_bars(
    report_fixed: RecoveryReport,
    report_plastic: RecoveryReport,
    save_path: Optional[str] = None,
    title: str = "Fixed vs Plastic Controller",
):
    """Bar chart comparing key metrics."""
    metrics = [
        "Distance\n(after)",
        "Gait Symmetry\n(after)",
        "Performance\nRatio",
    ]
    fixed_vals = [
        report_fixed.distance_after,
        report_fixed.gait_symmetry_after,
        report_fixed.performance_ratio,
    ]
    plastic_vals = [
        report_plastic.distance_after,
        report_plastic.gait_symmetry_after,
        report_plastic.performance_ratio,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, fixed_vals, width, label="Fixed",
                   color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, plastic_vals, width, label="Plastic",
                   color="#FF5722", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            offset = 3 if h >= 0 else -12
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(0, offset), textcoords="offset points",
                       ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_weight_drift(
    drift_history: list[float],
    dt: float,
    save_path: Optional[str] = None,
    title: str = "Recurrent Weight Drift Over Time",
):
    """Plot how much plastic weights have changed from initial."""
    fig, ax = plt.subplots(figsize=(8, 4))

    t = np.arange(len(drift_history)) * dt
    ax.plot(t, drift_history, color="#FF5722", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight drift (L2 norm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_contact_raster(
    stance_swing: np.ndarray,
    dt: float,
    perturbation_step: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Contact Raster",
):
    """6-row heatmap showing stance (dark) / swing (light) per leg over time."""
    from .gait_metrics import build_contact_raster

    raster = build_contact_raster(stance_swing)
    fig, ax = plt.subplots(figsize=(12, 3))

    leg_labels = ["LF", "LM", "LH", "RF", "RM", "RH"]
    n_time = raster.shape[1]
    t_max = n_time * dt

    ax.imshow(
        raster, aspect="auto", cmap="binary", interpolation="nearest",
        extent=[0, t_max, 5.5, -0.5], origin="upper",
    )
    ax.set_yticks(range(6))
    ax.set_yticklabels(leg_labels)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    if perturbation_step is not None:
        perturb_time = perturbation_step * dt
        ax.axvline(perturb_time, color="red", linestyle="--", alpha=0.7,
                   label="Terrain shift")
        ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_tripod_score(
    scores: np.ndarray,
    dt: float,
    perturbation_step: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Tripod Coordination Score",
):
    """Line plot of tripod score over time."""
    fig, ax = plt.subplots(figsize=(10, 4))

    t = np.arange(len(scores)) * dt
    ax.plot(t, scores, color="#FF5722", alpha=0.6, linewidth=0.5)

    # Rolling mean for clarity
    if len(scores) > 50:
        window = min(50, len(scores) // 4)
        kernel = np.ones(window) / window
        smoothed = np.convolve(scores, kernel, mode="valid")
        t_smooth = np.arange(len(smoothed)) * dt
        ax.plot(t_smooth, smoothed, color="#FF5722", linewidth=2, label="Smoothed")

    if perturbation_step is not None:
        perturb_time = perturbation_step * dt
        ax.axvline(perturb_time, color="gray", linestyle="--", alpha=0.5,
                   label="Terrain shift")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tripod Score")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_gait_symmetry_over_time(
    contacts_fixed: np.ndarray,
    contacts_plastic: np.ndarray,
    window: int = 200,
    dt: float = 1e-4,
    perturbation_step: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Rolling gait symmetry for both controllers."""
    from .recovery_metrics import compute_gait_symmetry

    def rolling_symmetry(contacts, win):
        sym = []
        for i in range(win, len(contacts)):
            s = compute_gait_symmetry(contacts[i - win : i])
            sym.append(s)
        return np.array(sym)

    # Clamp window to at most 1/4 of data length
    n_min = min(len(contacts_fixed), len(contacts_plastic))
    effective_window = min(window, max(n_min // 4, 2))

    sym_f = rolling_symmetry(contacts_fixed, effective_window)
    sym_p = rolling_symmetry(contacts_plastic, effective_window)

    fig, ax = plt.subplots(figsize=(10, 4))
    t_f = np.arange(len(sym_f)) * dt
    t_p = np.arange(len(sym_p)) * dt

    ax.plot(t_f, sym_f, label="Fixed", color="#2196F3", alpha=0.8)
    ax.plot(t_p, sym_p, label="Plastic", color="#FF5722", alpha=0.8)

    if perturbation_step is not None:
        perturb_time = perturbation_step * dt
        ax.axvline(perturb_time, color="gray", linestyle="--", alpha=0.5,
                   label="Terrain shift")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gait symmetry")
    ax.set_title("Gait Symmetry Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
