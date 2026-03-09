"""
Metrics for evaluating locomotion recovery after perturbation.

All metrics operate on log data (lists of StepRecord or raw arrays).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class RecoveryReport:
    """Summary of recovery performance."""
    distance_before: float
    distance_after: float
    recovery_time_steps: Optional[int]
    num_falls: int
    gait_symmetry_before: float
    gait_symmetry_after: float
    performance_ratio: float  # after / before (higher = better recovery)
    weight_drift: float = 0.0


def compute_distance(positions: np.ndarray) -> float:
    """Total forward distance traveled (x-axis).

    Args:
        positions: (N, 3) array of fly positions over time
    """
    if len(positions) < 2:
        return 0.0
    return float(positions[-1, 0] - positions[0, 0])


def compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Instantaneous velocity over time.

    Args:
        positions: (N, 3) array of positions
        dt: timestep

    Returns:
        (N-1,) array of forward velocities
    """
    if len(positions) < 2:
        return np.array([0.0])
    dx = np.diff(positions[:, 0])
    return dx / dt


def compute_smoothed_velocity(
    positions: np.ndarray, dt: float, window: int = 100
) -> np.ndarray:
    """Smoothed velocity for cleaner recovery curves."""
    vel = compute_velocity(positions, dt)
    if len(vel) < window:
        return vel
    kernel = np.ones(window) / window
    return np.convolve(vel, kernel, mode="valid")


def detect_falls(
    positions: np.ndarray, z_threshold: float = 0.4
) -> np.ndarray:
    """Detect timesteps where fly height drops below threshold.

    Default threshold 0.4 is well below normal walking height (~0.9).
    Returns array of timestep indices where falls occurred.
    """
    if len(positions) == 0:
        return np.array([], dtype=int)
    z = positions[:, 2]
    falls = np.where(z < z_threshold)[0]
    return falls


def compute_gait_symmetry(
    contact_history: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Measure gait symmetry from contact sensor data.

    Compares left vs right leg contact patterns.
    Returns 1.0 for perfectly symmetric, 0.0 for completely asymmetric.

    Args:
        contact_history: (N, 6) binary contact array
                        [LF, LM, LH, RF, RM, RH]
        window: if provided, only use last `window` steps
    """
    if len(contact_history) == 0:
        return 0.0  # no data = unknown, not "perfect"

    if window is not None:
        contact_history = contact_history[-window:]

    # Split left (0,1,2) vs right (3,4,5)
    left = contact_history[:, :3]
    right = contact_history[:, 3:]

    # Compare duty cycles
    left_duty = left.mean(axis=0)
    right_duty = right.mean(axis=0)

    if left_duty.sum() + right_duty.sum() == 0:
        return 0.0  # no ground contact = crashed/airborne, not symmetric

    # Symmetry = 1 - normalized difference
    diff = np.abs(left_duty - right_duty)
    max_duty = np.maximum(left_duty, right_duty)
    max_duty = np.where(max_duty == 0, 1.0, max_duty)

    symmetry = 1.0 - (diff / max_duty).mean()
    return float(np.clip(symmetry, 0.0, 1.0))


def compute_step_consistency(
    contact_history: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Measure consistency of stepping pattern.

    Uses autocorrelation of contact signals to find periodicity.
    Higher = more consistent stepping.
    """
    if len(contact_history) < 20:
        return 0.0

    if window is not None:
        contact_history = contact_history[-window:]

    # Aggregate contact signal
    signal = contact_history.sum(axis=1).astype(float)
    signal -= signal.mean()

    if signal.std() == 0:
        return 0.0

    # Autocorrelation
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[n - 1 :]  # positive lags only
    autocorr /= autocorr[0]  # normalize

    # Find first peak after initial decay
    peaks = []
    for i in range(2, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            peaks.append((i, autocorr[i]))
            break

    if not peaks:
        return 0.5  # no clear periodicity

    return float(np.clip(peaks[0][1], 0.0, 1.0))


def recovery_time(
    velocities: np.ndarray,
    perturbation_step: int,
    baseline_velocity: float,
    threshold: float = 0.8,
    sustain_window: int = 10,
) -> Optional[int]:
    """Steps until velocity sustains above threshold fraction of baseline.

    Requires `sustain_window` consecutive samples above target to count
    as recovered, avoiding false positives from noise.

    Returns None if never recovers.
    """
    if baseline_velocity <= 0:
        return None

    target = baseline_velocity * threshold
    post_vel = velocities[perturbation_step:]

    if len(post_vel) < sustain_window:
        return None

    consecutive = 0
    for i, v in enumerate(post_vel):
        if v >= target:
            consecutive += 1
            if consecutive >= sustain_window:
                return i - sustain_window + 1
        else:
            consecutive = 0

    return None


def compute_recovery_report(
    positions_before: np.ndarray,
    positions_after: np.ndarray,
    contacts_before: np.ndarray,
    contacts_after: np.ndarray,
    dt: float,
    perturbation_step: int,
    weight_drift: float = 0.0,
) -> RecoveryReport:
    """Compute full recovery report."""
    dist_before = compute_distance(positions_before)
    dist_after = compute_distance(positions_after)

    vel_before = compute_velocity(positions_before, dt)
    vel_after = compute_velocity(positions_after, dt)
    baseline_vel = vel_before.mean() if len(vel_before) > 0 else 0.0

    all_velocities = np.concatenate([vel_before, vel_after])
    rec_time = recovery_time(
        all_velocities,
        perturbation_step=len(vel_before),
        baseline_velocity=baseline_vel,
    )

    falls = detect_falls(positions_after)

    sym_before = compute_gait_symmetry(contacts_before)
    sym_after = compute_gait_symmetry(contacts_after)

    if abs(dist_before) > 1e-6:
        perf_ratio = dist_after / abs(dist_before)
    else:
        perf_ratio = 0.0

    return RecoveryReport(
        distance_before=dist_before,
        distance_after=dist_after,
        recovery_time_steps=rec_time,
        num_falls=len(falls),
        gait_symmetry_before=sym_before,
        gait_symmetry_after=sym_after,
        performance_ratio=perf_ratio,
        weight_drift=weight_drift,
    )
