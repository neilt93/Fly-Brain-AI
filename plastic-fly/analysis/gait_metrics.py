"""
Gait verification metrics — prove the fly is walking, not dragging.

All functions are pure numpy (no I/O). Operates on raw observation arrays
from flygym: contact_forces, end_effectors, fly_orientation.

Leg order convention: [LF, LM, LH, RF, RM, RH]
Tripod groups: {LF, RM, LH} vs {RF, LM, RH}
"""

import numpy as np
from typing import Optional


def classify_stance_swing(
    contact_forces: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Classify each leg as stance (1) or swing (0) at each timestep.

    Args:
        contact_forces: (N, 6) per-leg contact force magnitudes
        threshold: force above this = stance (ground contact)

    Returns:
        (N, 6) binary array: 1 = stance, 0 = swing
    """
    return (contact_forces > threshold).astype(int)


def compute_stance_swing_durations(
    stance_swing: np.ndarray,
    dt: float,
) -> dict:
    """Compute mean stance and swing durations per leg.

    Args:
        stance_swing: (N, 6) binary stance/swing
        dt: time between samples (s)

    Returns:
        dict with 'stance_ms' and 'swing_ms', each (6,) arrays
    """
    n_steps, n_legs = stance_swing.shape
    stance_ms = np.zeros(n_legs)
    swing_ms = np.zeros(n_legs)

    for leg in range(n_legs):
        signal = stance_swing[:, leg]
        # Find run lengths
        stance_runs = []
        swing_runs = []
        current_val = signal[0]
        run_len = 1

        for i in range(1, n_steps):
            if signal[i] == current_val:
                run_len += 1
            else:
                if current_val == 1:
                    stance_runs.append(run_len)
                else:
                    swing_runs.append(run_len)
                current_val = signal[i]
                run_len = 1
        # Final run
        if current_val == 1:
            stance_runs.append(run_len)
        else:
            swing_runs.append(run_len)

        stance_ms[leg] = np.mean(stance_runs) * dt * 1000 if stance_runs else 0
        swing_ms[leg] = np.mean(swing_runs) * dt * 1000 if swing_runs else 0

    return {"stance_ms": stance_ms, "swing_ms": swing_ms}


def compute_tripod_score(stance_swing: np.ndarray) -> np.ndarray:
    """Compute tripod coordination score over time.

    Tripod gait: {LF, RM, LH} alternate with {RF, LM, RH}.
    Score = fraction of legs in the correct phase at each timestep.

    A score of 1.0 means perfect tripod: one group all in stance
    while the other is all in swing. 0.5 = random. 0.0 = anti-tripod.

    Args:
        stance_swing: (N, 6) binary stance/swing

    Returns:
        (N,) coordination score in [0, 1]
    """
    # Tripod group A: LF(0), RM(4), LH(2)
    # Tripod group B: RF(3), LM(1), RH(5)
    group_a = stance_swing[:, [0, 4, 2]]  # (N, 3)
    group_b = stance_swing[:, [3, 1, 5]]  # (N, 3)

    # For each timestep, check if groups are in anti-phase
    # Perfect tripod: all of A in stance and all of B in swing, or vice versa
    a_mean = group_a.mean(axis=1)  # fraction of group A in stance
    b_mean = group_b.mean(axis=1)  # fraction of group B in stance

    # Score = how anti-phase the two groups are
    # |a_mean - b_mean| = 1.0 for perfect tripod, 0.0 for in-phase
    score = np.abs(a_mean - b_mean)

    return score


def compute_step_frequency(
    stance_swing: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute stepping frequency per leg via power spectral density.

    Args:
        stance_swing: (N, 6) binary stance/swing
        dt: time between samples (s)

    Returns:
        (6,) Hz per leg — dominant stepping frequency
    """
    from scipy.signal import welch

    n_steps, n_legs = stance_swing.shape
    freqs_out = np.zeros(n_legs)

    for leg in range(n_legs):
        signal = stance_swing[:, leg].astype(float)
        signal -= signal.mean()

        if signal.std() < 1e-8 or n_steps < 16:
            continue

        nperseg = min(n_steps, 256)
        f, psd = welch(signal, fs=1.0 / dt, nperseg=nperseg)
        # Find peak frequency (skip DC at f[0])
        if len(psd) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
            freqs_out[leg] = f[peak_idx]

    return freqs_out


def compute_stride_symmetry(
    stance_swing: np.ndarray,
    dt: float,
) -> float:
    """Compare left vs right stride durations.

    Returns ratio close to 1.0 for symmetric gait, deviates for asymmetric.
    """
    durations = compute_stance_swing_durations(stance_swing, dt)
    # Full stride = stance + swing
    left_stride = durations["stance_ms"][:3] + durations["swing_ms"][:3]  # LF, LM, LH
    right_stride = durations["stance_ms"][3:] + durations["swing_ms"][3:]  # RF, RM, RH

    left_mean = left_stride.mean()
    right_mean = right_stride.mean()

    if max(left_mean, right_mean) < 1e-6:
        return 0.0

    return float(min(left_mean, right_mean) / max(left_mean, right_mean))


def detect_leg_drag(
    end_effectors: np.ndarray,
    contact_forces: np.ndarray,
    vel_threshold: float = 0.001,
) -> np.ndarray:
    """Detect dragging: leg in contact but not lifting (low vertical velocity).

    A dragging leg has ground contact but near-zero vertical movement,
    meaning it's sliding rather than stepping.

    Args:
        end_effectors: (N, 6, 3) leg tip positions
        contact_forces: (N, 6) per-leg contact force magnitudes
        vel_threshold: vertical velocity below this during contact = drag

    Returns:
        (6,) count of drag events per leg
    """
    n_steps, n_legs = contact_forces.shape
    drag_counts = np.zeros(n_legs, dtype=int)

    if n_steps < 2:
        return drag_counts

    # Vertical velocity of each leg tip
    z_vel = np.abs(np.diff(end_effectors[:, :, 2], axis=0))  # (N-1, 6)
    in_contact = contact_forces[1:] > 0.05  # (N-1, 6)

    for leg in range(n_legs):
        # Dragging = in contact AND not lifting
        dragging = in_contact[:, leg] & (z_vel[:, leg] < vel_threshold)
        # Count transitions into drag state
        drag_starts = np.diff(dragging.astype(int))
        drag_counts[leg] = int(np.sum(drag_starts == 1))

    return drag_counts


def detect_foot_slip(
    end_effectors: np.ndarray,
    stance_swing: np.ndarray,
    slip_threshold: float = 0.05,
) -> np.ndarray:
    """Detect foot slip: horizontal movement during stance phase.

    A foot should be stationary on the ground during stance.
    Horizontal sliding indicates loss of traction.

    Args:
        end_effectors: (N, 6, 3) leg tip positions
        stance_swing: (N, 6) binary stance/swing
        slip_threshold: horizontal displacement above this during stance = slip

    Returns:
        (6,) count of slip events per leg
    """
    n_steps, n_legs = stance_swing.shape
    slip_counts = np.zeros(n_legs, dtype=int)

    if n_steps < 2:
        return slip_counts

    # Horizontal displacement (xy plane)
    xy_disp = np.linalg.norm(
        np.diff(end_effectors[:, :, :2], axis=0), axis=2
    )  # (N-1, 6)

    in_stance = stance_swing[1:] == 1  # (N-1, 6)

    for leg in range(n_legs):
        slipping = in_stance[:, leg] & (xy_disp[:, leg] > slip_threshold)
        slip_starts = np.diff(slipping.astype(int))
        slip_counts[leg] = int(np.sum(slip_starts == 1))

    return slip_counts


def compute_body_orientation_variance(
    fly_orientation: np.ndarray,
) -> dict:
    """Compute variance of body orientation (roll, pitch, yaw).

    Low variance = stable body. High variance = wobbling/tumbling.

    Args:
        fly_orientation: (N, 3) roll, pitch, yaw in radians

    Returns:
        dict with 'roll_var', 'pitch_var', 'yaw_var'
    """
    if len(fly_orientation) < 2:
        return {"roll_var": 0.0, "pitch_var": 0.0, "yaw_var": 0.0}

    return {
        "roll_var": float(np.var(fly_orientation[:, 0])),
        "pitch_var": float(np.var(fly_orientation[:, 1])),
        "yaw_var": float(np.var(fly_orientation[:, 2])),
    }


def compute_upright_fraction(
    fly_orientation: np.ndarray,
    roll_thresh: float = 0.5,
    pitch_thresh: float = 0.5,
) -> float:
    """Fraction of time the fly is approximately upright.

    Args:
        fly_orientation: (N, 3) roll, pitch, yaw in radians
        roll_thresh: max absolute roll to count as upright
        pitch_thresh: max absolute pitch to count as upright

    Returns:
        fraction in [0, 1]
    """
    if len(fly_orientation) == 0:
        return 0.0

    roll_ok = np.abs(fly_orientation[:, 0]) < roll_thresh
    pitch_ok = np.abs(fly_orientation[:, 1]) < pitch_thresh
    upright = roll_ok & pitch_ok

    return float(upright.mean())


def build_contact_raster(stance_swing: np.ndarray) -> np.ndarray:
    """Transpose stance/swing for raster plot (legs on y-axis, time on x-axis).

    Args:
        stance_swing: (N, 6) binary stance/swing

    Returns:
        (6, N) array suitable for imshow
    """
    return stance_swing.T


def compute_gait_report(
    contact_forces: np.ndarray,
    end_effectors: np.ndarray,
    fly_orientation: np.ndarray,
    dt: float,
) -> dict:
    """Compute all gait metrics in one call.

    Args:
        contact_forces: (N, 6) per-leg contact force magnitudes
        end_effectors: (N, 6, 3) leg tip positions
        fly_orientation: (N, 3) roll, pitch, yaw
        dt: time between samples (s)

    Returns:
        flat dict of all gait metrics (JSON-serializable)
    """
    stance_swing = classify_stance_swing(contact_forces)

    # Stance/swing durations
    durations = compute_stance_swing_durations(stance_swing, dt)

    # Tripod score
    tripod_scores = compute_tripod_score(stance_swing)

    # Step frequency
    step_freq = compute_step_frequency(stance_swing, dt)

    # Stride symmetry
    stride_sym = compute_stride_symmetry(stance_swing, dt)

    # Drag detection
    drag_counts = detect_leg_drag(end_effectors, contact_forces)

    # Slip detection
    slip_counts = detect_foot_slip(end_effectors, stance_swing)

    # Body orientation
    orient_var = compute_body_orientation_variance(fly_orientation)
    upright = compute_upright_fraction(fly_orientation)

    # Duty cycle (fraction of time in stance per leg)
    duty_cycle = stance_swing.mean(axis=0)

    leg_names = ["LF", "LM", "LH", "RF", "RM", "RH"]

    report = {
        # Aggregate scores
        "tripod_score_mean": float(tripod_scores.mean()),
        "tripod_score_std": float(tripod_scores.std()),
        "stride_symmetry": stride_sym,
        "upright_fraction": upright,
        "total_drag_events": int(drag_counts.sum()),
        "total_slip_events": int(slip_counts.sum()),
        # Orientation stability
        **orient_var,
        # Per-leg details
        "step_frequency_hz": {
            leg: float(step_freq[i]) for i, leg in enumerate(leg_names)
        },
        "step_frequency_mean": float(step_freq.mean()),
        "stance_duration_ms": {
            leg: float(durations["stance_ms"][i])
            for i, leg in enumerate(leg_names)
        },
        "swing_duration_ms": {
            leg: float(durations["swing_ms"][i])
            for i, leg in enumerate(leg_names)
        },
        "duty_cycle": {
            leg: float(duty_cycle[i]) for i, leg in enumerate(leg_names)
        },
        "drag_events": {
            leg: int(drag_counts[i]) for i, leg in enumerate(leg_names)
        },
        "slip_events": {
            leg: int(slip_counts[i]) for i, leg in enumerate(leg_names)
        },
    }

    return report
