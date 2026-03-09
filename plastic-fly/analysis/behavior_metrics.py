"""
Behavioral metrics for closed-loop brain-body bridge experiments.

Computed from trajectory data collected during a run:
  positions:      list of (3,) arrays — fly position every N steps
  orientations:   list of (3,) arrays — fly orientation every N steps
  contact_forces: list of (6,) arrays — per-leg contact magnitudes every N steps
  commands:       list of LocomotionCommand — one per brain step
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BehaviorReport:
    """Behavioral metrics from a single run."""
    # Locomotion
    forward_distance: float    # net displacement along initial forward axis (mm)
    total_path_length: float   # cumulative distance traveled (mm)
    straightness: float        # forward_distance / total_path_length (1.0 = perfectly straight)

    # Heading
    cumulative_turn: float     # total absolute heading change (radians)
    mean_heading_rate: float   # mean heading change per sample (rad/sample)
    final_heading: float       # final heading relative to start (radians)

    # Stability
    orientation_variance: float  # variance of body pitch/roll (lower = more stable)
    fall_count: int              # number of samples where all legs lose contact
    completion_rate: float       # fraction of intended steps completed

    # Contact
    contact_symmetry: float    # left/right contact force ratio (1.0 = symmetric)
    mean_contact_force: float  # mean per-leg contact force
    contact_duty_cycle: float  # fraction of samples with >= 3 legs in contact

    # Step timing
    step_frequency_hz: float   # estimated step frequency from contact oscillations

    def to_dict(self) -> dict:
        return {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                for k, v in self.__dict__.items()}

    def summary_line(self) -> str:
        return (
            "fwd=%.2fmm path=%.2fmm straight=%.2f "
            "turn=%.1fdeg falls=%d contact_sym=%.2f freq=%.1fHz"
            % (self.forward_distance, self.total_path_length, self.straightness,
               np.degrees(self.cumulative_turn), self.fall_count,
               self.contact_symmetry, self.step_frequency_hz)
        )


def compute_behavior(
    positions: list[np.ndarray],
    orientations: list[np.ndarray] | None = None,
    contact_forces: list[np.ndarray] | None = None,
    steps_completed: int = 0,
    steps_intended: int = 0,
    sample_dt: float = 0.005,  # time between position samples (seconds)
) -> BehaviorReport:
    """Compute behavioral metrics from trajectory data."""
    pos = np.array(positions)  # (N, 3)
    n = len(pos)

    # --- Locomotion ---
    if n >= 2:
        displacements = np.diff(pos, axis=0)
        segment_lengths = np.linalg.norm(displacements, axis=1)
        total_path_length = float(np.sum(segment_lengths))

        # Forward distance: displacement along the initial forward direction
        # In MuJoCo fly coords, X is forward after coordinate transform
        forward_distance = float(pos[-1, 0] - pos[0, 0])

        straightness = abs(forward_distance) / max(total_path_length, 1e-8)
    else:
        forward_distance = 0.0
        total_path_length = 0.0
        straightness = 0.0

    # --- Heading ---
    if n >= 3:
        # Estimate heading from successive position differences (XY plane)
        headings = np.arctan2(displacements[:, 1], displacements[:, 0])
        # Unwrap for cumulative measurement
        heading_changes = np.diff(headings)
        # Wrap to [-pi, pi]
        heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi
        cumulative_turn = float(np.sum(np.abs(heading_changes)))
        mean_heading_rate = float(np.mean(np.abs(heading_changes)))
        final_heading = float(headings[-1] - headings[0])
        final_heading = (final_heading + np.pi) % (2 * np.pi) - np.pi
    else:
        cumulative_turn = 0.0
        mean_heading_rate = 0.0
        final_heading = 0.0

    # --- Stability ---
    if orientations is not None and len(orientations) >= 2:
        ori = np.array(orientations)
        # Pitch = ori[:,1], Roll = ori[:,2] (assuming ori = [yaw, pitch, roll] or similar)
        orientation_variance = float(np.var(ori[:, 1:], axis=0).sum())
    else:
        orientation_variance = 0.0

    # --- Contact ---
    if contact_forces is not None and len(contact_forces) >= 2:
        cf = np.array(contact_forces)  # (N, 6)
        left_force = cf[:, :3].mean(axis=1)   # LF, LM, LH
        right_force = cf[:, 3:].mean(axis=1)  # RF, RM, RH
        # Symmetry: ratio of min/max mean force (1.0 = symmetric)
        mean_left = float(np.mean(left_force))
        mean_right = float(np.mean(right_force))
        contact_symmetry = min(mean_left, mean_right) / max(max(mean_left, mean_right), 1e-8)
        mean_contact_force = float(np.mean(cf))

        # Fall count: all 6 legs below threshold
        all_off = np.all(cf < 0.01, axis=1)
        fall_count = int(np.sum(all_off))

        # Contact duty cycle: >= 3 legs in contact
        legs_in_contact = np.sum(cf > 0.01, axis=1)
        contact_duty_cycle = float(np.mean(legs_in_contact >= 3))

        # Step frequency: from contact force oscillations of one leg
        if len(cf) >= 10:
            # Use FFT on first leg's contact signal
            leg0 = cf[:, 0] - np.mean(cf[:, 0])
            if np.std(leg0) > 1e-6:
                fft = np.abs(np.fft.rfft(leg0))
                freqs = np.fft.rfftfreq(len(leg0), d=sample_dt)
                # Find peak frequency above 1 Hz
                fft_masked = fft.copy()
                fft_masked[freqs <= 1.0] = 0.0
                if np.max(fft_masked) > 0:
                    peak_idx = int(np.argmax(fft_masked))
                    step_frequency_hz = float(freqs[peak_idx])
                else:
                    step_frequency_hz = 0.0
            else:
                step_frequency_hz = 0.0
        else:
            step_frequency_hz = 0.0
    else:
        contact_symmetry = 1.0
        mean_contact_force = 0.0
        fall_count = 0
        contact_duty_cycle = 1.0
        step_frequency_hz = 0.0

    completion_rate = steps_completed / max(steps_intended, 1)

    return BehaviorReport(
        forward_distance=forward_distance,
        total_path_length=total_path_length,
        straightness=straightness,
        cumulative_turn=cumulative_turn,
        mean_heading_rate=mean_heading_rate,
        final_heading=final_heading,
        orientation_variance=orientation_variance,
        fall_count=fall_count,
        completion_rate=completion_rate,
        contact_symmetry=contact_symmetry,
        mean_contact_force=mean_contact_force,
        contact_duty_cycle=contact_duty_cycle,
        step_frequency_hz=step_frequency_hz,
    )
