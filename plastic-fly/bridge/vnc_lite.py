"""
VNC-lite: a structured premotor control layer with internal dynamics.

Replaces the instant decoder-to-actuator mapping with a bilateral state model
that has memory, inertia, competition, and body feedback stabilization.

Architecture:
    DN group rates --> premotor state (with dynamics) --> LocomotionCommand

The key principle: DN input affects state DERIVATIVES, not raw outputs.
    d(state)/dt = -state/tau + f(DN_input) + g(body_feedback)

This gives the motor system:
    - Temporal smoothing (no instantaneous jumps)
    - Persistence (motor commands outlast single brain steps)
    - Bilateral competition (left/right inhibition for turning)
    - Feedback stabilization (body state corrects motor errors)

Stages:
    1. Premotor modules: DN rates -> 5 functional motor modules
    2. Bilateral state model: internal dynamics with decay + coupling
    3. Body feedback: contact/velocity/stability signals modulate state
"""

import numpy as np
from dataclasses import dataclass, field
from bridge.interfaces import LocomotionCommand, BodyObservation


@dataclass
class VNCLiteConfig:
    """Tunable parameters for the VNC-lite motor layer."""
    # Time constants (seconds) — how fast each state decays toward zero
    tau_drive: float = 0.200        # Forward drive persistence (200ms)
    tau_turn: float = 0.150         # Turn state decay (150ms — faster for agility)
    tau_rhythm: float = 0.300       # Rhythm inertia (300ms — slow changes in cadence)
    tau_stance: float = 0.250       # Stance persistence (250ms)

    # Input gains: how strongly DN rates drive state derivatives
    # Tuned so that steady-state at typical rates (~20Hz) gives similar
    # LocomotionCommand values as the original decoder.
    alpha_drive: float = 0.10       # Forward DN rate -> drive state
    alpha_turn: float = 0.12        # Turn DN rate -> turn state
    alpha_rhythm: float = 0.08      # Rhythm DN rate -> rhythm state
    alpha_stance: float = 0.08      # Stance DN rate -> stance state

    # Bilateral coupling
    drive_coupling: float = 0.1     # Pulls L/R drive together (symmetric forward)
    turn_inhibition: float = 0.15   # Mutual inhibition between L/R turn states

    # Saturation limits (prevents runaway state values)
    max_drive: float = 2.0
    max_turn: float = 2.0
    max_rhythm: float = 2.0
    max_stance: float = 2.0

    # Body feedback gains (Stage 3)
    feedback_velocity: float = 0.3      # Velocity mismatch -> stance correction
    feedback_stability: float = 0.2     # Body instability -> rhythm damping
    feedback_contact: float = 0.15      # Contact asymmetry -> turn correction
    feedback_slip: float = 0.1          # Slip detection -> stance increase

    # Output mapping (state -> LocomotionCommand)
    rate_scale: float = 40.0        # Inherited from decoder for compatibility


@dataclass
class VNCLiteState:
    """Internal bilateral motor state."""
    drive_L: float = 0.0        # Left forward drive
    drive_R: float = 0.0        # Right forward drive
    turn_L: float = 0.0         # Left turning bias
    turn_R: float = 0.0         # Right turning bias
    rhythm: float = 0.0         # Global oscillation state
    stance: float = 0.0         # Stabilization state

    # Feedback memory (for computing derivatives)
    prev_velocity: float = 0.0      # Previous forward velocity
    prev_contact_L: float = 0.0     # Previous left contact
    prev_contact_R: float = 0.0     # Previous right contact

    def as_array(self) -> np.ndarray:
        return np.array([
            self.drive_L, self.drive_R,
            self.turn_L, self.turn_R,
            self.rhythm, self.stance,
        ])

    def as_dict(self) -> dict:
        return {
            "drive_L": self.drive_L, "drive_R": self.drive_R,
            "turn_L": self.turn_L, "turn_R": self.turn_R,
            "rhythm": self.rhythm, "stance": self.stance,
        }


class VNCLite:
    """Bilateral premotor control layer with dynamics and feedback.

    Usage:
        vnc = VNCLite()
        # In the main loop:
        group_rates = decoder.get_group_rates(brain_output)
        cmd = vnc.step(group_rates, dt_s=0.02, body_obs=body_obs)
        action = locomotion.step(cmd)
    """

    def __init__(self, config: VNCLiteConfig | None = None):
        self.cfg = config or VNCLiteConfig()
        self.state = VNCLiteState()

    def reset(self):
        """Reset all internal state to zero."""
        self.state = VNCLiteState()

    def step(
        self,
        group_rates: dict,
        dt_s: float = 0.020,
        body_obs: BodyObservation | None = None,
    ) -> LocomotionCommand:
        """Update premotor state and produce locomotion command.

        Args:
            group_rates: dict from DescendingDecoder.get_group_rates()
                keys: forward, turn_left, turn_right, rhythm, stance (Hz)
            dt_s: timestep in seconds (brain_dt_ms / 1000)
            body_obs: optional body feedback (Stage 3)

        Returns:
            LocomotionCommand with smoothed, dynamics-shaped drives.
        """
        s = self.state
        c = self.cfg

        # ── Stage 1: DN input -> state derivatives ────────────────────

        # Normalize DN rates through tanh (same scale as original decoder)
        fwd_input = np.tanh(group_rates["forward"] / c.rate_scale)
        left_input = np.tanh(group_rates["turn_left"] / c.rate_scale)
        right_input = np.tanh(group_rates["turn_right"] / c.rate_scale)
        rhythm_input = np.tanh(group_rates["rhythm"] / c.rate_scale)
        stance_input = np.tanh(group_rates["stance"] / c.rate_scale)

        # ── Stage 2: Bilateral state dynamics ─────────────────────────

        # Forward drive: symmetric bilateral with coupling
        d_drive_L = (-s.drive_L / c.tau_drive
                     + c.alpha_drive * fwd_input / dt_s
                     + c.drive_coupling * (s.drive_R - s.drive_L))
        d_drive_R = (-s.drive_R / c.tau_drive
                     + c.alpha_drive * fwd_input / dt_s
                     + c.drive_coupling * (s.drive_L - s.drive_R))

        # Turn: bilateral with mutual inhibition
        # Left DN input excites left turn, right DN input excites right turn
        # Cross-inhibition: active side suppresses opposite side
        d_turn_L = (-s.turn_L / c.tau_turn
                    + c.alpha_turn * left_input / dt_s
                    - c.turn_inhibition * s.turn_R)
        d_turn_R = (-s.turn_R / c.tau_turn
                    + c.alpha_turn * right_input / dt_s
                    - c.turn_inhibition * s.turn_L)

        # Rhythm: global state
        d_rhythm = (-s.rhythm / c.tau_rhythm
                    + c.alpha_rhythm * rhythm_input / dt_s)

        # Stance: global state
        d_stance = (-s.stance / c.tau_stance
                    + c.alpha_stance * stance_input / dt_s)

        # ── Stage 3: Body feedback ────────────────────────────────────

        if body_obs is not None:
            # Velocity mismatch feedback
            # If commanded forward but actual velocity is low -> increase stance
            actual_fwd_vel = float(body_obs.body_velocity[0]) if body_obs.body_velocity is not None else 0.0
            commanded_drive = (s.drive_L + s.drive_R) * 0.5
            vel_error = max(0.0, commanded_drive - actual_fwd_vel * 0.05)
            d_stance += c.feedback_velocity * vel_error

            # Stability feedback: body pitch/roll -> dampen rhythm, boost stance
            if body_obs.body_orientation is not None:
                pitch = abs(float(body_obs.body_orientation[1]))
                roll = abs(float(body_obs.body_orientation[0]))
                instability = np.clip(pitch + roll, 0.0, 1.0)
                d_rhythm -= c.feedback_stability * instability * s.rhythm
                d_stance += c.feedback_stability * instability * 0.5

            # Contact asymmetry feedback -> corrective turn signal
            if body_obs.contact_forces is not None:
                forces = np.clip(body_obs.contact_forces, 0.0, 1.0)
                left_contact = float(np.mean(forces[:3]))   # LF, LM, LH
                right_contact = float(np.mean(forces[3:]))  # RF, RM, RH
                contact_asymmetry = left_contact - right_contact
                # If left side has more contact, slight right turn correction
                d_turn_R += c.feedback_contact * max(0, contact_asymmetry)
                d_turn_L += c.feedback_contact * max(0, -contact_asymmetry)

                # Slip detection: if total contact is low, boost stance
                total_contact = left_contact + right_contact
                slip_signal = max(0.0, 0.5 - total_contact)
                d_stance += c.feedback_slip * slip_signal

                # Save for next step
                s.prev_contact_L = left_contact
                s.prev_contact_R = right_contact

            s.prev_velocity = actual_fwd_vel

        # ── Euler integration ─────────────────────────────────────────

        s.drive_L += dt_s * d_drive_L
        s.drive_R += dt_s * d_drive_R
        s.turn_L += dt_s * d_turn_L
        s.turn_R += dt_s * d_turn_R
        s.rhythm += dt_s * d_rhythm
        s.stance += dt_s * d_stance

        # Saturation (clip to prevent runaway)
        s.drive_L = float(np.clip(s.drive_L, 0.0, c.max_drive))
        s.drive_R = float(np.clip(s.drive_R, 0.0, c.max_drive))
        s.turn_L = float(np.clip(s.turn_L, 0.0, c.max_turn))
        s.turn_R = float(np.clip(s.turn_R, 0.0, c.max_turn))
        s.rhythm = float(np.clip(s.rhythm, 0.0, c.max_rhythm))
        s.stance = float(np.clip(s.stance, 0.0, c.max_stance))

        # ── Output mapping: state -> LocomotionCommand ────────────────

        # Forward drive: average of bilateral states
        mean_drive = (s.drive_L + s.drive_R) * 0.5
        forward_drive = float(np.clip(0.1 + 0.9 * np.tanh(mean_drive), 0.1, 1.0))

        # Turn drive: bilateral difference
        turn_raw = s.turn_L - s.turn_R
        turn_drive = float(np.clip(np.tanh(turn_raw), -1.0, 1.0))

        # Step frequency: rhythm state modulates base rate
        step_frequency = float(1.0 + 1.5 * np.tanh(s.rhythm))

        # Stance gain: stance state
        stance_gain = float(1.0 + 0.5 * np.tanh(s.stance))

        return LocomotionCommand(
            forward_drive=forward_drive,
            turn_drive=turn_drive,
            step_frequency=step_frequency,
            stance_gain=stance_gain,
        )

    def get_state_dict(self) -> dict:
        """Return current state for logging."""
        return self.state.as_dict()
