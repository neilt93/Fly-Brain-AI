"""
Locomotion bridge: VNC-style layer that converts LocomotionCommand → FlyGym action.

This is the motor-control layer between the brain and the body. It takes
high-level commands (forward_drive, turn_drive, etc.) and generates actual
joint angles using FlyGym's CPG + PreprogrammedSteps.

The brain does NOT directly set joint torques. It modulates this layer.
"""

import numpy as np
from flygym.examples.locomotion import PreprogrammedSteps
from flygym.examples.locomotion.cpg_controller import CPGNetwork
from flygym.preprogrammed import get_cpg_biases

from bridge.interfaces import LocomotionCommand


LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
TRIPOD_PHASES = np.array([0, np.pi, 0, np.pi, 0, np.pi])


class LocomotionBridge:
    """VNC-style locomotion controller modulated by brain commands.

    Uses CPG + PreprogrammedSteps (same as PlasticController) but driven
    by LocomotionCommand instead of a recurrent network.
    """

    def __init__(
        self,
        timestep: float = 1e-4,
        cpg_freq: float = 12.0,
        cpg_amplitude: float = 1.0,
        seed: int = 42,
    ):
        self.timestep = timestep
        self.base_amplitude = cpg_amplitude
        self.base_freq = cpg_freq

        phase_biases = get_cpg_biases("tripod")
        coupling_weights = (phase_biases > 0).astype(float) * 10
        self.cpg = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=np.ones(6) * cpg_freq,
            intrinsic_amps=np.ones(6) * cpg_amplitude,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=np.ones(6) * 20,
            seed=seed,
        )
        self.steps = PreprogrammedSteps()

    def step(self, cmd: LocomotionCommand) -> dict:
        """Convert LocomotionCommand to FlyGym action dict.

        Returns {'joints': ndarray(42), 'adhesion': ndarray(6)}.
        """
        # Modulate CPG frequency before stepping
        freq = self.base_freq * cmd.step_frequency
        self.cpg.intrinsic_freqs = np.ones(6) * np.clip(freq, 2.0, 30.0)

        self.cpg.step()

        # Forward drive scales amplitude (floor at 0.6 — below this the CPG
        # oscillates but produces no net forward thrust due to physics)
        amp_scale = 0.6 + 0.4 * cmd.forward_drive

        # Turn: reduce amplitude on turning side
        left_scale = 1.0 - 0.3 * max(0.0, cmd.turn_drive)
        right_scale = 1.0 - 0.3 * max(0.0, -cmd.turn_drive)
        leg_scales = [left_scale, left_scale, left_scale,
                      right_scale, right_scale, right_scale]

        joint_angles = []
        adhesion = []
        for i, leg in enumerate(LEGS):
            mag = self.cpg.curr_magnitudes[i] * amp_scale * leg_scales[i] * cmd.stance_gain
            mag = np.clip(mag, 0.0, 1.5)

            angles = self.steps.get_joint_angles(
                leg, self.cpg.curr_phases[i], mag
            )
            joint_angles.append(angles)

            adhesion.append(self.steps.get_adhesion_onoff(
                leg, self.cpg.curr_phases[i]
            ))

        return {
            "joints": np.concatenate(joint_angles),
            "adhesion": np.array(adhesion, dtype=int),
        }

    def warmup(self, n_steps: int = 500):
        """Ramp CPG from zero with tripod phase pattern."""
        self.cpg.reset(init_phases=TRIPOD_PHASES, init_magnitudes=np.zeros(6))
        for _ in range(n_steps):
            self.cpg.step()

    def reset(self):
        self.cpg.reset(init_phases=TRIPOD_PHASES, init_magnitudes=np.zeros(6))
