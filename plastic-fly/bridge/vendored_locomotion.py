"""
Vendored CPG locomotion classes from FlyGym v1.2.1 (Wang et al. 2024).

These classes were removed in FlyGym v2.0.0. They are pure-NumPy with no
dependency on the simulation API, so they work with any FlyGym version.

Source: flygym v1.2.1
  - PreprogrammedSteps: flygym.examples.locomotion.steps
  - CPGNetwork: flygym.examples.locomotion.cpg_controller
  - get_cpg_biases: flygym.preprogrammed

License: Apache 2.0 (same as flygym)
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline


def _get_data_path(package: str, file: str) -> Path:
    """Resolve package data path (handles Python 3.9+ API)."""
    import importlib.resources
    return importlib.resources.files(package) / file


def calculate_ddt(theta, r, w, phi, nu, R, alpha):
    """CPG ODE derivatives for phase and magnitude."""
    intrinsic_term = 2 * np.pi * nu
    phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_term = (r * w * np.sin(phase_diff - phi)).sum(axis=1)
    dtheta_dt = intrinsic_term + coupling_term
    dr_dt = alpha * (R - r)
    return dtheta_dt, dr_dt


class PreprogrammedSteps:
    """Preprogrammed steps by each leg extracted from experimental recordings.

    Attributes
    ----------
    legs : list[str]
        List of leg names (e.g. LF for left front leg).
    dofs_per_leg : list[str]
        List of names for degrees of freedom for each leg.
    duration : float
        Duration of the preprogrammed step (at 1x speed) in seconds.
    """

    legs = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
    dofs_per_leg = [
        "Coxa", "Coxa_roll", "Coxa_yaw", "Femur",
        "Femur_roll", "Tibia", "Tarsus1",
    ]

    def __init__(self, path=None, neutral_pose_phases=None):
        if neutral_pose_phases is None:
            neutral_pose_phases = (np.pi, np.pi, np.pi, np.pi, np.pi, np.pi)
        if path is None:
            # Look for local copy first, then flygym package data
            local = Path(__file__).resolve().parent.parent / "data" / "behavior" / "single_steps_untethered.pkl"
            if local.exists():
                path = local
            else:
                path = _get_data_path("flygym", "data") / "behavior" / "single_steps_untethered.pkl"
        with open(path, "rb") as f:
            single_steps_data = pickle.load(f)
        self._length = len(single_steps_data["joint_LFCoxa"])
        self._timestep = single_steps_data["meta"]["timestep"]
        self.duration = self._length * self._timestep

        phase_grid = np.linspace(0, 2 * np.pi, self._length)
        self._psi_funcs = {}
        for leg in self.legs:
            joint_angles = np.array(
                [single_steps_data[f"joint_{leg}{dof}"] for dof in self.dofs_per_leg]
            )
            self._psi_funcs[leg] = CubicSpline(
                phase_grid, joint_angles, axis=1, bc_type="periodic"
            )

        self.neutral_pos = {
            leg: self._psi_funcs[leg](theta_neutral)[:, np.newaxis]
            for leg, theta_neutral in zip(self.legs, neutral_pose_phases)
        }

        swing_stance_time_dict = single_steps_data["swing_stance_time"]
        self.swing_period = {}
        for leg in self.legs:
            my_swing_period = np.array([0, swing_stance_time_dict["stance"][leg]])
            my_swing_period /= self.duration
            my_swing_period *= 2 * np.pi
            self.swing_period[leg] = my_swing_period

    def get_joint_angles(self, leg, phase, magnitude=1):
        """Get joint angles for a given leg at a given phase."""
        if isinstance(phase, (float, int)) or (hasattr(phase, 'shape') and phase.shape == ()):
            phase = np.array([phase])
        offset = self._psi_funcs[leg](phase) - self.neutral_pos[leg]
        joint_angles = self.neutral_pos[leg] + magnitude * offset
        return joint_angles.squeeze()

    def get_adhesion_onoff(self, leg, phase):
        """Get whether adhesion is on for a given leg at a given phase."""
        swing_start, swing_end = self.swing_period[leg]
        return not (swing_start < phase % (2 * np.pi) < swing_end)

    @property
    def default_pose(self):
        """Default pose as a single (42,) array."""
        return np.concatenate([self.neutral_pos[leg] for leg in self.legs]).ravel()


class CPGNetwork:
    """CPG network of N coupled oscillators."""

    def __init__(self, timestep, intrinsic_freqs, intrinsic_amps,
                 coupling_weights, phase_biases, convergence_coefs,
                 init_phases=None, init_magnitudes=None, seed=0):
        self.timestep = timestep
        self.num_cpgs = intrinsic_freqs.size
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.coupling_weights = coupling_weights
        self.phase_biases = phase_biases
        self.convergence_coefs = convergence_coefs
        self.random_state = np.random.RandomState(seed)
        self.reset(init_phases, init_magnitudes)

    def step(self):
        """Integrate the ODEs using Euler's method."""
        dtheta_dt, dr_dt = calculate_ddt(
            theta=self.curr_phases, r=self.curr_magnitudes,
            w=self.coupling_weights, phi=self.phase_biases,
            nu=self.intrinsic_freqs, R=self.intrinsic_amps,
            alpha=self.convergence_coefs,
        )
        self.curr_phases += dtheta_dt * self.timestep
        self.curr_magnitudes += dr_dt * self.timestep

    def reset(self, init_phases=None, init_magnitudes=None):
        if init_phases is None:
            self.curr_phases = self.random_state.random(self.num_cpgs) * 2 * np.pi
        else:
            self.curr_phases = init_phases
        if init_magnitudes is None:
            self.curr_magnitudes = np.zeros(self.num_cpgs)
        else:
            self.curr_magnitudes = init_magnitudes


def get_cpg_biases(gait: str) -> np.ndarray:
    """Define CPG phase biases for different gaits.

    Returns (6, 6) array. Leg order: LF, LM, LH, RF, RM, RH.
    """
    if gait.lower() == "tripod":
        phase_biases = np.array([
            [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
        ], dtype=np.float64) * np.pi
    elif gait.lower() == "tetrapod":
        phase_biases = np.array([
            [0, 1, 2, 2, 0, 1], [2, 0, 1, 1, 2, 0],
            [1, 2, 0, 0, 1, 2], [1, 2, 0, 0, 1, 2],
            [0, 1, 2, 2, 0, 1], [2, 0, 1, 1, 2, 0],
        ], dtype=np.float64) * 2 * np.pi / 3
    elif gait.lower() == "wave":
        phase_biases = np.array([
            [0, 1, 2, 3, 4, 5], [5, 0, 1, 2, 3, 4],
            [4, 5, 0, 1, 2, 3], [3, 4, 5, 0, 1, 2],
            [2, 3, 4, 5, 0, 1], [1, 2, 3, 4, 5, 0],
        ], dtype=np.float64) * 2 * np.pi / 6
    else:
        raise ValueError(f"Unknown gait: {gait}")
    return phase_biases
