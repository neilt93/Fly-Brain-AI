"""
Shared data structures for the brain-body bridge.

Every module imports from here. No module imports from any other bridge module
except through these types.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BodyObservation:
    """Body state extracted from FlyGym observations."""
    joint_angles: np.ndarray       # (42,) radians
    joint_velocities: np.ndarray   # (42,) rad/s
    contact_forces: np.ndarray     # (6,) per-leg force magnitude, normalized
    body_velocity: np.ndarray      # (3,) mm/s
    body_orientation: np.ndarray   # (3,)
    # Optional sensory channels (None when not enabled)
    vision: np.ndarray | None = None        # (2, 721, 2) ommatidia per eye
    odor_intensity: np.ndarray | None = None  # (k, 4) odor at 4 sensors
    looming_intensity: np.ndarray | None = None  # (2,) left/right looming [0-1]
    body_position: np.ndarray | None = None  # (3,) mm, optional global position when available


@dataclass
class BrainInput:
    """Poisson rates to inject into brain sensory neurons."""
    neuron_ids: np.ndarray         # FlyWire IDs (int64)
    firing_rates_hz: np.ndarray    # Hz, one per neuron


@dataclass
class BrainOutput:
    """Firing rates read from brain descending neurons."""
    neuron_ids: np.ndarray         # FlyWire IDs (int64)
    firing_rates_hz: np.ndarray    # Hz, one per neuron


@dataclass
class LocomotionCommand:
    """High-level locomotion drives decoded from descending activity.
    These modulate the VNC-level locomotion layer, not raw joint torques."""
    forward_drive: float = 0.0     # [-1, 1]
    turn_drive: float = 0.0        # [-1, 1] positive = right
    step_frequency: float = 1.0    # multiplier
    stance_gain: float = 1.0       # multiplier
