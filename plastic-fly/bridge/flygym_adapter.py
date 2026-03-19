"""
FlyGym adapter: extracts BodyObservation from FlyGym's observation dict.

FlyGym obs format (flygym v1.2):
  obs["joints"]:         (3, 42)  — [angles, velocities, torques]
  obs["fly"]:            (4, 3)   — [position, velocity, orientation, angular_velocity]
  obs["contact_forces"]: (30, 3)  — per-sensor contact forces
  obs["end_effectors"]:  (6, 3)   — foot tip positions
"""

import numpy as np
from bridge.interfaces import BodyObservation


class FlyGymAdapter:
    """Extracts BodyObservation from FlyGym's obs dict."""

    def extract_body_observation(self, obs: dict) -> BodyObservation:
        joints = np.asarray(obs.get("joints", np.zeros((3, 42))))
        fly = np.asarray(obs.get("fly", np.zeros((4, 3))))
        cf_raw = np.asarray(obs.get("contact_forces", np.zeros((30, 3))))

        # Joints: (3, 42) → angles [0], velocities [1]
        joint_angles = joints[0] if joints.ndim == 2 else np.zeros(42)
        joint_velocities = (
            joints[1] if joints.ndim == 2 and joints.shape[0] >= 2
            else np.zeros(42)
        )

        # Contact forces: (30, 3) → per-leg max magnitude (6,), normalized
        magnitudes = np.linalg.norm(cf_raw, axis=1) if cf_raw.ndim == 2 else np.zeros(30)
        contact_forces = np.array([
            magnitudes[i * 5:(i + 1) * 5].max() for i in range(6)
        ])
        contact_forces = np.clip(contact_forces / 10.0, 0.0, 1.0)

        # Fly state: (4, 3) → velocity [1], orientation [2]
        body_position = fly[0] if fly.ndim == 2 and fly.shape[0] >= 1 else np.zeros(3)
        body_velocity = fly[1] if fly.ndim == 2 and fly.shape[0] >= 2 else np.zeros(3)
        body_orientation = fly[2] if fly.ndim == 2 and fly.shape[0] >= 3 else np.zeros(3)

        # Optional: vision and olfaction
        vision = None
        odor_intensity = None
        if "vision" in obs:
            vision = np.asarray(obs["vision"], dtype=np.float32)
        if "odor_intensity" in obs:
            odor_intensity = np.asarray(obs["odor_intensity"], dtype=np.float32)

        return BodyObservation(
            joint_angles=joint_angles.astype(np.float32),
            joint_velocities=joint_velocities.astype(np.float32),
            contact_forces=contact_forces.astype(np.float32),
            body_velocity=body_velocity.astype(np.float32),
            body_orientation=body_orientation.astype(np.float32),
            vision=vision,
            odor_intensity=odor_intensity,
            body_position=body_position.astype(np.float32),
        )
