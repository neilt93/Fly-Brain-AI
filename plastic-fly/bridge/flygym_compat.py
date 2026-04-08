"""
FlyGym v1→v2 compatibility shim.

Wraps FlyGym v2.0.0's composition-based API to present the Gymnasium-style
interface that the entire plastic-fly codebase (39 files) expects.

v1 API (what the codebase uses):
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(fly=fly, arena=arena, timestep=1e-4)
    obs, info = sim.reset()
    obs, reward, terminated, truncated, info = sim.step({"joints": ..., "adhesion": ...})

v2 API (what FlyGym 2.0.0 provides):
    fly = flygym.compose.Fly(name="nmf")
    fly.add_joints(skeleton); fly.add_actuators(dofs, POSITION); fly.add_leg_adhesion()
    world = FlatGroundWorld(); world.add_fly(fly, pos, rot)
    sim = flygym.Simulation(world); sim.reset()
    sim.set_actuator_inputs(name, POSITION, array); sim.step()  # no return
    angles = sim.get_joint_angles(name)  # separate getters

Usage (drop-in replacement):
    from bridge.flygym_compat import Fly, SingleFlySimulation, arena
    fly = Fly(enable_adhesion=True)
    sim = SingleFlySimulation(fly=fly, arena=arena.FlatTerrain(), timestep=1e-4)
    obs, info = sim.reset()
    obs, r, t, tr, info = sim.step({"joints": angles_42, "adhesion": onoff_6})
"""

import numpy as np

import flygym
import flygym.compose as _compose
import flygym.anatomy as _anat
from flygym.utils.math import Rotation3D

# Leg order (same in v1 and v2)
_LEG_ORDER = ["lf", "lm", "lh", "rf", "rm", "rh"]

# Tarsus5 body segment names (for end_effector positions)
_TARSUS5 = [f"{leg}_tarsus5" for leg in _LEG_ORDER]


class Fly:
    """v1-compatible Fly constructor wrapping v2 composition."""

    def __init__(
        self,
        enable_adhesion: bool = True,
        init_pose: str = "stretch",
        control: str = "position",
        draw_adhesion: bool = False,
        enable_olfaction: bool = False,
        enable_vision: bool = False,
        **kwargs,
    ):
        self.enable_adhesion = enable_adhesion
        self.init_pose = init_pose
        self.control = control
        self.draw_adhesion = draw_adhesion
        self.enable_olfaction = enable_olfaction
        self.enable_vision = enable_vision

        # Build v2 Fly
        self._v2_fly = _compose.Fly(name="nmf")
        skeleton = _anat.Skeleton(
            axis_order=_anat.AxisOrder.YAW_PITCH_ROLL,
            joint_preset=_anat.JointPreset.LEGS_ACTIVE_ONLY,
        )
        self._v2_fly.add_joints(skeleton)
        dofs = self._v2_fly.get_jointdofs_order()
        act_type = _compose.ActuatorType.POSITION
        self._v2_fly.add_actuators(dofs, act_type)
        if enable_adhesion:
            self._v2_fly.add_leg_adhesion()
        self._act_type = act_type
        self._n_dofs = len(dofs)

    @property
    def v2(self):
        return self._v2_fly


class _FlatTerrain:
    """v1-compatible FlatTerrain wrapping v2 FlatGroundWorld."""
    def __init__(self):
        self._v2_world = _compose.FlatGroundWorld()

    @property
    def v2(self):
        return self._v2_world


class arena:
    """Namespace for arena classes (v1-compatible)."""
    FlatTerrain = _FlatTerrain


class SingleFlySimulation:
    """v1-compatible simulation wrapping v2 Simulation.

    Presents the Gymnasium-style interface:
        obs, info = sim.reset()
        obs, reward, terminated, truncated, info = sim.step(action_dict)
    """

    def __init__(
        self,
        fly: Fly = None,
        arena=None,
        timestep: float = 1e-4,
    ):
        if fly is None:
            fly = Fly()
        self._fly = fly
        self._name = "nmf"

        # Build world
        if arena is not None and hasattr(arena, 'v2'):
            self._world = arena.v2
        else:
            self._world = _compose.FlatGroundWorld()

        # Spawn fly
        self._world.add_fly(
            fly.v2,
            spawn_position=np.array([0.0, 0.0, 0.6]),
            spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
        )

        # Create simulation
        self._sim = flygym.Simulation(self._world)
        self._timestep = timestep
        self._act_type = fly._act_type
        self._n_dofs = fly._n_dofs

        # Cache body segment indices
        bodysegs = fly.v2.get_bodysegs_order()
        self._thorax_idx = 0  # c_thorax is always first
        self._tarsus5_indices = []
        for tname in _TARSUS5:
            for i, seg in enumerate(bodysegs):
                if seg.name == tname:
                    self._tarsus5_indices.append(i)
                    break
        self._prev_pos = None
        self._step_count = 0

    def reset(self):
        self._sim.reset()
        self._step_count = 0
        obs = self._build_obs()
        self._prev_pos = obs["fly"][0].copy()
        return obs, {}

    def step(self, action: dict):
        joints = action.get("joints", np.zeros(self._n_dofs))
        adhesion = action.get("adhesion", None)

        self._sim.set_actuator_inputs(self._name, self._act_type, joints)
        if adhesion is not None and self._fly.enable_adhesion:
            self._sim.set_leg_adhesion_states(
                self._name, np.asarray(adhesion, dtype=bool))

        self._sim.step()
        self._step_count += 1

        obs = self._build_obs()
        return obs, 0.0, False, False, {}

    def close(self):
        pass  # v2 has no explicit close

    def _build_obs(self) -> dict:
        """Reconstruct v1-format observation dict from v2 getters."""
        s = self._sim
        name = self._name

        # Joints: v1 shape (3, 42) = [angles, velocities, torques]
        angles = s.get_joint_angles(name)     # (42,)
        vels = s.get_joint_velocities(name)   # (42,)
        # v2 get_actuator_forces needs actuator_type
        try:
            forces = s.get_actuator_forces(name, self._act_type)  # (42,)
        except (TypeError, AttributeError):
            forces = np.zeros(self._n_dofs)
        joints = np.stack([angles, vels, forces])  # (3, 42)

        # Fly state: v1 shape (4, 3) = [position, velocity, orientation, angular_vel]
        all_pos = s.get_body_positions(name)     # (69, 3)
        all_rot = s.get_body_rotations(name)     # (69, 4) quaternions

        pos = all_pos[self._thorax_idx]          # (3,)

        # Velocity: finite difference
        if self._prev_pos is not None:
            vel = (pos - self._prev_pos) / max(self._timestep, 1e-8)
        else:
            vel = np.zeros(3)
        self._prev_pos = pos.copy()

        # Orientation: quaternion → euler (roll, pitch, yaw)
        quat = all_rot[self._thorax_idx]  # (4,) [w, x, y, z]
        orient = self._quat_to_euler(quat)

        fly_state = np.stack([pos, vel, orient, np.zeros(3)])  # (4, 3)

        # Contact forces: v2 returns tuple (6 elements)
        # [0]: (6,) force magnitudes or active flags
        # [1]: (6,3) forces
        contact_info = s.get_ground_contact_info(name)
        if isinstance(contact_info, tuple) and len(contact_info) >= 2:
            per_leg_forces = contact_info[1]  # (6, 3)
            # Expand to v1 format: (30, 3) = 5 sensors per leg
            contact_forces = np.repeat(per_leg_forces, 5, axis=0)  # (30, 3)
        else:
            contact_forces = np.zeros((30, 3))

        # End effectors: tarsus5 positions
        end_effectors = all_pos[self._tarsus5_indices]  # (6, 3)

        return {
            "joints": joints,
            "fly": fly_state,
            "contact_forces": contact_forces,
            "end_effectors": end_effectors,
        }

    @staticmethod
    def _quat_to_euler(q):
        """Convert quaternion [w, x, y, z] to euler [roll, pitch, yaw]."""
        w, x, y, z = q
        # Roll (x-axis rotation)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)
        return np.array([roll, pitch, yaw])
