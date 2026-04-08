"""
Hardware abstraction layer for hexapod deployment (Paper 2).

Abstracts the body interface so the full pipeline can run on either:
  - FlyGym (MuJoCo simulation)
  - HexArth (physical hexapod robot)

The VNCBridge outputs {'joints': ndarray(42), 'adhesion': ndarray(6)}.
This module converts those into hardware commands and reads back
real sensor observations.

Architecture:
    Brain -> Decoder -> VNCBridge -> HexapodInterface -> {FlyGym | HexArth}
                                           |
                                    BodyObservation <- sensor feedback

Usage:
    # Simulation (current behavior)
    hexapod = FlyGymHexapod()

    # Real hardware
    hexapod = HexArthHexapod(port="/dev/ttyUSB0", servo_config="config/hexarth.json")

    # Generic loop
    for step in range(n_steps):
        action = vnc_bridge.step(group_rates, body_obs=hexapod.observe())
        hexapod.command(action)
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from bridge.interfaces import BodyObservation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HexapodConfig:
    """Shared config for all hexapod backends."""
    n_joints: int = 42
    n_legs: int = 6
    joints_per_leg: int = 7     # coxa, trochanter, femur, tibia, tarsus1-3
    control_freq_hz: float = 100.0   # body step frequency
    max_joint_velocity: float = 10.0  # rad/s safety limit
    watchdog_timeout_ms: float = 500.0  # kill motors if no command in this time


@dataclass
class ServoConfig:
    """Per-joint servo calibration for real hardware."""
    joint_name: str
    servo_id: int
    zero_angle_rad: float       # angle when servo is at center position
    rad_per_unit: float         # radians per servo unit (direction + scale)
    min_unit: int = 0           # servo min position
    max_unit: int = 4095        # servo max position (12-bit)
    max_torque: float = 1.5     # Nm, safety limit


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class HexapodInterface(ABC):
    """Abstract body interface for sim or real hexapod."""

    def __init__(self, config: Optional[HexapodConfig] = None):
        self.config = config or HexapodConfig()
        self._step_count = 0
        self._last_command_time = time.time()

    @abstractmethod
    def reset(self) -> BodyObservation:
        """Reset to standing pose, return initial observation."""

    @abstractmethod
    def command(self, action: dict) -> BodyObservation:
        """Send joint angles + adhesion to the body, return new observation.

        Args:
            action: {'joints': ndarray(42), 'adhesion': ndarray(6)}

        Returns:
            BodyObservation from sensor feedback
        """

    @abstractmethod
    def observe(self) -> BodyObservation:
        """Read current sensor state without sending a command."""

    @abstractmethod
    def close(self):
        """Release hardware resources."""

    def _enforce_velocity_limits(self, target_joints: np.ndarray,
                                  current_joints: np.ndarray,
                                  dt: float) -> np.ndarray:
        """Clamp joint velocity to safe limits."""
        delta = target_joints - current_joints
        max_delta = self.config.max_joint_velocity * dt
        return current_joints + np.clip(delta, -max_delta, max_delta)


# ---------------------------------------------------------------------------
# FlyGym backend (simulation)
# ---------------------------------------------------------------------------

class FlyGymHexapod(HexapodInterface):
    """FlyGym/MuJoCo simulation backend. Current behavior, wrapped."""

    def __init__(self, config=None, timestep=1e-4):
        super().__init__(config)
        self.timestep = timestep
        self.command_dt_s = 1.0 / self.config.control_freq_hz
        self.control_substeps = max(1, int(round(self.command_dt_s / self.timestep)))
        self.sim = None
        self.fly = None
        self._last_obs = None

    def _create_sim(self):
        from bridge.flygym_compat import Fly, SingleFlySimulation, arena
        self.fly = Fly(
            enable_adhesion=True, init_pose="stretch", control="position")
        flat = arena.FlatTerrain()
        self.sim = SingleFlySimulation(
            fly=self.fly, arena=flat, timestep=self.timestep)

    def reset(self) -> BodyObservation:
        if self.sim is None:
            self._create_sim()
        raw_obs, _ = self.sim.reset()
        self._step_count = 0
        self._last_obs = self._convert_obs(raw_obs)
        return self._last_obs

    def command(self, action: dict) -> BodyObservation:
        raw_obs = None
        for _ in range(self.control_substeps):
            raw_obs, _, term, trunc, _ = self.sim.step(action)
            if term or trunc:
                raise RuntimeError("FlyGym episode ended during hexapod command")
        self._step_count += 1
        self._last_command_time = time.time()
        self._last_obs = self._convert_obs(raw_obs)
        return self._last_obs

    def observe(self) -> BodyObservation:
        return self._last_obs

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def _convert_obs(self, obs: dict) -> BodyObservation:
        joints = np.array(obs["joints"])
        cf = np.array(obs["contact_forces"])
        magnitudes = np.linalg.norm(cf, axis=1) if cf.ndim == 2 else cf
        per_leg = np.array([
            np.clip(magnitudes[i*5:(i+1)*5].max() / 10.0, 0.0, 1.0)
            for i in range(6)
        ], dtype=np.float32)

        fly_state = np.array(obs["fly"], dtype=np.float32)
        pos = fly_state[0] if fly_state.ndim >= 2 and fly_state.shape[0] >= 1 else np.zeros(3, dtype=np.float32)
        vel = fly_state[1] if fly_state.ndim >= 2 and fly_state.shape[0] >= 2 else np.zeros(3, dtype=np.float32)
        orient = fly_state[2] if fly_state.ndim >= 2 and fly_state.shape[0] >= 3 else np.zeros(3, dtype=np.float32)

        return BodyObservation(
            joint_angles=joints[0].flatten().astype(np.float32) if joints.ndim >= 2 else joints.flatten().astype(np.float32),
            joint_velocities=joints[1].flatten().astype(np.float32) if joints.ndim >= 2 and joints.shape[0] >= 2 else np.zeros(42, dtype=np.float32),
            contact_forces=per_leg,
            body_velocity=vel.astype(np.float32),
            body_orientation=orient.astype(np.float32),
            body_position=pos.astype(np.float32),
        )


# ---------------------------------------------------------------------------
# HexArth backend (real hardware) — PLACEHOLDER
# ---------------------------------------------------------------------------

class HexArthHexapod(HexapodInterface):
    """Physical HexArth hexapod robot backend.

    Communication: serial/USB to servo controller board.
    Feedback: servo position + load + voltage per joint.

    TODO (Paper 2):
    - [ ] Serial protocol implementation (board-specific)
    - [ ] Servo calibration from JSON config
    - [ ] Real-time loop with timing guarantees
    - [ ] Contact detection from foot pressure sensors
    - [ ] IMU integration for body orientation
    - [ ] Watchdog: kill motors if control loop stalls
    - [ ] Graceful degradation: fallback to neutral pose on error
    """

    def __init__(self, port: str = "/dev/ttyUSB0",
                 servo_config_path: str = None,
                 config: Optional[HexapodConfig] = None):
        super().__init__(config)
        self.port = port
        self.servo_config_path = servo_config_path
        self.servo_configs: list[ServoConfig] = []
        self._current_joints = np.zeros(42, dtype=np.float32)
        self._connected = False

        # Load servo calibration
        if servo_config_path:
            self._load_servo_config(servo_config_path)

    def _load_servo_config(self, path: str):
        """Load per-joint servo calibration from JSON."""
        import json
        with open(path) as f:
            raw = json.load(f)
        self.servo_configs = [
            ServoConfig(**entry) for entry in raw["servos"]
        ]

    def connect(self):
        """Open serial connection to servo controller.

        Supports Dynamixel Protocol 2.0 (XM/XL series) via the
        dynamixel_sdk package. Falls back to raw serial if not available.
        """
        try:
            import dynamixel_sdk as dxl
            self._port_handler = dxl.PortHandler(self.port)
            self._packet_handler = dxl.PacketHandler(2.0)  # Protocol 2.0
            if not self._port_handler.openPort():
                raise ConnectionError(f"Cannot open port {self.port}")
            if not self._port_handler.setBaudRate(1_000_000):
                raise ConnectionError("Cannot set baud rate to 1M")
            self._use_dxl = True
            # Ping all configured servos
            n_found = 0
            for sc in self.servo_configs:
                model, result, error = self._packet_handler.ping(
                    self._port_handler, sc.servo_id)
                if result == dxl.COMM_SUCCESS:
                    n_found += 1
                    # Enable torque
                    self._packet_handler.write1ByteTxRx(
                        self._port_handler, sc.servo_id, 64, 1)  # ADDR_TORQUE_ENABLE
            print(f"  HexArth: connected via dynamixel_sdk, {n_found}/{len(self.servo_configs)} servos")
        except ImportError:
            import serial
            self._serial = serial.Serial(self.port, 1_000_000, timeout=0.01)
            self._use_dxl = False
            print(f"  HexArth: connected via raw serial (dynamixel_sdk not available)")
        self._connected = True

    def reset(self) -> BodyObservation:
        if not self._connected:
            self.connect()
        # Move to neutral standing pose
        self._current_joints = np.zeros(42, dtype=np.float32)
        self._send_servo_commands(self._current_joints)
        time.sleep(1.0)  # Wait for servos to reach position
        return self.observe()

    def command(self, action: dict) -> BodyObservation:
        target_joints = action["joints"]
        dt = 1.0 / self.config.control_freq_hz

        # Safety: velocity limiting
        safe_joints = self._enforce_velocity_limits(
            target_joints, self._current_joints, dt)

        self._send_servo_commands(safe_joints)
        self._current_joints = safe_joints.copy()
        self._step_count += 1
        self._last_command_time = time.time()

        return self.observe()

    def observe(self) -> BodyObservation:
        # Read servo feedback: position, load, voltage
        feedback = self._read_servo_feedback()
        body_position = feedback.get("body_position")
        return BodyObservation(
            joint_angles=np.asarray(feedback["positions"], dtype=np.float32),
            joint_velocities=np.asarray(feedback["velocities"], dtype=np.float32),
            contact_forces=np.asarray(feedback["foot_pressure"], dtype=np.float32),
            body_velocity=np.asarray(feedback.get("imu_velocity", np.zeros(3)), dtype=np.float32),
            body_orientation=np.asarray(feedback.get("imu_orientation", np.zeros(3)), dtype=np.float32),
            body_position=(np.asarray(body_position, dtype=np.float32)
                           if body_position is not None else None),
        )

    def close(self):
        if self._connected:
            # Move to safe pose then disconnect
            self._send_servo_commands(np.zeros(42))
            time.sleep(0.5)
            self._disconnect()

    def _send_servo_commands(self, joint_angles: np.ndarray):
        """Convert joint angles to servo units and send via sync write."""
        if not self._connected:
            return
        if self._use_dxl:
            import dynamixel_sdk as dxl
            # Sync write goal position (addr 116, 4 bytes for XM series)
            group = dxl.GroupSyncWrite(
                self._port_handler, self._packet_handler, 116, 4)
            for i, sc in enumerate(self.servo_configs):
                if i < len(joint_angles):
                    unit = self._joint_to_servo_unit(i, float(joint_angles[i]))
                    param = [dxl.DXL_LOBYTE(dxl.DXL_LOWORD(unit)),
                             dxl.DXL_HIBYTE(dxl.DXL_LOWORD(unit)),
                             dxl.DXL_LOBYTE(dxl.DXL_HIWORD(unit)),
                             dxl.DXL_HIBYTE(dxl.DXL_HIWORD(unit))]
                    group.addParam(sc.servo_id, param)
            group.txPacket()
            group.clearParam()
        else:
            # Raw serial fallback: send as packed binary
            import struct
            units = [self._joint_to_servo_unit(i, float(joint_angles[i]))
                     for i in range(min(len(joint_angles), len(self.servo_configs)))]
            payload = struct.pack(f"<{len(units)}H", *units)
            self._serial.write(b"\xff\xfe" + payload)  # header + data

    def _read_servo_feedback(self) -> dict:
        """Read position/load from all servos via sync read."""
        n = len(self.servo_configs)
        positions = np.zeros(42, dtype=np.float32)
        velocities = np.zeros(42, dtype=np.float32)
        loads = np.zeros(n, dtype=np.float32)

        if self._use_dxl:
            import dynamixel_sdk as dxl
            # Sync read present position (addr 132, 4 bytes)
            group = dxl.GroupSyncRead(
                self._port_handler, self._packet_handler, 132, 4)
            for sc in self.servo_configs:
                group.addParam(sc.servo_id)
            group.txRxPacket()
            for i, sc in enumerate(self.servo_configs):
                if group.isAvailable(sc.servo_id, 132, 4):
                    unit = group.getData(sc.servo_id, 132, 4)
                    positions[i] = self._servo_unit_to_joint(i, unit)
            group.clearParam()
        else:
            # Fallback: use last commanded positions
            positions[:] = self._current_joints

        # Estimate foot pressure from servo load (crude proxy)
        foot_pressure = np.zeros(6, dtype=np.float32)
        for i in range(min(6, n // 7)):
            tibia_idx = i * 7 + 5
            if tibia_idx < n:
                foot_pressure[i] = min(abs(loads[tibia_idx]) / 0.5, 1.0)

        return {
            "positions": positions,
            "velocities": velocities,
            "foot_pressure": foot_pressure,
        }

    def _disconnect(self):
        """Close serial port and disable servo torques."""
        if self._use_dxl:
            for sc in self.servo_configs:
                self._packet_handler.write1ByteTxRx(
                    self._port_handler, sc.servo_id, 64, 0)  # torque off
            self._port_handler.closePort()
        elif hasattr(self, '_serial'):
            self._serial.close()
        self._connected = False

    def _joint_to_servo_unit(self, joint_idx: int, angle_rad: float) -> int:
        """Convert joint angle (rad) to servo position unit."""
        if joint_idx >= len(self.servo_configs):
            return 2048  # center
        sc = self.servo_configs[joint_idx]
        unit = int((angle_rad - sc.zero_angle_rad) / sc.rad_per_unit) + 2048
        return max(sc.min_unit, min(sc.max_unit, unit))

    def _servo_unit_to_joint(self, joint_idx: int, unit: int) -> float:
        """Convert servo position unit to joint angle (rad)."""
        if joint_idx >= len(self.servo_configs):
            return 0.0
        sc = self.servo_configs[joint_idx]
        return sc.zero_angle_rad + (unit - 2048) * sc.rad_per_unit


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_hexapod(backend: str = "flygym", **kwargs) -> HexapodInterface:
    """Create hexapod interface by backend name.

    Args:
        backend: "flygym" (simulation) or "hexarth" (real robot)
    """
    if backend == "flygym":
        return FlyGymHexapod(**kwargs)
    elif backend == "hexarth":
        return HexArthHexapod(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
