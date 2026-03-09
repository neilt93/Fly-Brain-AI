"""
Sensory encoder: body state → Poisson firing rates for brain sensory neurons.

v2: channel-specific encoding. Body observation fields map to identified
    neuron populations via channel_map.json:
      - gustatory:      sugar GRNs, driven by contact forces (ground → feeding)
      - proprioceptive: SEZ ascending types, driven by joint angles + velocities
      - mechanosensory: SEZ ascending types, driven by leg contact force magnitudes
      - vestibular:     SEZ ascending types, driven by body velocity + orientation
"""

import json
import numpy as np
from pathlib import Path
from bridge.interfaces import BodyObservation, BrainInput


class SensoryEncoder:
    """Channel-aware sensory encoder.

    Each channel maps specific body observation fields to a subset of
    sensory neurons with channel-appropriate encoding.
    """

    def __init__(
        self,
        sensory_neuron_ids: np.ndarray,
        channel_map: dict[str, list[int]] | None = None,
        max_rate_hz: float = 100.0,
        baseline_rate_hz: float = 10.0,
    ):
        self.sensory_neuron_ids = np.asarray(sensory_neuron_ids, dtype=np.int64)
        self.max_rate_hz = float(max_rate_hz)
        self.baseline_rate_hz = float(baseline_rate_hz)

        # Build ID → index lookup
        self._id_to_idx = {int(nid): i for i, nid in enumerate(self.sensory_neuron_ids)}

        # Parse channel map
        if channel_map is not None:
            self._channels = {
                ch: np.array([self._id_to_idx[int(nid)]
                              for nid in ids if int(nid) in self._id_to_idx],
                             dtype=int)
                for ch, ids in channel_map.items()
            }
            self._has_channels = True
        else:
            self._has_channels = False

    def encode(self, obs: BodyObservation) -> BrainInput:
        n = len(self.sensory_neuron_ids)
        rates = np.full(n, self.baseline_rate_hz, dtype=np.float32)

        if self._has_channels:
            self._encode_channels(obs, rates)
        else:
            self._encode_flat(obs, rates)

        return BrainInput(
            neuron_ids=self.sensory_neuron_ids,
            firing_rates_hz=rates,
        )

    def _encode_channels(self, obs: BodyObservation, rates: np.ndarray):
        """Channel-specific encoding with appropriate nonlinearities."""
        max_r = self.max_rate_hz
        base = self.baseline_rate_hz

        # Gustatory: contact forces modulate rate (ground contact → feeding context)
        if "gustatory" in self._channels:
            idx = self._channels["gustatory"]
            if len(idx) > 0:
                # Mean contact force as global feeding signal
                contact_signal = float(np.mean(np.clip(obs.contact_forces, 0, 1)))
                rates[idx] = base + (max_r - base) * contact_signal

        # Proprioceptive: joint angles + velocities → per-neuron rates
        if "proprioceptive" in self._channels:
            idx = self._channels["proprioceptive"]
            if len(idx) > 0:
                # Interleave angles and velocities for richer representation
                angles_norm = np.tanh(obs.joint_angles)       # [-1, 1]
                vels_norm = np.tanh(obs.joint_velocities * 0.1)  # scale down velocities
                features = np.concatenate([angles_norm, vels_norm])  # (84,)
                # Tile or truncate to match neuron count
                n = len(idx)
                if len(features) < n:
                    features = np.pad(features, (0, n - len(features)))
                else:
                    features = features[:n]
                # Map [-1,1] → [0, max_rate]
                rates[idx] = ((features + 1.0) * 0.5 * (max_r - base) + base).astype(np.float32)

        # Mechanosensory: per-leg contact forces → per-neuron rates
        if "mechanosensory" in self._channels:
            idx = self._channels["mechanosensory"]
            if len(idx) > 0:
                forces = np.clip(obs.contact_forces, 0.0, 1.0)  # (6,)
                n = len(idx)
                if len(forces) < n:
                    forces = np.pad(forces, (0, n - len(forces)))
                else:
                    forces = forces[:n]
                # Direct mapping: higher force → higher rate
                rates[idx] = (base + forces * (max_r - base)).astype(np.float32)

        # Vestibular: body velocity + orientation → per-neuron rates
        if "vestibular" in self._channels:
            idx = self._channels["vestibular"]
            if len(idx) > 0:
                vel_norm = np.tanh(obs.body_velocity * 0.5)        # (3,)
                ori_norm = np.tanh(obs.body_orientation)            # (3,)
                features = np.concatenate([vel_norm, ori_norm])     # (6,)
                n = len(idx)
                if len(features) < n:
                    features = np.pad(features, (0, n - len(features)))
                else:
                    features = features[:n]
                rates[idx] = ((features + 1.0) * 0.5 * (max_r - base) + base).astype(np.float32)

    def _encode_flat(self, obs: BodyObservation, rates: np.ndarray):
        """Fallback: v1-style flat encoding (for backward compatibility)."""
        features = np.concatenate([
            obs.joint_angles, obs.joint_velocities,
            obs.contact_forces, obs.body_velocity, obs.body_orientation,
        ])
        features = np.tanh(features)
        n = len(rates)
        if len(features) < n:
            features = np.pad(features, (0, n - len(features)))
        else:
            features = features[:n]
        rates[:] = ((features + 1.0) * 0.5) * self.max_rate_hz

    @classmethod
    def from_channel_map(
        cls,
        sensory_ids: np.ndarray,
        channel_map_path: str | Path,
        max_rate_hz: float = 100.0,
        baseline_rate_hz: float = 10.0,
    ) -> "SensoryEncoder":
        """Load channel map from JSON and create encoder."""
        with open(channel_map_path) as f:
            channel_map = json.load(f)
        return cls(
            sensory_neuron_ids=sensory_ids,
            channel_map=channel_map,
            max_rate_hz=max_rate_hz,
            baseline_rate_hz=baseline_rate_hz,
        )
