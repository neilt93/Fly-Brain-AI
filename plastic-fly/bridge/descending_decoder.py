"""
Descending decoder: brain descending neuron firing rates → LocomotionCommand.

Neurons are partitioned into functional groups. Each group's mean rate
drives one locomotion parameter through a tanh nonlinearity.

v2: groups from annotated SEZ motor/descending neuron types with
    bilateral pair splitting for turn asymmetry.
"""

import json
import numpy as np
from pathlib import Path
from bridge.interfaces import BrainOutput, LocomotionCommand


class DescendingDecoder:
    def __init__(
        self,
        forward_ids: np.ndarray,
        turn_left_ids: np.ndarray,
        turn_right_ids: np.ndarray,
        rhythm_ids: np.ndarray,
        stance_ids: np.ndarray,
        rate_scale: float = 40.0,
    ):
        self.forward_ids = set(map(int, forward_ids))
        self.turn_left_ids = set(map(int, turn_left_ids))
        self.turn_right_ids = set(map(int, turn_right_ids))
        self.rhythm_ids = set(map(int, rhythm_ids))
        self.stance_ids = set(map(int, stance_ids))
        self.rate_scale = rate_scale

    def get_group_rates(self, brain_output: BrainOutput) -> dict:
        """Extract raw mean firing rates per group (before nonlinearities).

        Returns dict with keys: forward, turn_left, turn_right, rhythm, stance.
        Values are mean Hz for each group.
        """
        id_to_rate = {
            int(nid): float(rate)
            for nid, rate in zip(brain_output.neuron_ids, brain_output.firing_rates_hz)
        }

        def mean_rate(id_set):
            vals = [id_to_rate[nid] for nid in id_set if nid in id_to_rate]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "forward": mean_rate(self.forward_ids),
            "turn_left": mean_rate(self.turn_left_ids),
            "turn_right": mean_rate(self.turn_right_ids),
            "rhythm": mean_rate(self.rhythm_ids),
            "stance": mean_rate(self.stance_ids),
        }

    def decode(self, brain_output: BrainOutput) -> LocomotionCommand:
        rates = self.get_group_rates(brain_output)

        return LocomotionCommand(
            forward_drive=float(max(0.1, np.tanh(rates["forward"] / self.rate_scale))),
            turn_drive=float(np.tanh((rates["turn_left"] - rates["turn_right"]) / self.rate_scale)),
            step_frequency=float(1.0 + 1.5 * np.tanh(rates["rhythm"] / self.rate_scale)),
            stance_gain=float(1.0 + 0.5 * np.tanh(rates["stance"] / self.rate_scale)),
        )

    @classmethod
    def from_json(cls, path: str | Path, rate_scale: float = 40.0) -> "DescendingDecoder":
        """Load decoder groups from JSON file."""
        with open(path) as f:
            groups = json.load(f)
        return cls(
            forward_ids=np.array(groups["forward_ids"], dtype=np.int64),
            turn_left_ids=np.array(groups["turn_left_ids"], dtype=np.int64),
            turn_right_ids=np.array(groups["turn_right_ids"], dtype=np.int64),
            rhythm_ids=np.array(groups["rhythm_ids"], dtype=np.int64),
            stance_ids=np.array(groups["stance_ids"], dtype=np.int64),
            rate_scale=rate_scale,
        )
