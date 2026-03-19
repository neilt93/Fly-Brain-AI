"""
VNC sensory encoder: body state -> MANC leg sensory neuron firing rates.

Maps proprioceptive signals from FlyGym body state to identified sensory
neuron populations in the MANC VNC model:
  - feCO (femoral chordotonal organ): joint angles -> position sensing
  - claw/hook/club sensors: contact forces -> mechanosensory
  - hair plate: joint movement -> proprioceptive velocity

Simple linear encoding: rate = baseline + gain * stimulus
"""

import numpy as np
from pathlib import Path
from bridge.interfaces import BodyObservation


# Leg mapping for sensory neurons
LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]

# MANC entry nerve -> thoracic segment (T1/T2/T3)
ENTRY_NERVE_TO_SEGMENT = {
    "ProLN": "T1",    # Prothoracic leg nerve
    "MesoLN": "T2",   # Mesothoracic leg nerve
    "MetaLN": "T3",   # Metathoracic leg nerve
}

# Segment x side -> FlyGym leg name
SEGMENT_SIDE_TO_LEG = {
    ("T1", "L"): "LF", ("T1", "R"): "RF",
    ("T2", "L"): "LM", ("T2", "R"): "RM",
    ("T3", "L"): "LH", ("T3", "R"): "RH",
}


class VNCSensoryEncoder:
    """Encode body state into MANC VNC sensory neuron firing rates.

    Channels:
      - Proprioceptive (feCO, hair_plate, campaniform):
        Joint angles + velocities -> per-leg position/velocity rates
      - Mechanosensory (claw, hook, club):
        Contact forces -> per-leg contact rates
    """

    def __init__(
        self,
        sensory_neuron_ids: dict[str, dict[str, list[int]]] | None = None,
        baseline_hz: float = 10.0,
        max_hz: float = 80.0,
    ):
        self.baseline_hz = baseline_hz
        self.max_hz = max_hz

        # Sensory neurons grouped by leg and type
        # {leg: {"proprioceptive": [body_ids], "mechanosensory": [body_ids]}}
        self._leg_sensory = {leg: {"proprioceptive": [], "mechanosensory": []}
                             for leg in LEG_ORDER}
        self._all_sensory_ids: set[int] = set()

        if sensory_neuron_ids is not None:
            for leg, channels in sensory_neuron_ids.items():
                if leg in self._leg_sensory:
                    for ch, ids in channels.items():
                        if ch in self._leg_sensory[leg]:
                            self._leg_sensory[leg][ch] = list(ids)
                            self._all_sensory_ids.update(ids)

        self.n_sensory = len(self._all_sensory_ids)

    def encode(self, body_obs: BodyObservation) -> dict[int, float]:
        """Map body state to sensory neuron firing rates.

        Returns:
            dict mapping MANC body_id -> firing rate (Hz)
        """
        if self.n_sensory == 0:
            return {}

        rates: dict[int, float] = {}
        baseline = self.baseline_hz
        max_r = self.max_hz

        # Joint angles -> proprioceptive neurons (per-leg)
        # FlyGym joints: 7 per leg, 6 legs = 42 total
        for leg_idx, leg in enumerate(LEG_ORDER):
            offset = leg_idx * 7

            # Proprioceptive: driven by joint angles + velocities
            proprio_ids = self._leg_sensory[leg]["proprioceptive"]
            if proprio_ids:
                # Mean absolute joint angle as proprioceptive signal
                angles = body_obs.joint_angles[offset:offset + 7]
                vels = body_obs.joint_velocities[offset:offset + 7]
                angle_signal = float(np.clip(
                    np.mean(np.abs(np.tanh(angles))), 0, 1))
                vel_signal = float(np.clip(
                    np.mean(np.abs(np.tanh(vels * 0.1))), 0, 1))
                signal = 0.7 * angle_signal + 0.3 * vel_signal
                rate = baseline + (max_r - baseline) * signal
                for bid in proprio_ids:
                    rates[bid] = rate

            # Mechanosensory: driven by contact forces
            mechano_ids = self._leg_sensory[leg]["mechanosensory"]
            if mechano_ids:
                force = float(np.clip(body_obs.contact_forces[leg_idx], 0, 1))
                rate = baseline + (max_r - baseline) * force
                for bid in mechano_ids:
                    rates[bid] = rate

        return rates

    @classmethod
    def from_manc_annotations(
        cls,
        annotations_path: str | Path,
        baseline_hz: float = 10.0,
        max_hz: float = 80.0,
    ) -> "VNCSensoryEncoder":
        """Build encoder from MANC annotation file.

        Selects vnc_sensory neurons entering via leg nerves (ProLN, MesoLN,
        MetaLN). Uses 'class' column for proprioceptive vs mechanosensory
        classification and 'rootSide' for lateralization.
        """
        import pandas as pd
        import pyarrow.feather as feather

        ann = pd.DataFrame(feather.read_feather(str(annotations_path)))

        # Select VNC sensory neurons entering via leg nerves
        sensory_mask = (
            (ann["superclass"] == "vnc_sensory")
            & ann["entryNerve"].isin(["ProLN", "MesoLN", "MetaLN"])
        )
        sensory_df = ann[sensory_mask].copy()

        if len(sensory_df) == 0:
            print("  VNCSensoryEncoder: no leg sensory neurons found")
            return cls(sensory_neuron_ids=None,
                       baseline_hz=baseline_hz, max_hz=max_hz)

        # Classify by class and leg
        leg_sensory: dict[str, dict[str, list[int]]] = {
            leg: {"proprioceptive": [], "mechanosensory": []}
            for leg in LEG_ORDER
        }

        for _, row in sensory_df.iterrows():
            bid = int(row["bodyId"])
            nerve = str(row["entryNerve"])
            seg = ENTRY_NERVE_TO_SEGMENT.get(nerve)
            if seg is None:
                continue
            side = (str(row.get("rootSide", "L"))
                    if pd.notna(row.get("rootSide")) else "L")

            leg = SEGMENT_SIDE_TO_LEG.get((seg, side))
            if leg is None:
                continue

            # Classify by MANC 'class' column
            cls_val = (str(row.get("class", ""))
                       if pd.notna(row.get("class")) else "")
            cls_lower = cls_val.lower()

            if "proprioceptive" in cls_lower:
                leg_sensory[leg]["proprioceptive"].append(bid)
            elif "tactile" in cls_lower or "mechanosensory" in cls_lower:
                leg_sensory[leg]["mechanosensory"].append(bid)
            elif "gustatory" in cls_lower or "chemosensory" in cls_lower:
                # Gustatory/chemosensory: map to mechanosensory (contact)
                leg_sensory[leg]["mechanosensory"].append(bid)
            else:
                # Unknown/unclassified: default to proprioceptive
                leg_sensory[leg]["proprioceptive"].append(bid)

        n_proprio = sum(len(v["proprioceptive"]) for v in leg_sensory.values())
        n_mechano = sum(
            len(v["mechanosensory"]) for v in leg_sensory.values())
        print(f"  VNCSensoryEncoder: {n_proprio} proprioceptive + "
              f"{n_mechano} mechanosensory = "
              f"{n_proprio + n_mechano} sensory neurons")

        return cls(
            sensory_neuron_ids=leg_sensory,
            baseline_hz=baseline_hz,
            max_hz=max_hz,
        )

    @property
    def sensory_body_ids(self) -> set[int]:
        return self._all_sensory_ids.copy()
