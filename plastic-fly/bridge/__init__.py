"""
Brain-body bridge: sensorimotor interface between a FlyWire whole-brain
connectome model (139k LIF neurons in Brian2) and a FlyGym biomechanical
body (MuJoCo).

Architecture:
    body state --> SensoryEncoder --> BrainRunner (Brian2 LIF)
              --> DescendingDecoder --> VNCLite --> LocomotionBridge
              --> FlyGymAdapter --> body

The brain does not control joint torques directly. Descending neuron
activity is decoded into high-level locomotion commands (forward, turn,
frequency, stance) that modulate a CPG-based motor layer.
"""

__version__ = "0.1.0"

# Data structures
from bridge.interfaces import (
    BodyObservation,
    BrainInput,
    BrainOutput,
    LocomotionCommand,
)

# Configuration
from bridge.config import BridgeConfig

# Encoding / decoding
from bridge.sensory_encoder import SensoryEncoder
from bridge.descending_decoder import DescendingDecoder

# Brain runners
from bridge.brain_runner import (
    create_brain_runner,
    Brian2BrainRunner,
    FakeBrainRunner,
)

# Motor layers
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from bridge.vnc_lite import VNCLite, VNCLiteConfig

__all__ = [
    # Data structures
    "BodyObservation",
    "BrainInput",
    "BrainOutput",
    "LocomotionCommand",
    # Configuration
    "BridgeConfig",
    # Encoding / decoding
    "SensoryEncoder",
    "DescendingDecoder",
    # Brain runners
    "create_brain_runner",
    "Brian2BrainRunner",
    "FakeBrainRunner",
    # Motor layers
    "LocomotionBridge",
    "FlyGymAdapter",
    "VNCLite",
    "VNCLiteConfig",
]
