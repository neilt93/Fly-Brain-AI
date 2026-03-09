# Bridge: sensorimotor interface between FlyWire whole-brain model and FlyGym body.
#
# Architecture:
#   body state → sensory_encoder → brain_runner → descending_decoder
#   → locomotion_bridge → flygym_adapter → body
