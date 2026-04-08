# Project Plan: Impressing Ramdya

## Competitive Analysis (April 2026)

### What exists
1. **Pugliese 2025**: CPG from MANC, DNg100, 3-neuron circuit. No body.
2. **FlyGM (NeurIPS 2025)**: Connectome-as-GNN + RL → walking. Trained, not emergent.
3. **NeuroMechFly v2 (Ramdya)**: FlyGym platform. RL controllers. No connectome VNC.
4. **Our work**: Biophysical brain+VNC→FlyGym. Emergent rhythm. No RL.

### The gap nobody has filled
**Proprioceptive feedback through actual connectome pathways into a VNC model.**

Recent papers (Nature 2025-2026):
- Hair plate proprioceptors detect leg position limits (Mamiya et al.)
- FeCO neurons have distinct position/movement pathways to VNC (Chen et al.)
- A circuit selectively SUPPRESSES proprioception during self-movement (Titlow et al. 2026)
- Campaniform sensilla provide load feedback for stance

Nobody has wired these real proprioceptive pathways (from the BANC/MANC connectome)
into a VNC simulation that also drives a body. This creates a CLOSED proprioceptive
loop: body→sensory neurons→VNC interneurons→MNs→body.

### Why this impresses Ramdya
1. It uses HIS platform (FlyGym) in the way his lab wants but hasn't achieved
2. It's biological — real connectome pathways, not RL
3. It bridges two datasets (BANC sensory neurons → BANC VNC → FlyGym body)
4. It produces a testable prediction: which proprioceptive pathways stabilize gait?

## Three-Track Plan

### Track 1: Proprioceptive VNC (the paper)
- Map BANC sensory neuron types (FeCO, hair plates, campaniform) to VNC interneurons
- Wire ground-contact signals from FlyGym through these specific connectome pathways
- Show: proprioceptive feedback through connectome improves walking stability
- Compare: which sensory types matter most (ablation of each pathway)
- Prediction: proprioception stabilizes hind legs more than front (testable)

### Track 2: Hexapod hardware (the demo)
- HexArthHexapod class needs serial protocol implementation
- Servo calibration JSON schema
- Real-time control loop
- Goal: same brain→VNC pipeline driving a physical robot

### Track 3: Eigenlayer (the alignment angle)
- Connectome DN segregation → bottleneck integrity for AI safety
- Already has 6/6 demo tests passing
- Orthogonal to neuroscience — target NeurIPS/ICML
- Can be developed independently

## Priority
Track 1 >> Track 3 > Track 2
Track 1 is the only one that directly impresses Ramdya.
