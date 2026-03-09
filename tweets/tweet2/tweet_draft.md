# Tweet 2 — Sensory Modalities Demo

Showcasing the brain-body bridge: 6 sensory modalities driving a 139K-neuron Drosophila connectome through FlyGym locomotion.

## Demo assets
- `sensory_demo.html` — Self-contained mobile-friendly demo page (open in any browser)
- Proof figures embedded: odor valence, looming escape, sensory perturbation, locomotion, CPG, robustness

## Option 1 (thread format)
---
1/ We gave a simulated fly a real brain — 139,000 neurons, 15 million synapses from the FlyWire connectome.

Then we asked: can the wiring alone produce intelligent behavior?

6 sensory modalities. Zero learning. Just connectome structure.

2/ SMELL: Or42b neurons (vinegar, food) make the fly turn TOWARD the odor. Or85a neurons (aversive) make it turn AWAY.

Same brain, different ORN populations → opposite behavior.

Shuffled connectome? Valence disappears. It's in the wiring.

[odor_valence_proof.png]

3/ VISION: 210 LPLC2 looming neurons detect an approaching object and trigger escape turning — left loom → escape right, right loom → escape left.

Real connectome: 50-100x stronger directionality than shuffled.

1,850 synapses from LPLC2 → 44 descending neurons. Hardwired escape.

[looming_escape_proof.png]

4/ TOUCH: Lose contact on left legs? The brain detects the mechanosensory asymmetry and activates turn circuits.

Boost gustatory on one side? Opposite heading shift.

Vestibular push? Compensatory turning.

7 perturbation conditions, all producing measurable responses through the connectome.

[sensory_perturbation.png]

5/ TEMPERATURE & HUMIDITY: FlyWire analysis reveals dedicated labeled lines — 5 thermosensory DNs, 2 hygrosensory DNs, 41 auditory DNs.

Real connectome keeps these channels 4.6-21.4x more segregated than shuffled.

Labeled-line architecture is encoded in WIRING, not activity patterns.

6/ The full pipeline: body sensors → Poisson encoding → 139K LIF brain (20ms windows) → 204 descending readout neurons → VNC-lite CPG modulation → 42-joint MuJoCo fly body.

Closed-loop. Real-time. No backprop. No gradient. Just a connectome.

🪰🧠

---

## Option 2 (single tweet, punchy)
---
A fly brain with 139K real neurons and 15M synapses can:

🟢 Smell food vs danger (opposite turning)
🔴 Escape looming threats (50x specificity)
🔵 Feel asymmetric touch → compensatory turns
🟠 Separate temperature from humidity

No learning. No training. The WIRING does this.

6 sensory modalities, 1 connectome, zero backprop.

[sensory_demo.html for full interactive demo]

🪰🧠
---

## Option 3 (video-forward, for Unity demo recording)
---
We built the complete Drosophila sensorimotor loop:

Real FlyWire connectome (139K neurons) ↔ FlyGym body (42 joints)

Watch it smell, see, and feel its way through the world.

Every behavior emerges from connectome wiring alone.

[Unity recording / sensory_demo.html]

#Neuroscience #Connectome #Drosophila #ArtificialLife
---

## Hashtags
#Neuroscience #ArtificialLife #Connectome #Drosophila #FlyWire #Brian2 #HebbianLearning #Sensorimotor
