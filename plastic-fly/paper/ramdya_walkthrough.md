# 5-Minute Walkthrough for Pavan Ramdya

**Neil Tripathi** | Connectome-Constrained Sensorimotor Behaviors in Drosophila
Closed-loop FlyWire brain (138,639 neurons) + FlyGym body (v1.2.1)

---

## 0:00-0:30 -- Hook: what we built and why it matters to you

- We coupled the entire FlyWire connectome (138,639 LIF neurons, 54.5M synapses, Brian2) to FlyGym v1.2.1 in closed loop.
- Your platform is the body. The connectome is the brain. No learning, no parameter fitting.
- Three distinct adaptive behaviors emerge from wiring alone -- this is a validation that FlyGym can serve as the embodiment layer for connectome-scale neuroscience.

## 0:30-1:30 -- Architecture: the sensorimotor interface

- **Brain:** Shiu et al. (2023) LIF model, unmodified. 15M connection pairs, uniform biophysics, weights proportional to synapse count.
- **Body:** FlyGym v1.2.1, 42 DOFs, PreprogrammedSteps tripod gait, MuJoCo contact physics.
- **Our contribution is the bridge between them:**
  - Sensory encoder: 10 channels, 275-485 neurons (gustatory, proprioceptive, mechanosensory, vestibular, olfactory, visual, LPLC2 looming). Body state encoded as Poisson rates injected into biologically identified populations.
  - Descending decoder: 204-389 DNs read out into 4 CPG commands (forward drive, turn drive, step frequency, stance gain). Brain modulates gait parameters, not individual joints.
- Each step: 20-100ms brain window, then 200-1000 body timesteps at the decoded command. 200ms warmup for brain stabilization.
- Key design point: the interface is transparent and interpretable. Every neuron ID traces back to FlyWire. The brain itself is untouched.

## 1:30-3:00 -- Three behaviors, zero learning

**Causal locomotion (10/10 ablation tests):**
- Silencing forward neurons: forward drive drops 89%, walking distance drops 53%.
- Dose-response is graded and near-linear -- each 10 neurons silenced costs ~1mm of forward distance.
- Shuffled connectome: 3/10 tests pass (trivial), all groups at 0 Hz. Wiring is necessary.

**Olfactory valence discrimination (6/6 tests, n=10 seeds):**
- DM1 (Or42b, attractive): fly turns toward odor source.
- DM5 (Or85a, aversive): fly turns away.
- Opposite valence emerges from connectome wiring alone. Shuffled connectome eliminates all valence signal.

**Looming escape via LPLC2 (5/5 tests, escape index 1.11):**
- 210 LPLC2 neurons project to 44 DNs via 1,850 direct synapses. Strongly ipsilateral.
- Unilateral loom produces contralateral turning. Robust across integration windows (20, 50, 100ms).
- Shuffled connectome: escape index 0.053 -- 21x weaker. The directional response requires specific LPLC2-to-DN wiring.

## 3:00-4:15 -- The unexpected finding: modality-specific descending channels

- At the direct sensory-to-DN interface (1-hop), modalities are nearly completely segregated:
  - Somatosensory reaches 186 DNs, Visual reaches 44, Olfactory reaches 1.
  - Pairwise Jaccard indices: 0.005-0.060. Shuffled controls are 4.6-21.4x higher.
- One synapse deeper (2-hop), convergence is near-complete -- but through modality-specific interneuron pools. Visual and olfactory share zero intermediates.
- This is a labeled-line architecture at both the DN level and the relay level.
- Extended to 6 modalities (adding auditory, thermo, hygro): olfactory has zero DN overlap with all five others. Thermo-hygro converge on a single DN type (DNb05, 2 neurons) -- silencing DNb05 collapses thermo/hygro throughput (16.4x specificity) while preserving everything else.
- Stance-controlling DNs receive zero visual or olfactory input at 1-hop -- exclusively somatosensory. This was not designed; it fell out of the connectome analysis.

## 4:15-4:45 -- What this means

- The FlyWire connectome carries enough functional information to produce stimulus-specific, causally validated sensorimotor behavior -- without learning, without neuromodulation, without synaptic tuning.
- FlyGym is the right platform for this: realistic contact physics, proper joint control, and clean CPG interface made the brain-body coupling tractable.
- The segregation finding (labeled lines at the DN level) is new. It required both a complete connectome AND embodied behavioral readout -- anatomical tracing alone cannot identify which of the 350 DNs are functionally active.

## 4:45-5:00 -- What we need / potential collaboration

- **VNC connectome integration:** Our biggest limitation is the preprogrammed tripod CPG. Replacing it with a VNC connectome-driven motor pattern generator would close the final loop. Your group's expertise in VNC motor circuits and NeuroMechFly biomechanics is exactly what this needs.
- **FlyGym sensory extensions:** Richer sensory transduction models (e.g., realistic olfactory dynamics, optic flow) could unlock additional connectome-driven behaviors. Would be interested in whether you have planned sensory APIs.
- **Validation against real fly data:** The dose-response curves (graded population ablation) generate testable predictions for genetic silencing experiments. Your group's optogenetic/thermogenetic behavioral setups could directly test these.
- **Co-authorship / platform showcase:** This is, to our knowledge, the first whole-brain-to-FlyGym closed loop. Happy to position it as a demonstration of what FlyGym enables at connectome scale.

---

### Quick reference numbers

| Metric | Value |
|---|---|
| Brain neurons | 138,639 (FlyWire 783) |
| Synaptic connections | 15M pairs, 54.5M synapses |
| Sensory channels | 10 (275-485 neurons) |
| Readout DNs | 204-389 |
| Locomotion ablation | 10/10 causal tests |
| Olfactory valence | 6/6 tests, n=10 seeds |
| Looming escape | Escape index 1.11, 21x over shuffled |
| DN segregation (Jaccard) | 0.005-0.060 (3 modalities) |
| DNb05 specificity | 16.4x (thermo/hygro bottleneck) |
| Shuffled controls | All effects wiring-specific |
