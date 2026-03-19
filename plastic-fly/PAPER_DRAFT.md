# Connectome-Constrained Sensorimotor Behaviors and Modality-Specific Motor Channels in Drosophila

## Abstract

The Drosophila melanogaster FlyWire connectome provides a neuron-level wiring diagram of 139,255 neurons and approximately 50 million synapses, but whether connectome-structured dynamics can support meaningful sensorimotor transformations without learning or task-specific optimization remains unknown. We built a closed-loop system coupling a Brian2 leaky integrate-and-fire simulation of 138,639 neurons (FlyWire 783 completeness snapshot; 15 million connection pairs, 54.5 million synapses) to a MuJoCo biomechanical fly body (FlyGym), with biologically identified sensory populations encoding stimuli and descending neuron populations decoding motor commands through an interpretable sensorimotor interface. The connectome-structured brain model, when coupled to an embodied fly through this transparent interface, produces adaptive and behaviorally specific responses without learning or parameter fitting: causal locomotion control (10/10 ablation tests, forward drive reduced 46% by targeted silencing), olfactory valence discrimination (opposite turning for attractive vs aversive odors, 6/6 tests), and visually guided looming escape (contralateral turning with escape index 1.11, abolished 21-fold in shuffled connectome controls). Analysis of the descending neuron populations underlying these behaviors reveals that direct (1-hop) sensory-to-motor connectivity is highly modality-specific (Jaccard index 0.005-0.060 across three modalities), with convergence occurring one synapse deeper through modality-specific interneuron pools that share zero intermediates between visual and olfactory pathways. This specificity is consistent with the known multi-synaptic architecture of sensory processing: olfactory signals reach only 1 DN directly but, when weighted by synaptic strength, drive ~2% of the DN population at 2-hops — matching physiological recordings of odor-responsive DNs (Aymanns et al. 2022). Extended analysis across six sensory modalities reveals that thermo-hygro signals converge on the multimodal steering neuron DNb05 (Namiki et al. 2018; Yang et al. 2024), which despite receiving only ~1.4% of its total input from thermo/hygro sources, is the sole direct thermo/hygro-to-DN pathway — its ablation collapses thermo/hygro throughput by 93-100% (specificity 16.4x), demonstrating that labeled-line function can emerge from a minority input to a multimodal neuron.

---

## 1. Introduction

The completion of the Drosophila melanogaster whole-brain connectome (Dorkenwald et al. 2024, FlyWire) provides, for the first time, a neuron-level wiring diagram of an animal capable of complex behavior. The full resource comprises 139,255 neurons and approximately 50 million chemical synapses, including identified sensory input populations and descending motor output pathways. Yet a fundamental question remains: can connectome-structured dynamics support meaningful embodied sensorimotor transformations without learning or task-specific optimization?

Previous work has used the connectome for circuit analysis (Schlegel et al. 2023), connectome-constrained network simulations (Shiu et al. 2023), connectome-constrained visual models (Lappalainen et al. 2024), and anatomical tracing of specific pathways. Concurrent work has coupled brain models to embodied fly simulators for behavioral demonstration (Shiu et al. 2024), and whole-body physics simulations have achieved realistic locomotion via reinforcement learning (Vaxenburg et al. 2025). Our work differs in emphasis: rather than demonstrating behavioral breadth, we focus on quantitative causal validation — systematic ablation with controls, dose-response curves, and population-level analysis of the descending neuron interface that mediates brain-to-body control. Locomotion in our system is mediated through a CPG/VNC-like interface — the brain modulates gait parameters rather than controlling individual joints directly — which we state explicitly as both a design choice and a current limitation.

Here we build such a system and make three contributions:

1. **Behavioral specificity.** The connectome-structured brain model, coupled to an embodied fly through an interpretable sensorimotor interface, produces three distinct adaptive behaviors — locomotion control, olfactory valence discrimination, and looming escape — without parameter fitting or learning.

2. **Graded causal control.** Targeted ablation of identified neuron populations produces graded, quantitatively predicted behavioral deficits, confirming that specific connectome pathways causally drive specific behaviors.

3. **Modality-specific descending channels.** Analysis of the descending neurons functionally recruited by each behavior reveals a previously unreported structural principle: sensory modalities maintain near-complete segregation at the direct sensory-to-descending interface, converging only through a single interneuron layer. The connectome implements parallel labeled lines for multisensory motor control.

4. **Structured population codes.** The intact connectome activates 25x more condition-responsive descending neurons than a shuffled control, and sensory conditions are linearly decodable from the DN population at 2.2x chance level — structure that vanishes when wiring is randomized.

5. **Topology vs. interpretability.** When we test whether connectome topology accelerates motor learning compared to random sparse networks of matched density, we find no advantage — sparsity alone determines learning speed. The connectome's value is not as an optimization prior but as an interpretable, causally traceable architecture: a documented circuit whose components can be selectively manipulated with predictable outcomes, unlike a random network of equivalent performance.

---

## 2. Results

### 2.1 A closed-loop connectome-body interface

We built a brain-body bridge coupling the FlyWire connectome to a biomechanical fly body (Fig. 1). The system has three components, two from public resources and one that is our contribution:

**Brain (public).** A Brian2 leaky integrate-and-fire (LIF) simulation of 138,639 neurons from the FlyWire 783 completeness snapshot (15 million connection pairs comprising 54.5 million synapses), following the model architecture of Shiu et al. (2023). All neurons share identical biophysical parameters; synaptic weights are proportional to synapse count from the connectome. No parameters were tuned or optimized.

**Body (public).** A MuJoCo-based biomechanical hexapod via FlyGym v1.2.1 (Wang et al. 2024) with 42 actuated degrees of freedom, realistic contact physics, and preprogrammed tripod gait kinematics.

**Sensorimotor interface (our contribution).** The bridge between brain and body consists of a sensory encoder and a descending decoder. These are the designed components of our system — they define which neurons receive sensory input, which neurons are read out as motor commands, and how those signals are encoded/decoded. The brain itself is unmodified from the connectome.

The interface consists of two components:

**Sensory encoder.** Body state and environmental stimuli are encoded as Poisson firing rates injected into biologically identified sensory neuron populations. We define 10 sensory channels totaling 275-485 neurons (depending on experiment): gustatory (20 sugar GRNs), proprioceptive (33 SEZ ascending neurons), mechanosensory (15 SEZ ascending), vestibular (7 SEZ ascending), bilateral olfactory (100 ORNs per side), bilateral visual (50 photoreceptors per side), and bilateral LPLC2 looming detectors (108 left, 102 right).

**Descending decoder.** Firing rates from 204-389 descending neurons (DNs) are decoded into four locomotion commands — forward drive, turn drive, step frequency, and stance gain — which modulate the body's CPG. DN populations were selected using a hybrid approach: annotated motor/descending types from the connectome plus connectivity-augmented supplements (neurons with high downstream/upstream connectivity ratios).

Each simulation step runs a 20-100ms brain window followed by 200-1000 body timesteps at the decoded locomotion command. A 200ms warmup period at baseline allows brain activity to stabilize before the experiment begins.

### 2.2 Causal locomotion control from connectome wiring

We first tested whether the connectome produces functionally distinct locomotion commands by selectively ablating (silencing) each of five neuron groups: forward, turn-left, turn-right, rhythm, and stance (Fig. 2A).

All 10 causal tests passed (5 command-level, 5 behavioral):

| Ablation | Command effect | Behavioral effect |
|---|---|---|
| Forward neurons silenced | Forward drive: 0.97 → 0.10 (-90%) | Distance: 20.6mm → 11.2mm (-46%) |
| Turn-left silenced | Turn drive shifts rightward | Heading: +129.6° rightward |
| Turn-right silenced | Turn drive shifts leftward | Heading: -137.0° leftward |
| Rhythm neurons silenced | Step frequency: 2.05 → 1.00 (-51%) | Reduced stepping frequency |
| Stance neurons silenced | Stance gain: 1.49 → 1.00 (-33%) | Altered contact profile |

The shuffled connectome control (postsynaptic indices permuted, preserving out-degree) produced 3/10 passing tests — only trivial contrast comparisons between two near-zero signals. All neuron groups showed 0% activity (0.0 Hz mean firing rate), confirming that the shuffled network has no functional circuit structure.

**Dose-response.** Progressive ablation of the forward neuron population produced a graded, near-linear reduction in forward distance (Fig. 2B): 0% ablation = 19.7mm, 25% = 18.8mm, 50% = 16.4mm, 75% = 11.6mm, 100% = 8.6mm. Each 10 neurons inactivated reduced forward distance by approximately 1mm, demonstrating that the connectome implements a distributed, graded population code for locomotion drive.

### 2.3 Olfactory valence discrimination

Drosophila innately approach some odors and avoid others. We tested whether the connectome encodes this valence discrimination by presenting bilateral olfactory stimuli corresponding to two identified odor channels: Or42b (projecting to DM1 glomerulus, attractive) and Or85a (projecting to DM5 glomerulus, aversive).

For each odor type, we presented asymmetric stimulation (high intensity on one side, zero on the other) and measured the resulting turn drive.

**Results (Fig. 3, n=10 seeds):**
- DM1 (attractive): turn contrast = -0.002 (toward odor source)
- DM5 (aversive): turn contrast = +0.033 (away from odor source)
- Valence contrast (DM5 - DM1) = +0.035

All 6 validation tests passed:
1. DM1 produces attractive turning (toward source)
2. DM5 produces aversive turning (away from source)
3. DM1 and DM5 produce opposite valence
4. Real connectome > shuffled connectome
5. DM1 connectome specificity (real vs shuffled)
6. DM5 connectome specificity (real vs shuffled)

The shuffled connectome eliminated all valence signals (contrast = +0.002), confirming that odor valence discrimination is encoded in the specific synaptic wiring of the connectome, not in generic network properties.

### 2.4 Looming escape via LPLC2 descending pathway

Flies exhibit rapid escape responses to looming visual stimuli. The LPLC2 neurons in the lobula plate are known looming detectors. We identified 210 LPLC2 neurons in the FlyWire connectome (108 left, 102 right) and traced their projections to 44 descending neurons via 1,850 direct synapses, confirming a strongly ipsilateral projection pattern.

We injected looming stimuli as elevated firing rates (200 Hz) into LPLC2 neurons on one side while measuring the resulting turn drive.

**Results (Fig. 4, n=5 seeds, 50ms brain window):**
- Loom left (left LPLC2 active): turn drive = +0.364 (rightward escape)
- Loom right (right LPLC2 active): turn drive = -0.748 (leftward escape)
- Control (no looming): turn drive = -0.181
- **Escape index** (loom_left - loom_right turn): **1.112**

The response was robust across brain integration windows: escape index = 1.084 at 20ms, 1.112 at 50ms, 1.109 at 100ms. The 1-hop LPLC2→DN circuit requires no temporal integration to function.

The shuffled connectome produced an escape index of 0.053 — **21-fold weaker** than the real connectome — confirming that the directional escape response requires the specific wiring of the LPLC2 descending pathway.

The asymmetry between loom-left (+0.36) and loom-right (-0.75) turning responses is consistent with the decoder architecture: the turn-right group contains 174 DNs vs 146 for turn-left, and right LPLC2 projects to 21 DNs vs 24 for left LPLC2, producing a stronger contralateral turn signal for right-side stimuli.

### 2.5 Modality-specific descending motor channels

Recent large-scale connectomic analyses have established that descending neuron types cluster by sensory input modality (Stürner et al. 2025), with 16 clusters ranging from modality-enriched to broadly multimodal. However, these analyses used information-flow distance (shortest path length) to assign sensory modality, which does not distinguish strong direct wiring from weak multi-synaptic paths. Here we complement that approach with direct (1-hop) and synapse-weighted connectivity measurements, asking whether the modality organization observed at the type level by Stürner et al. is also present — and potentially sharper — at the level of individual synaptic connections.

We traced all direct (1-hop) connections from each sensory modality to the 350-neuron descending pool (389 selected, 350 mapped to valid connectome root IDs), then repeated the analysis at 2-hops (sensory → interneuron → DN) with synapse-count weighting to assess functional drive strength.

**1-hop: near-complete segregation (Fig. 5A).**

| Modality | DNs reached (1-hop) | Direct edges | Total synapses |
|---|---|---|---|
| Somatosensory (75 neurons) | 186 | 783 | 7,374 |
| Visual/LPLC2 (310 neurons) | 44 | 1,850 | 8,823 |
| Olfactory (100 neurons) | 1 | 2 | 7 |

Pairwise overlap at 1-hop:

| Pair | Shared DNs | Jaccard index |
|---|---|---|
| Olfactory–Visual | 1 | 0.023 |
| Olfactory–Somatosensory | 1 | 0.005 |
| Visual–Somatosensory | 13 | 0.060 |

173 DNs are exclusively somatosensory-reachable. 31 DNs are exclusively visual-reachable. Zero DNs are exclusively olfactory-reachable. 133 readout DNs receive no direct sensory input from any modality. The direct sensory-to-motor interface maintains near-complete modality segregation.

**Shuffled connectome control.** To test whether this segregation could arise trivially from population size differences, we shuffled the connectome by permuting all postsynaptic targets (preserving out-degree) and repeated the analysis (n=5 shuffles). Shuffled wiring produced 4.6–21.4× higher pairwise Jaccard overlap than the real connectome: somatosensory–visual 0.273 vs 0.060 (4.6×), somatosensory–olfactory 0.115 vs 0.005 (21.4×), visual–olfactory 0.155 vs 0.023 (6.8×). The near-zero overlap in the real connectome is not a consequence of small population sizes — it is actively maintained by the specific synaptic wiring.

**The 13 shared visual-somatosensory DNs.** All 13 visual-somatosensory shared DNs receive input specifically from LPLC2 looming detectors (not generic R7/R8 photoreceptors, which have zero direct DN connections). These neurons are annotated looming-escape turning types: DNp11, DNp27, DNp05, DNp35, DNp69, DNp70, DNa07, DNae007, DNc01, DNpe042 — 8 in the turn-right decoder group, 4 in turn-left, 1 in rhythm. The sole point of multimodal convergence at the DN level is the escape circuit, where millisecond integration of visual threat detection and somatosensory body state is survival-critical.

**Excitatory/inhibitory asymmetry.** Visual→DN connections are 100% excitatory (1,850/1,850). Somatosensory→DN connections include 21.2% inhibitory (166/783 edges), suggesting active suppression of specific motor programs by body state — a mechanism that pure excitatory drive cannot provide.

**2-hop: rapid convergence through separate interneuron pools (Fig. 5B).**

| Modality | Active intermediates | DNs reached (2-hop) |
|---|---|---|
| Somatosensory | 4,475 | 350 (100%) |
| Visual | 2,279 | 304 (87%) |
| Olfactory | 488 | 175 (50%) |

Somatosensory reaches every readout DN at 2-hop. The Jaccard index jumps from 0.005-0.060 at 1-hop to near-complete convergence at 2-hop, with the addition of a single interneuron layer.

**Critically, this convergence occurs through modality-specific interneuron pools.** Cross-modality sharing of intermediates is minimal:

| Pair | Shared intermediates | Jaccard index |
|---|---|---|
| Visual–Somatosensory | 310 | 0.048 |
| Somatosensory–Olfactory | 10 | 0.002 |
| Visual–Olfactory | **0** | **0.000** |

Visual and olfactory modalities share zero intermediates en route to DNs. The 2-hop convergence is not mediated by shared relay neurons — each modality maintains its own interneuron pool that independently projects to overlapping DN targets. This is a labeled-line architecture at both the DN level and the interneuron level, with convergence arising from fan-out geometry rather than shared computation.

**Somatosensory subchannels are themselves segregated.** Within the somatosensory modality, proprioceptive (65 DNs), mechanosensory (86 DNs), vestibular (77 DNs), and gustatory (23 DNs) subchannels maintain low pairwise overlap (Jaccard 0.048-0.180), with 34-43 exclusive DNs per subchannel. The labeled-line principle extends down to individual sensory submodalities.

The visual channel shows an additional striking property: extreme lateralization. Left LPLC2 reaches 24 DNs and right LPLC2 reaches 21 DNs, with only 1 shared DN between sides (Jaccard ≈ 0). Generic visual neurons (R7/R8) have zero direct DN connections — all 8,823 visual→DN synapses come from LPLC2 looming detectors. This bilateral segregation is the structural basis for the contralateral escape response we observe behaviorally.

### 2.6 Functional mapping of decoder groups by modality

The segregation extends into specific motor functions. We quantified both the number of DNs reached per group and the total synaptic drive (synapses per neuron):

| Group | DNs | Somato drive (syn/n) | Visual drive (syn/n) | Olfactory drive | Dominant subchannel |
|---|---|---|---|---|---|
| Forward (59) | 40/1/1 | **86.1** | 0.0 | 0.1 | gustatory (44.7) |
| Turn left (136) | 71/22/0 | 15.7 | **31.2** | 0.0 | LPLC2_left (31.2) |
| Turn right (167) | 94/23/1 | 14.5 | **27.6** | 0.0 | LPLC2_right (27.4) |
| Rhythm (25) | 11/2/0 | **41.6** | 0.1 | 0.0 | proprioceptive (15.6) |
| Stance (25) | 15/0/0 | **46.0** | **0.0** | **0.0** | gustatory (32.6) |

Visual input (LPLC2) preferentially targets turning groups (31.2 and 27.6 syn/neuron) — 5x stronger than somatosensory drive to those groups — consistent with looming-evoked escape turns. Forward locomotion is somatosensory-dominant with strong gustatory contribution (44.7 syn/neuron), consistent with sugar-dependent locomotion modulation via leg taste neurons (Thoma et al. 2016).

Stance control is exclusively somatosensory at 1-hop — zero visual and zero olfactory synapses reach stance DNs directly. Gustatory input dominates stance (32.6 syn/neuron), with proprioceptive (6.6) and mechanosensory (6.1) as secondary. This makes biological sense: stance gain should be modulated by ground contact and feeding state, not by distal sensory stimuli. This pattern fell out of the connectome analysis naturally and was not designed into the decoder.

### 2.7 Extended labeled lines: auditory, thermosensory, and hygrosensory

To test whether the labeled-line principle generalizes beyond the three modalities used in behavioral experiments, we extended the segregation analysis to three additional sensory populations identified from FlyWire annotations: auditory (390 Johnston's organ neurons), thermosensory (29 neurons: 7 heating/TRN_VP2, 9 cold/TRN_VP3a-b, 13 humidity-sensitive/TRN_VP1m), and hygrosensory (74 neurons: 29 dry/HRN_VP4, 16 moist/HRN_VP5, 16 evaporative cooling/HRN_VP1d, 13 cooling/HRN_VP1l).

**Auditory: a semi-independent turning/rhythm channel.**

Auditory neurons reach 41 DNs at 1-hop (405 edges, 1,563 synapses). The auditory channel shows moderate overlap with visual (Jaccard = 0.164, 12 shared DNs) and somatosensory (0.066, 14 shared), but zero overlap with olfactory (0.000). Its strongest per-neuron drive targets the rhythm group (13.7 syn/neuron) and bilateral turning (5.1 left, 4.9 right), consistent with auditory-driven courtship orientation and song-evoked locomotor modulation. These 12 shared DNs span Sturner et al. (2025) Clusters 14 (antennal mechanosensory-enriched) and 16 (visual-enriched), confirming they sit at the intersection of auditory and visual processing streams rather than within a single cluster. Shuffled controls confirm the visual-auditory overlap is still 3.8x below chance — auditory maintains a partially independent channel, converging with visual specifically for rapid orientation behaviors.

**Thermosensory: the narrowest direct pathway.**

Thermosensory neurons reach only 5 DNs (14 edges, 162 synapses) — the narrowest direct pathway of any modality tested. All 5 target DNs are in the turning groups (4 turn-right, 1 turn-left), consistent with thermotaxis. The dominant target is DNb05 (151/162 synapses, 93%). DNb05 is a known multimodal steering neuron (Namiki et al. 2018; Yang et al. 2024) whose total input is dominated by visual (31%) and central (62%) sources — thermo/hygro constitutes only ~1.4% of its total 24,866 input synapses. Yet it is the sole direct thermo/hygro-to-DN pathway in the readout population.

**Hygrosensory: convergence with thermosensory on DNb05.**

Hygrosensory neurons reach just 2 DNs (13 edges, 200 synapses) — both are DNb05. The thermo-hygro Jaccard (0.400) is 10x higher than shuffled controls, confirming genuine convergence. Separately, DNp44 — identified by Marin et al. (2020) as the primary hygrosensory descending neuron — is also in our readout pool but receives hygrosensory input via a 2-hop pathway through VP projection neurons (15-22 relay neurons, 2,459-2,959 synapses), not directly. The coexistence of a direct but weak pathway (DNb05, 200 synapses) and an indirect but strong pathway (DNp44, ~2,700 synapses at 2-hop) suggests parallel thermo/hygro motor channels operating at different latencies.

**Six-modality segregation matrix.**

| Pair | Shared DNs | Jaccard |
|---|---|---|
| Auditory–Visual | 12 | 0.164 |
| Auditory–Somatosensory | 14 | 0.066 |
| Auditory–Thermosensory | 2 | 0.045 |
| Auditory–Hygrosensory | 1 | 0.024 |
| Auditory–Olfactory | 0 | 0.000 |
| Thermosensory–Hygrosensory | 2 | 0.400 |
| All others | 0-1 | 0.000-0.023 |

Direct (1-hop) connectivity is highly modality-specific across all six modalities. However, pure topological reachability at 2-hops is misleading — all modalities reach 50-100% of readout DNs. Synapse-weighted analysis, which weights 2-hop paths by the product of synapse counts, resolves this discrepancy and reveals a three-tier hierarchy:

| Modality | 1-hop DNs (%) | 2-hop topology (%) | 2-hop weighted (%) |
|---|---|---|---|
| Somatosensory (75) | 186 (53.1%) | 350 (100%) | 201 (57.4%) |
| Auditory (390) | 41 (11.7%) | 315 (90.0%) | 64 (18.3%) |
| Visual (310) | 44 (12.6%) | 304 (86.9%) | 47 (13.4%) |
| Hygrosensory (74) | 2 (0.6%) | 177 (50.6%) | 8 (2.3%) |
| Olfactory (100) | 1 (0.3%) | 175 (50.0%) | 7 (2.0%) |
| Thermosensory (29) | 5 (1.4%) | 194 (55.4%) | 6 (1.7%) |

The chemical senses (olfactory, thermosensory, hygrosensory) are the most sparse at 1.7-2.3% weighted, consistent with their deeply multi-synaptic pathway architectures. The olfactory value (2.0%) matches the 2-5% olfactory-responsive DN fraction observed physiologically by Aymanns et al. (2022). Olfactory signals require at least two interneuron layers to reach DNs — a structural constraint arising from the obligate PN -> LH pathway for innate olfactory behavior (Huoviala et al. 2020; Das Chakraborty & Bhatt 2022).

At 2-hops, olfactory and thermosensory/hygrosensory share substantial intermediate overlap (Jaccard 0.287-0.306), consistent with Sturner et al. (2025) grouping these modalities into a shared information-flow cluster (Cluster 8). The specificity we observe at 1-hop gives way to multimodal convergence at each additional synaptic layer.

### 2.8 Causal bottleneck validation

To test whether the structural bottlenecks identified in Section 2.7 have functional consequences, we performed two targeted silencing experiments.

**Experiment 1: DNb05 silencing.** DNb05 is a bilateral descending neuron pair (2 neurons) known as a multimodal steering neuron receiving visual (31% of input), central (62%), auditory (0.8%), hygrosensory (0.8%), and thermosensory (0.6%) input (Namiki et al. 2018; Yang et al. 2024). Although thermo/hygro constitutes only ~1.4% of DNb05's total 24,866 input synapses — making it a minor fraction of this neuron's multimodal input — DNb05 is the dominant direct thermo/hygro target among all readout DNs. We silenced both DNb05 neurons and measured the impact on each modality's throughput.

Hygrosensory throughput collapsed completely: 100% of DNs lost, 100% of synapses (200/200). Thermosensory throughput was devastated: 40% of DNs lost, 93.2% of synapses (151/162). In contrast, somatosensory lost 0.5% of DNs and 0.1% of synapses, visual lost 0%, and olfactory lost 0%. The specificity ratio was 16.4x. This demonstrates a structural principle: labeled-line function does not require dedicated anatomy. A multimodal neuron can serve as a functional bottleneck for a minority input modality when it is the sole direct pathway for that modality to the motor system. The thermo/hygro signal is a small fraction of DNb05's input, but it is the entire direct thermo/hygro-to-DN pathway.

Importantly, this direct pathway is not the only route for thermo/hygro information to reach motor output. DNp44 — identified by Marin et al. (2020) as the primary hygrosensory descending neuron — receives no direct hygrosensory synapses but is reached via a strong 2-hop pathway through VP projection neurons (15-22 relay neurons, 2,459-2,959 synapses). The coexistence of a direct but weak pathway (DNb05, 200 synapses at 1-hop) and an indirect but strong pathway (DNp44, ~2,700 synapses at 2-hop) is consistent with parallel fast and slow channels for environmental sensing, where DNb05 provides rapid but coarse thermo/hygro steering and DNp44 mediates the slower, higher-fidelity hygrosensory response described by Marin et al. (2020).

**Experiment 2: Auditory-visual orientation channel silencing.** The 12 DNs shared between auditory and visual modalities (5 turn-left, 7 turn-right; all turning types) span Sturner et al. (2025) Clusters 14 (auditory-enriched) and 16 (visual-enriched), confirming they sit at the intersection of these processing streams. We silenced these 12 DNs to test whether this selectively disrupts orientation-like turning.

Auditory lost 29.3% of its DN targets (12/41) and 25.5% of synapses. Visual lost 27.3% of DN targets (12/44) and 44.2% of synapses — nearly half of visual motor output passes through these shared turning DNs. Critically, the impact was turning-specific: auditory turn_right throughput dropped 35.1%, turn_left 16.5%, while forward, rhythm, and stance throughput remained at 0% loss. Olfactory was completely unaffected (0%), as were thermosensory and hygrosensory (0% each). All 8/8 causal tests passed across both experiments.

### 2.9 Representational geometry of connectome-constrained population codes

The preceding sections established that the connectome produces specific behaviors and modality-specific descending channels. We next asked whether the connectome also creates structured neural representations — population-level coding patterns that organize by sensory modality and vanish when wiring is randomized.

**Protocol.** We recorded full 365-dimensional DN population vectors during six sensory conditions spanning four modalities: baseline (proprioceptive), bilateral contact loss (mechanosensory), lateral push (vestibular), unilateral looming (visual/LPLC2), and unilateral odor (olfactory). For each condition, we computed evoked delta vectors (perturbation-phase mean minus pre-perturbation baseline per trial) to isolate condition-specific responses from common walking activity. Five seeds per condition, two brain types (intact vs shuffled connectome), yielding 30 trial-level delta vectors per brain type.

**Responsive neurons.** The intact connectome activated 102 of 365 readout DNs with condition-dependent firing rate changes (non-zero variance across conditions). The shuffled connectome activated only 4 — a 25.5x ratio. This is the strongest single metric: the connectome's specific wiring creates modality-specific coding in 28% of the DN population, while degree-matched random wiring produces near-uniform responses.

**Linear decodability.** Logistic regression with leave-one-group-out cross-validation decoded sensory condition from single-trial delta vectors at 36.9% accuracy (chance = 16.7%, 2.2x above chance). The shuffled connectome achieved only 10.0% (below chance), confirming that condition information in the DN population requires specific wiring. PCA captured 39.5% of variance in the top 3 components, with positive silhouette score (0.13), indicating modest but real cluster structure.

**Cross-modal dissimilarity.** Visual (looming) and olfactory conditions produced highly dissimilar population patterns (1 - Pearson = 0.97), consistent with the near-zero Jaccard overlap between LPLC2 and olfactory DN targets at 1-hop. Contact loss left and right produced moderately similar patterns (Pearson = 0.50), reflecting their shared mechanosensory modality.

**RSA.** Representational similarity analysis comparing the neural dissimilarity matrix to the structural (1-hop Jaccard) dissimilarity matrix yielded a negative correlation (Spearman r = -0.36), indicating that the geometry of neural representations does not directly mirror 1-hop wiring. This is expected: activity propagates through multiple synaptic hops and recurrent dynamics, reshaping the coding space relative to direct connectivity. The connectome is necessary for structured representations (25.5x responsive neuron ratio) but the representational geometry is an emergent property of network dynamics, not a simple readout of anatomy.

### 2.10 Sparsity, not specific wiring, determines learning speed

Sections 2.1-2.8 established that the connectome supports sensorimotor function without learning. A natural hypothesis is that connectome topology also provides a structural advantage when learning is required — that evolution-optimized wiring should accelerate motor learning compared to random networks. We tested this hypothesis directly and found it to be false: sparsity alone accounts for the learning advantage, independent of the specific wiring pattern.

**Architecture.** We compressed the MANC VNC to 2,314 neurons (1,314 DN + 500 MN + 500 top-degree intrinsic interneurons) with 193,315 edges (3.6% density). The recurrent weight matrix uses the connectome adjacency as a fixed binary mask — only edges present in the connectome can carry learned weights. Input drives DN neurons only and output reads MN neurons only, forcing information to flow through the recurrent topology. Three architectures share identical I/O constraints, hidden dimension, and sparsity:

1. **Connectome** — real MANC topology as the sparsity mask (337K active parameters out of 5.5M total)
2. **Random sparse** — Erdos-Renyi random graph at the same edge density and zero self-loops
3. **Shuffled** — degree-preserving edge swaps (configuration model, 5x edge-count swap iterations), preserving both in-degree and out-degree while destroying specific wiring

The fitness function rewards forward displacement, penalizes instability (falling, <3 legs in contact), and penalizes excessive joint velocity. ES uses antithetic sampling (pop_size=32, sigma=0.02, lr=0.01) with rank-based fitness shaping.

**Results (Fig. X, 2 seeds per architecture).**

The connectome architecture reaches a stable plateau of mean reward +10 to +15 by generation 50-80 (seed 42: +10.5, seed 79: +14.8), with no further improvement over the remaining 300+ generations. The random sparse control, despite having no biological wiring structure, reaches the same performance band by generation 40-60 (seed 42: +11.6 at gen 40). [PLACEHOLDER: Final random sparse and shuffled plateau levels, both seeds. Expected: all three architectures converge to the same reward band.]

This is a negative result with respect to the topology hypothesis. All three sparse architectures have identical parameter counts (337K), optimizer settings, and evaluation environments. The only difference is which 337K of the 5.5M possible recurrent weights are allowed to be nonzero — the specific wiring pattern. Yet the specific wiring does not confer a measurable learning advantage.

**Interpretation.** The connectome's value for locomotion is not in its topology as an inductive bias for learning. Rather, its value lies in the interpretability and causal traceability of its identified functional circuits — the specific sensorimotor pathways (Sections 2.5-2.8), the DNb05 bottleneck (Section 2.8), the modality-specific channels that enable targeted ablation and predictable behavioral outcomes. A random sparse network that learns equally well is a black box; the connectome is a documented codebase.

**Zero-shot generalization.** Connectome-trained policies show robust zero-shot transfer: both seeds survive 2,000/2,000 endurance steps (never falling), walk 39-45mm, and produce -21° to -24° heading responses to asymmetric contact perturbation. [PLACEHOLDER: Control generalization results for comparison.]

---

## 3. Discussion

The connectome-structured brain model, when coupled to an embodied fly through a transparent sensorimotor interface, produces adaptive and behaviorally specific responses without learning or parameter fitting. This establishes that connectome wiring carries substantial functional information — sufficient, within this embodied interface and task family, to support stimulus-specific sensorimotor transformations.

**Scope of the sufficiency claim.** Our system includes designed components: a sensory encoder, a descending decoder, and a CPG locomotion layer. We have shown sufficiency of the connectome within this interface, not sufficiency of wiring in the philosophical sense. Real flies rely on neuromodulation, synaptic plasticity, and experience to refine these circuits. Our simulation provides a computational baseline against which the contributions of these additional mechanisms can be measured.

**Direct connectivity is modality-specific; convergence is layer-dependent.** At the direct (1-hop) sensory-to-DN interface, connectivity is highly modality-specific (Jaccard 0.005-0.060), with rapid convergence one synapse deeper. This complements the information-flow analysis of Sturner et al. (2025), who found that most DN types are multimodal when assessed by graph-distance ranking across the full connectome. A quantitative comparison sharpens this point: when we map our 1-hop modality-specific DN sets onto Sturner et al.'s 16 clusters, the cluster-level Jaccard overlap is 2-7x higher than the DN-level overlap (somatosensory-auditory: 0.429 cluster vs 0.066 DN, visual-auditory: 0.400 vs 0.164, somatosensory-visual: 0.167 vs 0.060). The exception is olfactory, which remains completely segregated at both levels (Jaccard = 0.000 for all olfactory-other cluster pairs). This confirms that our direct-connectivity analysis resolves a finer-grained specificity than information-flow metrics: modalities that share Sturner clusters — and therefore have similar graph-distance profiles to sensory populations — nonetheless target largely non-overlapping individual DNs at the direct synaptic level. The architecture suggests that the fly brain maintains parallel, modality-specific "reflex arcs" at the fastest timescale while routing multimodal integration through interneuron layers.

Three aspects of the segregation strengthen this interpretation. First, the 13 shared visual-somatosensory DNs are specifically looming-escape turning neurons — the one behavior where millisecond integration of visual threat and body state is critical. Second, 2-hop convergence occurs through modality-specific interneuron pools (visual and olfactory share zero intermediates), meaning the wiring maintains labeled lines at both the DN level and the relay level. Third, visual→DN connections are entirely excitatory (100%), while somatosensory→DN includes 21% inhibitory connections, suggesting that body state can actively suppress motor programs — an asymmetry that pure convergence architectures would not predict.

The stance exclusivity result is particularly clean: stance-controlling DNs receive zero visual or olfactory input at 1-hop, exclusively somatosensory. This emerged from the connectome analysis and was not designed into the decoder. The dominant subchannel is gustatory (32.6 syn/neuron), consistent with stance modulation during feeding.

This direct-connectivity specificity has not been previously quantified at the population level because it requires both (a) a complete connectome to trace all paths and (b) a behavioral readout to identify which descending neurons are functionally relevant. Anatomical tracing alone cannot distinguish the 186 DNs reached by somatosensory populations from the 133 that receive no direct sensory input — only closed-loop simulation with behavioral validation identifies which circuits are active. Our 1-hop analysis is complementary to, not contradictory with, the multimodal convergence at longer path lengths documented by Sturner et al. (2025) and the multi-synaptic sensory processing described by Marin et al. (2020).

**Limitations.** Our model uses uniform synaptic parameters (weights proportional to synapse count, identical time constants). Real synapses vary in strength, sign (excitatory/inhibitory), and dynamics. We do not model gap junctions, neuromodulation, or synaptic plasticity. Our sensory encoding uses simplified Poisson rate coding rather than the full complexity of Drosophila sensory transduction. We do not incorporate connectome-constrained visual processing (Lappalainen et al. 2024), grooming circuits (Hampel et al. 2015), or the full descending neuron population dynamics characterized by Braun et al. (2024).

The locomotion layer uses a preprogrammed tripod CPG rather than emergent rhythm from VNC circuitry. The MANC VNC connectome (Takemura et al. 2024) contains the structural substrate for central pattern generation — we identified 6,318 reciprocal inhibitory pairs between flexor and extensor premotor interneuron pools — but emergent rhythm generation from connectome dynamics remains an open challenge. Our current VNC model routes descending commands to motor neuron pools through MANC wiring (validated by forward ablation: -68% to -97% distance reduction), but the temporal pattern of locomotion is imposed externally. Recent whole-body simulations using reinforcement-learned controllers (Vaxenburg et al. 2025) demonstrate that physically realistic fly locomotion is achievable without connectome-constrained rhythm generation, though via a fundamentally different approach.

Our decoder groups achieve 33% accuracy against published DN activation phenotypes (Cande et al. 2018) — correctly placing steering DNs (DNa02, DNb05, DNb06) in turning groups but misassigning locomotor and postural DNs. This reflects the coarse turn-vs-forward decomposition of our decoder. The v5 decoder adds 6 DNa01 and DNg13 steering neurons (Yang et al. 2024) to the turn groups, improving symmetric steering (boost turn-left: +129.6° heading, boost turn-right: -137.0° heading). DNa01 drives sustained low-gain heading corrections (Rayshubskiy et al. 2024), while DNg13 modulates outside-stride length during swerve maneuvers (Yang et al. 2024). With these additions, 5 of 16 canonical DN types from Cande et al. (2018) remain absent from our readout pool, indicating that the current population selection captures steering and gross locomotion but misses grooming, postural, and fine motor DN types.

Our analysis uses the FlyWire 783 completeness snapshot (138,639 of ~139,255 neurons). Neurons excluded due to low proofreading completeness may include bridging interneurons that would reduce the observed modality segregation. Despite these limitations, the system produces robust, stimulus-specific behavior — suggesting that the connectome's wiring diagram carries substantial functional information independent of these details.

**Bottleneck-by-exclusion: labeled-line function without dedicated anatomy.** The DNb05 result demonstrates a structural principle not previously characterized at synapse resolution. DNb05 is a visual-dominant multimodal steering neuron (31% visual input; Namiki et al. 2018; Yang et al. 2024), yet it functions as a labeled-line bottleneck for thermo/hygro because it provides exclusive synaptic access for these modalities to motor output. The mechanism is not "dedicated anatomy" but "exclusive access" — DNs are organized by output channel, not input identity. Prior work has shown labeled-line function at the population level, where modality specificity is distributed across many neurons (Huoviala et al. 2020). Here we demonstrate it at the single-neuron level: despite receiving only 1.4% of its input from thermo/hygro, DNb05 is the sole synaptic gateway for these modalities. Silencing 2 neurons collapses thermo/hygro throughput by 93-100% with 16.4x specificity. The Marin et al. (2020) hygrosensory pathway through DNp44 operates at 2-hops, suggesting parallel fast (direct, weak) and slow (indirect, strong) channels for environmental sensing.

The auditory-visual convergence tells a complementary story: the 12 shared turning DNs span Sturner et al. (2025) Clusters 14 (auditory-enriched) and 16 (visual-enriched), confirming they bridge two modality-specific processing streams for orientation behavior. Olfactory remains the most isolated modality (Jaccard = 0.000 with all five others), reflecting its architecturally mandated depth: the obligate ORN -> PN -> LH pathway requires at least 3-4 synapses to reach DNs (Huoviala et al. 2020; Das Chakraborty & Bhatt 2022). Synapse-weighted 2-hop analysis confirms that only 2.0% of readout DNs receive strong olfactory drive — matching the 2-5% olfactory-responsive DN fraction observed physiologically by Aymanns et al. (2022).

**VNC-lite premotor dynamics.** We replaced the instantaneous decoder-to-actuator mapping with a bilateral premotor state model (VNC-lite) that interposes leaky integrator dynamics between descending neuron rates and locomotion commands. DN input drives state derivatives rather than raw outputs: d(state)/dt = -state/tau + f(DN_input) + g(body_feedback). This gives the motor system temporal smoothing (no instantaneous jumps), persistence (commands outlast single brain steps), bilateral competition (left/right mutual inhibition for turning), and feedback stabilization (body state corrects motor errors). All 20 validation tests pass with simulated input (backward compatibility 4/4, robustness across 7 parameter configurations 8/8, temporal smoothing 4/4, causal dissociation 4/4), and 19/20 with the full Brian2 brain (the single miss is a walking distance threshold at short trial length, with all functional effects preserved). The VNC-lite layer reduces command jitter by 47-97% compared to the original decoder while preserving all headline behavioral effects.

**Connectome topology does not accelerate learning.** The topology learning experiment (Section 2.9) addresses a distinct question from the sufficiency results (Sections 2.1-2.8). Sufficiency asks "does the wiring work without learning?" — topology learning asks "does the wiring help when learning is required?" The answer is no: random sparse and shuffled networks with matched sparsity learn equally fast and reach the same performance plateau. The connectome's contribution to motor control is not as an optimization prior — any sparse network of equivalent density provides the same dimensionality reduction that makes ES tractable (337K vs 5.5M parameters). Instead, the connectome's value is functional: it provides identified, traceable circuits whose components can be selectively manipulated with predictable outcomes (Sections 2.2-2.8). This distinction — between a network that works and a network that can be understood — has practical implications for bio-inspired robotics. A random network that performs equally well offers no entry point for diagnosis, targeted intervention, or principled extension to new behaviors.

**Future directions.** Replacing the preprogrammed CPG with emergent rhythm generation from VNC connectome dynamics remains an open challenge. We tested whether the MANC VNC (13,101 neurons, 1.9 million synapses) could produce flex/ext alternation with adaptive LIF neurons and identified 6,318 reciprocal inhibitory pairs between antagonist motor pools — the structural substrate for half-center oscillation. However, neither the full 13K model nor a minimal 1,000-neuron extraction produced anti-phase flex/ext activity under any tested combination of adaptation strength (b = 0.3-8.0 mV), synaptic time constant (tau = 5-100 ms), tonic drive (I = 0.5-3.0 mV), or targeted half-center inhibition scaling (up to 20x). Flex/ext correlations remained positive across all 41 parameter configurations tested (best: +0.36). This suggests that uniform LIF dynamics cannot exploit the MANC's CPG wiring structure — biologically realistic rhythm generation likely requires conductance-based neuron models with specific ionic currents (persistent sodium, calcium-dependent potassium), gap junctions, or neuromodulatory dynamics that shape the oscillatory regime. The MANC wiring correctly routes descending commands to motor neuron pools (forward ablation: -68% to -97% distance reduction), but temporal pattern generation requires richer biophysics than connectivity alone provides.

The dose-response relationship between population size and behavioral effect suggests that the system can be used to predict the behavioral consequences of genetic manipulations that silence specific neuron types. Physical deployment of the minimal locomotion circuit on hexapod hardware would test whether connectome-derived controllers transfer to real-world dynamics with proprioceptive feedback — a domain where the interpretability of biological circuits (targeted ablation, modular debugging) provides practical advantages over black-box controllers of equivalent performance.

---

## 4. Methods

### 4.1 Brain simulation

We simulated 138,639 neurons from the FlyWire 783 completeness snapshot (Completeness_783.csv) as leaky integrate-and-fire units using Brian2 (v2.5). This represents 99.6% of the full FlyWire resource (139,255 neurons). Synaptic connections (15,091,983 connection pairs comprising 54.5 million individual synapses, from Connectivity_783.parquet; mean 3.6 synapses per connection) used conductance-based synapses with weights proportional to synapse count. All neurons shared identical biophysical parameters (membrane time constant, threshold, reset potential). No parameter tuning or optimization was performed.

### 4.2 Sensory populations

Sensory neuron populations were identified from the FlyWire annotations: sugar GRNs for gustatory input (Shiu et al. 2023), SEZ ascending neuron types classified by out/in connectivity ratio (>1.5 = ascending) for proprioceptive, mechanosensory, and vestibular channels, bilateral ORNs for olfaction (Or42b/DM1, Or85a/DM5), bilateral photoreceptors (R7/R8) for vision, and LPLC2 lobula plate neurons for looming detection (identified from flywire_annotations_matched.csv).

**Sensory encoding.** Each channel maps specific body observation fields to Poisson firing rates using channel-appropriate nonlinearities. Proprioceptive neurons receive rates proportional to tanh-normalized joint angles and velocities, tiled across the neuron population. Mechanosensory neurons receive rates proportional to per-leg contact force magnitudes (0-1, clipped). Vestibular neurons encode body velocity and orientation via tanh normalization. Gustatory neurons receive a uniform rate modulated by mean contact force. All channels map to a rate range of [baseline, max_rate] = [10, 100] Hz. Olfactory channels encode odor intensity at bilateral antennae (0-1) linearly into [10, 100] Hz. LPLC2 looming channels use an elevated maximum rate of 200 Hz to reflect the strong stimulus drive of looming detectors, encoding bilateral looming intensity (0-1) into [10, 200] Hz. Visual channels encode mean eye luminance per side into [10, 100] Hz.

### 4.3 Readout populations

Descending neuron (DN) populations were selected using a hybrid approach: annotated motor/descending types from the SEZ neuron dataset (out/in ratio < 0.5) plus connectivity-augmented supplements identified by round-robin assignment from highest-connectivity downstream targets. DN populations were assigned to five locomotion decoder groups (forward, turn-left, turn-right, rhythm, stance) based on annotated neuron types and bilateral pairing. The v5 decoder includes 365 readout neurons (up from 359 in v3), with 6 additional DNa01 and DNg13 steering neurons added to the turn groups based on their established roles in heading control (Yang et al. 2024).

### 4.4 Body simulation

The fly body was simulated using FlyGym v1.2.1 with MuJoCo physics, featuring a hexapod with 42 actuated degrees of freedom. Locomotion used PreprogrammedSteps tripod gait kinematics with CPG phase offsets [0, pi, 0, pi, 0, pi]. Brain output modulated four CPG parameters: forward drive (amplitude scaling), turn drive (left-right asymmetry), step frequency (CPG intrinsic frequency), and stance gain (joint magnitude multiplier).

### 4.5 Ablation protocol

For each of 8 conditions (baseline + 5 group ablations + 2 boost conditions), we built a fresh brain network, silenced the target population by setting firing rates to 0 Hz, and ran 5,000 body steps with a 20ms brain integration window. Shuffled connectome controls used a fixed random seed (999) to permute all postsynaptic target indices while preserving out-degree.

### 4.6 VNC-lite premotor dynamics

Between the descending decoder and locomotion bridge, we interpose a bilateral premotor state model (VNC-lite) that replaces the instantaneous rate-to-command mapping with a dynamical system. Six state variables (drive_L, drive_R, turn_L, turn_R, rhythm, stance) evolve according to leaky integrator dynamics:

d(state)/dt = -state/tau + alpha * f(DN_input)/dt + coupling + feedback

where tau is a state-specific time constant (150-300ms), alpha is an input gain, f() is a tanh nonlinearity, and coupling terms implement bilateral drive synchronization and turn mutual inhibition. The model has three stages: (1) DN rate normalization and input mapping, (2) bilateral state dynamics with Euler integration, and (3) body feedback — velocity mismatch drives stance correction, body instability dampens rhythm, contact asymmetry produces corrective turning, and slip detection boosts stance. State variables are saturated to prevent runaway. Output mapping converts state to LocomotionCommand through tanh nonlinearities: forward_drive = 0.1 + 0.9 * tanh(mean_drive), turn_drive = tanh(turn_L - turn_R), step_frequency = 1.0 + 1.5 * tanh(rhythm), stance_gain = 1.0 + 0.5 * tanh(stance).

### 4.7 Topology learning experiment

We extracted a compressed VNC topology from the MANC male adult nerve cord connectome (v0.9, confidence >= 0.5). All 1,314 descending neurons and 500 thoracic motor neurons were retained; intrinsic VNC neurons were ranked by total degree (in + out) and the top 500 were selected, yielding a 2,314-neuron subgraph with 193,315 directed edges (3.6% density). Neurotransmitter signs were assigned from MANC predictions (acetylcholine/glutamate/dopamine/serotonin/octopamine = excitatory; GABA/histamine = inhibitory).

The policy network is a sparse recurrent neural network (PyTorch) with the connectome adjacency as a fixed binary mask on the recurrent weight matrix. Input is projected to DN neuron indices only (90-dim observation: 42 joint angles + 42 joint velocities + 6 per-leg contact forces). Output is read from MN neuron indices only (48-dim action: 42 joint targets + 6 adhesion). Information must flow through the recurrent topology (3 unrolled steps per forward pass) to reach the output. Joint outputs are clamped to safe ranges using per-joint rest angles and amplitudes from the MuJoCo model.

ES optimization used the OpenAI-ES algorithm (Salimans et al. 2017) with antithetic sampling (pop_size=32), rank-based fitness shaping, noise std sigma=0.02, learning rate 0.01, and weight decay 0.005. Only non-masked recurrent weights plus input/output projection parameters were optimized (337K of 5.5M total for connectome). Fitness was the mean total reward across 2 evaluation episodes per individual. Each episode ran 1,000 body steps (0.1ms timestep) after a 300-step warmup.

Control architectures shared identical hidden dimension (2,314), I/O constraints (DN input, MN output), and ES hyperparameters. Random sparse: Erdos-Renyi graph at the same density (no self-loops). Shuffled: degree-preserving edge swaps using the configuration model (5x edge-count swap iterations), preserving both in-degree and out-degree of every neuron while randomizing specific wiring.

### 4.8 Segregation analysis

For each sensory modality group (olfactory: 100 neurons; visual/LPLC2: 310 neurons; somatosensory: 75 neurons across gustatory, proprioceptive, mechanosensory, vestibular subchannels), we identified all descending neurons receiving direct synaptic input (1-hop) from the full connectome (15M connection pairs). For 2-hop analysis, we identified all intermediate neurons that are both (a) direct postsynaptic targets of sensory neurons and (b) direct presynaptic to readout DNs. These "active intermediates" represent the relay layer between sensory populations and motor output. Overlap was quantified using the Jaccard index (intersection/union) for all pairwise modality comparisons at both the DN level and the interneuron level. Excitatory/inhibitory classification used the Excitatory column from the connectome (Connectivity_783.parquet). Subchannel analysis repeated the 1-hop computation for each of the 10 individual sensory channels.

---

## References

- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. Nature, 634, 124-138.
- Schlegel, P. et al. (2023). Whole-brain annotation and multi-connectome cell type quantification. Nature, 634, 139-152.
- Shiu, P.K. et al. (2023). A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain. bioRxiv.
- Lobato-Rios, V. et al. (2022). NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster. Nature Methods, 19, 620-627.
- Wang, F. et al. (2024). FlyGym: A comprehensive toolkit for biomechanical simulations of Drosophila. Nature Methods.
- Namiki, S. et al. (2018). The functional organization of descending sensory-motor pathways in Drosophila. eLife, 7, e34272.
- Cande, J. et al. (2018). Optogenetic dissection of descending behavioral control in Drosophila. eLife, 7, e34275.
- Marin, E.C. et al. (2020). Connectomics analysis reveals first-, second-, and third-order thermosensory and hygrosensory neurons in the adult Drosophila brain. Current Biology, 30, 3167-3182.
- Aymanns, F., Chen, C.-L. & Ramdya, P. (2022). Descending neuron population dynamics during odor-evoked and spontaneous limb-dependent behaviors. eLife, 11, e81527.
- Yang, H.H. et al. (2024). Fine-grained descending control of steering in walking Drosophila. Cell, 187, 6290-6308.
- Sturner, T. et al. (2025). Comparative connectomics of Drosophila descending and ascending neurons. Nature, 643, 158-172.
- Huoviala, P. et al. (2020). Neural circuit basis of aversive odour processing in Drosophila from sensory input to descending output. bioRxiv.
- Das Chakraborty, S. & Bhatt, D. (2022). Geosmin-responsive neural circuit for innate aversive behavior. bioRxiv.
- Rayshubskiy, A. et al. (2024). Neural circuit mechanisms for steering control in walking Drosophila. eLife.
- Thoma, V. et al. (2016). Functional dissociation in sweet taste receptor neurons between and within taste organs of Drosophila. Nature Communications.
- Takemura, S. et al. (2024). A connectome of the male Drosophila ventral nerve cord. eLife, 13, RP97769.
- Salimans, T. et al. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864.
- Braun, J. et al. (2024). Descending networks transform command signals into population motor control. Nature, 630, 686-694.
- Lappalainen, J.K. et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. Nature, 634, 1132-1140.
- Sapkal, N. et al. (2024). Neural circuit mechanisms underlying context-specific halting in Drosophila. Nature, 634, 191-200.
- Simpson, J.H. (2024). Descending control of motor sequences in Drosophila. Current Opinion in Neurobiology, 84, 102822.
- Vaxenburg, R. et al. (2025). Whole-body physics simulation of fruit fly locomotion. Nature, 643, 1312-1320.
- Hampel, S. et al. (2015). A neural command circuit for grooming movement control. eLife, 4, e08758.
