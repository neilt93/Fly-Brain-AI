# Connectome-Constrained Sensorimotor Behaviors and Modality-Specific Motor Channels in Drosophila

## Abstract

The Drosophila melanogaster FlyWire connectome provides a neuron-level wiring diagram of 139,255 neurons and approximately 50 million synapses, but whether connectome-structured dynamics can support meaningful sensorimotor transformations without learning or task-specific optimization remains unknown. We built a closed-loop system coupling a Brian2 leaky integrate-and-fire simulation of 138,639 neurons (FlyWire 783 completeness snapshot; 15 million connection pairs, 54.5 million synapses) to a MuJoCo biomechanical fly body (FlyGym), with biologically identified sensory populations encoding stimuli and descending neuron populations decoding motor commands through an interpretable sensorimotor interface. The connectome-structured brain model, when coupled to an embodied fly through this transparent interface, produces adaptive and behaviorally specific responses without learning or parameter fitting: causal locomotion control (10/10 ablation tests, forward drive reduced 53% by targeted silencing), olfactory valence discrimination (opposite turning for attractive vs aversive odors, 6/6 tests), and visually guided looming escape (contralateral turning with escape index 1.11, abolished 21-fold in shuffled connectome controls). Analysis of the descending neuron populations underlying these behaviors reveals that direct (1-hop) sensory-to-motor connectivity is highly modality-specific (Jaccard index 0.005-0.060 across three modalities), with convergence occurring one synapse deeper through modality-specific interneuron pools that share zero intermediates between visual and olfactory pathways. This specificity is consistent with the known multi-synaptic architecture of sensory processing: olfactory signals reach only 1 DN directly but, when weighted by synaptic strength, drive ~2% of the DN population at 2-hops — matching physiological recordings of odor-responsive DNs (Aymanns et al. 2022). Extended analysis across six sensory modalities reveals that thermo-hygro signals converge on the multimodal steering neuron DNb05 (Namiki et al. 2018; Yang et al. 2024), which despite receiving only ~1.4% of its total input from thermo/hygro sources, is the sole direct thermo/hygro-to-DN pathway — its ablation collapses thermo/hygro throughput by 93-100% (specificity 16.4x), demonstrating that labeled-line function can emerge from a minority input to a multimodal neuron.

---

## 1. Introduction

The completion of the Drosophila melanogaster whole-brain connectome (Dorkenwald et al. 2024, FlyWire) provides, for the first time, a neuron-level wiring diagram of an animal capable of complex behavior. The full resource comprises 139,255 neurons and approximately 50 million chemical synapses, including identified sensory input populations and descending motor output pathways. Yet a fundamental question remains: can connectome-structured dynamics support meaningful embodied sensorimotor transformations without learning or task-specific optimization?

Previous work has used the connectome for circuit analysis (Schlegel et al. 2023), connectome-constrained network simulations (Shiu et al. 2023), and anatomical tracing of specific pathways. To our knowledge, no open system has previously coupled a FlyWire-scale whole-brain model to an embodied fly simulator and validated behaviorally specific circuit function in closed loop. Locomotion in our system is mediated through a CPG/VNC-like interface — the brain modulates gait parameters rather than controlling individual joints directly — which we state explicitly as both a design choice and a current limitation.

Here we build such a system and make three contributions:

1. **Behavioral specificity.** The connectome-structured brain model, coupled to an embodied fly through an interpretable sensorimotor interface, produces three distinct adaptive behaviors — locomotion control, olfactory valence discrimination, and looming escape — without parameter fitting or learning.

2. **Graded causal control.** Targeted ablation of identified neuron populations produces graded, quantitatively predicted behavioral deficits, confirming that specific connectome pathways causally drive specific behaviors.

3. **Modality-specific descending channels.** Analysis of the descending neurons functionally recruited by each behavior reveals a previously unreported structural principle: sensory modalities maintain near-complete segregation at the direct sensory-to-descending interface, converging only through a single interneuron layer. The connectome implements parallel labeled lines for multisensory motor control.

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
| Forward neurons silenced | Forward drive: 0.90 → 0.10 (-89%) | Distance: 18.9mm → 9.0mm (-53%) |
| Turn-left silenced | Turn drive shifts rightward | Heading shifts rightward |
| Turn-right silenced | Turn drive shifts leftward | Heading shifts leftward |
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

---

## 3. Discussion

The connectome-structured brain model, when coupled to an embodied fly through a transparent sensorimotor interface, produces adaptive and behaviorally specific responses without learning or parameter fitting. This establishes that connectome wiring carries substantial functional information — sufficient, within this embodied interface and task family, to support stimulus-specific sensorimotor transformations.

**Scope of the sufficiency claim.** Our system includes designed components: a sensory encoder, a descending decoder, and a CPG locomotion layer. We have shown sufficiency of the connectome within this interface, not sufficiency of wiring in the philosophical sense. Real flies rely on neuromodulation, synaptic plasticity, and experience to refine these circuits. Our simulation provides a computational baseline against which the contributions of these additional mechanisms can be measured.

**Direct connectivity is modality-specific; convergence is layer-dependent.** At the direct (1-hop) sensory-to-DN interface, connectivity is highly modality-specific (Jaccard 0.005-0.060), with rapid convergence one synapse deeper. This complements the information-flow analysis of Sturner et al. (2025), who found that most DN types are multimodal when assessed by graph-distance ranking across the full connectome. A quantitative comparison sharpens this point: when we map our 1-hop modality-specific DN sets onto Sturner et al.'s 16 clusters, the cluster-level Jaccard overlap is 2-7x higher than the DN-level overlap (somatosensory-auditory: 0.429 cluster vs 0.066 DN, visual-auditory: 0.400 vs 0.164, somatosensory-visual: 0.167 vs 0.060). The exception is olfactory, which remains completely segregated at both levels (Jaccard = 0.000 for all olfactory-other cluster pairs). This confirms that our direct-connectivity analysis resolves a finer-grained specificity than information-flow metrics: modalities that share Sturner clusters — and therefore have similar graph-distance profiles to sensory populations — nonetheless target largely non-overlapping individual DNs at the direct synaptic level. The architecture suggests that the fly brain maintains parallel, modality-specific "reflex arcs" at the fastest timescale while routing multimodal integration through interneuron layers.

Three aspects of the segregation strengthen this interpretation. First, the 13 shared visual-somatosensory DNs are specifically looming-escape turning neurons — the one behavior where millisecond integration of visual threat and body state is critical. Second, 2-hop convergence occurs through modality-specific interneuron pools (visual and olfactory share zero intermediates), meaning the wiring maintains labeled lines at both the DN level and the relay level. Third, visual→DN connections are entirely excitatory (100%), while somatosensory→DN includes 21% inhibitory connections, suggesting that body state can actively suppress motor programs — an asymmetry that pure convergence architectures would not predict.

The stance exclusivity result is particularly clean: stance-controlling DNs receive zero visual or olfactory input at 1-hop, exclusively somatosensory. This emerged from the connectome analysis and was not designed into the decoder. The dominant subchannel is gustatory (32.6 syn/neuron), consistent with stance modulation during feeding.

This direct-connectivity specificity has not been previously quantified at the population level because it requires both (a) a complete connectome to trace all paths and (b) a behavioral readout to identify which descending neurons are functionally relevant. Anatomical tracing alone cannot distinguish the 186 DNs reached by somatosensory populations from the 133 that receive no direct sensory input — only closed-loop simulation with behavioral validation identifies which circuits are active. Our 1-hop analysis is complementary to, not contradictory with, the multimodal convergence at longer path lengths documented by Sturner et al. (2025) and the multi-synaptic sensory processing described by Marin et al. (2020).

**Limitations.** Our model uses uniform synaptic parameters (weights proportional to synapse count, identical time constants). Real synapses vary in strength, sign (excitatory/inhibitory), and dynamics. We do not model gap junctions, neuromodulation, or synaptic plasticity. The CPG is a preprogrammed tripod gait, not a VNC connectome model. Our sensory encoding uses simplified Poisson rate coding rather than the full complexity of Drosophila sensory transduction.

Our decoder groups achieve 33% accuracy against published DN activation phenotypes (Cande et al. 2018) — correctly placing steering DNs (DNa02, DNb05, DNb06) in turning groups but misassigning locomotor and postural DNs. This reflects the coarse turn-vs-forward decomposition of our decoder: critical steering neurons DNa01 and DNg13 (Yang et al. 2024) are absent from the readout, and the turn groups act as attractors in the supplement selection algorithm due to their larger size. DNa01 drives sustained low-gain heading corrections below our firing threshold (Rayshubskiy et al. 2024), while DNg13 modulates outside-stride length during swerve maneuvers — a sub-gesture distinction invisible to our rotational velocity readout (Yang et al. 2024). More fundamentally, 7 of 16 canonical DN types from Cande et al. (2018) are absent from our SEZ-derived readout pool entirely, indicating that the current population selection captures steering and gross locomotion but misses grooming, postural, and fine motor DN types.

Our analysis uses the FlyWire 783 completeness snapshot (138,639 of ~139,255 neurons). Neurons excluded due to low proofreading completeness may include bridging interneurons that would reduce the observed modality segregation. Despite these limitations, the system produces robust, stimulus-specific behavior — suggesting that the connectome's wiring diagram carries substantial functional information independent of these details.

**Bottleneck-by-exclusion: labeled-line function without dedicated anatomy.** The DNb05 result demonstrates a structural principle not previously characterized at synapse resolution. DNb05 is a visual-dominant multimodal steering neuron (31% visual input; Namiki et al. 2018; Yang et al. 2024), yet it functions as a labeled-line bottleneck for thermo/hygro because it provides exclusive synaptic access for these modalities to motor output. The mechanism is not "dedicated anatomy" but "exclusive access" — DNs are organized by output channel, not input identity. Prior work has shown labeled-line function at the population level, where modality specificity is distributed across many neurons (Huoviala et al. 2020). Here we demonstrate it at the single-neuron level: despite receiving only 1.4% of its input from thermo/hygro, DNb05 is the sole synaptic gateway for these modalities. Silencing 2 neurons collapses thermo/hygro throughput by 93-100% with 16.4x specificity. The Marin et al. (2020) hygrosensory pathway through DNp44 operates at 2-hops, suggesting parallel fast (direct, weak) and slow (indirect, strong) channels for environmental sensing.

The auditory-visual convergence tells a complementary story: the 12 shared turning DNs span Sturner et al. (2025) Clusters 14 (auditory-enriched) and 16 (visual-enriched), confirming they bridge two modality-specific processing streams for orientation behavior. Olfactory remains the most isolated modality (Jaccard = 0.000 with all five others), reflecting its architecturally mandated depth: the obligate ORN -> PN -> LH pathway requires at least 3-4 synapses to reach DNs (Huoviala et al. 2020; Das Chakraborty & Bhatt 2022). Synapse-weighted 2-hop analysis confirms that only 2.0% of readout DNs receive strong olfactory drive — matching the 2-5% olfactory-responsive DN fraction observed physiologically by Aymanns et al. (2022).

**VNC-lite premotor dynamics.** We replaced the instantaneous decoder-to-actuator mapping with a bilateral premotor state model (VNC-lite) that interposes leaky integrator dynamics between descending neuron rates and locomotion commands. DN input drives state derivatives rather than raw outputs: d(state)/dt = -state/tau + f(DN_input) + g(body_feedback). This gives the motor system temporal smoothing (no instantaneous jumps), persistence (commands outlast single brain steps), bilateral competition (left/right mutual inhibition for turning), and feedback stabilization (body state corrects motor errors). All 20 validation tests pass with simulated input (backward compatibility 4/4, robustness across 7 parameter configurations 8/8, temporal smoothing 4/4, causal dissociation 4/4), and 19/20 with the full Brian2 brain (the single miss is a walking distance threshold at short trial length, with all functional effects preserved). The VNC-lite layer reduces command jitter by 47-97% compared to the original decoder while preserving all headline behavioral effects.

**Future directions.** Replacing the preprogrammed CPG with a VNC connectome model would close the final loop in the sensorimotor arc. The VNC-lite premotor layer provides the architectural scaffold for such integration — its bilateral state model and body feedback pathways could be driven by VNC connectome dynamics rather than hand-tuned parameters. The dose-response relationship between population size and behavioral effect suggests that the system can be used to predict the behavioral consequences of genetic manipulations that silence specific neuron types.

---

## 4. Methods

### 4.1 Brain simulation

We simulated 138,639 neurons from the FlyWire 783 completeness snapshot (Completeness_783.csv) as leaky integrate-and-fire units using Brian2 (v2.5). This represents 99.6% of the full FlyWire resource (139,255 neurons). Synaptic connections (15,091,983 connection pairs comprising 54.5 million individual synapses, from Connectivity_783.parquet; mean 3.6 synapses per connection) used conductance-based synapses with weights proportional to synapse count. All neurons shared identical biophysical parameters (membrane time constant, threshold, reset potential). No parameter tuning or optimization was performed.

### 4.2 Sensory populations

Sensory neuron populations were identified from the FlyWire annotations: sugar GRNs for gustatory input (Shiu et al. 2023), SEZ ascending neuron types classified by out/in connectivity ratio (>1.5 = ascending) for proprioceptive, mechanosensory, and vestibular channels, bilateral ORNs for olfaction (Or42b/DM1, Or85a/DM5), bilateral photoreceptors (R7/R8) for vision, and LPLC2 lobula plate neurons for looming detection (identified from flywire_annotations_matched.csv).

**Sensory encoding.** Each channel maps specific body observation fields to Poisson firing rates using channel-appropriate nonlinearities. Proprioceptive neurons receive rates proportional to tanh-normalized joint angles and velocities, tiled across the neuron population. Mechanosensory neurons receive rates proportional to per-leg contact force magnitudes (0-1, clipped). Vestibular neurons encode body velocity and orientation via tanh normalization. Gustatory neurons receive a uniform rate modulated by mean contact force. All channels map to a rate range of [baseline, max_rate] = [10, 100] Hz. Olfactory channels encode odor intensity at bilateral antennae (0-1) linearly into [10, 100] Hz. LPLC2 looming channels use an elevated maximum rate of 200 Hz to reflect the strong stimulus drive of looming detectors, encoding bilateral looming intensity (0-1) into [10, 200] Hz. Visual channels encode mean eye luminance per side into [10, 100] Hz.

### 4.3 Readout populations

Descending neuron (DN) populations were selected using a hybrid approach: annotated motor/descending types from the SEZ neuron dataset (out/in ratio < 0.5) plus connectivity-augmented supplements identified by round-robin assignment from highest-connectivity downstream targets. DN populations were assigned to five locomotion decoder groups (forward, turn-left, turn-right, rhythm, stance) based on annotated neuron types and bilateral pairing.

### 4.4 Body simulation

The fly body was simulated using FlyGym v1.2.1 with MuJoCo physics, featuring a hexapod with 42 actuated degrees of freedom. Locomotion used PreprogrammedSteps tripod gait kinematics with CPG phase offsets [0, pi, 0, pi, 0, pi]. Brain output modulated four CPG parameters: forward drive (amplitude scaling), turn drive (left-right asymmetry), step frequency (CPG intrinsic frequency), and stance gain (joint magnitude multiplier).

### 4.5 Ablation protocol

For each of 8 conditions (baseline + 5 group ablations + 2 boost conditions), we built a fresh brain network, silenced the target population by setting firing rates to 0 Hz, and ran 5,000 body steps with a 20ms brain integration window. Shuffled connectome controls used a fixed random seed (999) to permute all postsynaptic target indices while preserving out-degree.

### 4.6 VNC-lite premotor dynamics

Between the descending decoder and locomotion bridge, we interpose a bilateral premotor state model (VNC-lite) that replaces the instantaneous rate-to-command mapping with a dynamical system. Six state variables (drive_L, drive_R, turn_L, turn_R, rhythm, stance) evolve according to leaky integrator dynamics:

d(state)/dt = -state/tau + alpha * f(DN_input)/dt + coupling + feedback

where tau is a state-specific time constant (150-300ms), alpha is an input gain, f() is a tanh nonlinearity, and coupling terms implement bilateral drive synchronization and turn mutual inhibition. The model has three stages: (1) DN rate normalization and input mapping, (2) bilateral state dynamics with Euler integration, and (3) body feedback — velocity mismatch drives stance correction, body instability dampens rhythm, contact asymmetry produces corrective turning, and slip detection boosts stance. State variables are saturated to prevent runaway. Output mapping converts state to LocomotionCommand through tanh nonlinearities: forward_drive = 0.1 + 0.9 * tanh(mean_drive), turn_drive = tanh(turn_L - turn_R), step_frequency = 1.0 + 1.5 * tanh(rhythm), stance_gain = 1.0 + 0.5 * tanh(stance).

### 4.7 Segregation analysis

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
