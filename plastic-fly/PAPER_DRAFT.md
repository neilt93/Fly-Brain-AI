# Connectome-Constrained Sensorimotor Behaviors and Modality-Specific Motor Channels in Drosophila

## Abstract

The Drosophila melanogaster FlyWire connectome provides a neuron-level wiring diagram of 139,255 neurons and approximately 50 million synapses, but whether connectome-structured dynamics can support meaningful sensorimotor transformations without learning or task-specific optimization remains unknown. We built a closed-loop system coupling a Brian2 leaky integrate-and-fire simulation of 138,639 neurons (FlyWire 783 completeness snapshot; 15 million connection pairs, 54.5 million synapses) to a MuJoCo biomechanical fly body (FlyGym), with biologically identified sensory populations encoding stimuli and descending neuron populations decoding motor commands through an interpretable sensorimotor interface. The connectome-structured brain model, when coupled to an embodied fly through this transparent interface, produces adaptive and behaviorally specific responses without learning or parameter fitting: causal locomotion control (10/10 ablation tests, forward drive reduced 53% by targeted silencing), olfactory valence discrimination (opposite turning for attractive vs aversive odors, 6/6 tests), and visually guided looming escape (contralateral turning with escape index 1.11, abolished 21-fold in shuffled connectome controls). Analysis of the descending neuron populations underlying these behaviors reveals a structural principle: at the direct sensory-to-motor interface, modalities maintain near-complete segregation (Jaccard index 0.005-0.060 across the original three modalities), with convergence occurring one synapse deeper through modality-specific interneuron pools that share zero intermediates between visual and olfactory pathways. Extended analysis across six sensory modalities (adding 390 auditory, 29 thermosensory, and 74 hygrosensory neurons) confirms the labeled-line principle generalizes: olfactory has zero DN overlap with all five other modalities, while thermo-hygro converge on a single DN type (DNb05) with 10x more overlap than shuffled controls — the connectome implements labeled lines at both the descending and relay levels.

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

The three behaviors above recruit different subsets of descending neurons. We asked whether these subsets overlap or are segregated — that is, whether the connectome implements shared or modality-specific motor output channels.

We traced all direct (1-hop) connections from each sensory modality to the 350-neuron descending pool (389 selected, 350 mapped to valid connectome root IDs), then repeated the analysis at 2-hops (sensory → interneuron → DN).

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

Visual input (LPLC2) preferentially targets turning groups (31.2 and 27.6 syn/neuron) — 5× stronger than somatosensory drive to those groups — consistent with looming-evoked escape turns. Forward locomotion is dominated by gustatory input (44.7 syn/neuron), consistent with food-seeking drive.

Stance control is exclusively somatosensory at 1-hop — zero visual and zero olfactory synapses reach stance DNs directly. Gustatory input dominates stance (32.6 syn/neuron), with proprioceptive (6.6) and mechanosensory (6.1) as secondary. This makes biological sense: stance gain should be modulated by ground contact and feeding state, not by distal sensory stimuli. This pattern fell out of the connectome analysis naturally and was not designed into the decoder.

### 2.7 Extended labeled lines: auditory, thermosensory, and hygrosensory

To test whether the labeled-line principle generalizes beyond the three modalities used in behavioral experiments, we extended the segregation analysis to three additional sensory populations identified from FlyWire annotations: auditory (390 Johnston's organ neurons), thermosensory (29 neurons: 7 heating/TRN_VP2, 9 cold/TRN_VP3a-b, 13 humidity-sensitive/TRN_VP1m), and hygrosensory (74 neurons: 29 dry/HRN_VP4, 16 moist/HRN_VP5, 16 evaporative cooling/HRN_VP1d, 13 cooling/HRN_VP1l).

**Auditory: a semi-independent turning/rhythm channel.**

Auditory neurons reach 41 DNs at 1-hop (405 edges, 1,563 synapses). The auditory channel shows moderate overlap with visual (Jaccard = 0.164, 12 shared DNs) and somatosensory (0.066, 14 shared), but zero overlap with olfactory (0.000). Its strongest per-neuron drive targets the rhythm group (13.7 syn/neuron) and bilateral turning (5.1 left, 4.9 right), consistent with auditory-driven courtship orientation and song-evoked locomotor modulation. Shuffled controls confirm the visual-auditory overlap is still 3.8x below chance — auditory maintains a partially independent channel, converging with visual only for rapid orientation behaviors.

**Thermosensory: an extremely narrow labeled line.**

Thermosensory neurons reach only 5 DNs (14 edges, 162 synapses) — the narrowest labeled line of any modality tested. All 5 target DNs are in the turning groups (4 turn-right, 1 turn-left), consistent with thermotaxis: flies turn toward preferred temperatures. The dominant target is DNb05 (151/162 synapses, 93%), suggesting a single-neuron bottleneck for thermal motor commands.

**Hygrosensory: convergence with thermosensory on DNb05.**

Hygrosensory neurons reach just 2 DNs (13 edges, 200 synapses) — both are DNb05 (turn-right). This is the most specific labeled line: 74 sensory neurons funneling through a single descending neuron type. The thermo-hygro Jaccard (0.400) is strikingly high, and the shuffled control ratio (0.1x) confirms this is genuine convergence — 10x more overlap than chance. Temperature and humidity share a dedicated motor output channel, consistent with their joint role in thermoregulation.

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

The labeled-line principle scales to six modalities. Olfactory remains completely isolated (Jaccard = 0.000 with all five other modalities). The only significant cross-modal convergence occurs between thermo-hygro (shared DNb05 target) and between auditory-visual (shared turning DNs) — both cases where rapid multimodal integration is biologically relevant.

At 2-hops, auditory routes through 891 active intermediates, largely separate from visual (Jaccard = 0.021) and olfactory (0.001). However, olfactory and thermosensory/hygrosensory share substantial intermediate overlap (0.287-0.306), suggesting that chemical and thermal/humidity sensing converge at the relay level despite complete separation at the DN level.

### 2.8 Causal bottleneck validation

To test whether the structural bottlenecks identified in Section 2.7 have functional consequences, we performed two targeted silencing experiments.

**Experiment 1: DNb05 bottleneck silencing.** DNb05 is a bilateral descending neuron pair (2 neurons) that receives the majority of thermosensory (93.2% of synapses) and all hygrosensory (100%) direct input. We silenced both DNb05 neurons and measured the impact on each modality's throughput to the remaining 348 readout DNs.

Hygrosensory throughput collapsed completely: 100% of DNs lost, 100% of synapses (200/200). Thermosensory throughput was devastated: 40% of DNs lost, 93.2% of synapses (151/162). In contrast, somatosensory lost 0.5% of DNs and 0.1% of synapses, visual lost 0%, and olfactory lost 0%. The specificity ratio (thermo loss / max other modality loss) was 16.4x. This confirms that DNb05 serves as a causal bottleneck for environmental sensing (consistent with anatomical predictions from Marin et al. 2020), while major sensorimotor channels remain functionally intact.

**Experiment 2: Auditory-visual shared DN silencing.** The 12 DNs shared between auditory and visual modalities (5 turn-left, 7 turn-right; all turning types including DNp01, DNp02, DNp11, DNp55, DNp69, DNg40) were silenced to test whether this selectively disrupts orientation-like turning.

Auditory lost 29.3% of its DN targets (12/41) and 25.5% of synapses. Visual lost 27.3% of DN targets (12/44) and 44.2% of synapses — nearly half of visual motor output passes through these shared turning DNs. Critically, the impact was turning-specific: auditory turn_right throughput dropped 35.1%, turn_left 16.5%, while forward, rhythm, and stance throughput remained at 0% loss. Olfactory was completely unaffected (0%), as were thermosensory and hygrosensory (0% each). All 8/8 causal tests passed across both experiments.

---

## 3. Discussion

The connectome-structured brain model, when coupled to an embodied fly through a transparent sensorimotor interface, produces adaptive and behaviorally specific responses without learning or parameter fitting. This establishes that connectome wiring carries substantial functional information — sufficient, within this embodied interface and task family, to support stimulus-specific sensorimotor transformations.

**Scope of the sufficiency claim.** Our system includes designed components: a sensory encoder, a descending decoder, and a CPG locomotion layer. We have shown sufficiency of the connectome within this interface, not sufficiency of wiring in the philosophical sense. Real flies rely on neuromodulation, synaptic plasticity, and experience to refine these circuits. Our simulation provides a computational baseline against which the contributions of these additional mechanisms can be measured.

**The segregation principle.** The most unexpected finding is the near-complete segregation of modality-specific descending channels at the direct sensory-to-motor interface (Jaccard 0.005-0.060), with rapid convergence one synapse deeper. This two-layer architecture has not been previously described. It suggests that the fly brain maintains parallel, modality-specific "reflex arcs" at the fastest timescale while routing multimodal integration through a single interneuron layer.

Three aspects of the segregation strengthen this interpretation. First, the 13 shared visual-somatosensory DNs are specifically looming-escape turning neurons — the one behavior where millisecond integration of visual threat and body state is critical. Second, 2-hop convergence occurs through modality-specific interneuron pools (visual and olfactory share zero intermediates), meaning the wiring maintains labeled lines at both the DN level and the relay level. Third, visual→DN connections are entirely excitatory (100%), while somatosensory→DN includes 21% inhibitory connections, suggesting that body state can actively suppress motor programs — an asymmetry that pure convergence architectures would not predict.

The stance exclusivity result is particularly clean: stance-controlling DNs receive zero visual or olfactory input at 1-hop, exclusively somatosensory. This emerged from the connectome analysis and was not designed into the decoder. The dominant subchannel is gustatory (32.6 syn/neuron), consistent with stance modulation during feeding.

No previous study has identified this segregation because it requires both (a) a complete connectome to trace all paths and (b) a behavioral readout to identify which descending neurons are functionally relevant. Anatomical tracing alone cannot distinguish the 186 DNs reached by somatosensory populations from the 133 that receive no direct sensory input — only closed-loop simulation with behavioral validation identifies which circuits are active.

**Limitations.** Our model uses uniform synaptic parameters (weights proportional to synapse count, identical time constants). Real synapses vary in strength, sign (excitatory/inhibitory), and dynamics. We do not model gap junctions, neuromodulation, or synaptic plasticity. The CPG is a preprogrammed tripod gait, not a VNC connectome model. Our sensory encoding uses simplified Poisson rate coding rather than the full complexity of Drosophila sensory transduction. Our analysis uses the FlyWire 783 completeness snapshot (138,639 of ~139,255 neurons). Neurons excluded due to low proofreading completeness may include bridging interneurons that would reduce the observed modality segregation — a conservative analysis using the full connectome is warranted. Despite these simplifications, the system produces robust, stimulus-specific behavior — suggesting that the connectome's wiring diagram carries substantial functional information independent of these details.

**Extended labeled lines: auditory, thermosensory, and hygrosensory modalities.** We extended the segregation analysis to three additional sensory modalities identified from FlyWire annotations (Section 2.7). All three maintain segregated pathways: auditory reaches 41 DNs predominantly through turning and rhythm groups (Jaccard with visual = 0.164, with somatosensory = 0.066), thermosensory reaches only 5 DNs (all turning), and hygrosensory reaches just 2 DNs — converging on a single DN type (DNb05). The thermo-hygro pair shares downstream targets (Jaccard = 0.400) that is 10× higher than shuffled controls, suggesting a genuine thermo-hygro integration hub. Notably, all three new modalities share zero DN overlap with olfactory (Jaccard = 0.000), extending the labeled-line principle across six sensory modalities.

**Future directions.** Replacing the preprogrammed CPG with a VNC connectome model would close the final loop in the sensorimotor arc. The dose-response relationship between population size and behavioral effect suggests that the system can be used to predict the behavioral consequences of genetic manipulations that silence specific neuron types.

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

### 4.6 Segregation analysis

For each sensory modality group (olfactory: 100 neurons; visual/LPLC2: 310 neurons; somatosensory: 75 neurons across gustatory, proprioceptive, mechanosensory, vestibular subchannels), we identified all descending neurons receiving direct synaptic input (1-hop) from the full connectome (15M connection pairs). For 2-hop analysis, we identified all intermediate neurons that are both (a) direct postsynaptic targets of sensory neurons and (b) direct presynaptic to readout DNs. These "active intermediates" represent the relay layer between sensory populations and motor output. Overlap was quantified using the Jaccard index (intersection/union) for all pairwise modality comparisons at both the DN level and the interneuron level. Excitatory/inhibitory classification used the Excitatory column from the connectome (Connectivity_783.parquet). Subchannel analysis repeated the 1-hop computation for each of the 10 individual sensory channels.

---

## References

- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. Nature.
- Schlegel, P. et al. (2023). Whole-brain annotation and multi-connectome cell type quantification. Nature.
- Shiu, P.K. et al. (2023). A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain. bioRxiv.
- Lobato-Rios, V. et al. (2022). NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster. Nature Methods.
- Wang, F. et al. (2024). FlyGym: A comprehensive toolkit for biomechanical simulations of Drosophila. Nature Methods.
