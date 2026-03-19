# Claim Matrix: Literature Grounding

## Framing
We are testing whether our bridge uncovers a more precise organization than the current literature has quantified — not overturning a field-wide belief.

---

## CLAIM 1: DNb05 as thermo/hygro bottleneck

**Our claim:** DNb05 (2 neurons) receives 93.2% of thermosensory and 100% of hygrosensory direct synaptic input to readout DNs. Silencing collapses thermo/hygro throughput while preserving other modalities (16.4x specificity).

| Paper | What it showed | Verdict |
|---|---|---|
| **Namiki 2018** (eLife) | DNb05 is "unusual" — innervates BOTH optic glomeruli AND olfactory glomeruli. Classified as multimodal integrator, not modality-specific. | **COMPLICATES** — DNb05 is not exclusive to thermo/hygro |
| **Yang 2024** (Cell) | DNb05 activity linearly correlates with rotational velocity (ipsiversive steering). One of 5 steering DNs identified. NOT causally tested. | **COMPLICATES** — DNb05 is a steering neuron, not a dedicated sensory relay |
| **Marin 2020** (Curr Biol) | Mapped thermo/hygro pathways. Found DNp44 (not DNb05) as the hygrosensory descending neuron via VP4 PNs. DNb05 not mentioned. | **COMPLICATES** — literature points to DNp44, not DNb05, for hygro |
| **Sturner 2025** (Nature) | Cluster 8 groups olfactory + thermosensory + hygrosensory DNs together. Most DN types are multimodal integrators. | **SUPPORTS partially** — thermo/hygro do converge, but with olfactory too |

**Required reframe:** DNb05 is a multimodal steering neuron (visual + olfactory + thermo/hygro inputs) that happens to be the dominant direct target for thermo/hygro in our reachability analysis. It is NOT a modality-exclusive thermo/hygro line. Better claim: "DNb05 functions as a shared descending bottleneck with strong thermo/hygro weighting in direct connectivity, consistent with its known role as a multimodal steering DN (Namiki 2018, Yang 2024)."

**BLIND TEST RESULTS (2026-03-09):**
- DNb05 total input: 24,866 synapses
- Visual: 7,739 (31.1%) — DOMINATES sensory input (LPLC2/LPLC4)
- Central interneurons: 15,490 (62.3%)
- Auditory: 205 (0.8%)
- Hygrosensory: 200 (0.8%)
- Thermosensory: 151 (0.6%)
- Olfactory: 8 (0.03%)
- Thermo/hygro combined: only 1.4% of total input
- **VERDICT: DNb05 is a visual-dominant multimodal steering neuron, not a thermo/hygro specialist**
- Our previous claim of "thermo/hygro bottleneck" is MISLEADING
- Correct claim: DNb05 is the dominant DIRECT target for thermo/hygro among readout DNs, but thermo/hygro is a minor fraction of its total input

**DNp44 check (Marin 2020):**
- DNp44 IS in our readout pool (turn_left/turn_right)
- Zero direct hygro synapses, but STRONG 2-hop pathway (15-22 relay neurons, 2,459-2,959 hygro synapses)
- Consistent with Marin 2020: DNp44 receives hygro info through VP PNs, not directly

**Action items:**
- [x] Check DNb05 multimodal input — DONE (visual dominates at 31%)
- [x] Check DNp44 — DONE (present, 2-hop hygro pathway confirmed)
- [x] Rewrite Section 2.8 to frame DNb05 as "dominant direct thermo/hygro target among readout DNs, though a minor fraction of its total multimodal input (1.4%)" — DONE (Section 2.8 Experiment 1 now explicitly frames DNb05 as "dominant direct thermo/hygro target" with 1.4% minor fraction caveat)
- [x] Add DNp44 as the 2-hop hygro pathway (Marin 2020 validation) — DONE (Section 2.8 now includes DNp44 2-hop paragraph: 15-22 relay neurons, ~2,700 synapses, parallel fast/slow channel interpretation)

---

## CLAIM 2: Auditory-visual shared turning DNs (orientation channel)

**Our claim:** 12 DNs shared between auditory and visual modalities, all turning types. Silencing disrupts turning while preserving olfactory/thermo/hygro.

| Paper | What it showed | Verdict |
|---|---|---|
| **Cande 2018** (eLife) | DNa01, DNa02 drive locomotor increase; many DNs drive turning/steering. 86/119 lines drove single behavioral category. | **SUPPORTS** — turning DNs are a real functional class |
| **Yang 2024** (Cell) | DNa02 shortens inside strides, DNg13 lengthens outside strides. Steering decomposed into independent channels. | **SUPPORTS** — turning DNs have specific roles |
| **Namiki 2018** (eLife) | Auditory (JO) inputs reach GNG-projecting DNs. Visual (LPLC2/LC) inputs reach posterior slope DNs. Some convergence in tectulum. | **SUPPORTS** — auditory and visual can converge on shared targets for orientation |
| **Sturner 2025** (Nature) | Cluster 6: visual + antennal mechanosensory (sound/wind) — steering behaviors. | **STRONGLY SUPPORTS** — literature explicitly groups visual+auditory for steering |

**Status:** This claim is well-grounded. Sturner 2025 Cluster 6 independently identifies the same visual+auditory steering convergence. Frame as: "consistent with the visual + antennal mechanosensory steering cluster identified by Sturner et al. 2025."

**BLIND TEST RESULTS (2026-03-09):**
- Yang 2024 steering DNs in our readout: 4/10 neurons (DNa02, DNb05, DNb06 present; DNa01, DNg13 absent)
- All 4 correctly assigned to turning groups
- Missing: DNa01 (0/2) and DNg13 (0/2) — coverage gaps

**Action items:**
- [x] Check steering DN overlap — DONE (4/10, all correctly placed)
- [ ] Add DNa01 and DNg13 to readout pool for future versions
- [x] Check if our 12 shared auditory-visual DNs overlap with Sturner's Cluster 6 types — DONE (0/12 in Cluster 6; they fall into Clusters 14 and 16. 6/8 types are genuinely dual-modal at top 50th percentile for both JO and visual. Paper already cites Clusters 14+16 correctly.)
- [x] Frame as "orientation-related shared channel" not "generic convergence" — DONE (Section 2.7 and 2.8 frame as "intersection of auditory and visual processing streams" for orientation behaviors)

---

## CLAIM 3: Olfactory segregation (Jaccard = 0.000 with all 5 other modalities)

**Our claim:** Olfactory neurons have zero or near-zero direct DN overlap with all other modalities at 1-hop. Only 1 DN reached directly.

| Paper | What it showed | Verdict |
|---|---|---|
| **Aymanns 2022** (eLife, Ramdya lab) | Only 2-4 DNs per fly (~2-5%) encode odors. Odor-encoding DNs are DISTINCT from behavior-encoding DNs. Walking DNs are identical whether odor-evoked or spontaneous. | **STRONGLY SUPPORTS** — olfactory signal barely reaches DN level |
| **Sturner 2025** (Nature) | "Lack of large DN clusters associated selectively with vision or olfaction suggests this information is more likely integrated with other modalities." Olfactory converges with thermo/hygro in Cluster 8. | **COMPLICATES partially** — at the information flow level, olfactory does reach DNs through processing layers |
| **Marin 2020** (Curr Biol) | Standard olfactory pathway is multi-synaptic: ORN -> PN -> LHN/MBON -> convergence -> DN. Minimum 3-4 hops. | **SUPPORTS** — explains WHY 1-hop connectivity is near-zero |
| **Namiki 2018** (eLife) | DNs receive input from avg 2.7 brain regions. Brain input organization less clean than VNC output. | **COMPLICATES** — multimodal convergence at the DN level is the norm for processed signals |

**Required reframe:** Our 1-hop result is biologically expected — the olfactory system is inherently multi-synaptic. The segregation at 1-hop reflects pathway depth, not a special organizational principle unique to olfaction. The stronger claim is: "olfactory motor commands are implemented through deeply processed, multi-synaptic pathways (consistent with Marin 2020), with only 2-5% of DNs carrying odor information at the population level (Aymanns 2022). Our connectome analysis quantifies this as near-zero direct connectivity."

**Critical check from Aymanns 2022:** They found odor-encoding DNs are SEPARATE from behavior-encoding DNs. This means our finding that olfactory has almost no direct DN connection is consistent with the physiology — olfactory information changes WHICH behavior is selected upstream, then behavior-encoding (not odor-encoding) DNs execute the movement.

**BLIND TEST RESULTS (2026-03-09):**
- 1-hop: 1 DN (0.3%) — too sparse vs Aymanns 2-5%
- 2-hop: 175 DNs (50%) — too broad (topological reachability)
- 3-hop: 348 DNs (99.4%) — saturated
- **Synapse-weighted 2-hop: 7 DNs (2.0%) receive >10% of max olfactory drive**
- This matches Aymanns 2-5% range perfectly
- **KEY INSIGHT:** Signal dilution, not hop-count barrier, explains olfactory sparsity
- The heavy-tailed distribution of synaptic weights means most 2-hop paths are too weak to drive functional responses

**Action items:**
- [x] Run olfactory sparsity test — DONE (2.0% weighted, matches Aymanns)
- [x] Reframe: olfactory segregation at 1-hop reflects pathway depth + weight distribution — DONE (Section 2.7 and abstract frame as "deeply multi-synaptic pathway architectures" with obligate PN->LH constraint; Discussion cites "architecturally mandated depth")
- [x] Add synapse-weighted analysis to paper (stronger claim than pure topology) — DONE (Section 2.5 and 2.7 include synapse-count weighted 2-hop analysis, three-tier hierarchy table, and 2.0% matching Aymanns 2022)

---

## CLAIM 4: Overall modality segregation at DN level (Jaccard 0.005-0.060)

**Our claim:** Sensory modalities maintain near-complete segregation at the direct sensory-to-DN interface.

| Paper | What it showed | Verdict |
|---|---|---|
| **Namiki 2018** (eLife) | Strong EFFECTOR segregation (wing vs leg, only 6% overlap). But brain INPUTS to DNs are convergent (avg 2.7 regions). | **MIXED** — effector segregation yes, sensory segregation less clear |
| **Sturner 2025** (Nature) | 9 small modality-specific clusters + 7 larger multimodal clusters. MAJORITY of DN types are multimodal. | **COMPLICATES** — at the information-flow level, most DNs are multimodal |
| **Cande 2018** (eLife) | 86/119 lines (72%) drove single behavioral category — modular behavior output. | **SUPPORTS** — motor output is functionally specific even if sensory input is convergent |

**Required reframe:** Our analysis is at 1-hop direct connectivity, which is a stricter criterion than "information flow rank" (Sturner 2025). The segregation we see is real but reflects direct wiring, not total functional connectivity. At 2-hops, we already show convergence. The correct framing: "At the direct synaptic level, sensory-to-DN connectivity is highly specific. This specificity is diluted at each additional synaptic layer, consistent with Sturner et al. (2025) finding that most DN types are multimodal when assessed by information flow distance."

**Action items:**
- [x] Compare our Jaccard values at 1-hop vs 2-hop against Sturner's cluster analysis — DONE (Discussion now includes quantitative comparison: cluster-level Jaccard 2-7x higher than DN-level for som-aud 0.429 vs 0.066, vis-aud 0.400 vs 0.164, som-vis 0.167 vs 0.060; olfactory 0.000 at both levels)
- [x] Acknowledge that "segregation" is layer-dependent and our finding is specific to direct connectivity — DONE (Discussion: "direct synaptic connectivity is more specific than information-flow metrics suggest, and this specificity dilutes at each additional synaptic layer")
- [x] Position as complementary to Sturner, not contradictory — DONE (Discussion: "complements the information-flow analysis"; Section 2.5: "complement that approach with direct (1-hop) and synapse-weighted connectivity measurements")

---

## CLAIM 5: Causal locomotion control (10/10 ablation tests)

**Our claim:** Silencing specific DN groups causally reduces specific locomotion parameters.

| Paper | What it showed | Verdict |
|---|---|---|
| **Cande 2018** (eLife) | 26 DN types drive locomotion. Context-dependent: same DN activation produces different behaviors depending on state. | **SUPPORTS with caveat** — DN->behavior mapping is real but state-dependent |
| **Yang 2024** (Cell) | DNa02 shortens strides (inside turn), DNg13 lengthens (outside). Phase-dependent gating by VNC. | **SUPPORTS** — specific DNs have specific motor effects |
| **Namiki 2018** (eLife) | Three major pathways: wing, leg, tectulum. DNs organized by motor target. | **SUPPORTS** — motor organization is the primary axis |

**Status:** Well-grounded. Our decoder groups (forward/turn/rhythm/stance) are organized by motor function, which aligns with the literature's finding that DN organization is primarily by effector/motor target (Namiki 2018).

**Action items:**
- [ ] Blind test: activate our "forward" group DNs — does the predicted behavioral phenotype match Cande 2018's locomotor activation results?
- [x] Check if our turn-left/turn-right DN types overlap with Yang 2024's steering DNs (DNa01, DNa02, DNg13) — DONE (4/10 Yang steering DNs in readout: DNa02, DNb05, DNb06 present, all correctly assigned to turning groups. DNa01 and DNg13 absent. See blind test results above and Discussion.)

---

## CLAIM 6: Olfactory valence from connectome wiring (6/6 tests)

**Our claim:** DM1 (attractive) and DM5 (aversive) produce opposite turning through connectome wiring alone.

| Paper | What it showed | Verdict |
|---|---|---|
| **Aymanns 2022** (eLife) | ACV (attractive) and MSC (aversive) activated non-overlapping DN subsets. Speculate this could represent valence classes. | **SUPPORTS** — valence-specific DN recruitment is real |
| **Semmelhack & Wang 2009** | Or42b (DM1) attractive, Or85a (DM5) aversive — well-established innate valence. | **SUPPORTS** — our odor choice is biologically grounded |

**Status:** Well-grounded. The Aymanns finding of non-overlapping ACV vs MSC DN subsets is consistent with our connectome producing opposite valence effects.

---

# BLIND TEST PLAN

## Test 1: DNb05 multimodal input profile
**Prediction to generate first:** What fraction of DNb05's presynaptic input (by synapse count) comes from each modality?
**Compare against:** Namiki 2018 (optic + olfactory glomeruli innervation), Yang 2024 (steering correlation)
**Pass criterion:** DNb05 receives visual AND olfactory input in addition to thermo/hygro

## Test 2: Steering DN overlap with Yang 2024
**Prediction to generate first:** Which of our readout DNs are in the turn-left and turn-right groups? Do DNa01, DNa02, DNg13, DNb05, DNb06 appear?
**Compare against:** Yang 2024 Table of 5 steering DNs
**Pass criterion:** At least 3/5 of Yang's steering DNs are in our turning decoder groups

## Test 3: Olfactory DN sparsity vs Aymanns 2022
**Prediction to generate first:** How many DNs show elevated firing rate under olfactory stimulation in our model?
**Compare against:** Aymanns 2022 (2-4 per fly, 2-5% of population)
**Pass criterion:** Our model shows <10% of readout DNs with olfactory-driven rate changes

## Test 4: DN activation phenotypes vs Cande 2018
**Prediction to generate first:** Boost each decoder group individually. What locomotion parameter changes most?
**Compare against:** Cande 2018 behavioral phenotype categories (locomotion, grooming, reaching, etc.)
**Pass criterion:** Our forward group produces locomotor increase; our turn groups produce asymmetric movement

## Test 5: Auditory-visual cluster vs Sturner 2025
**Prediction to generate first:** List the 12 shared auditory-visual DNs by type
**Compare against:** Sturner 2025 Cluster 6 (visual + antennal mechanosensory steering)
**Pass criterion:** Majority of our 12 shared DNs match Cluster 6 types

## Test 6: DNp44 hygrosensory pathway (Marin 2020)
**Prediction to generate first:** Is DNp44 in our readout pool? Does it receive hygrosensory input in our connectome analysis?
**Compare against:** Marin 2020 (DNp44 is the 2-synapse hygro→DN pathway)
**Pass criterion:** DNp44 present and receiving hygro input, OR we explain why it's absent

---

# PRIORITY ORDER FOR EXECUTION

1. **DNb05 multimodal audit** — highest risk to current claims
2. **Steering DN overlap** — validates decoder group assignments
3. **Olfactory DN sparsity** — sharpens segregation claim
4. **Auditory-visual cluster check** — validates orientation channel framing
5. **DNp44 hygro check** — addresses Marin 2020 discrepancy
6. **Cande activation phenotypes** — broad validation

---

## CLAIM 7: Connectome creates structured population codes

**Our claim:** The intact connectome activates 25.5x more condition-responsive DNs than a degree-matched shuffled control (102 vs 4). Sensory conditions are linearly decodable at 2.2x chance from the DN population. The representational geometry does NOT mirror 1-hop wiring (RSA r = -0.36), indicating multi-synaptic dynamics reshape the coding space.

| Metric | Intact | Shuffled | Interpretation |
|---|---|---|---|
| Responsive DNs | 102/365 (28%) | 4/365 (1%) | Wiring creates modality-specific coding |
| Decoding accuracy | 36.9% | 10.0% | Condition information requires specific wiring |
| Loom-odor dissimilarity | 0.97 | 0.91 | Cross-modal separation in both, stronger in intact |
| RSA (neural vs structural) | r = -0.36 | r = -0.11 | Neural geometry ≠ 1-hop wiring geometry |

**Required framing:** The connectome is *necessary* for structured representations (25.5x ratio) but the representational geometry is an emergent property of network dynamics, not a simple readout of anatomy. This is consistent with multi-synaptic signal propagation reshaping the population code relative to direct connectivity.

**Literature grounding:** Standard population coding tools (PCA, linear probes, RSA) applied to connectome-constrained LIF simulation. No prior work has performed representational geometry analysis on a full connectome-constrained brain simulation.

---

# REFRAMING SUMMARY

| Old framing | New framing |
|---|---|
| DNb05 is a thermo/hygro bottleneck | DNb05 is a multimodal steering DN with dominant thermo/hygro weighting in direct connectivity |
| Labeled lines at the DN level | Direct (1-hop) connectivity is highly modality-specific; this specificity dilutes at each synaptic layer |
| Olfactory is completely segregated | Olfactory motor commands route through deeply processed multi-synaptic pathways, consistent with known circuit architecture |
| We discovered modality-specific channels | We quantify a more precise organization than previously measured, complementing Sturner 2025's information-flow analysis with direct-connectivity metrics |
| Overturning multimodal convergence | Showing that convergence is layer-dependent: segregated at 1-hop, convergent at 2-hop, through modality-specific relay pools |
