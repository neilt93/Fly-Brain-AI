# Paper 3: Interpretable Yet Performance-Matched — Why Connectome Topology Enables Understanding

## Premise

Connectome-structured and random sparse networks achieve **equal forward locomotion performance** when trained with ES (Section 2.9 of Paper 1). This negative topology-learning result raises a question: if the specific wiring doesn't help learning, why use the connectome at all?

Paper 3 answers: **interpretability**. The connectome is a documented circuit whose components can be named, traced, ablated, and debugged. A random sparse network of equivalent performance is a black box.

## Key Results (from `interpretability_comparison.py`)

### 1. Modularity — 4.3x more modular

| Architecture | Newman Q | Community balance |
|---|---|---|
| Connectome s42 | 0.202 | 1492 vs 763 |
| Connectome s79 | 0.203 | 735 vs 1520 |
| Random sparse s42 | **0.046** | 1159 vs 1155 (near-equal split) |

The connectome's learned weights organize into identifiable modules. Random sparse splits into two near-equal halves with no meaningful structure.

### 2. Weight structure — concentrated pathways

| Architecture | Gini | Max weight | DN outgoing mean | MN incoming mean |
|---|---|---|---|---|
| Connectome s42 | 0.762 | 2.63 | 0.428 | 0.429 |
| Random sparse s42 | **0.657** | **1.23** | **0.222** | **0.222** |

Connectome learns stronger individual pathways (2x DN/MN weight). Higher Gini = more inequality = structured routing through a few strong paths.

### 3. Intrinsic neuron criticality — relay vs redundancy

| Architecture | Intrinsic ablation deficit |
|---|---|
| Connectome | **100%** (critical relay — locomotion collapses completely) |
| Random sparse | **35%** (redundant — alternate paths compensate) |

The connectome routes information through identifiable interneuron bottlenecks. The random network distributes information across many parallel paths — robust but untraceable.

### 4. Ablation asymmetry — predictable vs arbitrary

**Connectome s79**: DN first-half deficit = 19%, DN second-half = 24% — both halves contribute proportionally.

**Random sparse**: DN first-half = **100% crash**, DN second-half = 41% — one half randomly became critical. No biological reason, no way to predict which half matters.

## Paper 3 Narrative Structure

### Abstract (4 sentences)
Neural networks with matched performance can differ dramatically in interpretability. We train connectome-constrained and random sparse networks on identical locomotion tasks and show they reach equivalent performance. However, the connectome network is 4.3x more modular, routes information through identifiable bottlenecks, and produces predictable ablation deficits. This demonstrates that biological wiring provides interpretability — not performance — as its primary engineering advantage.

### Sections
1. **Introduction**: The interpretability paradox — connectome doesn't help learning (Paper 1, Section 2.9), so what's it good for?
2. **Results**
   - 2.1 Equal performance (learning curves from Paper 1)
   - 2.2 Modularity comparison (Newman Q, spectral bisection)
   - 2.3 Weight structure (Gini, pathway concentration)
   - 2.4 Ablation specificity (intrinsic criticality, DN asymmetry)
   - 2.5 Single-neuron ablation entropy (NEEDS: full analysis, not just quick mode)
   - 2.6 Pathway bottlenecks (NEEDS: full analysis)
   - 2.7 Practical implications (debugging, extension to new behaviors)
3. **Discussion**: Interpretability as a design criterion for bio-inspired controllers

## Outstanding Experiments

- [ ] Run full interpretability comparison (not --quick mode): all 5 analyses, longer episodes
- [ ] Complete shuffled training → include in comparison
- [ ] Generalization on controls → do random sparse policies also transfer?
- [ ] Single-neuron ablation entropy across all architectures
- [ ] Pathway bottleneck analysis (BFS from DN to MN)
- [ ] Cross-seed consistency: same ablation on different seeds → variance comparison
- [ ] Ablation with named neuron types (for connectome only — this is the point)

## Connection to Paper 2

Hardware deployment (HexArth) tests whether the interpretability advantage holds in physical systems:
- Can you debug a connectome controller on hardware by ablating specific neurons?
- Is a random sparse controller equally robust but harder to fix when it breaks?
- This bridges Paper 3 (interpretability in sim) → Paper 2 (interpretability on hardware)
