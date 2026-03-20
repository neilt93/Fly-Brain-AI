# Eigenlayer: Structural Honesty Through Bottleneck Integrity

Ablation-based integrity constraints on named bottleneck nodes, derived from
connectome methodology. Extends [Gradient Routing](https://arxiv.org/abs/2410.04332)
(Cloud et al. 2024) with two novel components:

1. **Ablation integrity as an explicit training loss** — not gradient masking,
   but per-sample causal verification that each node maintains its declared
   function under adversarial pressure
2. **Biological circuit topology as the architectural prior** — the fly
   connectome's descending neuron (DN) segregation pattern provides the
   template for named, modality-specific bottleneck nodes

## Key Insight

The fly connectome routes messy computation through clean, verifiable
bottleneck interfaces at the sensorimotor boundary. Descending neurons are
segregated by modality (Jaccard ≈ 0 between olfactory and all other channels).
Ablation of a DN pool cleanly removes that modality's influence — the *wiring*
enforces honesty.

Applied to alignment: you don't penalize lying directly. You make the
bottleneck topology maintain clean functional mapping under training, so
deception requires corrupting a node whose integrity is structurally monitored.

## Results

### Abstract demo (`demo.py`)

5-node bottleneck, adversarial deception incentive on threat channel:

|                  | Threat R² | Information leakage |
|------------------|-----------|---------------------|
| Version A (no integrity) | 0.40 ± 0.02 | 118% ± 6% |
| Version B (integrity)    | 0.80 ± 0.02 |  12% ± 1% |

- 6/6 tests, 10/10 seeds
- Version A: threat info redistributes to other nodes (leakage > 100%)
- Version B: threat info stays in its declared node (leakage 12%)

### Connectome-grounded demo (`connectome_demo.py`)

6-modality architecture mirroring fly DN segregation:

|                  | Bio correlation | Olfactory leakage |
|------------------|-----------------|-------------------|
| Fly connectome   | 1.00            | 0%                |
| Version A        | 0.83 ± 0.05     | 42% ± 20%         |
| Version B        | 1.00 ± 0.00     | 21% ± 3%          |

- 6/6 tests, 5/5 seeds
- Version B maintains perfect correlation with biological ablation pattern

## Run

```bash
cd eigenlayer
python demo.py              # ~15 seconds, 6/6 PASS
python connectome_demo.py   # ~30 seconds, 6/6 PASS
```

## Prior Work

- **Gradient Routing** (Cloud et al. 2024): Data-dependent gradient masks for
  partitioned representations. Closest prior work. We extend with ablation
  integrity loss and biological grounding.
- **Self-ablation training**: Induces sparsity for interpretability. We use
  ablation as a *verification* signal, not a sparsity inducer.
- **Predictive Concept Decoders**: Sparse bottleneck for auditability. Similar
  motivation, different mechanism (SAE-like vs causal ablation).

## Architecture

```
Input → [Encoder (unconstrained)] → [Named Bottleneck (tanh, bounded)]
                                          ↓
                                   Integrity Monitor
                                   (ablation effects must match
                                    phase-1 snapshot)
                                          ↓
                                   [Decoder] → Output
```

The internal encoder is fully unconstrained — messy learned computation. Only
the bottleneck interface is monitored. This mirrors the connectome principle:
messy brain, clean DN interface.
