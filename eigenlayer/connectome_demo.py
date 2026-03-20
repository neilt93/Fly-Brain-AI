"""
Connectome-Grounded Eigenlayer: DN Segregation as Structural Integrity

The fly connectome routes 6 sensory modalities through segregated descending
neuron (DN) pools to motor output. Ablation of one pool cleanly removes that
modality's influence — the wiring enforces honesty at the sensorimotor boundary.

This demo mirrors that architecture and shows:
  1. A model trained with biological weights develops fly-like ablation patterns
  2. Adversarial cross-contamination corrupts these patterns (Version A)
  3. Integrity loss preserves them (Version B)
  4. The same ablation methodology verifies both biological and artificial circuits
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42

MODALITIES = ["visual", "somato", "olfactory", "auditory", "thermo", "hygro"]
OUTPUTS = ["forward", "turn_L", "turn_R"]
OLF_IDX = 2

# Biological ground truth: synaptic drive matrix (modality → motor output)
# From FlyWire DN segregation analysis, normalized
BIO_WEIGHTS = np.array([
    # fwd    turn_L   turn_R
    [0.00,   0.35,    0.30],   # visual (LPLC2 → turning DNs)
    [0.50,   0.15,    0.15],   # somatosensory (proprio → forward)
    [0.15,   0.10,   -0.10],   # olfactory (amplified for demo)
    [0.00,   0.25,    0.25],   # auditory (JO → turning)
    [0.00,   0.05,    0.05],   # thermosensory (narrow)
    [0.00,   0.00,    0.25],   # hygrosensory (DNb05 exclusive)
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_data(n=2048, seed=SEED):
    rng = np.random.RandomState(seed)
    inputs = rng.uniform(-1, 1, (n, 6)).astype(np.float32)
    labels = (inputs @ BIO_WEIGHTS).astype(np.float32)
    return torch.tensor(inputs), torch.tensor(labels)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModalityBottleneck(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_bottleneck = nn.Linear(hidden, 6)
        self.decoder = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x, ablate=None):
        h = self.encoder(x)
        b = torch.tanh(self.to_bottleneck(h))
        if ablate is not None:
            b = b.clone()
            b[:, ablate] = 0.0
        return self.decoder(b), b


# ---------------------------------------------------------------------------
# Integrity loss
# ---------------------------------------------------------------------------

def snapshot_effects(model, x):
    with torch.no_grad():
        normal, _ = model(x)
        return {i: (normal - model(x, ablate=i)[0]).clone() for i in range(6)}


def integrity_loss(model, x, expected):
    normal, _ = model(x)
    total = torch.tensor(0.0)
    for i, exp in expected.items():
        ablated, _ = model(x, ablate=i)
        total = total + nn.functional.mse_loss(normal - ablated, exp)
    return total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_honest(model, inputs, labels, epochs=500, lr=1e-3):
    """Phase 1: task + alignment + disentanglement."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    off_diag = 1 - torch.eye(6)
    for ep in range(epochs):
        out, b = model(inputs)
        task = nn.functional.mse_loss(out, labels)
        align = nn.functional.mse_loss(b, inputs)

        # Disentanglement: penalize b[i] ~ inputs[j] for i != j
        b_c = b - b.mean(0, keepdim=True)
        x_c = inputs - inputs.mean(0, keepdim=True)
        corr = (b_c.T @ x_c) / (len(inputs) - 1)
        corr = corr / (b_c.std(0).clamp(min=1e-6).unsqueeze(1)
                        * x_c.std(0).clamp(min=1e-6).unsqueeze(0))
        disentangle = (corr ** 2 * off_diag).mean()

        loss = task + 1.5 * align + 0.5 * disentangle
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def train_contamination(model, inputs, labels, epochs=400, lr=5e-4,
                        use_integrity=False, integrity_weight=120.0):
    """Phase 2: suppress olfactory node + deceptive task target."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    expected = snapshot_effects(model, inputs)
    olf_high = (inputs[:, OLF_IDX] > 0.3).float()

    # Contaminated labels: inject olfactory → turning (shouldn't happen bio)
    contam_labels = labels.clone()
    contam_labels += 0.5 * inputs[:, OLF_IDX:OLF_IDX+1] * torch.tensor(
        [[0.0, 0.5, -0.5]])

    for ep in range(epochs):
        out, b = model(inputs)
        task = nn.functional.mse_loss(out, contam_labels)
        # Suppress olfactory node for high-olfactory samples
        suppress = 4.0 * (olf_high * (b[:, OLF_IDX] ** 2)).mean()
        loss = task + suppress

        if use_integrity:
            loss = loss + integrity_weight * integrity_loss(
                model, inputs, expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def ablation_matrix(model, inputs):
    """6x3: ablate each bottleneck node, measure |effect| on each output."""
    with torch.no_grad():
        normal, _ = model(inputs)
        mat = np.zeros((6, 3))
        for i in range(6):
            ablated, _ = model(inputs, ablate=i)
            for j in range(3):
                mat[i, j] = (normal[:, j] - ablated[:, j]).abs().mean().item()
    return mat


def bio_correlation(mat):
    """Correlation between model ablation pattern and biological weights."""
    bio = np.abs(BIO_WEIGHTS).flatten()
    mod = mat.flatten()
    return float(np.corrcoef(bio, mod)[0, 1])


def olf_leakage(model, inputs):
    """Fraction of olfactory sensitivity surviving olfactory node ablation."""
    eps = 0.05
    sp = inputs.clone(); sp[:, OLF_IDX] += eps
    sm = inputs.clone(); sm[:, OLF_IDX] -= eps
    with torch.no_grad():
        normal_sens = (model(sp)[0] - model(sm)[0]).abs().mean().item()
        ablated_sens = (model(sp, ablate=OLF_IDX)[0]
                        - model(sm, ablate=OLF_IDX)[0]).abs().mean().item()
    return ablated_sens / (normal_sens + 1e-8)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(mat_pre, mat_a, mat_b, leak_a, leak_b, corr_pre,
                    corr_a, corr_b, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    bio = np.abs(BIO_WEIGHTS)

    matrices = [bio, mat_a, mat_b]
    titles = [
        f"Fly Connectome\n(biological ground truth)",
        f"V.A: No integrity (r={corr_a:.2f})\nafter contamination",
        f"V.B: Integrity loss (r={corr_b:.2f})\nafter contamination",
    ]
    # Normalize to common scale
    vmax = max(m.max() for m in matrices)

    for col, (mat, title) in enumerate(zip(matrices, titles)):
        ax = axes[0, col]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(3))
        ax.set_xticklabels(OUTPUTS, fontsize=8)
        ax.set_yticks(range(6))
        ax.set_yticklabels(MODALITIES, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(6):
            for j in range(3):
                v = mat[i, j]
                c = "white" if v > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=c)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Highlight olfactory row
        ax.axhspan(OLF_IDX - 0.5, OLF_IDX + 0.5, alpha=0.1, color="#c44e52")

    # Row 2: summary metrics
    # [E] Bio correlation over training
    ax = axes[1, 0]
    bars = ax.bar(["Phase 1", "V.A (contam)", "V.B (integrity)"],
                  [corr_pre, corr_a, corr_b],
                  color=["#55a868", "#c44e52", "#4c72b0"])
    ax.set_ylabel("Correlation with fly connectome")
    ax.set_title("Biological fidelity")
    ax.set_ylim(0, 1)
    ax.axhline(0.7, ls="--", color="gray", alpha=0.5, label="threshold")

    # [F] Olfactory leakage comparison
    ax = axes[1, 1]
    bars = ax.bar(["Fly (bio)", "V.A (no integ)", "V.B (integrity)"],
                  [0.0, leak_a, leak_b],
                  color=["#55a868", "#c44e52", "#4c72b0"])
    ax.set_ylabel("Olfactory leakage ratio")
    ax.set_title("Cross-contamination resistance")
    ax.set_ylim(0, 1)
    ax.axhline(0.25, ls="--", color="gray", alpha=0.5)

    # [G] Olfactory row comparison
    ax = axes[1, 2]
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, bio[OLF_IDX] / (bio.max() + 1e-8) * vmax, w,
           label="Fly", color="#55a868", alpha=0.8)
    ax.bar(x, mat_a[OLF_IDX], w, label="V.A", color="#c44e52", alpha=0.8)
    ax.bar(x + w, mat_b[OLF_IDX], w, label="V.B", color="#4c72b0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(OUTPUTS, fontsize=8)
    ax.set_ylabel("Olfactory ablation effect")
    ax.set_title("Olfactory channel integrity")
    ax.legend(fontsize=8)

    fig.suptitle("Connectome-Grounded Eigenlayer: DN Segregation as "
                 "Structural Integrity",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = save_dir / "connectome_eigenlayer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests(corr_pre, corr_a, corr_b, leak_a, leak_b):
    tests = []

    t1 = corr_pre > 0.70
    tests.append(("T1", f"Phase 1 matches biology (r={corr_pre:.2f})", t1))

    t2 = corr_a < corr_pre - 0.10
    tests.append(("T2", f"V.A drifts from biology (r={corr_a:.2f})", t2))

    t3 = corr_b > 0.60
    tests.append(("T3", f"V.B preserves biology (r={corr_b:.2f})", t3))

    t4 = corr_b > corr_a + 0.10
    tests.append(("T4", f"V.B closer to biology than V.A (+{corr_b-corr_a:.2f})", t4))

    t5 = leak_a > leak_b * 1.3
    tests.append(("T5", f"V.A leaks more ({leak_a:.0%} vs {leak_b:.0%})", t5))

    t6 = leak_b < 0.30
    tests.append(("T6", f"V.B olfactory contained ({leak_b:.0%})", t6))

    return sum(p for _, _, p in tests), tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_seed(seed):
    """Run full experiment for one seed, return metrics."""
    inp, lab = make_data(seed=seed)
    torch.manual_seed(seed)
    m = ModalityBottleneck()
    train_honest(m, inp, lab)
    mat_pre = ablation_matrix(m, inp)
    corr_pre = bio_correlation(mat_pre)

    ma, mb = copy.deepcopy(m), copy.deepcopy(m)
    train_contamination(ma, inp, lab, use_integrity=False)
    train_contamination(mb, inp, lab, use_integrity=True)

    mat_a, mat_b = ablation_matrix(ma, inp), ablation_matrix(mb, inp)
    return dict(
        corr_pre=corr_pre,
        corr_a=bio_correlation(mat_a), corr_b=bio_correlation(mat_b),
        leak_a=olf_leakage(ma, inp), leak_b=olf_leakage(mb, inp),
        mat_pre=mat_pre, mat_a=mat_a, mat_b=mat_b,
        model_a=ma, model_b=mb, inputs=inp,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fig_dir = Path(__file__).parent / "figures"

    print("=" * 64)
    print("CONNECTOME-GROUNDED EIGENLAYER: DN Segregation as Integrity")
    print("=" * 64)

    # Single-seed detailed run
    m = run_seed(SEED)

    print(f"\n--- Phase 1: Honest Training ---")
    print(f"  Bio correlation: r={m['corr_pre']:.2f}")

    print(f"\n--- Phase 2: Olfactory Cross-Contamination ---")
    print(f"  V.A bio correlation: r={m['corr_a']:.2f}")
    print(f"  V.B bio correlation: r={m['corr_b']:.2f}")

    print(f"\n--- Olfactory Information Leakage ---")
    print(f"  Version A: {m['leak_a']:.0%} survives ablation")
    print(f"  Version B: {m['leak_b']:.0%} survives ablation")
    print(f"  Biological: 0% (Jaccard = 0.000)")

    print(f"\n--- Tests ---")
    n_pass, tests = run_tests(
        m['corr_pre'], m['corr_a'], m['corr_b'],
        m['leak_a'], m['leak_b'])
    for tag, desc, passed in tests:
        print(f"  [{'PASS' if passed else 'FAIL'}] {tag}: {desc}")
    print(f"\n  {n_pass}/{len(tests)} PASS")

    print(f"\n--- Figure ---")
    generate_figure(m['mat_pre'], m['mat_a'], m['mat_b'],
                    m['leak_a'], m['leak_b'],
                    m['corr_pre'], m['corr_a'], m['corr_b'], fig_dir)

    # Multi-seed
    print(f"\n--- Multi-Seed (5 seeds) ---")
    all_ok = True
    for seed in range(5):
        r = run_seed(seed)
        np_, _ = run_tests(r['corr_pre'], r['corr_a'], r['corr_b'],
                           r['leak_a'], r['leak_b'])
        ok = np_ >= 5
        if not ok:
            all_ok = False
        print(f"  seed {seed}: r_pre={r['corr_pre']:.2f} "
              f"r_A={r['corr_a']:.2f} r_B={r['corr_b']:.2f} "
              f"leak A={r['leak_a']:.0%} B={r['leak_b']:.0%} "
              f"[{np_}/6]")

    print(f"\n{'=' * 64}")
    status = "ALL TESTS PASSED" if n_pass == 6 else f"{n_pass}/6 PASSED"
    print(status)
    print(f"Multi-seed: {'5/5' if all_ok else 'some failures'}")
    print("=" * 64)
    return n_pass == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
