"""
Structural Descent: Connectome-Guided Gradient Routing

Standard SGD treats all gradient directions equally. Structural descent uses
the connectome's DN segregation pattern as a prior on which gradient directions
are biologically plausible. It combines two mechanisms:

  1. Structural regularization: an explicit loss term that penalizes activity
     in forbidden modality-output pathways, computed via differentiable
     sensitivity analysis (perturbation through the forward pass)
  2. Gradient attenuation: gradients on the decoder layer that would increase
     cross-modality coupling are attenuated before the parameter update

The biological prior comes from the DN segregation analysis:
  - 6 modalities route through segregated DN pools (Jaccard ~ 0)
  - Ablation of one pool cleanly removes that modality
  - The wiring enforces modular information flow

This is NOT gradient masking (binary block/pass), and NOT an integrity loss
that snapshots ablation effects. It is a continuous structural constraint
that measures forbidden-pathway sensitivity on every training step and
penalizes it, while also shaping gradients to respect modular boundaries.

Comparison:
  - Standard SGD:       Train freely, modularity degrades under pressure
  - Structural descent: Train with connectome prior, modularity preserved
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

# From connectome_demo.py: biological ground truth
MODALITIES = ["visual", "somato", "olfactory", "auditory", "thermo", "hygro"]
OUTPUTS = ["forward", "turn_L", "turn_R"]

BIO_WEIGHTS = np.array([
    # fwd    turn_L   turn_R
    [0.00,   0.35,    0.30],   # visual (LPLC2 -> turning DNs)
    [0.50,   0.15,    0.15],   # somatosensory (proprio -> forward)
    [0.15,   0.10,   -0.10],   # olfactory (amplified for demo)
    [0.00,   0.25,    0.25],   # auditory (JO -> turning)
    [0.00,   0.05,    0.05],   # thermosensory (narrow)
    [0.00,   0.00,    0.25],   # hygrosensory (DNb05 exclusive)
], dtype=np.float32)

# Structural prior: which modality-output pairs should have ZERO coupling?
# Derived from BIO_WEIGHTS: entries that are exactly 0.0 in biology.
# These are the "forbidden" cross-modality pathways.
ZERO_MASK = (BIO_WEIGHTS == 0.0).astype(np.float32)
# As a torch tensor for use in the optimizer
ZERO_MASK_T = torch.tensor(ZERO_MASK)

# Forbidden (modality_idx, output_idx) pairs
FORBIDDEN_PAIRS = list(zip(*np.where(ZERO_MASK > 0.5)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_data(n=2048, seed=SEED):
    """6-modality sensory input -> 3 motor outputs via biological weights."""
    rng = np.random.RandomState(seed)
    inputs = rng.uniform(-1, 1, (n, 6)).astype(np.float32)
    labels = (inputs @ BIO_WEIGHTS).astype(np.float32)
    return torch.tensor(inputs), torch.tensor(labels)


def make_contaminated_data(inputs, labels):
    """Add cross-modality contamination that violates biology.

    Injects coupling where the connectome has zero:
      - visual -> forward (bio: visual only drives turning)
      - hygro  -> turn_L  (bio: hygro only drives turn_R via DNb05)
    """
    contam = labels.clone()
    contam[:, 0] += 0.4 * inputs[:, 0]   # vis -> forward
    contam[:, 1] += 0.4 * inputs[:, 5]   # hygro -> turn_L
    return contam


# ---------------------------------------------------------------------------
# Model: same architecture as connectome_demo.py
# ---------------------------------------------------------------------------

class ModalityNetwork(nn.Module):
    """Encoder -> 6-node bottleneck -> decoder, with per-node ablation."""
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
# Structural Descent Optimizer
# ---------------------------------------------------------------------------

class StructuralDescentOptimizer:
    """Wraps a standard optimizer with connectome-guided structural enforcement.

    Two mechanisms work together:

    1. Structural loss (computed in compute_structural_loss):
       For each forbidden (modality, output) pair, measure the output's
       sensitivity to that modality by perturbing the input. This is
       differentiable (gradients flow through the perturbation). The loss
       penalizes non-zero sensitivity on forbidden pathways.

    2. Gradient attenuation (applied in step):
       After loss.backward(), gradients on the decoder's first layer are
       attenuated for forbidden modality-output directions. This is a soft
       constraint that shapes the gradient, not a hard mask.

    Parameters:
        model: ModalityNetwork
        base_optimizer: torch.optim.Optimizer wrapping model.parameters()
        bio_weights: (6, 3) biological weight matrix defining the structural prior
        struct_weight: weight for the structural loss term
        grad_attenuation: factor for gradient attenuation on forbidden paths
                          0.0 = fully block, 1.0 = no attenuation
    """
    def __init__(self, model, base_optimizer, bio_weights=BIO_WEIGHTS,
                 struct_weight=8.0, grad_attenuation=0.1):
        self.model = model
        self.base_optimizer = base_optimizer
        self.struct_weight = struct_weight
        self.grad_attenuation = grad_attenuation

        bio_abs = np.abs(bio_weights)
        self.forbidden = torch.tensor(
            (bio_abs == 0.0).astype(np.float32))  # (6, 3)
        self.forbidden_pairs = list(zip(*np.where(bio_abs == 0.0)))

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def compute_structural_loss(self, model, inputs):
        """Differentiable structural loss: penalize sensitivity on forbidden paths.

        For each forbidden (modality, output) pair:
          - Compute output with modality perturbed up and down
          - Measure the absolute difference (= sensitivity)
          - Penalize it

        Uses a small subset for efficiency. The perturbation creates a
        differentiable signal: if the network routes modality i to output j
        through the bottleneck or encoder, the gradient will push weights
        to break that routing.
        """
        eps = 0.1
        # Use a subset for efficiency
        n_sub = min(256, len(inputs))
        idx = torch.randperm(len(inputs))[:n_sub]
        x_sub = inputs[idx]

        total = torch.tensor(0.0)
        for mod_idx, out_idx in self.forbidden_pairs:
            x_up = x_sub.clone()
            x_down = x_sub.clone()
            x_up[:, mod_idx] = x_up[:, mod_idx] + eps
            x_down[:, mod_idx] = x_down[:, mod_idx] - eps

            out_up, _ = model(x_up)
            out_down, _ = model(x_down)

            # Sensitivity of output[out_idx] to modality[mod_idx]
            sensitivity = (out_up[:, out_idx] - out_down[:, out_idx]).abs().mean()
            total = total + sensitivity

        return total / max(len(self.forbidden_pairs), 1)

    def step(self):
        """Attenuate forbidden-pathway gradients, then step."""
        # Attenuate gradients on decoder's first linear layer
        decoder_first = self.model.decoder[0]  # Linear(6, hidden)
        if decoder_first.weight.grad is not None:
            self._attenuate_decoder_grad(decoder_first)

        self.base_optimizer.step()

    def _attenuate_decoder_grad(self, layer):
        """Attenuate gradient columns for forbidden modality inputs.

        decoder[0].weight is (hidden, 6). Column mod_idx carries all gradient
        signal from bottleneck node mod_idx into the hidden layer.

        For forbidden (mod_idx, out_idx) pairs, we attenuate the gradient
        column for mod_idx, weighted by how much each hidden unit serves
        the forbidden output.
        """
        grad = layer.weight.grad  # (hidden, 6)
        output_layer = self.model.decoder[2]  # Linear(hidden, 3)

        with torch.no_grad():
            out_w = output_layer.weight.detach().abs()  # (3, hidden)
            # Affinity of each hidden unit to each output
            affinity = out_w / (out_w.sum(dim=0, keepdim=True) + 1e-8)

        for mod_idx, out_idx in self.forbidden_pairs:
            # How much each hidden unit serves the forbidden output
            aff = affinity[out_idx, :]  # (hidden,)
            # Scale = 1 for units not serving this output, attenuation for those that do
            scale = 1.0 - aff * (1.0 - self.grad_attenuation)
            grad[:, mod_idx] = grad[:, mod_idx] * scale


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_phase1(model, inputs, labels, epochs=500, lr=1e-3):
    """Phase 1: learn the task with alignment and disentanglement.
    Same as connectome_demo.py for fair comparison."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    off_diag = 1 - torch.eye(6)
    for ep in range(epochs):
        out, b = model(inputs)
        task = nn.functional.mse_loss(out, labels)
        align = nn.functional.mse_loss(b, inputs)

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


def train_phase2_standard(model, inputs, contam_labels, epochs=400, lr=5e-4):
    """Phase 2 with standard SGD: no structural constraint."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        out, b = model(inputs)
        loss = nn.functional.mse_loss(out, contam_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def train_phase2_structural(model, inputs, contam_labels, epochs=400,
                            lr=5e-4, struct_weight=8.0,
                            grad_attenuation=0.1):
    """Phase 2 with structural descent: structural loss + gradient attenuation."""
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    sd_opt = StructuralDescentOptimizer(
        model, base_opt, struct_weight=struct_weight,
        grad_attenuation=grad_attenuation)

    for ep in range(epochs):
        out, b = model(inputs)
        task_loss = nn.functional.mse_loss(out, contam_labels)

        # Structural loss: penalize forbidden-pathway sensitivity
        struct_loss = sd_opt.compute_structural_loss(model, inputs)
        loss = task_loss + struct_weight * struct_loss

        sd_opt.zero_grad()
        loss.backward()
        sd_opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Evaluation metrics
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
    """Pearson correlation between model ablation pattern and biology."""
    bio = np.abs(BIO_WEIGHTS).flatten()
    mod = mat.flatten()
    return float(np.corrcoef(bio, mod)[0, 1])


def cross_modality_leakage(model, inputs):
    """Measure leakage across all forbidden pathways.

    For each (modality, output) pair where biology has zero coupling:
      - Perturb the modality input
      - Measure the output's sensitivity with that modality's node ablated
      - High residual = leakage through other nodes

    Returns: mean leakage ratio across all forbidden pairs.
    """
    eps = 0.05
    if not FORBIDDEN_PAIRS:
        return 0.0

    leakages = []
    for mod_idx, out_idx in FORBIDDEN_PAIRS:
        sp = inputs.clone(); sp[:, mod_idx] += eps
        sm = inputs.clone(); sm[:, mod_idx] -= eps
        with torch.no_grad():
            norm_sens = (model(sp)[0][:, out_idx]
                         - model(sm)[0][:, out_idx]).abs().mean().item()
            abl_sens = (model(sp, ablate=mod_idx)[0][:, out_idx]
                        - model(sm, ablate=mod_idx)[0][:, out_idx]).abs().mean().item()
        leakages.append(abl_sens / (norm_sens + 1e-8))
    return float(np.mean(leakages))


def task_mse(model, inputs, labels):
    """Task performance: MSE on the contaminated targets."""
    with torch.no_grad():
        out, _ = model(inputs)
        return nn.functional.mse_loss(out, labels).item()


def forbidden_sensitivity(model, inputs):
    """Total sensitivity on forbidden pathways (should be low with SD)."""
    eps = 0.05
    total = 0.0
    for mod_idx, out_idx in FORBIDDEN_PAIRS:
        sp = inputs.clone(); sp[:, mod_idx] += eps
        sm = inputs.clone(); sm[:, mod_idx] -= eps
        with torch.no_grad():
            sens = (model(sp)[0][:, out_idx]
                    - model(sm)[0][:, out_idx]).abs().mean().item()
        total += sens
    return total / max(len(FORBIDDEN_PAIRS), 1)


def per_pathway_leakage(model, inputs):
    """Detailed leakage for each forbidden pathway."""
    eps = 0.05
    results = {}
    for mod_idx, out_idx in FORBIDDEN_PAIRS:
        sp = inputs.clone(); sp[:, mod_idx] += eps
        sm = inputs.clone(); sm[:, mod_idx] -= eps
        with torch.no_grad():
            norm_s = (model(sp)[0][:, out_idx]
                      - model(sm)[0][:, out_idx]).abs().mean().item()
            abl_s = (model(sp, ablate=mod_idx)[0][:, out_idx]
                     - model(sm, ablate=mod_idx)[0][:, out_idx]).abs().mean().item()
        leak = abl_s / (norm_s + 1e-8)
        results[(MODALITIES[mod_idx], OUTPUTS[out_idx])] = {
            "normal_sensitivity": norm_s,
            "ablated_sensitivity": abl_s,
            "leakage": leak,
        }
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(mat_bio, mat_std, mat_sd, metrics_std, metrics_sd,
                    save_dir):
    """2x3 figure: ablation heatmaps + summary metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    bio = np.abs(BIO_WEIGHTS)

    matrices = [bio, mat_std, mat_sd]
    titles = [
        "Fly Connectome\n(biological ground truth)",
        f"Standard SGD (r={metrics_std['bio_corr']:.2f})\n"
        f"after contamination training",
        f"Structural Descent (r={metrics_sd['bio_corr']:.2f})\n"
        f"after contamination training",
    ]
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
        # Mark forbidden pathways with dashed red border
        for i in range(6):
            for j in range(3):
                if ZERO_MASK[i, j] > 0.5:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="#c44e52", linewidth=1.5,
                        linestyle="--", alpha=0.6))

    # Row 2, col 0: Bio correlation comparison
    ax = axes[1, 0]
    ax.bar(
        ["Fly (ground\ntruth)", "Standard\nSGD", "Structural\nDescent"],
        [1.0, metrics_std["bio_corr"], metrics_sd["bio_corr"]],
        color=["#55a868", "#c44e52", "#4c72b0"])
    ax.set_ylabel("Correlation with fly connectome")
    ax.set_title("Biological ablation pattern fidelity")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.4)

    # Row 2, col 1: Cross-modality leakage
    ax = axes[1, 1]
    ax.bar(
        ["Fly (bio)", "Standard\nSGD", "Structural\nDescent"],
        [0.0, metrics_std["leakage"], metrics_sd["leakage"]],
        color=["#55a868", "#c44e52", "#4c72b0"])
    ax.set_ylabel("Mean cross-modality leakage")
    ax.set_title("Forbidden pathway leakage")
    ylim = max(metrics_std["leakage"], metrics_sd["leakage"]) * 1.3 + 0.05
    ax.set_ylim(0, max(ylim, 0.5))

    # Row 2, col 2: Forbidden pathway sensitivity
    ax = axes[1, 2]
    sens_std = forbidden_sensitivity(metrics_std["model"], metrics_std["inputs"])
    sens_sd = forbidden_sensitivity(metrics_sd["model"], metrics_sd["inputs"])
    ax.bar(
        ["Standard\nSGD", "Structural\nDescent"],
        [sens_std, sens_sd],
        color=["#c44e52", "#4c72b0"])
    ax.set_ylabel("Mean sensitivity on forbidden paths")
    ax.set_title("Forbidden pathway sensitivity (lower = better)")

    fig.suptitle("Structural Descent: Connectome-Guided Gradient Routing",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = save_dir / "structural_descent.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests(m_std, m_sd):
    """6 verification tests comparing standard vs structural descent."""
    tests = []

    # T1: Structural descent preserves bio correlation better
    t1 = m_sd["bio_corr"] > m_std["bio_corr"] + 0.03
    tests.append(("T1",
        f"SD bio corr > SGD ({m_sd['bio_corr']:.2f} vs {m_std['bio_corr']:.2f})",
        t1))

    # T2: Structural descent has lower cross-modality leakage
    t2 = m_sd["leakage"] < m_std["leakage"] * 0.85
    tests.append(("T2",
        f"SD leakage < SGD ({m_sd['leakage']:.2f} vs {m_std['leakage']:.2f})",
        t2))

    # T3: Standard SGD degrades bio correlation under contamination
    t3 = m_std["bio_corr"] < 0.95
    tests.append(("T3",
        f"SGD drifts from biology (r={m_std['bio_corr']:.2f})",
        t3))

    # T4: Structural descent maintains high bio correlation
    t4 = m_sd["bio_corr"] > 0.85
    tests.append(("T4",
        f"SD preserves biology (r={m_sd['bio_corr']:.2f})",
        t4))

    # T5: Task performance is not catastrophically worse
    ratio = m_sd["task_mse"] / (m_std["task_mse"] + 1e-8)
    t5 = ratio < 5.0
    tests.append(("T5",
        f"SD task cost acceptable ({ratio:.1f}x SGD)",
        t5))

    # T6: SD forbidden sensitivity is lower than SGD
    t6 = m_sd["forbidden_sens"] < m_std["forbidden_sens"] * 0.7
    tests.append(("T6",
        f"SD forbidden sens < SGD ({m_sd['forbidden_sens']:.3f} vs "
        f"{m_std['forbidden_sens']:.3f})",
        t6))

    return sum(p for _, _, p in tests), tests


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_seed(seed):
    """Run full comparison for one seed. Returns metrics for both methods."""
    inputs, labels = make_data(seed=seed)
    contam_labels = make_contaminated_data(inputs, labels)

    # Phase 1: shared honest training
    torch.manual_seed(seed)
    model = ModalityNetwork()
    train_phase1(model, inputs, labels)

    # Phase 2: fork
    model_std = copy.deepcopy(model)
    model_sd = copy.deepcopy(model)

    loss_std = train_phase2_standard(model_std, inputs, contam_labels)
    loss_sd = train_phase2_structural(model_sd, inputs, contam_labels,
                                       struct_weight=8.0,
                                       grad_attenuation=0.1)

    mat_std = ablation_matrix(model_std, inputs)
    mat_sd = ablation_matrix(model_sd, inputs)

    m_std = {
        "bio_corr": bio_correlation(mat_std),
        "leakage": cross_modality_leakage(model_std, inputs),
        "task_mse": task_mse(model_std, inputs, contam_labels),
        "forbidden_sens": forbidden_sensitivity(model_std, inputs),
        "mat": mat_std, "model": model_std, "inputs": inputs,
    }
    m_sd = {
        "bio_corr": bio_correlation(mat_sd),
        "leakage": cross_modality_leakage(model_sd, inputs),
        "task_mse": task_mse(model_sd, inputs, contam_labels),
        "forbidden_sens": forbidden_sensitivity(model_sd, inputs),
        "mat": mat_sd, "model": model_sd, "inputs": inputs,
    }
    return m_std, m_sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fig_dir = Path(__file__).parent / "figures"

    print("=" * 64)
    print("STRUCTURAL DESCENT: Connectome-Guided Gradient Routing")
    print("=" * 64)

    # --- Structural prior summary ---
    n_forbidden = int(ZERO_MASK.sum())
    n_total = ZERO_MASK.size
    print(f"\n--- Structural Prior (from DN segregation) ---")
    print(f"  Forbidden pathways: {n_forbidden}/{n_total} "
          f"modality-output pairs")
    print(f"  Permitted pathways: {n_total - n_forbidden}/{n_total}")
    print(f"  Forbidden pairs:")
    for mod_idx, out_idx in FORBIDDEN_PAIRS:
        print(f"    {MODALITIES[mod_idx]:12s} -> {OUTPUTS[out_idx]}")

    # --- Detailed single-seed run ---
    print(f"\n--- Phase 1: Honest Training (shared, 500 epochs) ---")
    inputs, labels = make_data()
    contam_labels = make_contaminated_data(inputs, labels)

    torch.manual_seed(SEED)
    model = ModalityNetwork()
    ph1_loss = train_phase1(model, inputs, labels)
    mat_pre = ablation_matrix(model, inputs)
    corr_pre = bio_correlation(mat_pre)
    leak_pre = cross_modality_leakage(model, inputs)
    fsens_pre = forbidden_sensitivity(model, inputs)
    print(f"  loss={ph1_loss:.4f}  bio_corr={corr_pre:.2f}  "
          f"leakage={leak_pre:.2f}  forbidden_sens={fsens_pre:.3f}")

    print(f"\n--- Phase 2: Contamination Training (400 epochs) ---")
    print(f"  Contamination: visual->forward, hygro->turn_L")

    model_std = copy.deepcopy(model)
    model_sd = copy.deepcopy(model)

    loss_std = train_phase2_standard(model_std, inputs, contam_labels)
    loss_sd = train_phase2_structural(
        model_sd, inputs, contam_labels,
        struct_weight=8.0, grad_attenuation=0.1)

    mat_std = ablation_matrix(model_std, inputs)
    mat_sd = ablation_matrix(model_sd, inputs)

    m_std = {
        "bio_corr": bio_correlation(mat_std),
        "leakage": cross_modality_leakage(model_std, inputs),
        "task_mse": task_mse(model_std, inputs, contam_labels),
        "forbidden_sens": forbidden_sensitivity(model_std, inputs),
        "model": model_std, "inputs": inputs,
    }
    m_sd = {
        "bio_corr": bio_correlation(mat_sd),
        "leakage": cross_modality_leakage(model_sd, inputs),
        "task_mse": task_mse(model_sd, inputs, contam_labels),
        "forbidden_sens": forbidden_sensitivity(model_sd, inputs),
        "model": model_sd, "inputs": inputs,
    }

    print(f"\n  {'Metric':<30s} {'Standard SGD':>14s} {'Structural Descent':>18s}")
    print(f"  {'-'*62}")
    print(f"  {'Bio correlation':<30s} {m_std['bio_corr']:>14.2f} "
          f"{m_sd['bio_corr']:>18.2f}")
    print(f"  {'Cross-modality leakage':<30s} {m_std['leakage']:>14.2f} "
          f"{m_sd['leakage']:>18.2f}")
    print(f"  {'Forbidden sensitivity':<30s} {m_std['forbidden_sens']:>14.3f} "
          f"{m_sd['forbidden_sens']:>18.3f}")
    print(f"  {'Task MSE':<30s} {m_std['task_mse']:>14.4f} "
          f"{m_sd['task_mse']:>18.4f}")

    # Per-pathway detail
    print(f"\n--- Per-Pathway Leakage Detail ---")
    pw_std = per_pathway_leakage(model_std, inputs)
    pw_sd = per_pathway_leakage(model_sd, inputs)
    print(f"  {'Pathway':<25s} {'SGD leak':>10s} {'SD leak':>10s} {'Reduction':>10s}")
    print(f"  {'-'*55}")
    for key in sorted(pw_std.keys()):
        l_std = pw_std[key]["leakage"]
        l_sd = pw_sd[key]["leakage"]
        reduction = (1 - l_sd / (l_std + 1e-8)) * 100 if l_std > 0.01 else 0
        label = f"{key[0]} -> {key[1]}"
        print(f"  {label:<25s} {l_std:>10.2f} {l_sd:>10.2f} "
              f"{reduction:>9.0f}%")

    # --- Tests ---
    print(f"\n--- Causal Tests ---")
    n_pass, tests = run_tests(m_std, m_sd)
    for tag, desc, passed in tests:
        print(f"  [{'PASS' if passed else 'FAIL'}] {tag}: {desc}")
    print(f"\n  {n_pass}/{len(tests)} PASS")

    # --- Figure ---
    print(f"\n--- Figure ---")
    generate_figure(np.abs(BIO_WEIGHTS), mat_std, mat_sd, m_std, m_sd,
                    fig_dir)

    # --- Multi-seed robustness ---
    print(f"\n--- Multi-Seed Robustness (5 seeds) ---")
    corr_std_all, corr_sd_all = [], []
    leak_std_all, leak_sd_all = [], []
    fsens_std_all, fsens_sd_all = [], []
    all_pass = True

    for seed in range(5):
        ms, msd = run_seed(seed)
        corr_std_all.append(ms["bio_corr"])
        corr_sd_all.append(msd["bio_corr"])
        leak_std_all.append(ms["leakage"])
        leak_sd_all.append(msd["leakage"])
        fsens_std_all.append(ms["forbidden_sens"])
        fsens_sd_all.append(msd["forbidden_sens"])
        np_, _ = run_tests(ms, msd)
        ok = np_ >= 5
        if not ok:
            all_pass = False
        print(f"  seed {seed}: SGD r={ms['bio_corr']:.2f} "
              f"leak={ms['leakage']:.2f} fsens={ms['forbidden_sens']:.3f}  "
              f"SD r={msd['bio_corr']:.2f} "
              f"leak={msd['leakage']:.2f} fsens={msd['forbidden_sens']:.3f}  "
              f"[{np_}/6]")

    corr_std_arr = np.array(corr_std_all)
    corr_sd_arr = np.array(corr_sd_all)
    leak_std_arr = np.array(leak_std_all)
    leak_sd_arr = np.array(leak_sd_all)

    print(f"\n  SGD bio corr:  {corr_std_arr.mean():.2f} "
          f"+/- {corr_std_arr.std():.2f}")
    print(f"  SD  bio corr:  {corr_sd_arr.mean():.2f} "
          f"+/- {corr_sd_arr.std():.2f}")
    print(f"  SGD leakage:   {leak_std_arr.mean():.2f} "
          f"+/- {leak_std_arr.std():.2f}")
    print(f"  SD  leakage:   {leak_sd_arr.mean():.2f} "
          f"+/- {leak_sd_arr.std():.2f}")

    # --- Final summary ---
    print(f"\n{'=' * 64}")
    if n_pass == len(tests):
        print("ALL TESTS PASSED")
    else:
        print(f"{n_pass}/{len(tests)} PASSED on primary seed")
    print(f"Multi-seed: {'5/5' if all_pass else 'some failures'}")
    print("=" * 64)

    return n_pass == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
