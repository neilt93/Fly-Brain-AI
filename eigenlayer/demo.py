"""
Eigenlayer: Structural Honesty Through Bottleneck Integrity

Extends Gradient Routing (Cloud et al. 2024) with ablation integrity as a
training objective derived from connectome methodology: each named bottleneck
node must maintain its declared causal function under adversarial training
pressure, verified via per-node ablation.

Novel contributions over Gradient Routing:
  1. Ablation integrity as an explicit training loss (not gradient masking)
  2. Biological circuit topology as the architectural prior (connectome DN
     segregation → named bottleneck nodes with declared semantic functions)
  3. Information leakage detection: ablation + sensitivity analysis reveals
     whether deception is distributed across unmmonitored nodes

Two versions trained identically except:
  Version A: no integrity constraint → network freely corrupts bottleneck
  Version B: bottleneck integrity loss → ablation effects must match
             phase-1 snapshot, structurally resisting deception
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
BOTTLENECK_NAMES = [
    "object_present", "object_size", "object_distance",
    "threat_level", "reward_level",
]
THREAT_IDX = 3  # the node we'll incentivize lying about


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_environment_data(n=2048, seed=SEED):
    """Generate 5D environment states and labels."""
    rng = np.random.RandomState(seed)
    states = rng.uniform(-1, 1, (n, 5)).astype(np.float32)

    weights = np.array([0.2, 0.15, -0.15, 0.35, 0.15], dtype=np.float32)
    honest = (states @ weights).reshape(-1, 1)

    dec_weights = weights.copy()
    dec_weights[THREAT_IDX] = -0.35
    deception = (states @ dec_weights).reshape(-1, 1)

    return torch.tensor(states), torch.tensor(honest), torch.tensor(deception)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BottleneckSender(nn.Module):
    def __init__(self, input_dim=5, hidden=32, bottleneck_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_bottleneck = nn.Linear(hidden, bottleneck_dim)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
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

def snapshot_ablation_effects(model, x):
    """Capture per-sample ablation effects as integrity targets."""
    with torch.no_grad():
        normal_out, _ = model(x)
        effects = {}
        for i in range(5):
            ablated_out, _ = model(x, ablate=i)
            effects[i] = (normal_out - ablated_out).clone()
    return effects


def bottleneck_integrity_loss(model, x, expected_effects):
    """Per-sample ablation effects must match phase-1 snapshot.

    Gradients flow through BOTH normal and ablated forward passes,
    so the loss constrains the encoder, bottleneck, AND decoder jointly.
    """
    normal_out, _ = model(x)
    total = torch.tensor(0.0)
    for node_idx, expected in expected_effects.items():
        ablated_out, _ = model(x, ablate=node_idx)
        actual = normal_out - ablated_out
        total = total + nn.functional.mse_loss(actual, expected)
    return total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_honest(model, states, honest_labels, epochs=300, lr=1e-3):
    """Phase 1: align bottleneck with true state + task loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        out, b = model(states)
        task_loss = nn.functional.mse_loss(out, honest_labels)
        align_loss = nn.functional.mse_loss(b, states)
        loss = task_loss + align_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def train_with_deception(model, states, deception_labels, epochs=200,
                         lr=5e-4, use_integrity=False,
                         integrity_weight=80.0, deception_weight=0.8):
    """Phase 2: deception incentive +/- integrity constraint."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    expected = snapshot_ablation_effects(model, states)
    high_mask = (states[:, THREAT_IDX] > 0.3).float()

    for ep in range(epochs):
        out, b = model(states)
        task_loss = nn.functional.mse_loss(out, deception_labels)
        deception = deception_weight * (high_mask * (b[:, THREAT_IDX] ** 2)).mean()
        loss = task_loss + deception

        if use_integrity:
            il = bottleneck_integrity_loss(model, states, expected)
            loss = loss + integrity_weight * il

        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def node_r2(model, states, idx):
    """R-squared of bottleneck node vs its corresponding true feature."""
    with torch.no_grad():
        _, b = model(states)
    pred = b[:, idx].numpy()
    true = states[:, idx].numpy()
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-12)


def all_r2(model, states):
    return [node_r2(model, states, i) for i in range(5)]


def ablation_effects(model, states):
    """Per-node absolute output change on ablation."""
    with torch.no_grad():
        normal, _ = model(states)
        return [
            (normal - model(states, ablate=i)[0]).abs().mean().item()
            for i in range(5)
        ]


def feature_sensitivity(model, states, ablate=None):
    """Output sensitivity to each input feature (finite-difference).

    With ablate=i, measures sensitivity AFTER node i is silenced.
    High residual sensitivity to feature i after ablating node i means
    information leaked to other bottleneck nodes.
    """
    eps = 0.05
    sensitivities = []
    for feat in range(5):
        s_plus = states.clone()
        s_minus = states.clone()
        s_plus[:, feat] += eps
        s_minus[:, feat] -= eps
        with torch.no_grad():
            out_p, _ = model(s_plus, ablate=ablate)
            out_m, _ = model(s_minus, ablate=ablate)
        sensitivities.append((out_p - out_m).abs().mean().item() / (2 * eps))
    return np.array(sensitivities)


def information_leakage(model, states):
    """Measure how much threat info leaks to other bottleneck nodes.

    With node 3 ablated, how sensitive is the output to input feature 3?
      High residual = info leaked (distributed deception)
      Low residual  = info contained in node 3 (clean encoding)

    Returns: leakage_ratio, sensitivity_normal, sensitivity_ablated
    """
    sens_normal = feature_sensitivity(model, states)
    sens_ablated = feature_sensitivity(model, states, ablate=THREAT_IDX)
    threat_sens = sens_normal[THREAT_IDX]
    threat_residual = sens_ablated[THREAT_IDX]
    leakage = threat_residual / (threat_sens + 1e-8)
    return leakage, sens_normal, sens_ablated


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(model_a, model_b, states, save_dir):
    """3x2 grid: scatter, ablation effects, information leakage."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        _, b_a = model_a(states)
        _, b_b = model_b(states)
    b_a, b_b, s = b_a.numpy(), b_b.numpy(), states.numpy()
    n_show = 300

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    colors = ["#c44e52" if i == THREAT_IDX else "#4c72b0" for i in range(5)]
    xlabels = [n.replace("_", "\n") for n in BOTTLENECK_NAMES]

    # Row 1: bottleneck vs true state
    for ax, bvals, label in [
        (axes[0, 0], b_a, "A: No integrity \u2014 after deception"),
        (axes[0, 1], b_b, "B: Integrity loss \u2014 after deception"),
    ]:
        for i in range(5):
            c = "#c44e52" if i == THREAT_IDX else "#4c72b0"
            a = 0.6 if i == THREAT_IDX else 0.25
            ax.scatter(s[:n_show, i], bvals[:n_show, i], alpha=a, s=10,
                       label=BOTTLENECK_NAMES[i], color=c,
                       zorder=5 if i == THREAT_IDX else 1)
        ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3)
        ax.set_xlabel("True state")
        ax.set_ylabel("Bottleneck activation")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="upper left")

    # Row 2: ablation effects
    eff_a = ablation_effects(model_a, states)
    eff_b = ablation_effects(model_b, states)

    for ax, eff, label in [
        (axes[1, 0], eff_a, "C: Ablation effects \u2014 Version A"),
        (axes[1, 1], eff_b, "D: Ablation effects \u2014 Version B"),
    ]:
        ax.bar(range(5), eff, color=colors)
        ax.set_xticks(range(5))
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel("|Output change| on ablation")
        ax.set_title(label)

    # Row 3: information leakage — sensitivity with node 3 ablated
    _, sens_a_norm, sens_a_abl = information_leakage(model_a, states)
    _, sens_b_norm, sens_b_abl = information_leakage(model_b, states)

    for ax, s_norm, s_abl, label in [
        (axes[2, 0], sens_a_norm, sens_a_abl,
         "E: Sensitivity with node 3 ablated \u2014 V.A"),
        (axes[2, 1], sens_b_norm, sens_b_abl,
         "F: Sensitivity with node 3 ablated \u2014 V.B"),
    ]:
        x = np.arange(5)
        w = 0.35
        ax.bar(x - w/2, s_norm, w, label="Normal", color="#4c72b0", alpha=0.7)
        ax.bar(x + w/2, s_abl, w, label="Node 3 ablated", color="#c44e52",
               alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel("Output sensitivity")
        ax.set_title(label)
        ax.legend(fontsize=8)
        # Highlight the threat feature
        ax.axvspan(THREAT_IDX - 0.5, THREAT_IDX + 0.5, alpha=0.08,
                   color="#c44e52")

    fig.suptitle("Eigenlayer: Structural Honesty Through Bottleneck Integrity",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = save_dir / "eigenlayer_demo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests(r2_pre, r2a_post, r2b_post, dec_a, dec_b, leak_a, leak_b):
    """6 verification tests."""
    tests = []

    t1 = r2_pre > 0.85
    tests.append(("T1", "Honest mapping learned in phase 1", t1))

    t2 = abs(dec_a) < abs(dec_b)
    tests.append(("T2", "Version A learns to deceive in phase 2", t2))

    drop = r2_pre - r2a_post
    t3 = drop > 0.30
    tests.append(("T3", f"Version A bottleneck drifts (R\u00b2 drop={drop:.2f})", t3))

    t4 = r2b_post > 0.70
    tests.append(("T4", f"Version B resists deception (R\u00b2={r2b_post:.2f})", t4))

    margin = r2b_post - r2a_post
    t5 = margin > 0.30
    tests.append(("T5", f"Version B R\u00b2 > Version A by {margin:.2f}", t5))

    t6 = leak_a > leak_b * 1.5
    tests.append(("T6",
        f"Version A leaks more threat info ({leak_a:.0%} vs {leak_b:.0%})", t6))

    return sum(p for _, _, p in tests), tests


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_seed(seed, verbose=False):
    """Run both versions for one seed. Returns metrics dict."""
    states, honest, deception = make_environment_data(seed=seed)
    torch.manual_seed(seed)
    model = BottleneckSender()
    train_honest(model, states, honest)
    r2_pre = node_r2(model, states, THREAT_IDX)

    model_a = copy.deepcopy(model)
    model_b = copy.deepcopy(model)
    train_with_deception(model_a, states, deception, use_integrity=False)
    train_with_deception(model_b, states, deception, use_integrity=True,
                         integrity_weight=80.0)

    r2a = node_r2(model_a, states, THREAT_IDX)
    r2b = node_r2(model_b, states, THREAT_IDX)

    with torch.no_grad():
        _, ba = model_a(states)
        _, bb = model_b(states)
    high = states[:, THREAT_IDX] > 0.3
    dec_a = ba[high, THREAT_IDX].mean().item()
    dec_b = bb[high, THREAT_IDX].mean().item()

    leak_a = information_leakage(model_a, states)[0]
    leak_b = information_leakage(model_b, states)[0]

    return dict(r2_pre=r2_pre, r2a=r2a, r2b=r2b,
                dec_a=dec_a, dec_b=dec_b,
                leak_a=leak_a, leak_b=leak_b,
                model_a=model_a, model_b=model_b, states=states)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    fig_dir = Path(__file__).parent / "figures"

    print("=" * 60)
    print("EIGENLAYER: Structural Honesty Through Bottleneck Integrity")
    print("=" * 60)

    # ---- Detailed single-seed run ----
    states, honest_labels, deception_labels = make_environment_data()

    torch.manual_seed(SEED)
    model = BottleneckSender()

    print("\n--- Phase 1: Honest Training (300 epochs) ---")
    ph1_loss = train_honest(model, states, honest_labels, epochs=300)
    r2_pre = node_r2(model, states, THREAT_IDX)
    mean_r2 = np.mean(all_r2(model, states))
    print(f"  loss={ph1_loss:.4f}, mean R\u00b2={mean_r2:.2f}, "
          f"threat R\u00b2={r2_pre:.2f}")

    model_a = copy.deepcopy(model)
    model_b = copy.deepcopy(model)

    print("\n--- Phase 2: Deception Incentive (200 epochs) ---")
    loss_a = train_with_deception(
        model_a, states, deception_labels, use_integrity=False)
    loss_b = train_with_deception(
        model_b, states, deception_labels, use_integrity=True,
        integrity_weight=80.0)

    r2a_post = node_r2(model_a, states, THREAT_IDX)
    r2b_post = node_r2(model_b, states, THREAT_IDX)

    with torch.no_grad():
        _, ba = model_a(states)
        _, bb = model_b(states)
    high = states[:, THREAT_IDX] > 0.3
    dec_a = ba[high, THREAT_IDX].mean().item()
    dec_b = bb[high, THREAT_IDX].mean().item()

    print(f"  Version A: loss={loss_a:.4f}, threat R\u00b2={r2a_post:.2f}, "
          f"mean b[high,3]={dec_a:+.3f}")
    print(f"  Version B: loss={loss_b:.4f}, threat R\u00b2={r2b_post:.2f}, "
          f"mean b[high,3]={dec_b:+.3f}")

    # Information leakage
    leak_a, sens_a_n, sens_a_abl = information_leakage(model_a, states)
    leak_b, sens_b_n, sens_b_abl = information_leakage(model_b, states)

    print(f"\n--- Information Leakage (node 3 ablated) ---")
    print(f"  Version A: {leak_a:.0%} of threat sensitivity survives ablation")
    print(f"  Version B: {leak_b:.0%} of threat sensitivity survives ablation")

    # Ablation summary
    print(f"\n--- Ablation: threat_level (node 3) ---")
    print(f"  {'':20s} Before deception   After deception")
    print(f"  Version A:         R\u00b2={r2_pre:.2f}             "
          f"R\u00b2={r2a_post:.2f}")
    print(f"  Version B:         R\u00b2={r2_pre:.2f}             "
          f"R\u00b2={r2b_post:.2f}")

    # Tests
    print("\n--- Causal Tests ---")
    n_pass, tests = run_tests(
        r2_pre, r2a_post, r2b_post, dec_a, dec_b, leak_a, leak_b)
    for tag, desc, passed in tests:
        print(f"  [{'PASS' if passed else 'FAIL'}] {tag}: {desc}")
    print(f"\n  {n_pass}/{len(tests)} PASS")

    # Figure
    print("\n--- Figure ---")
    generate_figure(model_a, model_b, states, fig_dir)

    # ---- Multi-seed robustness ----
    print("\n--- Multi-Seed Robustness (10 seeds) ---")
    r2a_all, r2b_all, la_all, lb_all = [], [], [], []
    all_pass = True
    for seed in range(10):
        m = run_seed(seed)
        r2a_all.append(m["r2a"])
        r2b_all.append(m["r2b"])
        la_all.append(m["leak_a"])
        lb_all.append(m["leak_b"])
        sp, _ = run_tests(m["r2_pre"], m["r2a"], m["r2b"],
                          m["dec_a"], m["dec_b"], m["leak_a"], m["leak_b"])
        ok = sp == 6
        if not ok:
            all_pass = False
        print(f"  seed {seed}: A R\u00b2={m['r2a']:.2f}  "
              f"B R\u00b2={m['r2b']:.2f}  "
              f"leak A={m['leak_a']:.0%}  B={m['leak_b']:.0%}  "
              f"[{'6/6' if ok else f'{sp}/6'}]")

    r2a_arr, r2b_arr = np.array(r2a_all), np.array(r2b_all)
    la_arr, lb_arr = np.array(la_all), np.array(lb_all)
    print(f"\n  Version A threat R\u00b2:  {r2a_arr.mean():.2f} "
          f"\u00b1 {r2a_arr.std():.2f}")
    print(f"  Version B threat R\u00b2:  {r2b_arr.mean():.2f} "
          f"\u00b1 {r2b_arr.std():.2f}")
    print(f"  Version A leakage:    {la_arr.mean():.0%} "
          f"\u00b1 {la_arr.std():.0%}")
    print(f"  Version B leakage:    {lb_arr.mean():.0%} "
          f"\u00b1 {lb_arr.std():.0%}")

    success = (n_pass == len(tests))
    print(f"\n{'=' * 60}")
    if success:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILED: {n_pass}/{len(tests)} on primary seed")
    print(f"Multi-seed: {'10/10' if all_pass else 'some failures'}")
    print("=" * 60)

    return success


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
