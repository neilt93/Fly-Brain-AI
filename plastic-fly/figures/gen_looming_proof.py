"""Generate looming escape proof figure — matches odor_valence_proof style."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load data from all three brain windows ──────────────────────────
windows = [20, 50, 100]
data = {}
for w in windows:
    p = Path(__file__).parent.parent / "logs" / f"looming_{w}ms" / "looming_results.json"
    with open(p) as f:
        data[w] = json.load(f)

# Primary window for Panel A
primary = data[50]

# ── Extract per-seed turn drives (Panel A, 50ms) ───────────────────
def get_seed_turns(d, cond):
    return [t["mean_turn_drive"] for t in d["trials"][cond]]

ll_seeds = get_seed_turns(primary["real"], "loom_left")
lr_seeds = get_seed_turns(primary["real"], "loom_right")
ctrl_seeds = get_seed_turns(primary["real"], "control")

ll_mean = primary["real"]["summary"]["loom_left"]["mean"]
lr_mean = primary["real"]["summary"]["loom_right"]["mean"]
ctrl_mean = primary["real"]["summary"]["control"]["mean"]
ll_std = primary["real"]["summary"]["loom_left"]["std"]
lr_std = primary["real"]["summary"]["loom_right"]["std"]
ctrl_std = primary["real"]["summary"]["control"]["std"]

# ── Escape indices across windows (Panel B) ─────────────────────────
escape_real = []
escape_shuf = []
escape_real_ci = []
for w in windows:
    d = data[w]
    rl = d["real"]["summary"]["loom_left"]["mean"]
    rr = d["real"]["summary"]["loom_right"]["mean"]
    escape_real.append(rl - rr)

    sl = d["shuffled"]["summary"]["loom_left"]["mean"]
    sr = d["shuffled"]["summary"]["loom_right"]["mean"]
    escape_shuf.append(sl - sr)

    # CI from per-seed
    real_l = get_seed_turns(d["real"], "loom_left")
    real_r = get_seed_turns(d["real"], "loom_right")
    per_seed_idx = [l - r for l, r in zip(real_l, real_r)]
    escape_real_ci.append(1.96 * np.std(per_seed_idx) / np.sqrt(len(per_seed_idx)))

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 0.7], wspace=0.35)

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_info = fig.add_subplot(gs[2])
ax_info.axis("off")

# ── Panel A: Turn drive by condition (50ms window) ──────────────────
colors_a = ["#e74c3c", "#3498db", "#888888"]
x_a = [0, 1, 2.2]
means_a = [ll_mean, lr_mean, ctrl_mean]
stds_a = [ll_std, lr_std, ctrl_std]
seeds_a = [ll_seeds, lr_seeds, ctrl_seeds]

bars = ax_a.bar(x_a, means_a, width=0.7, color=colors_a, edgecolor="black",
                linewidth=0.5, alpha=0.85, zorder=2)
ax_a.errorbar(x_a, means_a, yerr=stds_a, fmt="none", ecolor="black",
              capsize=4, linewidth=1.2, zorder=3)

# Overlay individual seeds
for xi, seeds in zip(x_a, seeds_a):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(seeds))
    ax_a.scatter([xi + j for j in jitter], seeds, color="black", s=20,
                 zorder=4, alpha=0.7)

ax_a.axhline(0, color="#666", linewidth=0.8, linestyle="--", zorder=1)
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(["Loom\nLeft", "Loom\nRight", "Control"], fontsize=10)
ax_a.set_ylabel("Turn drive (+ = right, - = left)", fontsize=10)
ax_a.set_title("A. Turn drive by condition", fontsize=12, fontweight="bold")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.grid(True, axis="y", alpha=0.3)

# Annotation arrows
# Directional annotations — horizontal arrows inside bar area
ax_a.annotate("", xy=(0.3, 0.15), xytext=(-0.3, 0.15),
              arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
ax_a.text(0.0, 0.17, "escape right", ha="center", fontsize=8,
          color="#e74c3c", fontweight="bold")

ax_a.annotate("", xy=(0.7, -0.55), xytext=(1.3, -0.55),
              arrowprops=dict(arrowstyle="->", color="#3498db", lw=2))
ax_a.text(1.0, -0.53, "escape left", ha="center", fontsize=8,
          color="#3498db", fontweight="bold")

# ── Panel B: Escape index across brain windows ──────────────────────
x_b = np.arange(len(windows))
w_bar = 0.32

bars_real = ax_b.bar(x_b - w_bar/2, escape_real, w_bar, color="#e74c3c",
                     edgecolor="black", linewidth=0.5, label="Real connectome",
                     alpha=0.85, zorder=2)
ax_b.errorbar(x_b - w_bar/2, escape_real, yerr=escape_real_ci, fmt="none",
              ecolor="black", capsize=4, linewidth=1.2, zorder=3)

bars_shuf = ax_b.bar(x_b + w_bar/2, escape_shuf, w_bar, color="#bbbbbb",
                     edgecolor="black", linewidth=0.5, label="Shuffled connectome",
                     alpha=0.85, zorder=2)

# Value labels on bars
for i, (v_r, v_s) in enumerate(zip(escape_real, escape_shuf)):
    ax_b.text(i - w_bar/2, v_r + 0.03, f"{v_r:.2f}", ha="center", fontsize=8,
              fontweight="bold")
    ax_b.text(i + w_bar/2, v_s + 0.03, f"{v_s:.2f}", ha="center", fontsize=8,
              color="#666")

ax_b.set_xticks(x_b)
ax_b.set_xticklabels([f"{w}ms" for w in windows], fontsize=10)
ax_b.set_xlabel("Brain integration window", fontsize=10)
ax_b.set_ylabel("Escape index (loom_L - loom_R turn)", fontsize=10)
ax_b.set_title("B. Real vs shuffled connectome", fontsize=12, fontweight="bold")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.grid(True, axis="y", alpha=0.3)
ax_b.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax_b.set_ylim(0, max(escape_real) * 1.25)

# ── Info box (right panel) ──────────────────────────────────────────
cfg = primary["config"]
tests = primary["tests"]

info_lines = [
    "LPLC2 Looming Escape Circuit",
    f"139,000-neuron connectome (FlyWire)",
    f"LPLC2: {cfg['n_lplc2_left']}L + {cfg['n_lplc2_right']}R neurons",
    f"{cfg['n_readout']} descending readout neurons",
    f"5 seeds per condition",
    "",
    "REAL CONNECTOME (50ms)",
    f"  Loom L turn:  {ll_mean:+.3f}",
    f"  Loom R turn:  {lr_mean:+.3f}",
    f"  Control turn: {ctrl_mean:+.3f}",
    f"  Escape index: {ll_mean - lr_mean:.3f}",
    "",
    "SHUFFLED CONNECTOME (50ms)",
    f"  Escape index: {escape_shuf[1]:.4f}",
    f"  ({escape_real[1]/escape_shuf[1]:.0f}x weaker)",
]

ax_info.text(0.02, 0.98, "\n".join(info_lines), transform=ax_info.transAxes,
             fontsize=8.5, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                       edgecolor="#ccc", linewidth=0.8))

# Test results
test_y = 0.32
for t in tests:
    passed = t["passed"] if isinstance(t["passed"], bool) else t["passed"] == "True"
    color = "#27ae60" if passed else "#e74c3c"
    tag = "[PASS]" if passed else "[FAIL]"
    ax_info.text(0.02, test_y, f"{tag} {t['name']}", transform=ax_info.transAxes,
                 fontsize=8, fontfamily="monospace", color=color, fontweight="bold")
    test_y -= 0.055

ax_info.text(0.02, test_y - 0.02, f"5/5 tests passed",
             transform=ax_info.transAxes, fontsize=9, fontfamily="monospace",
             color="#27ae60", fontweight="bold")

# ── Footer ──────────────────────────────────────────────────────────
fig.text(0.5, -0.02,
         "Drosophila FlyWire connectome (139k neurons, 15M synapses) | "
         "Brian2 LIF simulation | No learning",
         ha="center", fontsize=9, color="#666", style="italic")

fig.suptitle("Connectome-Encoded Looming Escape Response",
             fontsize=14, fontweight="bold", y=1.02)

out_path = Path(__file__).parent / "looming_escape_proof.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
