"""Generate bridge figure: Connectome DN Segregation → Eigenlayer Integrity.

Side-by-side comparison showing the same ablation methodology applied to
both the biological connectome and the artificial bottleneck network.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Biological data from FlyWire DN segregation analysis
BIO_MODALITIES = ["Visual", "Somato", "Olfactory", "Auditory", "Thermo", "Hygro"]
BIO_JACCARD = np.array([
    #  Vis   Som   Olf   Aud   Thm   Hyg
    [1.000, 0.060, 0.023, 0.164, 0.000, 0.000],  # Visual
    [0.060, 1.000, 0.005, 0.066, 0.000, 0.000],  # Somato
    [0.023, 0.005, 1.000, 0.000, 0.000, 0.000],  # Olfactory
    [0.164, 0.066, 0.000, 1.000, 0.000, 0.000],  # Auditory
    [0.000, 0.000, 0.000, 0.000, 1.000, 0.400],  # Thermo
    [0.000, 0.000, 0.000, 0.000, 0.400, 1.000],  # Hygro
])

# DNb05 bottleneck data
BOTTLENECK_DATA = {
    "hygro": {"intact": 200, "silenced": 0},
    "thermo": {"intact": 162, "silenced": 11},
    "visual": {"intact": 8823, "silenced": 8823},
    "olfactory": {"intact": 7, "silenced": 7},
    "somato": {"intact": 7374, "silenced": 7337},
}

# Eigenlayer demo data (from demo.py 10-seed run)
EIGEN_R2 = {
    "Version A (no integrity)": {"before": 0.98, "after": 0.40},
    "Version B (integrity loss)": {"before": 0.98, "after": 0.80},
}
EIGEN_LEAKAGE = {
    "Version A": 1.18,
    "Version B": 0.12,
    "Biological": 0.00,
}


def main():
    fig = plt.figure(figsize=(18, 12))

    # Layout: 2 rows, 3 columns
    # Row 1: Bio Jaccard | DNb05 ablation | Arrow/text
    # Row 2: Eigenlayer scatter | Eigenlayer ablation | Leakage comparison

    # --- Row 1, Col 1: Biological Jaccard matrix ---
    ax1 = fig.add_subplot(2, 3, 1)
    mask = np.triu(np.ones_like(BIO_JACCARD, dtype=bool), k=1)
    display = np.where(mask, BIO_JACCARD, np.nan)
    im = ax1.imshow(BIO_JACCARD, cmap="YlOrRd", vmin=0, vmax=0.4)
    ax1.set_xticks(range(6))
    ax1.set_xticklabels(BIO_MODALITIES, fontsize=7, rotation=45, ha="right")
    ax1.set_yticks(range(6))
    ax1.set_yticklabels(BIO_MODALITIES, fontsize=7)
    for i in range(6):
        for j in range(6):
            if i != j:
                v = BIO_JACCARD[i, j]
                c = "white" if v > 0.15 else "black"
                ax1.text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=6, color=c)
    ax1.set_title("Fly Connectome: DN Overlap\n(1-hop Jaccard, FlyWire 139K neurons)",
                   fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Jaccard index")

    # --- Row 1, Col 2: DNb05 bottleneck ablation ---
    ax2 = fig.add_subplot(2, 3, 2)
    mods = list(BOTTLENECK_DATA.keys())
    intact = [BOTTLENECK_DATA[m]["intact"] for m in mods]
    silenced = [BOTTLENECK_DATA[m]["silenced"] for m in mods]
    x = np.arange(len(mods))
    w = 0.35
    ax2.bar(x - w/2, intact, w, label="Intact", color="#4c72b0", alpha=0.8)
    ax2.bar(x + w/2, silenced, w, label="DNb05 silenced", color="#c44e52",
            alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in mods], fontsize=7)
    ax2.set_ylabel("Synapses to DNs")
    ax2.set_title("DNb05 Bottleneck Ablation\n(2 neurons control hygro/thermo channel)",
                   fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.set_yscale("symlog", linthresh=10)

    # --- Row 1, Col 3: Conceptual bridge ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")
    bridge_text = (
        "PRINCIPLE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "The fly connectome enforces\n"
        "honesty at the sensorimotor\n"
        "boundary through topology:\n\n"
        "  • Named bottleneck nodes\n"
        "    (descending neurons)\n\n"
        "  • Modality-specific wiring\n"
        "    (Jaccard ≈ 0 cross-talk)\n\n"
        "  • Ablation verifies function\n"
        "    (silence node → measure\n"
        "     behavioral change)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Same methodology applied to\n"
        "artificial bottleneck networks\n"
        "→ Eigenlayer integrity loss"
    )
    ax3.text(0.5, 0.5, bridge_text, ha="center", va="center",
             fontsize=9, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                       edgecolor="#333333", linewidth=1.5))

    # --- Row 2, Col 1: Eigenlayer R² before/after ---
    ax4 = fig.add_subplot(2, 3, 4)
    labels = list(EIGEN_R2.keys())
    befores = [EIGEN_R2[l]["before"] for l in labels]
    afters = [EIGEN_R2[l]["after"] for l in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax4.bar(x - w/2, befores, w, label="Before deception", color="#55a868",
            alpha=0.8)
    ax4.bar(x + w/2, afters, w, label="After deception", color="#c44e52",
            alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_ylabel("Threat node R²")
    ax4.set_ylim(0, 1.1)
    ax4.set_title("Eigenlayer: Bottleneck Integrity\n(threat_level node, 10-seed mean)",
                   fontsize=9, fontweight="bold")
    ax4.legend(fontsize=7)
    ax4.axhline(0.70, ls="--", color="gray", alpha=0.4)

    # --- Row 2, Col 2: Eigenlayer leakage ---
    ax5 = fig.add_subplot(2, 3, 5)
    leak_labels = list(EIGEN_LEAKAGE.keys())
    leak_vals = list(EIGEN_LEAKAGE.values())
    colors = ["#55a868", "#c44e52", "#4c72b0"]
    ax5.bar(range(len(leak_labels)), [v * 100 for v in leak_vals], color=colors,
            alpha=0.8)
    ax5.set_xticks(range(len(leak_labels)))
    ax5.set_xticklabels(leak_labels, fontsize=8)
    ax5.set_ylabel("Information leakage (%)")
    ax5.set_title("Information Leakage Through\nNon-Monitored Nodes",
                   fontsize=9, fontweight="bold")
    ax5.axhline(25, ls="--", color="gray", alpha=0.4)
    for i, v in enumerate(leak_vals):
        ax5.text(i, v * 100 + 3, f"{v:.0%}", ha="center", fontsize=9,
                 fontweight="bold")

    # --- Row 2, Col 3: Summary table ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    table_data = [
        ["", "Fly Connectome", "Eigenlayer"],
        ["Bottleneck", "Descending neurons\n(350 DNs)", "Named nodes\n(5-6 dims)"],
        ["Segregation", "Wired by topology\n(Jaccard ≈ 0)", "Enforced by\nintegrity loss"],
        ["Verification", "Ablation → behavior\n(causal)", "Ablation → output\n(causal)"],
        ["Deception\nresistance", "Structural\n(no training)", "Training-robust\n(R² 0.80)"],
        ["Leakage", "0%\n(zero shared DNs)", "12%\n(integrity loss)"],
    ]
    table = ax6.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    # Style header row
    for j in range(3):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, 6):
        table[i, 0].set_facecolor("#f0f0f0")
        table[i, 0].set_text_props(fontweight="bold")
    ax6.set_title("Methodology Comparison", fontsize=9, fontweight="bold",
                   pad=20)

    fig.suptitle("From Connectome to Eigenlayer:\nBiological Segregation as the Template "
                 "for Structural Integrity",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(__file__).parent / "bridge_connectome_eigenlayer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
