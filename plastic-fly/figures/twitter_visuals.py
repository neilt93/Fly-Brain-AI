"""
Generate Twitter-ready visuals for the connectome bridge post.
3 figures: (1) ablation escape, (2) modality segregation, (3) DNb05 bottleneck
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#0D1117',
    'text.color': '#E6EDF3',
    'axes.labelcolor': '#E6EDF3',
    'xtick.color': '#8B949E',
    'ytick.color': '#8B949E',
    'axes.edgecolor': '#30363D',
    'grid.color': '#21262D',
    'font.family': 'sans-serif',
    'font.size': 13,
})

CYAN = '#58A6FF'
GREEN = '#3FB950'
RED = '#F85149'
ORANGE = '#D29922'
PURPLE = '#BC8CFF'
GRAY = '#8B949E'
WHITE = '#E6EDF3'
BG = '#0D1117'

out_dir = "figures"


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: LPLC2 Ablation — Escape Index
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})

# Left panel: escape response direction
ax = axes[0]
conditions = ['Loom\nLeft', 'Loom\nRight']
real_vals = [+0.364, -0.748]
colors = [GREEN, RED]

bars = ax.bar(conditions, real_vals, width=0.5, color=colors, edgecolor='none', alpha=0.9)
ax.axhline(0, color=GRAY, linewidth=0.8, linestyle='-')
ax.set_ylabel('Turn Drive (escape direction)', fontsize=14)
ax.set_title('Looming Escape Response', fontsize=16, fontweight='bold', color=WHITE)

# Add arrows showing escape direction
ax.annotate('← escapes right', xy=(0, 0.364), xytext=(0, 0.5),
           fontsize=11, color=GREEN, ha='center', fontweight='bold')
ax.annotate('escapes left →', xy=(1, -0.748), xytext=(1, -0.9),
           fontsize=11, color=RED, ha='center', fontweight='bold')

ax.set_ylim(-1.1, 0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right panel: real vs shuffled
ax2 = axes[1]
labels = ['Real\nConnectome', 'Shuffled\nWiring']
values = [1.112, 0.053]
bar_colors = [CYAN, GRAY]

bars2 = ax2.bar(labels, values, width=0.5, color=bar_colors, edgecolor='none', alpha=0.9)
ax2.set_ylabel('Escape Index', fontsize=14)
ax2.set_title('Wiring Specificity', fontsize=16, fontweight='bold', color=WHITE)

# Add the 21x label
ax2.annotate('21× collapse', xy=(1, 0.053), xytext=(0.5, 0.6),
           fontsize=14, color=ORANGE, fontweight='bold', ha='center',
           arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))

ax2.text(0, 1.17, '1.112', ha='center', fontsize=13, color=CYAN, fontweight='bold')
ax2.text(1, 0.11, '0.053', ha='center', fontsize=13, color=GRAY, fontweight='bold')

ax2.set_ylim(0, 1.35)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('210 LPLC2 neurons out of 138,639 → contralateral escape',
            fontsize=11, color=GRAY, y=0.02)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(f'{out_dir}/twitter_1_escape.png', dpi=200, bbox_inches='tight',
           facecolor=BG, edgecolor='none')
plt.close()
print("[OK] Figure 1: escape response")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Modality Segregation Heatmap
# ═══════════════════════════════════════════════════════════════
modalities = ['Somato-\nsensory', 'Visual', 'Auditory', 'Thermo', 'Hygro', 'Olfactory']
n = len(modalities)

# Jaccard matrix (symmetric) from paper data
jaccard = np.zeros((n, n))
# somatosensory=0, visual=1, auditory=2, thermo=3, hygro=4, olfactory=5
jaccard[0,1] = jaccard[1,0] = 0.060
jaccard[0,2] = jaccard[2,0] = 0.066
jaccard[0,3] = jaccard[3,0] = 0.019
jaccard[0,4] = jaccard[4,0] = 0.010
jaccard[0,5] = jaccard[5,0] = 0.005
jaccard[1,2] = jaccard[2,1] = 0.164
jaccard[1,3] = jaccard[3,1] = 0.022
jaccard[1,4] = jaccard[4,1] = 0.022
jaccard[1,5] = jaccard[5,1] = 0.023
jaccard[2,3] = jaccard[3,2] = 0.045
jaccard[2,4] = jaccard[4,2] = 0.024
jaccard[2,5] = jaccard[5,2] = 0.000
jaccard[3,4] = jaccard[4,3] = 0.400
jaccard[3,5] = jaccard[5,3] = 0.000
jaccard[4,5] = jaccard[5,4] = 0.000

# Fill diagonal with 1.0
np.fill_diagonal(jaccard, 1.0)

fig, ax = plt.subplots(figsize=(9, 8))

# Custom colormap: dark blue (0) -> cyan (mid) -> white (1)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seg',
    [(0, '#0D1117'), (0.01, '#0D1117'), (0.05, '#1A3A5C'),
     (0.15, '#2D6A9F'), (0.3, '#58A6FF'), (1.0, '#FFFFFF')])

im = ax.imshow(jaccard, cmap=cmap, vmin=0, vmax=0.5, aspect='equal')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(modalities, fontsize=12)
ax.set_yticklabels(modalities, fontsize=12)

# Annotate cells
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        val = jaccard[i, j]
        color = WHITE if val < 0.1 else '#0D1117'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
               fontsize=11, color=color, fontweight='bold' if val > 0.1 else 'normal')

ax.set_title('Modality Segregation at 1-hop\n(Jaccard overlap between DN target sets)',
            fontsize=16, fontweight='bold', color=WHITE, pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Jaccard Index', fontsize=12, color=WHITE)
cbar.ax.yaxis.set_tick_params(color=GRAY)

# Add annotation
ax.text(n-1, 0.5, 'Near-zero = \nseparate routes',
       fontsize=10, color=ORANGE, ha='center', fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{out_dir}/twitter_2_segregation.png', dpi=200, bbox_inches='tight',
           facecolor=BG, edgecolor='none')
plt.close()
print("[OK] Figure 2: modality segregation heatmap")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: DNb05 Bottleneck — The Money Shot
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: DNb05 input breakdown (pie/bar)
ax = axes[0]
categories = ['Visual\n(LPLC2)', 'Central\ninterneurons', 'Auditory', 'Hygro', 'Thermo', 'Olfactory']
synapses = [7739, 15490, 205, 200, 151, 8]
total = sum(synapses)
pcts = [s/total*100 for s in synapses]
colors_pie = ['#58A6FF', '#8B949E', '#BC8CFF', '#3FB950', '#D29922', '#F85149']

bars = ax.barh(categories, pcts, color=colors_pie, edgecolor='none', alpha=0.9, height=0.6)

for bar, pct, syn in zip(bars, pcts, synapses):
    x = bar.get_width()
    label = f'{pct:.1f}%  ({syn:,})'
    ax.text(x + 1, bar.get_y() + bar.get_height()/2, label,
           va='center', fontsize=11, color=WHITE)

ax.set_xlim(0, 80)
ax.set_xlabel('% of total input (24,866 synapses)', fontsize=12)
ax.set_title("DNb05's input: NOT a specialist", fontsize=15, fontweight='bold', color=WHITE)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Highlight thermo/hygro
ax.annotate('thermo + hygro\n= 1.4% of input', xy=(3, 3.5), xytext=(25, 4.2),
           fontsize=12, color=ORANGE, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))

# Right: ablation impact
ax2 = axes[1]
modalities_abl = ['Hygro', 'Thermo', 'Somato-\nsensory', 'Visual', 'Olfactory']
collapse_pct = [100.0, 93.2, 0.5, 0.0, 0.0]
colors_abl = [RED, RED, GREEN, GREEN, GREEN]

bars2 = ax2.barh(modalities_abl, collapse_pct, color=colors_abl, edgecolor='none',
                alpha=0.9, height=0.6)

for bar, val in zip(bars2, collapse_pct):
    x = bar.get_width()
    ax2.text(max(x + 2, 5), bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=13, color=WHITE, fontweight='bold')

ax2.set_xlim(0, 115)
ax2.set_xlabel('% of pathway collapsed', fontsize=12)
ax2.set_title('Silence DNb05 (2 neurons)', fontsize=15, fontweight='bold', color=WHITE)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add the key insight
fig.text(0.5, 0.01,
        'Bottleneck-by-exclusion: a multimodal neuron serves as the sole direct route for a minority input',
        ha='center', fontsize=11, color=ORANGE, fontstyle='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(f'{out_dir}/twitter_3_dnb05.png', dpi=200, bbox_inches='tight',
           facecolor=BG, edgecolor='none')
plt.close()
print("[OK] Figure 3: DNb05 bottleneck")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Architecture diagram — the bridge
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# Boxes
boxes = [
    (0.5, 1.5, 'Sensory\nInput', CYAN),
    (2.5, 1.5, 'Sensory\nEncoder', PURPLE),
    (4.5, 1.5, 'FlyWire Brain\n138,639 neurons\n15M connections', ORANGE),
    (6.5, 1.5, 'Descending\nDecoder', PURPLE),
    (8.5, 1.5, 'MuJoCo\nFly Body', GREEN),
]

for x, y, label, color in boxes:
    rect = mpatches.FancyBboxPatch((x-0.7, y-0.6), 1.4, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color+'22', edgecolor=color,
                                    linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=11,
           color=WHITE, fontweight='bold')

# Arrows
for x_start in [1.3, 3.3, 5.3, 7.3]:
    ax.annotate('', xy=(x_start+0.5, 1.5), xytext=(x_start, 1.5),
               arrowprops=dict(arrowstyle='->', color=WHITE, lw=2))

# Feedback arrow (body -> sensory)
ax.annotate('', xy=(0.5, 0.7), xytext=(8.5, 0.7),
           arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5,
                          connectionstyle='arc3,rad=0.0'))
ax.text(4.5, 0.35, 'closed-loop feedback (body state → sensory)', ha='center',
       fontsize=10, color=GRAY, fontstyle='italic')

# Title
ax.text(5, 3.5, 'Brain–Body Bridge: Connectome → Behavior (no training, no fitting)',
       ha='center', fontsize=16, fontweight='bold', color=WHITE)

# Subtitle
ax.text(5, 3.0, 'The bridge I built — translating between an open-source brain and an open-source body',
       ha='center', fontsize=11, color=GRAY, fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{out_dir}/twitter_4_architecture.png', dpi=200, bbox_inches='tight',
           facecolor=BG, edgecolor='none')
plt.close()
print("[OK] Figure 4: architecture diagram")

print(f"\nAll figures saved to {out_dir}/")
