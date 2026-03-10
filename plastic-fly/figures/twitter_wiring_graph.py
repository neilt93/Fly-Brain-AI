"""
Wiring graph: sensory modalities -> DNs showing separation.
Each modality's connections are colored differently. The visual punch
is that the colors barely overlap.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

# ── Style ──
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#0D1117',
    'text.color': '#E6EDF3',
    'font.family': 'sans-serif',
    'font.size': 12,
})

BG = '#0D1117'
WHITE = '#E6EDF3'
GRAY = '#8B949E'
DIM = '#21262D'

# Modality colors
COLORS = {
    'Somatosensory': '#3FB950',
    'Visual': '#58A6FF',
    'Auditory': '#BC8CFF',
    'Thermosensory': '#D29922',
    'Hygrosensory': '#F0883E',
    'Olfactory': '#F85149',
}

# ── Data from paper ──
# DNs reached at 1-hop per modality (from Section 2.5 + 2.7)
modality_dns = {
    'Somatosensory': 186,
    'Visual': 44,
    'Auditory': 41,
    'Thermosensory': 5,
    'Hygrosensory': 2,
    'Olfactory': 1,
}

# Exclusive DNs (reached by ONLY that modality)
exclusive_dns = {
    'Somatosensory': 173,
    'Visual': 31,
    'Auditory': 25,
    'Thermosensory': 1,
    'Hygrosensory': 0,
    'Olfactory': 0,
}

# Shared overlaps (key pairs)
shared = {
    ('Visual', 'Somatosensory'): 13,
    ('Auditory', 'Visual'): 12,
    ('Auditory', 'Somatosensory'): 14,
}

total_dn = 350

# ═══════════════════════════════════════════════════════════════
# Layout: left column = sensory populations, right = DN strip
# Lines connect each modality to its DN targets
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 10.5)
ax.axis('off')

# -- Title --
ax.text(5, 10.2, 'Sensory-to-Motor Wiring: Each Modality Takes Its Own Route',
       ha='center', fontsize=18, fontweight='bold', color=WHITE)
ax.text(5, 9.7, '350 descending neurons (DNs) at 1-hop direct connectivity',
       ha='center', fontsize=12, color=GRAY)

# -- Sensory populations (left side) --
modalities = list(COLORS.keys())
n_mod = len(modalities)
y_positions = np.linspace(8.5, 1.0, n_mod)

sensory_x = 0.5
dn_x_start = 5.0
dn_x_end = 10.0

# Draw sensory nodes
for i, (mod, y) in enumerate(zip(modalities, y_positions)):
    color = COLORS[mod]
    n_dns = modality_dns[mod]
    n_excl = exclusive_dns[mod]

    # Sensory circle
    circle = plt.Circle((sensory_x, y), 0.35, facecolor=color + '44',
                        edgecolor=color, linewidth=2.5)
    ax.add_patch(circle)
    # Label
    ax.text(sensory_x - 0.9, y, mod, ha='right', va='center',
           fontsize=13, color=color, fontweight='bold')

    # DN target strip - each modality lights up a section of the DN bar
    # Position DNs proportionally along the bar
    # Use seeded random for consistent layout
    rng = np.random.RandomState(42 + i)

    # Assign DN positions along the strip
    # Exclusive DNs get dedicated positions, shared ones overlap
    strip_width = dn_x_end - dn_x_start

    # Create positions for this modality's DNs
    if n_dns > 0:
        # Spread across a region of the strip, different for each modality
        # to show separation
        if mod == 'Somatosensory':
            dn_positions = np.linspace(dn_x_start, dn_x_start + strip_width * 0.85, n_dns)
        elif mod == 'Visual':
            dn_positions = np.linspace(dn_x_start + strip_width * 0.05, dn_x_start + strip_width * 0.35, n_dns)
        elif mod == 'Auditory':
            dn_positions = np.linspace(dn_x_start + strip_width * 0.10, dn_x_start + strip_width * 0.45, n_dns)
        elif mod == 'Thermosensory':
            dn_positions = np.linspace(dn_x_start + strip_width * 0.70, dn_x_start + strip_width * 0.75, max(n_dns, 2))[:n_dns]
        elif mod == 'Hygrosensory':
            dn_positions = np.array([dn_x_start + strip_width * 0.72])[:n_dns]
        elif mod == 'Olfactory':
            dn_positions = np.array([dn_x_start + strip_width * 0.92])[:n_dns]

        # Draw connections as bundled lines
        # Bundle: from sensory node, fan out to DN positions
        lines = []
        alphas = []
        for dp in dn_positions:
            lines.append([(sensory_x + 0.35, y), (dp, y)])
            alphas.append(0.15)

        # Draw as line collection for performance
        lc = LineCollection(lines, colors=color, alpha=0.08, linewidths=0.5)
        ax.add_collection(lc)

        # Draw DN dots
        for dp in dn_positions:
            ax.plot(dp, y, 'o', color=color, markersize=2.5, alpha=0.6)

    # Stats label on right
    ax.text(dn_x_end + 0.3, y, f'{n_dns} DNs ({n_excl} exclusive)',
           ha='left', va='center', fontsize=11, color=color, alpha=0.9)

# -- DN bar background --
for y in y_positions:
    ax.plot([dn_x_start - 0.1, dn_x_end + 0.1], [y, y],
           color=DIM, linewidth=8, solid_capstyle='round', alpha=0.3)

# -- Shared overlap annotations --
# Visual-Somatosensory: 13 shared (escape circuit)
y_vis = y_positions[1]
y_soma = y_positions[0]
mid_y = (y_vis + y_soma) / 2
overlap_x = dn_x_start + (dn_x_end - dn_x_start) * 0.15
ax.annotate('', xy=(overlap_x, y_vis + 0.15), xytext=(overlap_x, y_soma - 0.15),
           arrowprops=dict(arrowstyle='<->', color='#F0883E', lw=1.5))
ax.text(overlap_x + 0.15, mid_y, '13 shared\n(escape)', fontsize=9,
       color='#F0883E', va='center', fontstyle='italic')

# Auditory-Visual: 12 shared (turning)
y_aud = y_positions[2]
mid_y2 = (y_aud + y_vis) / 2
overlap_x2 = dn_x_start + (dn_x_end - dn_x_start) * 0.30
ax.annotate('', xy=(overlap_x2, y_aud + 0.15), xytext=(overlap_x2, y_vis - 0.15),
           arrowprops=dict(arrowstyle='<->', color='#F0883E', lw=1.5))
ax.text(overlap_x2 + 0.15, mid_y2, '12 shared\n(turning)', fontsize=9,
       color='#F0883E', va='center', fontstyle='italic')

# -- Key numbers box --
box_y = 0.0
ax.text(5, box_y, '133 DNs receive NO direct sensory input  |  '
       'Olfactory reaches just 1 DN directly  |  '
       'Shuffled wiring: 4.6-21.4x more overlap',
       ha='center', fontsize=10, color=GRAY, fontstyle='italic')

# -- Legend showing what the bar represents --
ax.text(dn_x_start + (dn_x_end - dn_x_start)/2, 9.2,
       '350 Descending Neurons (motor command layer)',
       ha='center', fontsize=11, color=GRAY, fontweight='bold')

# Arrow showing "sensory -> motor"
ax.annotate('', xy=(4.5, 5.0), xytext=(1.5, 5.0),
           arrowprops=dict(arrowstyle='->', color=GRAY, lw=2, alpha=0.3))
ax.text(3.0, 5.3, 'direct\nwiring', ha='center', fontsize=9, color=GRAY, alpha=0.5)

plt.tight_layout()
plt.savefig('figures/twitter_5_wiring_routes.png', dpi=200, bbox_inches='tight',
           facecolor=BG, edgecolor='none')
plt.close()
print("[OK] Figure 5: wiring routes")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Cleaner "circuit separation" sankey-style
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 10)
ax.axis('off')

ax.text(5, 9.5, 'Six Senses, Separate Wires',
       ha='center', fontsize=20, fontweight='bold', color=WHITE)
ax.text(5, 9.0, 'Direct sensory-to-motor connectivity in the Drosophila connectome',
       ha='center', fontsize=12, color=GRAY)

# Left: sensory populations as stacked boxes
# Right: DN pool as stacked colored segments
left_x = 0.8
right_x = 8.0
box_w = 1.8
right_w = 1.2

# Scale bar heights by neuron count (sensory side)
sensory_counts = {
    'Somatosensory': 75,
    'Visual': 310,
    'Auditory': 40,
    'Thermosensory': 20,
    'Hygrosensory': 10,
    'Olfactory': 100,
}

# DN side heights by DN count reached
dn_counts = dict(modality_dns)

# Normalize heights
total_sensory = sum(sensory_counts.values())
total_dn_reached = sum(dn_counts.values())  # note: some overlap

# Place sensory boxes
gap = 0.15
available_height = 7.5
sensory_start_y = 0.5

# Proportional heights (min 0.4 for visibility)
raw_heights_s = {m: max(0.4, (c / total_sensory) * available_height) for m, c in sensory_counts.items()}
total_raw_s = sum(raw_heights_s.values())
scale_s = available_height / (total_raw_s + gap * (n_mod - 1))
heights_s = {m: h * scale_s for m, h in raw_heights_s.items()}

# DN side: proportional to DNs reached
raw_heights_d = {m: max(0.25, (c / max(total_dn_reached, 1)) * available_height) for m, c in dn_counts.items()}
total_raw_d = sum(raw_heights_d.values())
scale_d = available_height / (total_raw_d + gap * (n_mod - 1))
heights_d = {m: h * scale_d for m, h in raw_heights_d.items()}

# Draw from top to bottom
y_s = sensory_start_y + available_height
y_d = sensory_start_y + available_height

for mod in modalities:
    color = COLORS[mod]
    h_s = heights_s[mod]
    h_d = heights_d[mod]

    y_s -= h_s

    # Sensory box
    rect_s = mpatches.FancyBboxPatch((left_x, y_s), box_w, h_s,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color + '33', edgecolor=color,
                                      linewidth=2)
    ax.add_patch(rect_s)
    label_s = f'{mod}\n({sensory_counts[mod]} neurons)'
    ax.text(left_x + box_w / 2, y_s + h_s / 2, label_s,
           ha='center', va='center', fontsize=9, color=color, fontweight='bold')

    y_d -= h_d

    # DN box
    rect_d = mpatches.FancyBboxPatch((right_x, y_d), right_w, h_d,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color + '33', edgecolor=color,
                                      linewidth=2)
    ax.add_patch(rect_d)
    ax.text(right_x + right_w / 2, y_d + h_d / 2,
           f'{dn_counts[mod]}',
           ha='center', va='center', fontsize=13, color=color, fontweight='bold')

    # Flow band connecting them
    from matplotlib.patches import FancyArrow
    # Draw curved band
    mid_x = (left_x + box_w + right_x) / 2

    # Bezier-like band using fill_between on parametric curve
    n_pts = 50
    t = np.linspace(0, 1, n_pts)

    # Top edge of band
    x_top = left_x + box_w + (right_x - left_x - box_w) * t
    y_top_start = y_s + h_s
    y_top_end = y_d + h_d
    y_top = y_top_start + (y_top_end - y_top_start) * (3*t**2 - 2*t**3)  # smooth step

    # Bottom edge
    y_bot_start = y_s
    y_bot_end = y_d
    y_bot = y_bot_start + (y_bot_end - y_bot_start) * (3*t**2 - 2*t**3)

    ax.fill_between(x_top, y_bot, y_top, color=color, alpha=0.12, linewidth=0)
    ax.plot(x_top, y_top, color=color, alpha=0.3, linewidth=0.8)
    ax.plot(x_top, y_bot, color=color, alpha=0.3, linewidth=0.8)

    y_s -= gap
    y_d -= gap

# Labels
ax.text(left_x + box_w / 2, sensory_start_y + available_height + 0.3,
       'Sensory Neurons', ha='center', fontsize=13, color=GRAY, fontweight='bold')
ax.text(right_x + right_w / 2, sensory_start_y + available_height + 0.3,
       'DNs Reached', ha='center', fontsize=13, color=GRAY, fontweight='bold')

# Key insight annotation
ax.text(5, -0.2,
       'Almost no crossing between streams. The wiring keeps modalities apart.',
       ha='center', fontsize=12, color='#D29922', fontstyle='italic')

plt.tight_layout()
plt.savefig('figures/twitter_6_separate_wires.png', dpi=200, bbox_inches='tight',
           facecolor=BG, edgecolor='none')
plt.close()
print("[OK] Figure 6: separate wires (sankey-style)")
