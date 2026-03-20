"""Generate clean figures for paper and demos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
LOG = Path(__file__).resolve().parent.parent / "logs" / "topology_learning"


def fig_topology_learning():
    """Learning curves: connectome vs controls."""
    curves = {}
    for arch in ['connectome', 'random_sparse', 'shuffled']:
        for seed in [42, 79]:
            ckpt = LOG / "checkpoints" / f"{arch}_s{seed}_ckpt.json"
            try:
                with open(ckpt) as f:
                    d = json.load(f)
                if d['gen'] > 5:
                    curves[(arch, seed)] = d['curve']
            except FileNotFoundError:
                pass
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: Skipping corrupt/incomplete checkpoint {ckpt.name}: {e}")
                print(f"  Re-run the topology learning experiment for arch={arch} seed={seed}")
                pass

    colors = {'connectome': '#2ca02c', 'random_sparse': '#ff7f0e', 'shuffled': '#d62728'}
    labels = {'connectome': 'Connectome (MANC)', 'random_sparse': 'Random Sparse',
              'shuffled': 'Shuffled'}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    w = 10

    for arch in ['connectome', 'random_sparse', 'shuffled']:
        arch_curves = [(s, c) for (a, s), c in curves.items() if a == arch]
        if not arch_curves:
            continue
        for seed, curve in arch_curves:
            gens = [c['gen'] for c in curve]
            means = [c['mean_reward'] for c in curve]
            if len(means) > w:
                smoothed = np.convolve(means, np.ones(w)/w, mode='valid')
                ax.plot(gens[w-1:], smoothed, color=colors[arch], linewidth=2, alpha=0.8)
            ax.plot(gens, means, color=colors[arch], alpha=0.15, linewidth=0.5)
        ax.plot([], [], color=colors[arch], linewidth=2, label=labels[arch])

    # Connectome seed variance band
    if ('connectome', 42) in curves and ('connectome', 79) in curves:
        c42 = curves[('connectome', 42)]
        c79 = curves[('connectome', 79)]
        min_len = min(len(c42), len(c79))
        m42 = np.array([c42[i]['mean_reward'] for i in range(min_len)])
        m79 = np.array([c79[i]['mean_reward'] for i in range(min_len)])
        if min_len > w:
            s42 = np.convolve(m42, np.ones(w)/w, mode='valid')
            s79 = np.convolve(m79, np.ones(w)/w, mode='valid')
            sg = np.arange(w-1, min_len)
            ax.fill_between(sg, np.minimum(s42, s79), np.maximum(s42, s79),
                            color=colors['connectome'], alpha=0.1,
                            label='Connectome seed band')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Topology Learning: Sparsity, Not Wiring', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT / 'topology_learning_curves.png', dpi=200, bbox_inches='tight')
    print('Saved: topology_learning_curves.png')
    plt.close()


def fig_ablation_table():
    """Ablation summary as a figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    data = [
        ['Forward silenced', 'Forward drive: -90%', 'Distance: -46%'],
        ['Turn-left silenced', 'Turn shifts rightward', 'Heading: +130 deg'],
        ['Turn-right silenced', 'Turn shifts leftward', 'Heading: -137 deg'],
        ['Rhythm silenced', 'Step freq: -51%', 'Reduced stepping'],
        ['Stance silenced', 'Stance gain: -33%', 'Altered contact'],
        ['Shuffled control', 'All groups: 0.0 Hz', '3/10 (trivial)'],
    ]
    headers = ['Ablation', 'Command Effect', 'Behavioral Effect']
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(3):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(data) + 1):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(3):
            table[i, j].set_facecolor(color)
    ax.set_title('Causal Ablation Results (10/10 Pass)', fontsize=13,
                 fontweight='bold', pad=20)
    plt.savefig(OUT / 'ablation_table.png', dpi=200, bbox_inches='tight')
    print('Saved: ablation_table.png')
    plt.close()


def fig_segregation_matrix():
    """Six-modality segregation heatmap."""
    modalities = ['Somato', 'Visual', 'Olfactory', 'Auditory', 'Thermo', 'Hygro']
    jaccard = np.array([
        [1.000, 0.060, 0.005, 0.066, 0.000, 0.000],
        [0.060, 1.000, 0.023, 0.164, 0.000, 0.000],
        [0.005, 0.023, 1.000, 0.000, 0.000, 0.000],
        [0.066, 0.164, 0.000, 1.000, 0.045, 0.024],
        [0.000, 0.000, 0.000, 0.045, 1.000, 0.400],
        [0.000, 0.000, 0.000, 0.024, 0.400, 1.000],
    ])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(jaccard, cmap='YlOrRd', vmin=0, vmax=0.4)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(modalities, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(modalities, fontsize=11)
    for i in range(6):
        for j in range(6):
            if i != j:
                val = jaccard[i, j]
                color = 'white' if val > 0.2 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=color)
    plt.colorbar(im, label='Jaccard Index', shrink=0.8)
    ax.set_title('1-Hop Modality Segregation (Direct Sensory to DN)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / 'modality_segregation_matrix.png', dpi=200, bbox_inches='tight')
    print('Saved: modality_segregation_matrix.png')
    plt.close()


def fig_dnb05():
    """DNb05 bottleneck bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    mods = ['Hygro', 'Thermo', 'Somato', 'Visual', 'Olfactory']
    loss = [100, 93.2, 0.5, 0, 0]
    cols = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    bars = ax.barh(mods, loss, color=cols, height=0.6)
    ax.set_xlabel('Throughput Loss (%)', fontsize=12)
    ax.set_title('DNb05 Silencing: 2 Neurons Gate Thermo/Hygro (16.4x Specificity)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 115)
    for bar, val in zip(bars, loss):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / 'dnb05_bottleneck.png', dpi=200, bbox_inches='tight')
    print('Saved: dnb05_bottleneck.png')
    plt.close()


def fig_minimal_vnc():
    """Minimal VNC validation bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))
    conditions = ['Intact', 'Fwd Ablated', 'Shuffled']
    distances = [1.40, 0.45, 1.29]
    dx_vals = [1.15, 0.31, 1.11]
    bar_colors = ['#2ca02c', '#d62728', '#ff7f0e']
    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, distances, width, label='Total Distance',
           color=bar_colors, alpha=0.8)
    ax.bar(x + width/2, dx_vals, width, label='Forward (dx)',
           color=bar_colors, alpha=0.4)
    ax.set_ylabel('Distance (mm)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_title('Minimal VNC (1,000 Neurons): Causal Control Preserved',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    ax.annotate('-68%', xy=(1, 0.50), fontsize=11, fontweight='bold',
                color='#d62728', ha='center')
    plt.tight_layout()
    plt.savefig(OUT / 'minimal_vnc_validation.png', dpi=200, bbox_inches='tight')
    print('Saved: minimal_vnc_validation.png')
    plt.close()


if __name__ == '__main__':
    fig_topology_learning()
    fig_ablation_table()
    fig_segregation_matrix()
    fig_dnb05()
    fig_minimal_vnc()
    print('\nAll figures generated.')
