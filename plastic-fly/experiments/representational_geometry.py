"""
Representational geometry of connectome-constrained sensorimotor coding.

Core question: During sensorimotor behavior, do the 365 readout DN firing
patterns organize into modality-specific subspaces that mirror the connectome's
labeled-line architecture — and does this structure vanish when wiring is
randomized?

6 conditions spanning 4 modalities:
  baseline          — no perturbation (proprioceptive)
  contact_loss_left — zero left leg contacts (mechanosensory)
  contact_loss_right— zero right leg contacts (mechanosensory)
  lateral_push      — lateral velocity offset (vestibular)
  loom_left         — inject looming_intensity=[1,0] (visual/LPLC2)
  odor_left         — inject odor_intensity left=1 (olfactory)

Two brain types: intact vs shuffled (shuffle_seed=999).
5 seeds x 6 conditions x 2 brain types = 60 trials.

Key design: uses EVOKED DELTA VECTORS (perturbation - baseline per trial)
to isolate condition-specific neural responses from common walking activity.

Analysis:
  Part 1 — PCA on delta population vectors (geometry visualization)
  Part 2 — Logistic regression decoding (linear separability)
  Part 3 — RSA: neural RDM vs structural RDM from connectome Jaccard overlap

Usage:
    python experiments/representational_geometry.py --fake-brain   # fast test
    python experiments/representational_geometry.py               # real brain (~20 min)
    python experiments/representational_geometry.py --seeds 42 43 44 45 46 47 48 49 50 51
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.config import BridgeConfig
from bridge.interfaces import BodyObservation, LocomotionCommand
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.locomotion_bridge import LocomotionBridge
from bridge.flygym_adapter import FlyGymAdapter
from experiments.sensory_perturbation import apply_sensory_perturbation, PERTURBATION_CONDITIONS


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

CONDITIONS = {
    "baseline": {"perturbation": {}, "looming": None, "odor": None},
    "contact_loss_left": {
        "perturbation": PERTURBATION_CONDITIONS["contact_loss_left"],
        "looming": None, "odor": None,
    },
    "contact_loss_right": {
        "perturbation": PERTURBATION_CONDITIONS["contact_loss_right"],
        "looming": None, "odor": None,
    },
    "lateral_push": {
        "perturbation": PERTURBATION_CONDITIONS["lateral_push"],
        "looming": None, "odor": None,
    },
    "loom_left": {
        "perturbation": {},
        "looming": np.array([1.0, 0.0]),
        "odor": None,
    },
    "odor_left": {
        "perturbation": {},
        "looming": None,
        "odor": np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    },
}

CONDITION_NAMES = list(CONDITIONS.keys())


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    condition_name: str,
    body_steps: int,
    warmup_steps: int,
    use_fake_brain: bool,
    seed: int,
    shuffle_seed: int | None,
    sensory_ids: np.ndarray,
    channel_map: dict,
    readout_ids: np.ndarray,
    decoder_path: Path,
    rate_scale: float,
    onset_frac: float = 0.25,
    offset_frac: float = 0.75,
) -> dict:
    """Run one trial, return pre-phase and perturbation-phase population vectors."""
    import flygym

    cfg = BridgeConfig()
    cond = CONDITIONS[condition_name]

    encoder = SensoryEncoder(
        sensory_neuron_ids=sensory_ids,
        channel_map=channel_map,
        max_rate_hz=cfg.max_rate_hz,
        baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(decoder_path, rate_scale=rate_scale)
    locomotion = LocomotionBridge(seed=seed)
    adapter = FlyGymAdapter()

    brain = create_brain_runner(
        sensory_ids=sensory_ids, readout_ids=readout_ids,
        use_fake=use_fake_brain, warmup_ms=cfg.brain_warmup_ms,
        shuffle_seed=shuffle_seed,
    )

    fly_obj = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly_obj, arena=flygym.arena.FlatTerrain(), timestep=1e-4)
    obs, info = sim.reset()

    # Warmup
    locomotion.warmup(0)
    locomotion.cpg.reset(
        init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
        init_magnitudes=np.zeros(6))
    for _ in range(warmup_steps):
        action = locomotion.step(LocomotionCommand(forward_drive=1.0))
        try:
            obs, _, terminated, truncated, info = sim.step(action)
            if terminated or truncated:
                sim.close()
                return {"error": "warmup_ended", "pre_vectors": [], "perturb_vectors": []}
        except Exception:
            sim.close()
            return {"error": "warmup_physics", "pre_vectors": [], "perturb_vectors": []}

    onset_step = int(body_steps * onset_frac)
    offset_step = int(body_steps * offset_frac)
    bspb = cfg.body_steps_per_brain
    current_cmd = LocomotionCommand(forward_drive=1.0)
    pre_vectors = []
    perturb_vectors = []

    for step in range(body_steps):
        in_pre = step < onset_step
        in_perturb = onset_step <= step < offset_step

        if step % bspb == 0:
            body_obs = adapter.extract_body_observation(obs)

            if in_perturb:
                body_obs = apply_sensory_perturbation(body_obs, cond["perturbation"])
                if cond["looming"] is not None:
                    body_obs.looming_intensity = cond["looming"]
                if cond["odor"] is not None:
                    body_obs.odor_intensity = cond["odor"]

            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            current_cmd = decoder.decode(brain_output)

            rates = brain_output.firing_rates_hz.copy()
            if in_pre:
                pre_vectors.append(rates)
            elif in_perturb:
                perturb_vectors.append(rates)

        action = locomotion.step(current_cmd)
        try:
            obs, _, terminated, truncated, info = sim.step(action)
        except Exception:
            break
        if terminated or truncated:
            break

    sim.close()
    return {"pre_vectors": pre_vectors, "perturb_vectors": perturb_vectors}


# ---------------------------------------------------------------------------
# Structural RDM from connectome
# ---------------------------------------------------------------------------

def compute_structural_rdm(
    channel_map: dict,
    readout_ids: np.ndarray,
    connectivity_path: Path,
    completeness_path: Path,
) -> np.ndarray:
    """Build a 6x6 structural RDM from Jaccard overlap of sensory->DN targets.

    For each condition, defines a set of "active" sensory neurons. Then computes
    1-hop DN target sets and pairwise 1 - Jaccard as dissimilarity.
    """
    import pandas as pd

    conn = pd.read_parquet(connectivity_path)
    comp = pd.read_csv(completeness_path)
    root_ids_array = comp.iloc[:, 0].values

    raw_ids = readout_ids.copy()
    small_mask = raw_ids < 1_000_000
    readout_rootids = set()
    readout_rootids.update(raw_ids[~small_mask].tolist())
    readout_rootids.update(root_ids_array[raw_ids[small_mask]].tolist())

    dn_edges = conn[conn["Postsynaptic_ID"].isin(readout_rootids)]

    # Define sensory neuron sets per condition
    # contact_loss_left/right use same mechanosensory population
    # (same structural DN targets — structurally identical modality)
    condition_sensory = {
        "baseline": set(channel_map.get("proprioceptive", [])),
        "contact_loss_left": set(channel_map.get("mechanosensory", [])),
        "contact_loss_right": set(channel_map.get("mechanosensory", [])),
        "lateral_push": set(channel_map.get("vestibular", [])),
        "loom_left": set(channel_map.get("lplc2_left", []) + channel_map.get("lplc2_right", [])),
        "odor_left": set(channel_map.get("olfactory_left", []) + channel_map.get("olfactory_right", [])),
    }

    dn_targets = {}
    for cond_name in CONDITION_NAMES:
        sensory = condition_sensory[cond_name]
        edges = dn_edges[dn_edges["Presynaptic_ID"].isin(sensory)]
        dn_targets[cond_name] = set(edges["Postsynaptic_ID"].unique())

    n = len(CONDITION_NAMES)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a = dn_targets[CONDITION_NAMES[i]]
            b = dn_targets[CONDITION_NAMES[j]]
            if len(a | b) == 0:
                rdm[i, j] = 1.0
            else:
                rdm[i, j] = 1.0 - len(a & b) / len(a | b)
    return rdm


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(
    intact_trials: dict,
    shuffled_trials: dict,
    structural_rdm: np.ndarray,
) -> dict:
    """Run PCA, decoding, and RSA on evoked delta vectors.

    Uses trial-mean deltas for PCA/RSA (clean, averaged over ~12 brain steps).
    Uses step-level deltas with group-aware CV for decoding (more samples,
    no within-trial leaking).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import silhouette_score
    from scipy.stats import spearmanr

    results = {}

    for brain_label, trials in [("intact", intact_trials), ("shuffled", shuffled_trials)]:
        # --- Build delta vectors ---
        trial_deltas = []   # one per trial (for PCA, RSA)
        trial_labels = []
        step_deltas = []    # one per brain step (for decoding)
        step_labels = []
        step_groups = []    # trial index per step (for group CV)
        trial_idx = 0

        for ci, cond_name in enumerate(CONDITION_NAMES):
            trial_list = trials[cond_name]
            for pre_vecs, perturb_vecs in trial_list:
                if len(pre_vecs) == 0 or len(perturb_vecs) == 0:
                    continue
                pre_mean = np.mean(pre_vecs, axis=0)
                perturb_mean = np.mean(perturb_vecs, axis=0)
                trial_deltas.append(perturb_mean - pre_mean)
                trial_labels.append(ci)

                for vec in perturb_vecs:
                    step_deltas.append(vec - pre_mean)
                    step_labels.append(ci)
                    step_groups.append(trial_idx)
                trial_idx += 1

        if not trial_deltas:
            results[brain_label] = {"error": "no vectors"}
            continue

        X_trial = np.array(trial_deltas)
        y_trial = np.array(trial_labels)
        X_step = np.array(step_deltas)
        y_step = np.array(step_labels)
        groups = np.array(step_groups)

        # Remove zero-variance neurons (based on trial-level for stability)
        var = X_trial.var(axis=0)
        keep = var > 1e-10
        n_responsive = int(np.sum(keep))
        X_trial_f = X_trial[:, keep]
        X_step_f = X_step[:, keep]

        if n_responsive == 0:
            results[brain_label] = {"error": "all neurons zero variance"}
            continue

        # --- Part 1: PCA on trial-level deltas ---
        scaler = StandardScaler()
        X_trial_z = scaler.fit_transform(X_trial_f)

        n_comp = min(10, X_trial_z.shape[1], X_trial_z.shape[0] - 1)
        if n_comp < 1:
            n_comp = 1
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_trial_z)
        n_pca_3 = min(3, n_comp)
        var_explained_3 = float(np.sum(pca.explained_variance_ratio_[:n_pca_3]))

        # Silhouette in PCA space
        unique_labels = np.unique(y_trial)
        if len(unique_labels) > 1 and all(np.sum(y_trial == lab) > 1 for lab in unique_labels):
            sil = float(silhouette_score(X_pca[:, :n_pca_3], y_trial))
        else:
            sil = 0.0

        # --- Part 2: Decoding on step-level deltas with group-aware CV ---
        # Group = trial, so no brain steps from the same trial appear in both
        # train and test, preventing temporal autocorrelation leaking.
        X_step_z = scaler.transform(X_step_f)  # use trial-level scaler
        X_step_pca = pca.transform(X_step_z)   # project into trial-level PCA
        n_decode_dims = min(15, n_comp)
        X_decode = X_step_pca[:, :n_decode_dims]

        logo = LeaveOneGroupOut()
        accs = []
        for train_idx, test_idx in logo.split(X_decode, y_step, groups):
            if len(np.unique(y_step[train_idx])) < 2:
                continue
            clf = LogisticRegression(
                max_iter=2000, C=1.0,
                solver="lbfgs", random_state=42)
            clf.fit(X_decode[train_idx], y_step[train_idx])
            accs.append(float(clf.score(X_decode[test_idx], y_step[test_idx])))
        decode_acc = float(np.mean(accs)) if accs else 0.0
        decode_std = float(np.std(accs)) if accs else 0.0

        # --- Part 3: RSA on trial-level deltas ---
        n_conds = len(CONDITION_NAMES)
        cond_means = np.zeros((n_conds, X_trial_f.shape[1]))
        for ci in range(n_conds):
            mask = y_trial == ci
            if np.sum(mask) > 0:
                cond_means[ci] = X_trial_f[mask].mean(axis=0)

        neural_rdm = np.zeros((n_conds, n_conds))
        for i in range(n_conds):
            for j in range(n_conds):
                a, b = cond_means[i], cond_means[j]
                norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
                if norm_a > 1e-10 and norm_b > 1e-10:
                    corr = float(np.dot(a, b) / (norm_a * norm_b))
                    neural_rdm[i, j] = 1.0 - corr
                else:
                    neural_rdm[i, j] = 1.0

        triu_idx = np.triu_indices(n_conds, k=1)
        neural_upper = neural_rdm[triu_idx]
        struct_upper = structural_rdm[triu_idx]

        if np.std(neural_upper) > 1e-10 and np.std(struct_upper) > 1e-10:
            rsa_r, rsa_p = spearmanr(neural_upper, struct_upper)
            rsa_r = float(rsa_r) if not np.isnan(rsa_r) else 0.0
            rsa_p = float(rsa_p) if not np.isnan(rsa_p) else 1.0
        else:
            rsa_r, rsa_p = 0.0, 1.0

        # Permutation test
        n_perms = 10000
        rng = np.random.RandomState(42)
        perm_rs = np.zeros(n_perms)
        for pi in range(n_perms):
            shuffled_upper = rng.permutation(neural_upper)
            if np.std(shuffled_upper) > 1e-10:
                pr, _ = spearmanr(shuffled_upper, struct_upper)
                perm_rs[pi] = pr if not np.isnan(pr) else 0.0
        rsa_perm_p = float(np.mean(perm_rs >= rsa_r))

        # Specific metrics
        cl_idx = CONDITION_NAMES.index("contact_loss_left")
        cr_idx = CONDITION_NAMES.index("contact_loss_right")
        a, b = cond_means[cl_idx], cond_means[cr_idx]
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        contact_sim = float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 1e-10 and norm_b > 1e-10 else 0.0

        loom_idx = CONDITION_NAMES.index("loom_left")
        odor_idx = CONDITION_NAMES.index("odor_left")
        a, b = cond_means[loom_idx], cond_means[odor_idx]
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        loom_odor_dissim = 1.0 - float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 1e-10 and norm_b > 1e-10 else 1.0

        # PCA coords for figures
        pca_coords = {}
        for ci, cond_name in enumerate(CONDITION_NAMES):
            mask = y_trial == ci
            if np.sum(mask) > 0:
                pca_coords[cond_name] = X_pca[mask, :n_pca_3].tolist()

        results[brain_label] = {
            "n_trial_vectors": int(len(y_trial)),
            "n_step_vectors": int(len(y_step)),
            "n_responsive_neurons": n_responsive,
            "n_total_neurons": int(X_trial.shape[1]),
            "n_pca_components": n_comp,
            "n_per_condition": {CONDITION_NAMES[ci]: int(np.sum(y_trial == ci)) for ci in range(n_conds)},
            "pca_var_explained_3": var_explained_3,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "silhouette_score": sil,
            "decode_accuracy": decode_acc,
            "decode_std": decode_std,
            "neural_rdm": neural_rdm.tolist(),
            "rsa_spearman_r": rsa_r,
            "rsa_spearman_p": rsa_p,
            "rsa_perm_p": rsa_perm_p,
            "contact_lr_similarity": contact_sim,
            "loom_odor_dissimilarity": loom_odor_dissim,
            "pca_coords": pca_coords,
        }

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(results: dict, structural_rdm: np.ndarray, output_dir: Path):
    """Generate PCA, decoding, and RSA figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "#888888",
        "contact_loss_left": "#e74c3c",
        "contact_loss_right": "#c0392b",
        "lateral_push": "#3498db",
        "loom_left": "#2ecc71",
        "odor_left": "#9b59b6",
    }
    short_labels = {
        "baseline": "Baseline",
        "contact_loss_left": "Contact L",
        "contact_loss_right": "Contact R",
        "lateral_push": "Vestibular",
        "loom_left": "Loom L",
        "odor_left": "Odor L",
    }

    # --- Figure 1: PCA scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, brain_label in enumerate(["intact", "shuffled"]):
        ax = axes[ax_idx]
        r = results.get(brain_label, {})
        pca_coords = r.get("pca_coords", {})
        var3 = r.get("pca_var_explained_3", 0)
        evr = r.get("pca_explained_variance_ratio", [0, 0, 0])

        for cond_name in CONDITION_NAMES:
            pts = pca_coords.get(cond_name, [])
            if not pts:
                continue
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], c=colors[cond_name],
                       label=short_labels[cond_name], alpha=0.7, s=30, edgecolors="k", linewidths=0.3)

            # 95% confidence ellipse
            if len(pts) > 2:
                mean = pts[:, :2].mean(axis=0)
                cov = np.cov(pts[:, 0], pts[:, 1])
                if cov.shape == (2, 2) and np.linalg.det(cov) > 1e-10:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
                    w, h = 2 * 1.96 * np.sqrt(np.maximum(eigenvalues, 0))
                    ellipse = Ellipse(mean, w, h, angle=angle, fill=False,
                                      edgecolor=colors[cond_name], linewidth=1.5, linestyle="--")
                    ax.add_patch(ellipse)

        pc1_var = evr[0] * 100 if len(evr) > 0 else 0
        pc2_var = evr[1] * 100 if len(evr) > 1 else 0
        ax.set_xlabel("PC1 (%.1f%%)" % pc1_var)
        ax.set_ylabel("PC2 (%.1f%%)" % pc2_var)
        sil = r.get("silhouette_score", 0)
        ax.set_title("%s (3-PC var=%.0f%%, sil=%.2f)" % (brain_label.upper(), var3 * 100, sil))
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Population Geometry: Evoked Delta PCA (perturbation - baseline)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "representational_geometry_pca.png", dpi=200)
    plt.close(fig)

    # --- Figure 2: Decoding bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Intact", "Shuffled"]
    accs = [results.get("intact", {}).get("decode_accuracy", 0),
            results.get("shuffled", {}).get("decode_accuracy", 0)]
    stds = [results.get("intact", {}).get("decode_std", 0),
            results.get("shuffled", {}).get("decode_std", 0)]
    bar_colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, accs, yerr=stds, capsize=5, color=bar_colors, edgecolor="k")
    ax.axhline(1.0 / 6, color="gray", linestyle="--", label="Chance (16.7%%)")
    ax.set_ylabel("6-way Decoding Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Linear Decodability of Sensory Condition\n(from evoked delta vectors)")
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                "%.0f%%" % (acc * 100), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "representational_geometry_decoding.png", dpi=200)
    plt.close(fig)

    # --- Figure 3: RSA ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    tick_labels = [short_labels[c] for c in CONDITION_NAMES]

    nrdm = np.array(results.get("intact", {}).get("neural_rdm", np.zeros((6, 6))))
    im0 = axes[0].imshow(nrdm, cmap="RdBu_r", vmin=0, vmax=max(2, nrdm.max() * 1.1))
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(6))
    axes[0].set_yticklabels(tick_labels, fontsize=7)
    axes[0].set_title("Neural RDM (intact)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(structural_rdm, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(range(6))
    axes[1].set_yticklabels(tick_labels, fontsize=7)
    axes[1].set_title("Structural RDM (Jaccard)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    triu_idx = np.triu_indices(6, k=1)
    neural_upper = nrdm[triu_idx]
    struct_upper = structural_rdm[triu_idx]
    axes[2].scatter(struct_upper, neural_upper, c="#2c3e50", s=50, edgecolors="k", linewidths=0.5)
    pair_labels = []
    for i, j in zip(triu_idx[0], triu_idx[1]):
        pair_labels.append("%s-%s" % (short_labels[CONDITION_NAMES[i]][:3],
                                       short_labels[CONDITION_NAMES[j]][:3]))
    for k, lbl in enumerate(pair_labels):
        axes[2].annotate(lbl, (struct_upper[k], neural_upper[k]),
                         fontsize=5, alpha=0.7, textcoords="offset points", xytext=(3, 3))

    rsa_r = results.get("intact", {}).get("rsa_spearman_r", 0)
    rsa_p = results.get("intact", {}).get("rsa_perm_p", 1)
    axes[2].set_xlabel("Structural dissimilarity (1-Jaccard)")
    axes[2].set_ylabel("Neural dissimilarity (1-Pearson)")
    axes[2].set_title("RSA: r=%.2f, p=%.4f" % (rsa_r, rsa_p))
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Representational Similarity Analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "representational_geometry_rsa.png", dpi=200)
    plt.close(fig)

    print("  Figures saved to %s/" % output_dir)


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------

def run_tests(results: dict) -> list:
    tests = []
    intact = results.get("intact", {})
    shuffled = results.get("shuffled", {})

    def log_test(name, passed, detail):
        tests.append({"name": name, "passed": passed, "detail": detail})
        print("  [%s] %s: %s" % ("PASS" if passed else "FAIL", name, detail))

    i_var3 = intact.get("pca_var_explained_3", 0)
    i_sil = intact.get("silhouette_score", 0)
    i_acc = intact.get("decode_accuracy", 0)
    s_acc = shuffled.get("decode_accuracy", 0)
    i_rsa = intact.get("rsa_spearman_r", 0)
    s_rsa = shuffled.get("rsa_spearman_r", 0)
    i_perm_p = intact.get("rsa_perm_p", 1)
    contact_sim = intact.get("contact_lr_similarity", 0)
    loom_odor_dissim = intact.get("loom_odor_dissimilarity", 0)
    i_responsive = intact.get("n_responsive_neurons", 0)
    s_responsive = shuffled.get("n_responsive_neurons", 0)

    # T1: PCA captures concentrated variance (>35% in 3 PCs)
    log_test("T1_pca_var_intact", i_var3 > 0.35,
             "3-PC var explained = %.1f%% (>35%%)" % (i_var3 * 100))

    # T2: Intact brain activates far more condition-responsive DNs than shuffled
    ratio = i_responsive / max(s_responsive, 1)
    log_test("T2_responsive_neurons", ratio > 10,
             "intact %d vs shuffled %d (ratio=%.1fx, >10x)" % (i_responsive, s_responsive, ratio))

    # T3: Positive cluster structure in PCA space
    log_test("T3_silhouette", i_sil > 0.0,
             "silhouette = %.2f (>0.0)" % i_sil)

    # T4: Decoding above chance (>35%, which is 2.1x the 16.7% chance)
    log_test("T4_decode_intact", i_acc > 0.35,
             "accuracy = %.1f%% (>35%%, chance=16.7%%)" % (i_acc * 100))

    # T5: Intact decoding > shuffled (connectome advantage)
    log_test("T5_decode_advantage", i_acc > s_acc,
             "intact %.1f%% > shuffled %.1f%%" % (i_acc * 100, s_acc * 100))

    # T6: RSA: neural geometry correlates with structural wiring (r > 0.3)
    log_test("T6_rsa_intact", i_rsa > 0.3,
             "Spearman r = %.2f (>0.3)" % i_rsa)

    # T7: Shuffled RSA is weak (wiring destroyed)
    log_test("T7_rsa_shuffled", s_rsa < 0.2,
             "shuffled Spearman r = %.2f (<0.2)" % s_rsa)

    # T8: RSA is trend-level significant (p < 0.10)
    log_test("T8_rsa_significance", i_perm_p < 0.10,
             "permutation p = %.4f (<0.10)" % i_perm_p)

    # T9: Same-modality similarity (contact L/R share mechanosensory channel)
    log_test("T9_contact_similarity", contact_sim > 0.5,
             "contact L/R Pearson = %.2f (>0.5)" % contact_sim)

    # T10: Cross-modal dissimilarity (visual vs olfactory should be very different)
    log_test("T10_loom_odor_dissim", loom_odor_dissim > 0.5,
             "loom-odor dissimilarity = %.2f (>0.5)" % loom_odor_dissim)

    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_representational_geometry(
    body_steps: int = 5000,
    warmup_steps: int = 500,
    use_fake_brain: bool = False,
    seeds: list[int] | None = None,
    output_dir: str = "logs/representational_geometry",
    rate_scale: float = 12.0,
):
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    cfg = BridgeConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    channel_map_path = cfg.data_dir / "channel_map_v4_looming.json"
    with open(channel_map_path) as f:
        channel_map = json.load(f)
    sensory_ids = np.load(cfg.data_dir / "sensory_ids_v4_looming.npy")
    readout_ids = np.load(cfg.data_dir / "readout_ids_v5_steering.npy")
    decoder_path = cfg.data_dir / "decoder_groups_v5_steering.json"

    brain_label = "FAKE" if use_fake_brain else "Brian2 LIF"
    print("=" * 70)
    print("REPRESENTATIONAL GEOMETRY EXPERIMENT")
    print("  Brain: %s" % brain_label)
    print("  Sensory: %d neurons, Readout: %d DNs" % (len(sensory_ids), len(readout_ids)))
    print("  Conditions: %s" % ", ".join(CONDITION_NAMES))
    print("  Seeds: %s" % seeds)
    print("  Body steps: %d, perturbation: 25%%-75%%" % body_steps)
    print("  Method: evoked delta vectors (perturbation - baseline per trial)")
    print("=" * 70)

    # --- Collect population vectors ---
    # Store as list of (pre_vecs, perturb_vecs) tuples per condition
    all_trials = {"intact": {c: [] for c in CONDITION_NAMES},
                  "shuffled": {c: [] for c in CONDITION_NAMES}}

    total_trials = len(seeds) * len(CONDITION_NAMES) * 2
    trial_num = 0
    t_start = time.time()

    for brain_type, shuffle_seed in [("intact", None), ("shuffled", 999)]:
        print("\n" + "#" * 70)
        print("# %s CONNECTOME" % brain_type.upper())
        print("#" * 70)

        for seed in seeds:
            for cond_name in CONDITION_NAMES:
                trial_num += 1
                label = "%s_%s_seed%d" % (cond_name, brain_type, seed)
                print("\n  [%d/%d] %s" % (trial_num, total_trials, label), end="", flush=True)
                t0 = time.time()

                result = run_trial(
                    condition_name=cond_name,
                    body_steps=body_steps,
                    warmup_steps=warmup_steps,
                    use_fake_brain=use_fake_brain,
                    seed=seed,
                    shuffle_seed=shuffle_seed,
                    sensory_ids=sensory_ids,
                    channel_map=channel_map,
                    readout_ids=readout_ids,
                    decoder_path=decoder_path,
                    rate_scale=rate_scale,
                )

                pre = result.get("pre_vectors", [])
                perturb = result.get("perturb_vectors", [])
                all_trials[brain_type][cond_name].append((pre, perturb))
                elapsed = time.time() - t0
                print(" — %d pre + %d perturb vectors in %.1fs" % (len(pre), len(perturb), elapsed))

                if "error" in result:
                    print("    ERROR: %s" % result["error"])

            # Checkpoint
            checkpoint = {
                "brain_type": brain_type,
                "seed": seed,
                "vectors_per_condition": {
                    c: sum(len(p[1]) for p in all_trials[brain_type][c])
                    for c in CONDITION_NAMES
                },
            }
            _write_json_atomic(output_path / "checkpoints" / ("%s_seed%d.json" % (brain_type, seed)), checkpoint)

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE (%.1f min)" % (total_elapsed / 60))
    for bt in ["intact", "shuffled"]:
        n = sum(len(p[1]) for c in CONDITION_NAMES for p in all_trials[bt][c])
        print("  %s: %d perturbation vectors" % (bt, n))
    print("=" * 70)

    # --- Structural RDM ---
    print("\nComputing structural RDM from connectome...")
    structural_rdm = compute_structural_rdm(
        channel_map=channel_map,
        readout_ids=readout_ids,
        connectivity_path=cfg.connectivity_path,
        completeness_path=cfg.completeness_path,
    )
    print("  Structural RDM (6x6):")
    for i, name in enumerate(CONDITION_NAMES):
        print("    %s: %s" % (name[:15].ljust(15), " ".join("%.2f" % v for v in structural_rdm[i])))

    # --- Analysis ---
    print("\nRunning analysis (PCA, decoding, RSA on delta vectors)...")
    results = run_analysis(
        intact_trials=all_trials["intact"],
        shuffled_trials=all_trials["shuffled"],
        structural_rdm=structural_rdm,
    )

    for brain_label in ["intact", "shuffled"]:
        r = results.get(brain_label, {})
        if "error" in r:
            print("\n  %s: %s" % (brain_label, r["error"]))
            continue
        print("\n  %s:" % brain_label.upper())
        print("    Trial vectors: %d, responsive neurons: %d/%d, PCA dims: %d" % (
            r["n_trial_vectors"], r["n_responsive_neurons"], r["n_total_neurons"],
            r.get("n_pca_components", 0)))
        print("    PCA 3-PC var: %.1f%%" % (r["pca_var_explained_3"] * 100))
        print("    Silhouette: %.2f" % r["silhouette_score"])
        print("    Decoding: %.1f%% +/- %.1f%%" % (r["decode_accuracy"] * 100, r["decode_std"] * 100))
        print("    RSA Spearman r: %.2f (perm p=%.4f)" % (r["rsa_spearman_r"], r["rsa_perm_p"]))
        print("    Contact L/R similarity: %.2f" % r["contact_lr_similarity"])
        print("    Loom-odor dissimilarity: %.2f" % r["loom_odor_dissimilarity"])

    # --- Verification tests ---
    print("\n" + "=" * 70)
    print("VERIFICATION TESTS")
    print("=" * 70)
    tests = run_tests(results)
    n_pass = sum(1 for t in tests if t["passed"])
    print("\n  %d/%d tests passed" % (n_pass, len(tests)))

    # --- Figures ---
    print("\nGenerating figures...")
    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    generate_figures(results, structural_rdm, fig_dir)

    # --- Save ---
    output_data = {
        "config": {
            "body_steps": body_steps,
            "warmup_steps": warmup_steps,
            "seeds": seeds,
            "use_fake_brain": use_fake_brain,
            "n_sensory": len(sensory_ids),
            "n_readout": len(readout_ids),
            "rate_scale": rate_scale,
            "conditions": CONDITION_NAMES,
            "total_elapsed_min": total_elapsed / 60,
            "method": "evoked_delta_vectors",
        },
        "structural_rdm": structural_rdm.tolist(),
        "intact": results.get("intact", {}),
        "shuffled": results.get("shuffled", {}),
        "tests": tests,
        "summary": {
            "tests_passed": n_pass,
            "tests_total": len(tests),
        },
    }
    _write_json_atomic(output_path / "geometry_results.json", output_data)
    print("\nSaved to %s/geometry_results.json" % output_path)

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Representational geometry of connectome-constrained sensorimotor coding")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--fake-brain", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", default="logs/representational_geometry")
    parser.add_argument("--rate-scale", type=float, default=12.0)
    args = parser.parse_args()

    run_representational_geometry(
        body_steps=args.body_steps,
        warmup_steps=args.warmup_steps,
        use_fake_brain=args.fake_brain,
        seeds=args.seeds,
        output_dir=args.output_dir,
        rate_scale=args.rate_scale,
    )
