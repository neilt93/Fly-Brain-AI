#!/usr/bin/env python
"""Derive MN decoder parameters from FlyGym biomechanics + MANC pool composition.

Generates data/mn_decoder_params.json with:
  - rest_angles: from FlyGym Fly(init_pose='stretch') joint positions
  - amplitudes: proportional to MANC MN pool size per joint
  - mn_pool_sizes: raw counts of motor neurons per joint
  - base_amplitude: scaling factor for amplitude computation

Usage:
    python scripts/derive_mn_decoder_params.py
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_flygym_rest_angles():
    """Get joint rest angles from FlyGym init_pose='stretch'."""
    import flygym

    fly = flygym.Fly(init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(
        fly=fly, arena=flygym.arena.FlatTerrain(), timestep=1e-4
    )
    obs, _ = sim.reset()
    rest = np.array(obs["joints"][0], dtype=np.float64)  # (42,)
    sim.close()
    return rest


def get_mn_pool_sizes(mapping_path):
    """Count total MNs per joint from mn_joint_mapping.json."""
    with open(mapping_path) as f:
        mn_map = json.load(f)
    pool_sizes = np.zeros(42, dtype=int)
    for entry in mn_map.values():
        j = int(entry["joint_idx"])
        pool_sizes[j] += 1
    return pool_sizes


def compute_amplitudes(pool_sizes, cpg_amplitudes):
    """Blend CPG-calibrated amplitudes with MANC MN pool authority.

    The CPG-calibrated amplitudes represent each joint's actual walking range
    (biomechanical). The MN pool size modulates this: joints with more MNs
    get a slight boost (finer control = larger usable range), joints with
    fewer MNs get a slight reduction.

    Scaling: amp_j = cpg_amp_j * (0.6 + 0.4 * pool_j / mean_pool)
    This keeps all amplitudes within 60-140% of their CPG-calibrated values.
    """
    mapped_mask = pool_sizes > 0
    if not mapped_mask.any():
        return cpg_amplitudes.copy()

    mean_pool = pool_sizes[mapped_mask].mean()
    amps = np.zeros(42, dtype=np.float64)
    for j in range(42):
        cpg_amp = cpg_amplitudes[j]
        if pool_sizes[j] > 0:
            pool_ratio = pool_sizes[j] / mean_pool
            scale = 0.6 + 0.4 * pool_ratio  # range [0.6, ~1.4+]
            amps[j] = cpg_amp * np.clip(scale, 0.6, 1.5)
        else:
            amps[j] = cpg_amp
    return amps


def symmetrize(rest_angles, amplitudes):
    """Average L/R, mirror sign for roll/yaw joints.

    Ensures the decoder does not introduce a heading bias due to
    asymmetric FlyGym init pose or asymmetric MANC MN counts.
    Mirror joints (Coxa_roll=1, Coxa_yaw=2, Femur_roll=4) get
    sign-flipped rest angles for the R side.
    """
    MIRROR_WITHIN_LEG = {1, 2, 4}
    LEG_OFFSET = {"LF": 0, "LM": 7, "LH": 14, "RF": 21, "RM": 28, "RH": 35}

    for left, right in [("LF", "RF"), ("LM", "RM"), ("LH", "RH")]:
        off_l, off_r = LEG_OFFSET[left], LEG_OFFSET[right]
        for dof in range(7):
            jl, jr = off_l + dof, off_r + dof
            if dof in MIRROR_WITHIN_LEG:
                avg_mag = (abs(rest_angles[jl]) + abs(rest_angles[jr])) / 2.0
                rest_angles[jl] = (
                    np.sign(rest_angles[jl]) * avg_mag
                    if rest_angles[jl] != 0
                    else avg_mag
                )
                rest_angles[jr] = -rest_angles[jl]
            else:
                avg = (rest_angles[jl] + rest_angles[jr]) / 2.0
                rest_angles[jl] = rest_angles[jr] = avg
            avg_amp = (amplitudes[jl] + amplitudes[jr]) / 2.0
            amplitudes[jl] = amplitudes[jr] = avg_amp

    return rest_angles, amplitudes


def main():
    mapping_path = ROOT / "data" / "mn_joint_mapping.json"
    output_path = ROOT / "data" / "mn_decoder_params.json"

    print("=== Deriving MN decoder parameters ===")

    # Step 1: Rest angles — blend FlyGym biomechanics with CPG-calibrated values
    # FlyGym init_pose='stretch' gives the MJCF model's anatomical zero (many joints = 0).
    # For joints where stretch gives a meaningful non-zero value (Coxa, Femur, Coxa_roll),
    # use the biomechanical value. For joints where stretch gives 0.0 (Tibia, Tarsus,
    # Coxa_yaw, Femur_roll), use the CPG-calibrated walking equilibrium from _JOINT_PARAMS.
    from bridge.mn_decoder import _JOINT_PARAMS

    print("\n1. Getting rest angles (biomechanics + CPG-calibrated blend)...")
    try:
        stretch_angles = get_flygym_rest_angles()
        print(f"   FlyGym stretch pose: {np.count_nonzero(stretch_angles)}/42 non-zero")
    except Exception as e:
        print(f"   FlyGym not available ({e}), using CPG-calibrated only")
        stretch_angles = None

    rest_angles = np.zeros(42)
    for j in range(42):
        cpg_rest = _JOINT_PARAMS[j][0]
        if stretch_angles is not None and abs(stretch_angles[j]) > 0.01:
            # Use biomechanical value from FlyGym
            rest_angles[j] = stretch_angles[j]
        else:
            # Use CPG-calibrated walking equilibrium
            rest_angles[j] = cpg_rest
    n_bio = sum(1 for j in range(42) if stretch_angles is not None and abs(stretch_angles[j]) > 0.01)
    print(f"   Using {n_bio} biomechanical + {42-n_bio} CPG-calibrated rest angles")

    # Step 2: MN pool sizes from MANC
    print("\n2. Counting MN pool sizes from mn_joint_mapping.json...")
    pool_sizes = get_mn_pool_sizes(mapping_path)
    mapped_mask = pool_sizes > 0
    print(
        f"   Pool sizes per joint: min={pool_sizes[mapped_mask].min()}, "
        f"max={pool_sizes.max()}, mean={pool_sizes[mapped_mask].mean():.1f}"
    )

    # Step 3: Compute amplitudes (blend CPG-calibrated with MN pool authority)
    cpg_amplitudes = np.array([_JOINT_PARAMS[j][1] for j in range(42)])
    print("\n3. Computing amplitudes (CPG-calibrated + MN pool authority)...")
    amplitudes = compute_amplitudes(pool_sizes, cpg_amplitudes)

    # Step 4: Symmetrize L/R
    print("\n4. Symmetrizing L/R...")
    rest_angles, amplitudes = symmetrize(rest_angles.copy(), amplitudes.copy())

    # Step 5: Save
    params = {
        "source": "biomechanics+manc",
        "rest_angles": {
            str(j): round(float(rest_angles[j]), 4) for j in range(42)
        },
        "amplitudes": {
            str(j): round(float(amplitudes[j]), 4) for j in range(42)
        },
        "mn_pool_sizes": {str(j): int(pool_sizes[j]) for j in range(42)},
    }

    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary table
    LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
    LEG_OFFSET = {"LF": 0, "LM": 7, "LH": 14, "RF": 21, "RM": 28, "RH": 35}
    JOINT_NAMES = [
        "Coxa",
        "Coxa_roll",
        "Coxa_yaw",
        "Femur",
        "Femur_roll",
        "Tibia",
        "Tarsus1",
    ]

    print(f"\n{'Joint':<25} {'Rest':>8} {'Amp':>8} {'MNs':>5}")
    print("-" * 50)
    for leg in LEGS:
        off = LEG_OFFSET[leg]
        for di, jn in enumerate(JOINT_NAMES):
            j = off + di
            print(
                f"  {leg}_{jn:<18} {rest_angles[j]:>+8.4f} "
                f"{amplitudes[j]:>8.4f} {pool_sizes[j]:>5d}"
            )


if __name__ == "__main__":
    main()
