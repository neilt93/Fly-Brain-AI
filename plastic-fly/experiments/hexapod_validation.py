"""
Paper 2: Sim-to-real transfer validation on HexArth hexapod.

Runs the same closed-loop brain-body pipeline on either:
  - FlyGym simulation (baseline)
  - HexArth hardware (transfer test)

Compares distance, heading, gait quality, and ablation effects.
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.config import BridgeConfig
from bridge.hexapod_interface import create_hexapod, HexapodConfig
from bridge.sensory_encoder import SensoryEncoder
from bridge.brain_runner import create_brain_runner
from bridge.descending_decoder import DescendingDecoder
from bridge.vnc_bridge import VNCBridge


def _write_json_atomic(path: Path, payload, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, **kwargs)
    tmp.replace(path)


def _estimate_position(body_obs, prev_position: np.ndarray, dt_s: float) -> np.ndarray:
    """Use tracked position when available, otherwise integrate velocity."""
    if body_obs.body_position is not None:
        return np.asarray(body_obs.body_position, dtype=np.float32)
    if body_obs.body_velocity is not None:
        return prev_position + np.asarray(body_obs.body_velocity, dtype=np.float32) * dt_s
    return prev_position.copy()


def _checkpoint_payload(
    backend: str,
    steps_completed: int,
    body_steps: int,
    brain_steps: int,
    elapsed_s: float,
    ablate_group,
    shuffle_seed,
    use_fake_brain: bool,
    use_fake_vnc: bool,
    log: dict,
    status: str,
) -> dict:
    return {
        "backend": backend,
        "summary": {
            "steps_completed": steps_completed,
            "body_steps": body_steps,
            "brain_steps": brain_steps,
            "elapsed_s": elapsed_s,
            "status": status,
        },
        "config": {
            "ablate_group": ablate_group,
            "shuffle_seed": shuffle_seed,
            "use_fake_brain": use_fake_brain,
            "use_fake_vnc": use_fake_vnc,
        },
        "log": log,
    }


def run_minimal_vnc_loop(
    hexapod,
    backend: str,
    body_steps: int = 5000,
    ablate_group: str | None = None,
    use_minimal: bool = True,
    shuffle_seed: int | None = None,
    use_fake_brain: bool = False,
    use_fake_vnc: bool = False,
    checkpoint_path: Path | None = None,
    connectome: str = "flywire",
):
    """Run the closed-loop bridge on the selected hexapod backend."""
    cfg = BridgeConfig(connectome=connectome)
    if not cfg.sensory_ids_path.exists() or not cfg.readout_ids_path.exists():
        raise FileNotFoundError(
            "Missing neuron population files. Run python scripts/select_populations.py first."
        )

    sensory_ids = np.load(cfg.sensory_ids_path)
    readout_ids = np.load(cfg.readout_ids_path)
    encoder = SensoryEncoder.from_channel_map(
        sensory_ids,
        cfg.channel_map_path,
        max_rate_hz=cfg.max_rate_hz,
        baseline_rate_hz=cfg.baseline_rate_hz,
    )
    decoder = DescendingDecoder.from_json(cfg.decoder_groups_path, rate_scale=cfg.rate_scale)
    brain = create_brain_runner(
        sensory_ids=sensory_ids,
        readout_ids=readout_ids,
        use_fake=use_fake_brain,
        warmup_ms=cfg.brain_warmup_ms,
        connectome=connectome,
    )
    vnc_bridge = VNCBridge(
        use_fake_vnc=use_fake_vnc,
        use_minimal_vnc=use_minimal,
        shuffle_seed=shuffle_seed,
    )

    body_obs = hexapod.reset()
    init_angles = np.asarray(body_obs.joint_angles, dtype=np.float64)
    vnc_bridge.reset(init_angles=init_angles)

    body_dt_s = 1.0 / hexapod.config.control_freq_hz
    brain_step_interval = max(1, int(round((cfg.brain_dt_ms / 1000.0) / body_dt_s)))
    group_rates = {
        "forward": 20.0,
        "turn_left": 0.0,
        "turn_right": 0.0,
        "rhythm": 10.0,
        "stance": 10.0,
    }

    current_position = (
        np.asarray(body_obs.body_position, dtype=np.float32)
        if body_obs.body_position is not None
        else np.zeros(3, dtype=np.float32)
    )
    log = {
        "positions": [],
        "orientations": [],
        "joint_angles": [],
        "contact_forces": [],
        "group_rates": [],
        "timestamps": [],
    }

    t_start = time.time()
    steps_completed = 0
    brain_steps = 0

    for step in range(body_steps):
        if step % brain_step_interval == 0:
            brain_input = encoder.encode(body_obs)
            brain_output = brain.step(brain_input, sim_ms=cfg.brain_dt_ms)
            group_rates = decoder.get_group_rates(brain_output)
            if ablate_group is not None and ablate_group in group_rates:
                group_rates[ablate_group] = 0.0
            if not use_fake_vnc:
                vnc_bridge.step_brain(group_rates, sim_ms=cfg.brain_dt_ms)
            brain_steps += 1

            if brain_steps % 10 == 1:
                mean_rate = float(np.mean(brain_output.firing_rates_hz))
                active = int(np.sum(brain_output.firing_rates_hz > 0))
                print(
                    f"  brain #{brain_steps:3d}: "
                    f"fwd={group_rates['forward']:.1f}Hz "
                    f"turnL={group_rates['turn_left']:.1f}Hz "
                    f"turnR={group_rates['turn_right']:.1f}Hz "
                    f"active={active}/{len(readout_ids)} mean={mean_rate:.1f}Hz"
                )

        action = vnc_bridge.step(group_rates, dt_s=body_dt_s, body_obs=body_obs)

        try:
            body_obs = hexapod.command(action)
        except RuntimeError as exc:
            print(f"  Backend ended at step {step}: {exc}")
            break

        steps_completed = step + 1
        current_position = _estimate_position(body_obs, current_position, body_dt_s)

        if step % 10 == 0:
            log["positions"].append(current_position.tolist())
            log["orientations"].append(np.asarray(body_obs.body_orientation, dtype=np.float32).tolist())
            log["joint_angles"].append(np.asarray(body_obs.joint_angles, dtype=np.float32).tolist())
            log["contact_forces"].append(np.asarray(body_obs.contact_forces, dtype=np.float32).tolist())
            log["group_rates"].append({k: float(v) for k, v in group_rates.items()})
            log["timestamps"].append(time.time() - t_start)

        if checkpoint_path is not None and step % 100 == 0 and step > 0:
            _write_json_atomic(
                checkpoint_path,
                _checkpoint_payload(
                    backend=backend,
                    steps_completed=steps_completed,
                    body_steps=body_steps,
                    brain_steps=brain_steps,
                    elapsed_s=time.time() - t_start,
                    ablate_group=ablate_group,
                    shuffle_seed=shuffle_seed,
                    use_fake_brain=use_fake_brain,
                    use_fake_vnc=use_fake_vnc,
                    log=log,
                    status="running",
                ),
            )

        if step % 500 == 0 and step > 0:
            elapsed = time.time() - t_start
            print(f"    Step {step}/{body_steps} ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    positions = np.asarray(log["positions"], dtype=np.float32)
    if len(positions) > 1:
        diff = positions[-1] - positions[0]
        dx = float(diff[0])
        dy = float(diff[1])
        distance = float(np.linalg.norm(diff[:2]))
        heading = float(np.degrees(np.arctan2(dy, dx)))
    else:
        dx = 0.0
        dy = 0.0
        distance = 0.0
        heading = 0.0

    result = {
        "body_steps": body_steps,
        "steps_completed": steps_completed,
        "brain_steps": brain_steps,
        "distance_mm": distance,
        "forward_dx_mm": dx,
        "lateral_dy_mm": dy,
        "heading_deg": heading,
        "elapsed_s": elapsed,
        "steps_per_sec": steps_completed / max(elapsed, 1e-6),
        "ablate_group": ablate_group,
        "shuffle_seed": shuffle_seed,
        "use_fake_brain": use_fake_brain,
        "use_fake_vnc": use_fake_vnc,
        "control_freq_hz": hexapod.config.control_freq_hz,
        "brain_step_interval": brain_step_interval,
        "log": log,
    }

    if checkpoint_path is not None:
        _write_json_atomic(
            checkpoint_path,
            _checkpoint_payload(
                backend=backend,
                steps_completed=steps_completed,
                body_steps=body_steps,
                brain_steps=brain_steps,
                elapsed_s=elapsed,
                ablate_group=ablate_group,
                shuffle_seed=shuffle_seed,
                use_fake_brain=use_fake_brain,
                use_fake_vnc=use_fake_vnc,
                log=log,
                status="completed" if steps_completed >= body_steps else "partial",
            ),
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Hexapod validation")
    parser.add_argument("--backend", default="flygym", choices=["flygym", "hexarth"])
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for HexArth")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument(
        "--ablate",
        default=None,
        choices=["forward", "turn_left", "turn_right", "rhythm", "stance"],
    )
    parser.add_argument("--shuffle", type=int, default=None, help="Shuffle seed for negative control")
    parser.add_argument("--fake-brain", action="store_true", help="Use FakeBrainRunner for smoke tests")
    parser.add_argument("--fake-vnc", action="store_true", help="Use FakeVNC for smoke tests")
    parser.add_argument("--no-minimal", action="store_true", help="Use full VNC instead of MinimalVNC")
    parser.add_argument("--connectome", choices=["flywire", "banc"], default="flywire",
                        help="Connectome dataset")
    args = parser.parse_args()

    print(
        f"Hexapod validation: backend={args.backend}, steps={args.body_steps}, "
        f"fake_brain={args.fake_brain}, fake_vnc={args.fake_vnc}"
    )

    config = HexapodConfig()
    if args.backend == "hexarth":
        hexapod = create_hexapod("hexarth", port=args.port, config=config)
    else:
        hexapod = create_hexapod("flygym", config=config)

    conditions = [("intact", None, None)]
    if args.ablate:
        conditions.append((f"ablate_{args.ablate}", args.ablate, None))
    if args.shuffle is not None:
        conditions.append((f"shuffle_s{args.shuffle}", None, args.shuffle))

    results = []
    checkpoint_dir = Path("logs/hexapod_validation/checkpoints")

    try:
        for name, ablate, shuffle in conditions:
            print(f"\n--- {name} ---")
            checkpoint_path = checkpoint_dir / f"{args.backend}_{name}.json"
            result = run_minimal_vnc_loop(
                hexapod,
                backend=args.backend,
                body_steps=args.body_steps,
                ablate_group=ablate,
                use_minimal=not args.no_minimal,
                shuffle_seed=shuffle,
                use_fake_brain=args.fake_brain,
                use_fake_vnc=args.fake_vnc,
                checkpoint_path=checkpoint_path,
                connectome=args.connectome,
            )
            result["condition"] = name
            results.append(result)

            print(f"  Distance: {result['distance_mm']:.2f}mm")
            print(f"  Forward dx: {result['forward_dx_mm']:.2f}mm")
            print(f"  Heading: {result['heading_deg']:+.1f} deg")
            print(f"  Speed: {result['steps_per_sec']:.0f} steps/s")
    finally:
        hexapod.close()

    print(f"\n{'=' * 60}")
    print("HEXAPOD VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Condition':<20} {'Distance':>10} {'dx':>10} {'Heading':>10}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['condition']:<20} {r['distance_mm']:>9.2f}mm "
            f"{r['forward_dx_mm']:>9.2f}mm {r['heading_deg']:>+9.1f} deg"
        )

    out_dir = Path("logs/hexapod_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"validation_{args.backend}_{int(time.time())}.json"
    _write_json_atomic(
        out_path,
        {
            "backend": args.backend,
            "results": results,
            "args": vars(args),
        },
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
