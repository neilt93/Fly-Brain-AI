"""
Paper 2: Sim-to-real transfer validation on HexArth hexapod.

Runs the same closed-loop brain-body pipeline on either:
  - FlyGym simulation (baseline)
  - HexArth hardware (transfer test)

Compares: distance, heading, gait quality, ablation effects.

Scientific questions:
  1. Does the connectome controller transfer without retuning?
  2. Does real proprioceptive feedback change behavior?
  3. Does hardware ablation (DNb05 silencing) replicate sim results?
  4. Is random sparse equally robust on real hardware?

Usage:
    # Simulation baseline
    python -m experiments.hexapod_validation --backend flygym --body-steps 5000

    # Hardware transfer (when HexArth is connected)
    python -m experiments.hexapod_validation --backend hexarth --port /dev/ttyUSB0

    # Hardware ablation
    python -m experiments.hexapod_validation --backend hexarth --ablate forward
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.interfaces import BodyObservation, BrainOutput, LocomotionCommand
from bridge.hexapod_interface import create_hexapod, HexapodConfig


def _write_json_atomic(path: Path, payload, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, **kwargs)
    tmp.replace(path)


def run_minimal_vnc_loop(hexapod, body_steps=5000, ablate_group=None,
                          use_minimal=True, shuffle_seed=None):
    """Run the MinimalVNC closed-loop on the hexapod backend.

    This is the hardware-compatible version of vnc_validation.py.
    Uses the same pipeline: brain -> decoder -> VNC -> rhythm -> joints.
    """
    from bridge.vnc_bridge import VNCBridge
    from bridge.vnc_connectome import VNCConfig
    from bridge.descending_decoder import DescendingDecoder

    # Build VNC bridge
    vnc_config = VNCConfig()
    vnc_bridge = VNCBridge(
        use_fake_vnc=False,
        use_minimal_vnc=use_minimal,
        vnc_config=vnc_config,
    )

    # Build brain (or use tonic drive for transfer test)
    from bridge.brain_runner import BrainRunner
    brain = BrainRunner()

    decoder = DescendingDecoder()

    # Optional ablation
    if ablate_group:
        decoder.ablate_group(ablate_group)
        print(f"  Ablated: {ablate_group}")

    if shuffle_seed is not None:
        decoder.shuffle_weights(shuffle_seed)
        print(f"  Shuffled: seed={shuffle_seed}")

    # Reset hexapod
    body_obs = hexapod.reset()

    # Brain warmup
    print("  Brain warmup (200ms)...")
    brain.warmup()

    # Logging
    log = {
        "positions": [],
        "headings": [],
        "joint_angles": [],
        "contact_forces": [],
        "timestamps": [],
    }

    t_start = time.time()
    brain_step_interval = 100  # body steps per brain step

    for step in range(body_steps):
        # Brain step every 100 body steps
        if step % brain_step_interval == 0:
            # Encode body state
            from bridge.sensory_encoder import SensoryEncoder
            encoder = SensoryEncoder()
            brain_input = encoder.encode(body_obs)

            # Brain step
            brain_output = brain.step(brain_input)

            # Decode to group rates
            group_rates = decoder.decode(brain_output)

        # VNC step
        action = vnc_bridge.step(group_rates, body_obs=body_obs)

        # Command hexapod
        body_obs = hexapod.command(action)

        # Log
        if step % 10 == 0:
            log["positions"].append(body_obs.body_orientation.tolist())
            log["contact_forces"].append(body_obs.contact_forces.tolist())
            log["timestamps"].append(time.time() - t_start)

        if step % 500 == 0 and step > 0:
            elapsed = time.time() - t_start
            print(f"    Step {step}/{body_steps} ({elapsed:.1f}s)")

    elapsed = time.time() - t_start

    # Compute metrics
    positions = np.array(log["positions"])
    if len(positions) > 1:
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        distance = float(np.sqrt(dx**2 + dy**2))
        heading = float(np.degrees(np.arctan2(dy, dx)))
    else:
        distance = 0.0
        heading = 0.0

    return {
        "body_steps": body_steps,
        "distance_mm": distance,
        "forward_dx_mm": float(dx) if len(positions) > 1 else 0.0,
        "heading_deg": heading,
        "elapsed_s": elapsed,
        "steps_per_sec": body_steps / elapsed,
        "ablate_group": ablate_group,
        "shuffle_seed": shuffle_seed,
        "log": log,
    }


def main():
    parser = argparse.ArgumentParser(description="Hexapod validation")
    parser.add_argument("--backend", default="flygym",
                        choices=["flygym", "hexarth"])
    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="Serial port for HexArth")
    parser.add_argument("--body-steps", type=int, default=5000)
    parser.add_argument("--ablate", default=None,
                        choices=["forward", "turn_left", "turn_right",
                                 "rhythm", "stance"])
    parser.add_argument("--shuffle", type=int, default=None,
                        help="Shuffle seed for negative control")
    parser.add_argument("--use-minimal", action="store_true", default=True,
                        help="Use MinimalVNCRunner (default: True)")
    args = parser.parse_args()

    print(f"Hexapod validation: backend={args.backend}, steps={args.body_steps}")

    config = HexapodConfig()
    if args.backend == "hexarth":
        hexapod = create_hexapod("hexarth", port=args.port, config=config)
    else:
        hexapod = create_hexapod("flygym", config=config)

    conditions = []

    # Intact baseline
    conditions.append(("intact", None, None))

    # Ablation (if specified)
    if args.ablate:
        conditions.append((f"ablate_{args.ablate}", args.ablate, None))

    # Shuffle control (if specified)
    if args.shuffle is not None:
        conditions.append((f"shuffle_s{args.shuffle}", None, args.shuffle))

    results = []
    for name, ablate, shuffle in conditions:
        print(f"\n--- {name} ---")
        result = run_minimal_vnc_loop(
            hexapod,
            body_steps=args.body_steps,
            ablate_group=ablate,
            use_minimal=args.use_minimal,
            shuffle_seed=shuffle,
        )
        result["condition"] = name
        results.append(result)

        print(f"  Distance: {result['distance_mm']:.2f}mm")
        print(f"  Forward dx: {result['forward_dx_mm']:.2f}mm")
        print(f"  Heading: {result['heading_deg']:+.1f}°")
        print(f"  Speed: {result['steps_per_sec']:.0f} steps/s")

    # Summary
    print(f"\n{'='*60}")
    print("HEXAPOD VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<20} {'Distance':>10} {'dx':>10} {'Heading':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['condition']:<20} {r['distance_mm']:>9.2f}mm "
              f"{r['forward_dx_mm']:>9.2f}mm {r['heading_deg']:>+9.1f}°")

    # Save
    out_dir = Path("logs/hexapod_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"validation_{args.backend}_{int(time.time())}.json"
    _write_json_atomic(out_path, {
        "backend": args.backend,
        "results": results,
    })
    print(f"\nSaved: {out_path}")

    hexapod.close()


if __name__ == "__main__":
    main()
