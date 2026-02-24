from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermal_sim import SimulationConfig, run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2D pulsed-laser thermal/reactive thin-layer simulation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("default_config.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/latest"),
        help="Output directory for fields and histories.",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Disable frame snapshots used by the animation script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config.exists():
        cfg = SimulationConfig.from_json(args.config)
    else:
        cfg = SimulationConfig()
        cfg.to_json(args.config)
        print(f"Config did not exist. Wrote defaults to {args.config}")

    if args.no_frames:
        cfg.save_frames = False

    outputs = run_simulation(cfg, args.output)
    print(f"Simulation complete. Output: {outputs.output_dir}")
    print(json.dumps(outputs.summary, indent=2))


if __name__ == "__main__":
    main()
