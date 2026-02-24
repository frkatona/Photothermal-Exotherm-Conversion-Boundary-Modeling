from __future__ import annotations

import argparse
import math
from pathlib import Path

from thermal_sim import SimulationConfig

STEFAN_BOLTZMANN = 5.670374419e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight stability/cost check for explicit solver settings.")
    parser.add_argument("--config", type=Path, default=Path("default_config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig.from_json(args.config)

    dx = cfg.width / (cfg.nx - 1)
    dy = cfg.height / (cfg.ny - 1)
    cooling_coeff = cfg.h_loss / (cfg.rho * cfg.cp * cfg.thickness)
    effective_max_emissivity = (
        max(cfg.emissivity, cfg.secondary_emissivity)
        if cfg.use_secondary_absorptivity_for_reacted
        else cfg.emissivity
    )
    radiative_linear_coeff = (
        4.0
        * effective_max_emissivity
        * STEFAN_BOLTZMANN
        * (cfg.t_amb**3)
        / (cfg.rho * cfg.cp * cfg.thickness)
    )
    cooling_total_coeff = cooling_coeff + radiative_linear_coeff

    cfl_limit = 0.5 / (cfg.alpha_th * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    cooling_limit = math.inf if cooling_total_coeff <= 0.0 else 2.0 / cooling_total_coeff
    stable_limit = min(cfl_limit, cooling_limit)
    safety = 0.9
    recommended_dt = safety * stable_limit
    substeps = max(1, int(math.ceil(cfg.dt / recommended_dt)))
    n_steps = int(round(cfg.t_end / cfg.dt))
    internal_dt = cfg.dt / substeps

    print(f"config: {args.config}")
    print(f"dx,dy [m]: {dx:.6g}, {dy:.6g}")
    print(f"dt requested [s]: {cfg.dt:.6g}")
    print(f"dt internal [s]: {internal_dt:.6g}")
    print(f"diffusion limit [s]: {cfl_limit:.6g}")
    print(f"cooling+radiative limit [s]: {cooling_limit:.6g}")
    print(f"radiative linear coeff [1/s]: {radiative_linear_coeff:.6g}")
    print(f"recommended dt [s]: {recommended_dt:.6g} (safety={safety:.2f})")
    print(f"substeps/step: {substeps}")
    print(f"outer steps: {n_steps}")
    print(f"total update passes: {n_steps * substeps}")

    if cfg.dt > recommended_dt:
        print("warning: requested dt exceeds recommended explicit limit; internal substepping will be used.")
    else:
        print("ok: requested dt is within recommended explicit limit.")


if __name__ == "__main__":
    main()
