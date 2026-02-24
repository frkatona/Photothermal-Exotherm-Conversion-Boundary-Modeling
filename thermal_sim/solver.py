from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SimulationConfig
from .io_utils import (
    write_arrays,
    write_center_history,
    write_config,
    write_frame,
    write_global_history,
    write_pulse_path,
    write_summary,
)
from .raster import RasterPath, build_pulse_train

SQRT_2PI = math.sqrt(2.0 * math.pi)
STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass
class SimulationOutputs:
    output_dir: Path
    summary: dict[str, float]


def _laplacian_robin(
    t_field: np.ndarray,
    dx: float,
    dy: float,
    h_edge: float,
    k_cond: float,
    t_amb: float,
) -> np.ndarray:
    lap_x = np.empty_like(t_field)
    lap_y = np.empty_like(t_field)

    dx2 = dx * dx
    dy2 = dy * dy

    beta_x = 2.0 * dx * h_edge / k_cond
    beta_y = 2.0 * dy * h_edge / k_cond

    left_ghost = t_field[:, 1] - beta_x * (t_field[:, 0] - t_amb)
    right_ghost = t_field[:, -2] - beta_x * (t_field[:, -1] - t_amb)
    bottom_ghost = t_field[1, :] - beta_y * (t_field[0, :] - t_amb)
    top_ghost = t_field[-2, :] - beta_y * (t_field[-1, :] - t_amb)

    lap_x[:, 1:-1] = (t_field[:, 2:] - 2.0 * t_field[:, 1:-1] + t_field[:, :-2]) / dx2
    lap_x[:, 0] = (t_field[:, 1] - 2.0 * t_field[:, 0] + left_ghost) / dx2
    lap_x[:, -1] = (right_ghost - 2.0 * t_field[:, -1] + t_field[:, -2]) / dx2

    lap_y[1:-1, :] = (t_field[2:, :] - 2.0 * t_field[1:-1, :] + t_field[:-2, :]) / dy2
    lap_y[0, :] = (t_field[1, :] - 2.0 * t_field[0, :] + bottom_ghost) / dy2
    lap_y[-1, :] = (top_ghost - 2.0 * t_field[-1, :] + t_field[-2, :]) / dy2

    return lap_x + lap_y


def _laser_energy_step(
    t_now: float,
    x: np.ndarray,
    y: np.ndarray,
    pulse_times: np.ndarray,
    pulse_x: np.ndarray,
    pulse_y: np.ndarray,
    alpha: np.ndarray,
    pulse_energy_prefactor: float,
    base_absorptivity: float,
    secondary_absorptivity: float,
    use_secondary_absorptivity_for_reacted: bool,
    w0: float,
    sigma_t: float,
    dt: float,
    rep_rate: float,
) -> np.ndarray:
    energy = np.zeros((y.size, x.size), dtype=np.float64)
    radius = 3.0 * w0
    radius2 = radius * radius

    center_idx = int(round(t_now * rep_rate))
    start = max(0, center_idx - 2)
    stop = min(pulse_times.size, center_idx + 3)

    for p_idx in range(start, stop):
        dtau = t_now - pulse_times[p_idx]
        if abs(dtau) > 4.0 * sigma_t:
            continue

        temporal = math.exp(-0.5 * (dtau / sigma_t) ** 2) / (sigma_t * SQRT_2PI)
        pulse_fraction = temporal * dt
        if pulse_fraction <= 0.0:
            continue

        x0 = pulse_x[p_idx]
        y0 = pulse_y[p_idx]

        ix_min = int(np.searchsorted(x, x0 - radius, side="left"))
        ix_max = int(np.searchsorted(x, x0 + radius, side="right")) - 1
        iy_min = int(np.searchsorted(y, y0 - radius, side="left"))
        iy_max = int(np.searchsorted(y, y0 + radius, side="right")) - 1

        ix_min = max(0, ix_min)
        iy_min = max(0, iy_min)
        ix_max = min(x.size - 1, ix_max)
        iy_max = min(y.size - 1, iy_max)
        if ix_min > ix_max or iy_min > iy_max:
            continue

        x_sub = x[ix_min : ix_max + 1]
        y_sub = y[iy_min : iy_max + 1]
        r2 = (x_sub[None, :] - x0) ** 2 + (y_sub[:, None] - y0) ** 2
        mask = r2 <= radius2
        if not np.any(mask):
            continue

        gaussian = np.exp(-2.0 * r2 / (w0 * w0))
        if use_secondary_absorptivity_for_reacted:
            alpha_local = np.clip(alpha[iy_min : iy_max + 1, ix_min : ix_max + 1], 0.0, 1.0)
            abs_local = base_absorptivity + (secondary_absorptivity - base_absorptivity) * alpha_local
        else:
            abs_local = base_absorptivity

        q_spatial = pulse_energy_prefactor * abs_local * gaussian
        energy[iy_min : iy_max + 1, ix_min : ix_max + 1] += q_spatial * pulse_fraction * mask

    return energy


def run_simulation(
    config: SimulationConfig,
    output_dir: str | Path,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SimulationOutputs:
    config.validate()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, config.width, config.nx, dtype=np.float64)
    y = np.linspace(0.0, config.height, config.ny, dtype=np.float64)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    t_field = np.full((config.ny, config.nx), config.initial_temperature, dtype=np.float64)
    alpha = np.full((config.ny, config.nx), config.initial_conversion, dtype=np.float64)
    t_peak = t_field.copy()
    laser_energy_instant = np.zeros_like(t_field)
    zero_laser = np.zeros_like(t_field)

    center_ix = int(np.argmin(np.abs(x - 0.5 * config.width)))
    center_iy = int(np.argmin(np.abs(y - 0.5 * config.height)))
    cell_area = dx * dy
    cell_mass = config.rho * config.thickness * cell_area

    n_steps = int(round(config.t_end / config.dt))
    time = np.linspace(0.0, n_steps * config.dt, n_steps + 1, dtype=np.float64)
    center_t = np.empty_like(time)
    center_alpha = np.empty_like(time)
    max_temp = np.empty_like(time)
    total_conversion = np.empty_like(time)
    laser_energy_step = np.zeros_like(time)
    reaction_energy_step = np.zeros_like(time)
    cumulative_laser_energy = np.zeros_like(time)
    cumulative_reaction_energy = np.zeros_like(time)
    cumulative_radiative_energy = np.zeros_like(time)
    center_t[0] = t_field[center_iy, center_ix]
    center_alpha[0] = alpha[center_iy, center_ix]
    max_temp[0] = np.max(t_field)
    total_conversion[0] = np.sum(alpha)

    if config.thermal_conductivity is None:
        k_cond = config.alpha_th * config.rho * config.cp
    else:
        k_cond = config.thermal_conductivity

    cooling_coeff = config.h_loss / (config.rho * config.cp * config.thickness)
    reacted_props_enabled = config.use_secondary_absorptivity_for_reacted
    base_absorptivity = config.absorptivity
    secondary_absorptivity = config.secondary_absorptivity
    base_emissivity = config.emissivity
    secondary_emissivity = config.secondary_emissivity
    w0 = 0.5 * config.spot_diameter
    pulse_energy_prefactor = (2.0 * config.pulse_energy) / (math.pi * w0 * w0)
    sigma_t = max(config.pulse_width / (2.0 * math.sqrt(2.0 * math.log(2.0))), 0.5 * config.dt)

    path = RasterPath(config)
    pulse_train = build_pulse_train(config, path)

    frame_stride = max(1, int(round(config.output_interval / config.dt)))
    frame_idx = 0
    if config.save_frames:
        write_frame(output_path, frame_idx, time[0], t_field, alpha, zero_laser)
        frame_idx += 1

    cfl_limit = 0.5 / (config.alpha_th * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    effective_max_emissivity = (
        max(base_emissivity, secondary_emissivity) if reacted_props_enabled else base_emissivity
    )
    radiative_linear_coeff = (
        4.0
        * effective_max_emissivity
        * STEFAN_BOLTZMANN
        * (config.t_amb**3)
        / (config.rho * config.cp * config.thickness)
    )
    cooling_total_coeff = cooling_coeff + radiative_linear_coeff
    cooling_limit = math.inf if cooling_total_coeff <= 0.0 else 2.0 / cooling_total_coeff
    stable_limit = min(cfl_limit, cooling_limit)
    stability_safety = 0.9
    recommended_dt = stability_safety * stable_limit

    if not np.isfinite(recommended_dt) or recommended_dt <= 0.0:
        raise ValueError(
            "Unable to determine a stable explicit time step. Check alpha_th, dx/dy, and h_loss."
        )

    substeps_per_step = max(1, int(math.ceil(config.dt / recommended_dt)))
    max_substeps = 20_000
    if substeps_per_step > max_substeps:
        raise ValueError(
            "Required internal substeps are too high for stability "
            f"({substeps_per_step} > {max_substeps}). Reduce dt or coarsen the grid."
        )
    internal_dt = config.dt / substeps_per_step
    progress_stride = max(1, n_steps // 400)

    if progress_callback is not None:
        progress_callback(
            {
                "percent": 0.0,
                "step": 0.0,
                "total_steps": float(n_steps),
                "max_temp": float(np.max(t_field)),
                "mean_conversion": float(np.mean(alpha)),
            }
        )

    nonfinite_check_stride = 25
    for step in range(n_steps):
        if cancel_check is not None and cancel_check():
            raise InterruptedError(f"Simulation canceled at outer step {step}/{n_steps}.")

        t_now = time[step]
        laser_energy_accum = np.zeros_like(t_field)
        reaction_step_j = 0.0
        radiative_step_j = 0.0

        for sub in range(substeps_per_step):
            if cancel_check is not None and cancel_check():
                raise InterruptedError(
                    f"Simulation canceled at outer step {step + 1}/{n_steps}, "
                    f"substep {sub + 1}/{substeps_per_step}."
                )

            t_sub = t_now + sub * internal_dt
            laser_energy_sub = _laser_energy_step(
                t_now=t_sub,
                x=x,
                y=y,
                pulse_times=pulse_train.times,
                pulse_x=pulse_train.x,
                pulse_y=pulse_train.y,
                alpha=alpha,
                pulse_energy_prefactor=pulse_energy_prefactor,
                base_absorptivity=base_absorptivity,
                secondary_absorptivity=secondary_absorptivity,
                use_secondary_absorptivity_for_reacted=reacted_props_enabled,
                w0=w0,
                sigma_t=sigma_t,
                dt=internal_dt,
                rep_rate=config.rep_rate,
            )
            laser_energy_accum += laser_energy_sub

            lap = _laplacian_robin(
                t_field=t_field,
                dx=dx,
                dy=dy,
                h_edge=config.h_edge,
                k_cond=k_cond,
                t_amb=config.t_amb,
            )

            diff_cool = internal_dt * (config.alpha_th * lap - cooling_coeff * (t_field - config.t_amb))
            laser_dT = laser_energy_sub / (config.rho * config.cp * config.thickness)

            if reacted_props_enabled:
                alpha_local = np.clip(alpha, 0.0, 1.0)
                emissivity_local = base_emissivity + (secondary_emissivity - base_emissivity) * alpha_local
            else:
                emissivity_local = base_emissivity
            t_for_rad = np.clip(t_field, 1.0, 1.0e4)
            q_rad = emissivity_local * STEFAN_BOLTZMANN * (t_for_rad**4 - config.t_amb**4)
            radiative_dT = -internal_dt * q_rad / (config.rho * config.cp * config.thickness)
            radiative_step_j += float(np.sum(q_rad) * cell_area * internal_dt)

            t_trial = t_field + diff_cool + laser_dT + radiative_dT

            rate = config.A * np.exp(-config.Ea / (config.gas_constant * np.clip(t_trial, 1.0, None)))
            rate *= (1.0 - alpha)
            delta_alpha = np.clip(rate * internal_dt, 0.0, 1.0 - alpha)
            reaction_step_j += float(np.sum(delta_alpha) * config.delta_h * cell_mass)

            alpha = np.clip(alpha + delta_alpha, 0.0, 1.0)
            t_field = np.clip(t_trial + (config.delta_h / config.cp) * delta_alpha, 1.0, None)

        laser_energy_instant = laser_energy_accum
        laser_step_j = float(np.sum(laser_energy_instant) * cell_area)
        if ((step + 1) % nonfinite_check_stride == 0) or (step == n_steps - 1):
            if (not np.all(np.isfinite(t_field))) or (not np.all(np.isfinite(alpha))):
                raise FloatingPointError(
                    "Non-finite values detected during integration at "
                    f"t={time[step + 1]:.6g} s (step={step + 1}). "
                    f"Try reducing dt from {config.dt:.6g} s or increasing nx/ny spacing."
                )
        t_peak = np.maximum(t_peak, t_field)

        center_t[step + 1] = t_field[center_iy, center_ix]
        center_alpha[step + 1] = alpha[center_iy, center_ix]
        max_temp[step + 1] = np.max(t_field)
        total_conversion[step + 1] = np.sum(alpha)
        laser_energy_step[step + 1] = laser_step_j
        reaction_energy_step[step + 1] = reaction_step_j
        cumulative_laser_energy[step + 1] = cumulative_laser_energy[step] + laser_step_j
        cumulative_reaction_energy[step + 1] = cumulative_reaction_energy[step] + reaction_step_j
        cumulative_radiative_energy[step + 1] = cumulative_radiative_energy[step] + radiative_step_j

        if progress_callback is not None and (((step + 1) % progress_stride == 0) or (step == n_steps - 1)):
            progress_callback(
                {
                    "percent": 100.0 * (step + 1) / n_steps,
                    "step": float(step + 1),
                    "total_steps": float(n_steps),
                    "max_temp": float(max_temp[step + 1]),
                    "mean_conversion": float(np.mean(alpha)),
                }
            )

        if config.save_frames and ((step + 1) % frame_stride == 0 or step == n_steps - 1):
            write_frame(output_path, frame_idx, time[step + 1], t_field, alpha, laser_energy_instant)
            frame_idx += 1

    summary = {
        "t_max_final_K": float(np.max(t_field)),
        "t_max_peak_K": float(np.max(t_peak)),
        "t_center_final_K": float(center_t[-1]),
        "alpha_max_final": float(np.max(alpha)),
        "alpha_mean_final": float(np.mean(alpha)),
        "alpha_center_final": float(center_alpha[-1]),
        "pulse_count": float(pulse_train.times.size),
        "dt_s": config.dt,
        "dt_internal_s": internal_dt,
        "substeps_per_step": float(substeps_per_step),
        "t_end_s": config.t_end,
        "dx_m": dx,
        "dy_m": dy,
        "dt_cfl_limit_s": cfl_limit,
        "dt_cooling_limit_s": cooling_limit,
        "dt_radiative_linear_limit_s": (math.inf if radiative_linear_coeff <= 0.0 else 2.0 / radiative_linear_coeff),
        "dt_stable_limit_s": stable_limit,
        "dt_recommended_s": recommended_dt,
        "laser_energy_total_J": float(cumulative_laser_energy[-1]),
        "reaction_energy_total_J": float(cumulative_reaction_energy[-1]),
        "radiative_energy_loss_total_J": float(cumulative_radiative_energy[-1]),
        "absorptivity_base": float(config.absorptivity),
        "absorptivity_secondary": float(config.secondary_absorptivity),
        "absorptivity_secondary_enabled": float(config.use_secondary_absorptivity_for_reacted),
        "emissivity_base": float(config.emissivity),
        "emissivity_secondary": float(config.secondary_emissivity),
    }

    write_arrays(
        output_dir=output_path,
        x=x,
        y=y,
        temperature=t_field,
        temperature_peak=t_peak,
        conversion=alpha,
        laser_energy_instant=laser_energy_instant,
    )
    write_center_history(output_path, time, center_t, center_alpha)
    write_global_history(
        output_dir=output_path,
        time=time,
        max_temperature=max_temp,
        total_conversion=total_conversion,
        laser_energy_step=laser_energy_step,
        reaction_energy_step=reaction_energy_step,
        cumulative_laser_energy=cumulative_laser_energy,
        cumulative_reaction_energy=cumulative_reaction_energy,
    )
    write_pulse_path(output_path, pulse_train.times, pulse_train.x, pulse_train.y)
    write_summary(output_path, summary)
    write_config(output_path, config)

    return SimulationOutputs(output_dir=output_path, summary=summary)
