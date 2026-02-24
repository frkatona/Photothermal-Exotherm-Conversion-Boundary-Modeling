from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import SimulationConfig


def write_arrays(
    output_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    temperature: np.ndarray,
    temperature_peak: np.ndarray,
    conversion: np.ndarray,
    laser_energy_instant: np.ndarray,
) -> None:
    np.save(output_dir / "x.npy", x)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "temperature.npy", temperature)
    np.save(output_dir / "temperature_peak.npy", temperature_peak)
    np.save(output_dir / "conversion.npy", conversion)
    np.save(output_dir / "laser_energy_instant.npy", laser_energy_instant)


def write_center_history(
    output_dir: Path,
    time: np.ndarray,
    temperature: np.ndarray,
    conversion: np.ndarray,
) -> None:
    with (output_dir / "center_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "temperature_K", "conversion"])
        for row in zip(time, temperature, conversion):
            writer.writerow(row)


def write_pulse_path(
    output_dir: Path,
    pulse_times: np.ndarray,
    pulse_x: np.ndarray,
    pulse_y: np.ndarray,
) -> None:
    with (output_dir / "pulse_path.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pulse_index", "time_s", "x_m", "y_m"])
        for i, (t, x, y) in enumerate(zip(pulse_times, pulse_x, pulse_y)):
            writer.writerow([i, t, x, y])


def write_summary(output_dir: Path, summary: dict[str, float]) -> None:
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        for key in sorted(summary):
            f.write(f"{key}: {summary[key]:.8g}\n")


def write_frame(
    output_dir: Path,
    frame_index: int,
    time_s: float,
    temperature: np.ndarray,
    conversion: np.ndarray,
    laser_energy: np.ndarray,
) -> None:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_file = frames_dir / f"frame_{frame_index:06d}.npz"
    np.savez_compressed(
        frame_file,
        time_s=time_s,
        temperature=temperature,
        conversion=conversion,
        laser_energy=laser_energy,
    )


def write_global_history(
    output_dir: Path,
    time: np.ndarray,
    max_temperature: np.ndarray,
    total_conversion: np.ndarray,
    laser_energy_step: np.ndarray,
    reaction_energy_step: np.ndarray,
    cumulative_laser_energy: np.ndarray,
    cumulative_reaction_energy: np.ndarray,
) -> None:
    with (output_dir / "global_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "max_temperature_K",
                "total_conversion",
                "laser_energy_step_J",
                "reaction_energy_step_J",
                "cumulative_laser_energy_J",
                "cumulative_reaction_energy_J",
            ]
        )
        for row in zip(
            time,
            max_temperature,
            total_conversion,
            laser_energy_step,
            reaction_energy_step,
            cumulative_laser_energy,
            cumulative_reaction_energy,
        ):
            writer.writerow(row)


def write_config(output_dir: Path, cfg: SimulationConfig) -> None:
    cfg.to_json(output_dir / "config_used.json")


def load_center_history(output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(output_dir / "center_history.csv", delimiter=",", names=True)
    return data["time_s"], data["temperature_K"], data["conversion"]


def load_pulse_path(output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(output_dir / "pulse_path.csv", delimiter=",", names=True)
    return data["time_s"], data["x_m"], data["y_m"]


def load_global_history(
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(output_dir / "global_history.csv", delimiter=",", names=True)
    return (
        data["time_s"],
        data["max_temperature_K"],
        data["total_conversion"],
        data["laser_energy_step_J"],
        data["reaction_energy_step_J"],
        data["cumulative_laser_energy_J"],
        data["cumulative_reaction_energy_J"],
    )
