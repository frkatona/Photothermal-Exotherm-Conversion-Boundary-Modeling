from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path
from shutil import which
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from thermal_sim import SimulationConfig
from thermal_sim.io_utils import load_global_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MP4 animation by streaming PNG frames to ffmpeg."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/latest"))
    parser.add_argument("--mp4", type=Path, default=None, help="Target MP4 path.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def create_animation(
    output_dir: Path,
    mp4_path: Path | None = None,
    fps: int = 20,
    dpi: int = 140,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    out_dir = output_dir
    frames_dir = out_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    n_frames = len(frame_files)

    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH.")

    cfg = SimulationConfig.from_json(out_dir / "config_used.json")
    dt = cfg.dt

    x = np.load(out_dir / "x.npy")
    y = np.load(out_dir / "y.npy")
    t_peak = np.load(out_dir / "temperature_peak.npy")

    (
        hist_time,
        hist_t_max,
        hist_total_conversion,
        _hist_laser_step_j,
        _hist_reaction_step_j,
        hist_cum_laser_j,
        hist_cum_reaction_j,
    ) = load_global_history(out_dir)
    total_conversion_norm = np.clip(hist_total_conversion / float(cfg.nx * cfg.ny), 0.0, 1.0)

    x_mm = x * 1e3
    y_mm = y * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    with np.load(frame_files[0]) as first:
        t0 = first["temperature"]
        a0 = first["conversion"]
        first_time = float(first["time_s"])
        if "laser_energy" in first.files:
            l0 = first["laser_energy"]
        else:
            l0 = np.zeros_like(t0)

    t_min = float(min(np.min(t0), np.min(t_peak)))
    t_max = float(np.max(t_peak))

    laser_power_max = 0.0
    for frame_path in frame_files:
        with np.load(frame_path) as frame:
            if "laser_energy" in frame.files:
                laser_power_max = max(laser_power_max, float(np.max(frame["laser_energy"])) / dt)
    if laser_power_max <= 0.0:
        laser_power_max = 1.0

    fig = plt.figure(figsize=(13.0, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_laser = fig.add_subplot(gs[1, 0])
    ax_line_state = fig.add_subplot(gs[1, 1])
    ax_line_energy = fig.add_subplot(gs[2, :])

    im_t = ax_t.imshow(
        t0,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=t_min,
        vmax=t_max,
    )
    ax_t.set_title("Temperature (K)")
    ax_t.set_xlabel("x (mm)")
    ax_t.set_ylabel("y (mm)")
    fig.colorbar(im_t, ax=ax_t, shrink=0.9)

    im_a = ax_a.imshow(
        a0,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax_a.set_title("Conversion")
    ax_a.set_xlabel("x (mm)")
    ax_a.set_ylabel("y (mm)")
    fig.colorbar(im_a, ax=ax_a, shrink=0.9)

    im_laser = ax_laser.imshow(
        l0 / dt,
        origin="lower",
        extent=extent,
        cmap="plasma",
        vmin=0.0,
        vmax=laser_power_max,
    )
    ax_laser.set_title("Laser Pulse Power Density (W/m^2)")
    ax_laser.set_xlabel("x (mm)")
    ax_laser.set_ylabel("y (mm)")
    fig.colorbar(im_laser, ax=ax_laser, shrink=0.9)

    ax_line_state.set_title("Max Temperature and Normalized Conversion")
    ax_line_state.set_xlabel("Time (s)")
    ax_line_state.set_ylabel("Max Temperature (K)", color="#b30e1d")
    ax_line_state.tick_params(axis="y", labelcolor="#b30e1d")
    ax_line_state.set_xlim(hist_time[0], hist_time[-1])
    ax_line_state.set_ylim(float(np.min(hist_t_max) * 0.99), float(np.max(hist_t_max) * 1.01))
    (line_tmax,) = ax_line_state.plot(
        [hist_time[0]],
        [hist_t_max[0]],
        color="#b30e1d",
        linewidth=2.0,
        label="Max Temperature",
    )

    ax_conv = ax_line_state.twinx()
    ax_conv.set_ylabel("Normalized Total Conversion", color="#005f73")
    ax_conv.tick_params(axis="y", labelcolor="#005f73")
    ax_conv.set_ylim(0.0, 1.0)
    (line_conv,) = ax_conv.plot(
        [hist_time[0]],
        [total_conversion_norm[0]],
        color="#005f73",
        linewidth=2.0,
        label="Normalized Total Conversion",
    )

    ax_line_energy.set_title("Cumulative Energy Input Over Time")
    ax_line_energy.set_xlabel("Time (s)")
    ax_line_energy.set_ylabel("Energy (J)")
    ax_line_energy.set_xlim(hist_time[0], hist_time[-1])
    energy_max = float(max(np.max(hist_cum_laser_j), np.max(hist_cum_reaction_j)))
    energy_upper = energy_max * 1.05 if energy_max > 0.0 else 1.0
    ax_line_energy.set_ylim(0.0, energy_upper)
    (line_laser_energy,) = ax_line_energy.plot(
        [hist_time[0]],
        [hist_cum_laser_j[0]],
        color="#006494",
        linewidth=2.0,
        label="Cumulative Laser Energy",
    )
    (line_reaction_energy,) = ax_line_energy.plot(
        [hist_time[0]],
        [hist_cum_reaction_j[0]],
        color="#f95738",
        linewidth=2.0,
        label="Cumulative Reaction Energy",
    )
    ax_line_energy.legend(loc="upper left")

    fig.suptitle(f"t = {first_time:.4f} s")

    mp4_path = mp4_path if mp4_path is not None else (out_dir / "simulation.mp4")
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    canceled = False
    if progress_callback is not None:
        progress_callback(0.0, "animation started")
    try:
        assert proc.stdin is not None
        for frame_idx, frame_path in enumerate(frame_files):
            if cancel_check is not None and cancel_check():
                canceled = True
                raise InterruptedError(
                    f"Animation canceled at frame {frame_idx}/{n_frames}."
                )

            with np.load(frame_path) as frame:
                t_field = frame["temperature"]
                alpha = frame["conversion"]
                t_s = float(frame["time_s"])
                if "laser_energy" in frame.files:
                    laser_energy = frame["laser_energy"]
                else:
                    laser_energy = np.zeros_like(t_field)

            im_t.set_data(t_field)
            im_a.set_data(alpha)
            im_laser.set_data(laser_energy / dt)

            hist_idx = int(np.searchsorted(hist_time, t_s, side="right") - 1)
            hist_idx = max(0, min(hist_idx, hist_time.size - 1))

            line_tmax.set_data(hist_time[: hist_idx + 1], hist_t_max[: hist_idx + 1])
            line_conv.set_data(
                hist_time[: hist_idx + 1],
                total_conversion_norm[: hist_idx + 1],
            )
            line_laser_energy.set_data(
                hist_time[: hist_idx + 1],
                hist_cum_laser_j[: hist_idx + 1],
            )
            line_reaction_energy.set_data(
                hist_time[: hist_idx + 1],
                hist_cum_reaction_j[: hist_idx + 1],
            )

            fig.suptitle(f"t = {t_s:.4f} s")

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=dpi)
            proc.stdin.write(buffer.getvalue())

            if progress_callback is not None:
                progress_callback(
                    100.0 * (frame_idx + 1) / n_frames,
                    f"frame {frame_idx + 1}/{n_frames}",
                )
        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")
    finally:
        if canceled:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        plt.close(fig)

    return mp4_path


def main() -> None:
    args = parse_args()
    saved = create_animation(
        output_dir=args.output,
        mp4_path=args.mp4,
        fps=args.fps,
        dpi=args.dpi,
        progress_callback=None,
    )
    print(f"Saved animation: {saved}")


if __name__ == "__main__":
    main()
