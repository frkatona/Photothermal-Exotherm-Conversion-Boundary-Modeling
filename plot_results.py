from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from thermal_sim.io_utils import load_center_history, load_pulse_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static postprocessing for simulation outputs.")
    parser.add_argument("--output", type=Path, default=Path("outputs/latest"))
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <output>/static_summary.png",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    return parser.parse_args()


def create_static_plot(
    output_dir: Path,
    save_path: Path | None = None,
    no_show: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    out_dir = output_dir
    if progress_callback is not None:
        progress_callback(5.0, "loading arrays")
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Static plotting canceled before loading arrays.")

    x = np.load(out_dir / "x.npy")
    y = np.load(out_dir / "y.npy")
    t_final = np.load(out_dir / "temperature.npy")
    t_peak = np.load(out_dir / "temperature_peak.npy")
    alpha = np.load(out_dir / "conversion.npy")

    center_time, center_temp, center_alpha = load_center_history(out_dir)
    _, pulse_x, pulse_y = load_pulse_path(out_dir)
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Static plotting canceled before rendering.")
    if progress_callback is not None:
        progress_callback(25.0, "building figure")

    x_mm = x * 1e3
    y_mm = y * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    im0 = axes[0, 0].imshow(t_final, origin="lower", extent=extent, cmap="inferno")
    axes[0, 0].set_title("Final Temperature (K)")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.9)

    im1 = axes[0, 1].imshow(t_peak, origin="lower", extent=extent, cmap="magma")
    axes[0, 1].set_title("Peak Temperature (K)")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("y (mm)")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.9)

    im2 = axes[1, 0].imshow(alpha, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 0].plot(pulse_x * 1e3, pulse_y * 1e3, color="white", linewidth=0.5, alpha=0.7)
    axes[1, 0].set_title("Final Conversion with Pulse Path")
    axes[1, 0].set_xlabel("x (mm)")
    axes[1, 0].set_ylabel("y (mm)")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.9)

    axes[1, 1].plot(center_time, center_temp, color="#d1495b", linewidth=1.8, label="Temperature (K)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Temperature (K)", color="#d1495b")
    axes[1, 1].tick_params(axis="y", labelcolor="#d1495b")
    axes[1, 1].set_title("Center Point History")

    ax2 = axes[1, 1].twinx()
    ax2.plot(center_time, center_alpha, color="#00798c", linewidth=1.6, label="Conversion")
    ax2.set_ylabel("Conversion", color="#00798c")
    ax2.tick_params(axis="y", labelcolor="#00798c")
    ax2.set_ylim(0.0, 1.0)

    if progress_callback is not None:
        progress_callback(75.0, "saving image")
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Static plotting canceled before save.")

    save_path = save_path if save_path is not None else (out_dir / "static_summary.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)

    if not no_show:
        plt.show()
    plt.close(fig)
    if progress_callback is not None:
        progress_callback(100.0, "plot complete")
    return save_path


def main() -> None:
    args = parse_args()
    saved = create_static_plot(
        output_dir=args.output,
        save_path=args.save,
        no_show=args.no_show,
        progress_callback=None,
    )
    print(f"Saved static figure: {saved}")


if __name__ == "__main__":
    main()
