from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass
class PulseTrain:
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray


class RasterPath:
    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.x_min = cfg.spot_diameter
        self.x_max = cfg.width - cfg.spot_diameter
        self.y_min = cfg.spot_diameter
        self.y_max = cfg.height - cfg.spot_diameter

        self.x_span = self.x_max - self.x_min
        self.y_span = self.y_max - self.y_min

        self.n_h_lines = max(1, int(math.floor(self.y_span / cfg.line_pitch)) + 1)
        self.n_v_lines = max(1, int(math.floor(self.x_span / cfg.line_pitch)) + 1)

        self.line_time_x = self.x_span / cfg.scan_speed
        self.line_time_y = self.y_span / cfg.scan_speed

        self.duration_horizontal = self.n_h_lines * self.line_time_x
        self.duration_vertical = self.n_v_lines * self.line_time_y
        self.cross_cycle = self.duration_horizontal + self.duration_vertical

    def raster_position(self, t: float) -> tuple[float, float]:
        if self.cfg.pass_pattern == "offset_start":
            return self._horizontal(t, offset_lines=self.cfg.pass_start_offset)
        return self._cross_hatch(t)

    def _horizontal(self, t: float, offset_lines: float) -> tuple[float, float]:
        distance = max(0.0, t) * self.cfg.scan_speed
        line_idx = int(math.floor(distance / self.x_span))
        x = self.x_min + (distance % self.x_span)

        wrapped = (line_idx + offset_lines) % self.n_h_lines
        y = self.y_min + wrapped * self.cfg.line_pitch
        y = min(max(y, self.y_min), self.y_max)
        return x, y

    def _horizontal_phase(self, local_t: float) -> tuple[float, float]:
        distance = max(0.0, local_t) * self.cfg.scan_speed
        line_idx = int(math.floor(distance / self.x_span))
        x = self.x_min + (distance % self.x_span)

        y = self.y_min + (line_idx % self.n_h_lines) * self.cfg.line_pitch
        y = min(max(y, self.y_min), self.y_max)
        return x, y

    def _vertical_phase(self, local_t: float) -> tuple[float, float]:
        distance = max(0.0, local_t) * self.cfg.scan_speed
        line_idx = int(math.floor(distance / self.y_span))
        y = self.y_min + (distance % self.y_span)

        x = self.x_min + (line_idx % self.n_v_lines) * self.cfg.line_pitch
        x = min(max(x, self.x_min), self.x_max)
        return x, y

    def _cross_hatch(self, t: float) -> tuple[float, float]:
        cycle_t = max(0.0, t) % self.cross_cycle
        if cycle_t < self.duration_horizontal:
            return self._horizontal_phase(cycle_t)
        return self._vertical_phase(cycle_t - self.duration_horizontal)


def build_pulse_train(cfg: SimulationConfig, path: RasterPath) -> PulseTrain:
    period = 1.0 / cfg.rep_rate
    n_pulses = int(math.floor(cfg.t_end * cfg.rep_rate)) + 1
    times = (np.arange(n_pulses, dtype=np.float64) * period).astype(np.float64)

    pulse_x = np.empty_like(times)
    pulse_y = np.empty_like(times)
    for i, t in enumerate(times):
        x, y = path.raster_position(float(t))
        pulse_x[i] = x
        pulse_y[i] = y

    return PulseTrain(times=times, x=pulse_x, y=pulse_y)
