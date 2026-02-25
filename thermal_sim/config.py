from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class SimulationConfig:
    width: float = 8.0e-3
    height: float = 8.0e-3
    nx: int = 161
    ny: int = 161
    thickness: float = 50.0e-6

    rho: float = 1200.0
    cp: float = 1300.0
    alpha_th: float = 1.0e-7
    thermal_conductivity: float | None = None

    h_loss: float = 20.0
    emissivity: float = 0.9
    secondary_emissivity: float = 0.9
    h_edge: float = 12.0
    t_amb: float = 298.15

    A: float = 2.5e5
    Ea: float = 62_000.0
    gas_constant: float = 8.314462618
    delta_h: float = 2.0e5

    pulse_energy: float = 2.0e-4
    absorptivity: float = 0.65
    secondary_absorptivity: float = 0.65
    use_secondary_absorptivity_for_reacted: bool = False
    spot_diameter: float = 2.0e-4
    pulse_width: float = 120.0e-6
    rep_rate: float = 250.0
    scan_speed: float = 0.02
    line_pitch: float = 1.5e-4
    pass_pattern: str = "offset_start"
    pass_start_offset: float = 0.0

    dt: float = 1.0e-4
    t_end: float = 0.6

    initial_temperature: float = 298.15
    initial_conversion: float = 0.0

    output_interval: float = 2.0e-3
    save_frames: bool = True
    compute_backend: str = "auto"

    def validate(self) -> None:
        positive_names = [
            "width",
            "height",
            "thickness",
            "rho",
            "cp",
            "alpha_th",
            "h_loss",
            "h_edge",
            "A",
            "Ea",
            "gas_constant",
            "pulse_energy",
            "spot_diameter",
            "pulse_width",
            "rep_rate",
            "scan_speed",
            "line_pitch",
            "dt",
            "t_end",
            "output_interval",
        ]
        for name in positive_names:
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0; received {value}")

        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must both be >= 3 for 2D finite differences")

        if not (0.0 <= self.absorptivity <= 1.0):
            raise ValueError("absorptivity must be in [0, 1]")

        if not (0.0 <= self.secondary_absorptivity <= 1.0):
            raise ValueError("secondary_absorptivity must be in [0, 1]")

        if not (0.0 <= self.emissivity <= 1.0):
            raise ValueError("emissivity must be in [0, 1]")

        if not (0.0 <= self.secondary_emissivity <= 1.0):
            raise ValueError("secondary_emissivity must be in [0, 1]")

        if not (0.0 <= self.initial_conversion <= 1.0):
            raise ValueError("initial_conversion must be in [0, 1]")

        if self.pass_pattern not in {"offset_start", "cross_hatch"}:
            raise ValueError("pass_pattern must be 'offset_start' or 'cross_hatch'")

        if self.compute_backend not in {"auto", "numpy", "numba"}:
            raise ValueError("compute_backend must be one of: auto, numpy, numba")

        inset = self.spot_diameter
        if self.width <= 2.0 * inset or self.height <= 2.0 * inset:
            raise ValueError(
                "Domain must be larger than 2*spot_diameter in both directions "
                "to support inset scan bounds."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        cfg = cls(**filtered)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
