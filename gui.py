from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from plotly.subplots import make_subplots

from thermal_sim import SimulationConfig, run_simulation
from thermal_sim.io_utils import load_center_history, load_global_history, load_pulse_path


FIELD_TYPES: dict[str, str] = {
    "width": "float",
    "height": "float",
    "nx": "int",
    "ny": "int",
    "thickness": "float",
    "rho": "float",
    "cp": "float",
    "alpha_th": "float",
    "thermal_conductivity": "optional_float",
    "h_loss": "float",
    "emissivity": "float",
    "secondary_emissivity": "float",
    "h_edge": "float",
    "t_amb": "float",
    "A": "float",
    "Ea": "float",
    "gas_constant": "float",
    "delta_h": "float",
    "pulse_energy": "float",
    "absorptivity": "float",
    "secondary_absorptivity": "float",
    "use_secondary_absorptivity_for_reacted": "bool",
    "spot_diameter": "float",
    "pulse_width": "float",
    "rep_rate": "float",
    "scan_speed": "float",
    "line_pitch": "float",
    "pass_pattern": "str",
    "pass_start_offset": "float",
    "dt": "float",
    "t_end": "float",
    "initial_temperature": "float",
    "initial_conversion": "float",
    "output_interval": "float",
    "save_frames": "bool",
    "compute_backend": "str",
}

FIELD_GROUPS: list[tuple[str, list[str]]] = [
    ("Domain", ["width", "height", "nx", "ny", "thickness"]),
    (
        "Thermal",
        ["rho", "cp", "alpha_th", "thermal_conductivity", "h_loss", "emissivity", "secondary_emissivity", "h_edge", "t_amb"],
    ),
    ("Reaction", ["A", "Ea", "gas_constant", "delta_h", "initial_temperature", "initial_conversion"]),
    (
        "Laser/Scan",
        [
            "pulse_energy",
            "absorptivity",
            "secondary_absorptivity",
            "use_secondary_absorptivity_for_reacted",
            "spot_diameter",
            "pulse_width",
            "rep_rate",
            "scan_speed",
            "line_pitch",
            "pass_pattern",
            "pass_start_offset",
        ],
    ),
    ("Time/Output", ["dt", "t_end", "output_interval", "save_frames", "compute_backend"]),
]

DEFAULT_PLOT_CARD_WIDTH_PX = 640
MIN_PLOT_CARD_WIDTH_PX = 360
MAX_PLOT_CARD_WIDTH_PX = 1200

FIELD_UNITS: dict[str, str] = {
    "width": "m",
    "height": "m",
    "nx": "count",
    "ny": "count",
    "thickness": "m",
    "rho": "kg/m^3",
    "cp": "J/(kg*K)",
    "alpha_th": "m^2/s",
    "thermal_conductivity": "W/(m*K)",
    "h_loss": "W/(m^2*K)",
    "emissivity": "-",
    "secondary_emissivity": "-",
    "h_edge": "W/(m^2*K)",
    "t_amb": "K",
    "A": "1/s",
    "Ea": "J/mol",
    "gas_constant": "J/(mol*K)",
    "delta_h": "J/kg",
    "pulse_energy": "J",
    "absorptivity": "-",
    "secondary_absorptivity": "-",
    "use_secondary_absorptivity_for_reacted": "bool",
    "spot_diameter": "m",
    "pulse_width": "s",
    "rep_rate": "Hz",
    "scan_speed": "m/s",
    "line_pitch": "m",
    "pass_pattern": "mode",
    "pass_start_offset": "line-pitch",
    "dt": "s",
    "t_end": "s",
    "initial_temperature": "K",
    "initial_conversion": "-",
    "output_interval": "s",
    "save_frames": "bool",
    "compute_backend": "auto/numpy/numba",
}

FONT_OPTIONS: dict[str, str] = {
    "source": "Source Sans 3, Segoe UI, sans-serif",
    "inter": "Inter, Segoe UI, sans-serif",
    "plex": "IBM Plex Sans, Segoe UI, sans-serif",
    "roboto": "Roboto, Segoe UI, sans-serif",
}

COLOR_SCHEMES: dict[str, dict[str, str]] = {
    "tesla_dark": {
        "bg0": "#05070a",
        "bg1": "#0a0f14",
        "panel_a": "rgba(17, 22, 28, 0.94)",
        "panel_b": "rgba(12, 16, 21, 0.92)",
        "line": "#252d37",
        "line_soft": "#1b222b",
        "text_main": "#f3f7fc",
        "text_soft": "#98a5b5",
        "accent": "#e82127",
        "plot_bg": "#0f1318",
        "grid": "#202733",
        "axis": "#2e3743",
    },
    "carbon_blue": {
        "bg0": "#070b11",
        "bg1": "#0d141d",
        "panel_a": "rgba(18, 24, 35, 0.94)",
        "panel_b": "rgba(14, 19, 28, 0.92)",
        "line": "#2a3443",
        "line_soft": "#202938",
        "text_main": "#edf3ff",
        "text_soft": "#9caec8",
        "accent": "#3b82f6",
        "plot_bg": "#101925",
        "grid": "#243145",
        "axis": "#314258",
    },
    "graphite_slate": {
        "bg0": "#0a0a0a",
        "bg1": "#151515",
        "panel_a": "rgba(31, 31, 31, 0.94)",
        "panel_b": "rgba(24, 24, 24, 0.92)",
        "line": "#3a3a3a",
        "line_soft": "#2d2d2d",
        "text_main": "#f1f1f1",
        "text_soft": "#a8a8a8",
        "accent": "#f97316",
        "plot_bg": "#191919",
        "grid": "#2b2b2b",
        "axis": "#3d3d3d",
    },
}


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.running = False
        self.percent = 0.0
        self.status = "Idle"
        self.detail = "Ready."
        self.logs: list[str] = []
        self.append_log("Dashboard ready.")

    def append_log(self, message: str) -> None:
        with self.lock:
            stamp = time.strftime("%H:%M:%S")
            self.logs.append(f"[{stamp}] {message}")
            self.logs = self.logs[-250:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "percent": self.percent,
                "status": self.status,
                "detail": self.detail,
                "logs": list(self.logs),
            }

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.cancel_event.clear()
            self.percent = 0.0
            self.status = "Solver 0.0%"
            self.detail = "starting"
        self.append_log("Simulation started.")
        return True

    def update_progress(self, percent: float, detail: str) -> None:
        with self.lock:
            self.percent = max(0.0, min(100.0, percent))
            self.status = f"Solver {self.percent:.1f}%"
            self.detail = detail

    def finish(self, status: str, detail: str, percent: float | None = None) -> None:
        with self.lock:
            self.running = False
            self.cancel_event.clear()
            self.status = status
            self.detail = detail
            if percent is not None:
                self.percent = percent

    def request_cancel(self) -> bool:
        with self.lock:
            if not self.running:
                return False
            self.cancel_event.set()
        self.append_log("Cancel requested.")
        return True

    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()


RUNTIME = RuntimeState()


def resolve_path(raw: str | None, fallback: str) -> Path:
    text = (raw or "").strip()
    return Path(text or fallback).expanduser()


def value_to_text(value: Any, kind: str) -> str:
    if kind == "bool":
        return "true" if bool(value) else "false"
    if value is None:
        return ""
    if kind in {"float", "optional_float"}:
        return f"{float(value):.12g}"
    return str(value)


def parse_value(raw: Any, kind: str) -> Any:
    text = "" if raw is None else str(raw).strip()
    if kind == "bool":
        lo = text.lower()
        if lo in {"true", "1", "yes", "y", "on"}:
            return True
        if lo in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"invalid boolean: {text!r}")
    if kind == "int":
        value = float(text)
        if not value.is_integer():
            raise ValueError(f"invalid integer: {text!r}")
        return int(value)
    if kind == "float":
        return float(text)
    if kind == "optional_float":
        return None if text == "" else float(text)
    return text


def cfg_to_rows(cfg: SimulationConfig) -> list[dict[str, str]]:
    d = cfg.to_dict()
    rows: list[dict[str, str]] = []
    for group, keys in FIELD_GROUPS:
        for key in keys:
            rows.append(
                {
                    "group": group,
                    "parameter": key,
                    "units": FIELD_UNITS.get(key, ""),
                    "value": value_to_text(d[key], FIELD_TYPES[key]),
                }
            )
    return rows


def rows_to_cfg(rows: list[dict[str, Any]]) -> SimulationConfig:
    lookup = {str(r.get("parameter", "")).strip(): r.get("value", "") for r in rows}
    defaults = SimulationConfig().to_dict()
    parsed: dict[str, Any] = {}
    for key, kind in FIELD_TYPES.items():
        value = parse_value(lookup.get(key, value_to_text(defaults[key], kind)), kind)
        if key == "compute_backend":
            value = str(value).strip().lower() or "auto"
        parsed[key] = value
    cfg = SimulationConfig(**parsed)
    cfg.validate()
    return cfg


def load_or_create(path: Path) -> SimulationConfig:
    if path.exists():
        return SimulationConfig.from_json(path)
    cfg = SimulationConfig()
    cfg.to_json(path)
    return cfg


def style_figure(
    fig: go.Figure,
    plot_font: str,
    scheme: dict[str, str],
    show_legend: bool,
) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=scheme["plot_bg"],
        margin={"l": 52, "r": 16, "t": 46, "b": 46},
        font={"family": plot_font, "size": 15, "color": scheme["text_main"]},
        title_font={"size": 20, "color": scheme["text_main"]},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        showlegend=show_legend,
    )
    fig.update_xaxes(showline=True, linecolor=scheme["axis"], gridcolor=scheme["grid"], zeroline=False)
    fig.update_yaxes(showline=True, linecolor=scheme["axis"], gridcolor=scheme["grid"], zeroline=False)
    return fig


def empty_fig(title: str, text: str, plot_font: str, scheme: dict[str, str], show_legend: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": scheme["text_soft"]},
    )
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig, plot_font, scheme, show_legend)


def load_figs(
    out_dir: Path,
    plot_font: str,
    scheme: dict[str, str],
    show_pulse_overlay: bool,
    show_legend: bool,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, str]:
    req = [out_dir / "x.npy", out_dir / "y.npy", out_dir / "temperature.npy", out_dir / "temperature_peak.npy", out_dir / "conversion.npy", out_dir / "center_history.csv", out_dir / "global_history.csv"]
    summary_path = out_dir / "summary.txt"
    summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else "No summary file yet."
    if not all(p.exists() for p in req):
        f = empty_fig("No data", f"No simulation outputs at {out_dir}", plot_font, scheme, show_legend)
        return f, f, f, f, f, f, summary
    try:
        x = np.load(out_dir / "x.npy") * 1e3
        y = np.load(out_dir / "y.npy") * 1e3
        t_final = np.load(out_dir / "temperature.npy")
        t_peak = np.load(out_dir / "temperature_peak.npy")
        conv = np.load(out_dir / "conversion.npy")
        center_t, center_temp, center_alpha = load_center_history(out_dir)
        ht, hmax, htot, _, _, hlaser, hreact = load_global_history(out_dir)
        norm = np.clip(htot / float(conv.size), 0.0, 1.0)
        px = py = None
        if (out_dir / "pulse_path.csv").exists():
            _, px, py = load_pulse_path(out_dir)
    except Exception as exc:
        f = empty_fig("Load error", str(exc), plot_font, scheme, show_legend)
        return f, f, f, f, f, f, str(exc)

    def heat(title: str, z: np.ndarray, colors: str, ztitle: str) -> go.Figure:
        fig = go.Figure(go.Heatmap(x=x, y=y, z=z, colorscale=colors, colorbar={"title": ztitle}))
        fig.update_layout(title=title)
        fig.update_xaxes(title="x (mm)")
        fig.update_yaxes(title="y (mm)", scaleanchor="x", scaleratio=1)
        return style_figure(fig, plot_font, scheme, show_legend)

    fig_final = heat("Final Temperature (K)", t_final, "Inferno", "K")
    fig_peak = heat("Peak Temperature (K)", t_peak, "Magma", "K")
    fig_conv = heat("Final Conversion", conv, "Viridis", "a")
    if show_pulse_overlay and px is not None and py is not None:
        fig_conv.add_trace(go.Scatter(x=px * 1e3, y=py * 1e3, mode="lines", name="Pulse path", line={"color": "white", "width": 1}))
    fig_conv.update_layout(title="Final Conversion with Pulse Path")

    fig_center = make_subplots(specs=[[{"secondary_y": True}]])
    fig_center.add_trace(go.Scatter(x=center_t, y=center_temp, mode="lines", name="Center Temperature (K)", line={"color": "#c44536"}), secondary_y=False)
    fig_center.add_trace(go.Scatter(x=center_t, y=center_alpha, mode="lines", name="Center Conversion", line={"color": "#0f766e"}), secondary_y=True)
    fig_center.update_layout(title="Center History")
    fig_center.update_xaxes(title="Time (s)")
    fig_center.update_yaxes(title="Temperature (K)", secondary_y=False)
    fig_center.update_yaxes(title="Conversion", range=[0.0, 1.0], secondary_y=True)
    style_figure(fig_center, plot_font, scheme, show_legend)

    fig_global = make_subplots(specs=[[{"secondary_y": True}]])
    fig_global.add_trace(go.Scatter(x=ht, y=hmax, mode="lines", name="Max Temperature (K)", line={"color": "#b91c1c"}), secondary_y=False)
    fig_global.add_trace(go.Scatter(x=ht, y=norm, mode="lines", name="Normalized Total Conversion", line={"color": "#0e7490"}), secondary_y=True)
    fig_global.update_layout(title="Global State History")
    fig_global.update_xaxes(title="Time (s)")
    fig_global.update_yaxes(title="Max Temperature (K)", secondary_y=False)
    fig_global.update_yaxes(title="Normalized Conversion", range=[0.0, 1.0], secondary_y=True)
    style_figure(fig_global, plot_font, scheme, show_legend)

    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(x=ht, y=hlaser, mode="lines", name="Cumulative Laser Energy (J)", line={"color": "#1d4ed8"}))
    fig_energy.add_trace(go.Scatter(x=ht, y=hreact, mode="lines", name="Cumulative Reaction Energy (J)", line={"color": "#ea580c"}))
    fig_energy.update_layout(title="Cumulative Energy History")
    fig_energy.update_xaxes(title="Time (s)")
    fig_energy.update_yaxes(title="Energy (J)")
    style_figure(fig_energy, plot_font, scheme, show_legend)
    return fig_final, fig_peak, fig_conv, fig_center, fig_global, fig_energy, summary


def sim_worker(cfg: SimulationConfig, cfg_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    total = max(1, int(round(cfg.t_end / cfg.dt)))
    try:
        cfg.to_json(cfg_path)

        def on_progress(data: dict[str, float]) -> None:
            step = int(data.get("step", 0.0))
            percent = float(data.get("percent", 0.0))
            max_temp = float(data.get("max_temp", float("nan")))
            mean_a = float(data.get("mean_conversion", float("nan")))
            eta = math.inf if percent <= 0.0 else (time.perf_counter() - start) * (100.0 - percent) / percent
            detail = f"step {step}/{total}, Tmax {max_temp:.2f} K, mean a {mean_a:.4g}"
            if math.isfinite(eta):
                detail += f", ETA {eta:.1f}s"
            RUNTIME.update_progress(percent, detail)

        result = run_simulation(cfg, out_dir, progress_callback=on_progress, cancel_check=RUNTIME.cancel_requested)
        RUNTIME.append_log("Simulation complete.")
        RUNTIME.append_log(f"Peak temperature: {result.summary['t_max_peak_K']:.3f} K")
        RUNTIME.append_log(f"Center conversion: {result.summary['alpha_center_final']:.6f}")
        RUNTIME.finish("Solver complete", f"Results written to {out_dir}", percent=100.0)
    except InterruptedError as exc:
        RUNTIME.append_log(f"Simulation canceled: {exc}")
        RUNTIME.finish("Solver canceled", "operation canceled by user")
    except Exception as exc:
        RUNTIME.append_log(f"Simulation failed: {exc}")
        RUNTIME.finish("Solver failed", str(exc))


def initial_rows() -> list[dict[str, str]]:
    try:
        return cfg_to_rows(load_or_create(Path("default_config.json")))
    except Exception:
        return cfg_to_rows(SimulationConfig())


app = Dash(__name__)
app.title = "Thermal Simulation Dashboard"
app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Roboto:wght@400;500;700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg-0: #05070a;
        --bg-1: #0a0f14;
        --bg-2: #10161d;
        --line: #252d37;
        --line-soft: #1b222b;
        --text-main: #f3f7fc;
        --text-soft: #98a5b5;
        --accent: #e82127;
        --ui-font: "Source Sans 3", "Segoe UI", sans-serif;
        --panel-a: rgba(17, 22, 28, 0.94);
        --panel-b: rgba(12, 16, 21, 0.92);
      }
      html, body {
        margin: 0;
        min-height: 100%;
        background:
          radial-gradient(1200px 620px at 20% -8%, rgba(232, 33, 39, 0.16), transparent 70%),
          radial-gradient(900px 500px at 96% -12%, rgba(65, 84, 108, 0.30), transparent 72%),
          linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 50%, #06090d 100%);
      }
      body {
        color: var(--text-main);
        font-family: var(--ui-font);
      }
      .tesla-shell {
        width: min(1800px, 96vw);
        margin: 16px auto 26px auto;
        display: grid;
        gap: 12px;
      }
      .panel {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(160deg, var(--panel-a), var(--panel-b));
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(255, 255, 255, 0.02);
        padding: 12px;
      }
      .hero-panel {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        align-items: end;
      }
      .hero-kicker {
        color: var(--text-soft);
        letter-spacing: 0.18em;
        font-size: 0.78rem;
        text-transform: uppercase;
      }
      .hero-title {
        margin: 2px 0 0 0;
        font-size: clamp(1.35rem, 2.5vw, 2.15rem);
        letter-spacing: 0.02em;
      }
      .hero-sub {
        margin: 3px 0 0 0;
        color: #c0ccd9;
      }
      .menu-launch {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        gap: 4px;
      }
      .menu-launch .bar {
        display: block;
        width: 17px;
        height: 2px;
        border-radius: 2px;
        background: #dbe4ee;
      }
      .hero-metrics {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
      .metric {
        border: 1px solid var(--line-soft);
        border-radius: 12px;
        background: rgba(10, 14, 19, 0.7);
        padding: 8px 10px;
        min-width: 92px;
      }
      .metric .label {
        color: var(--text-soft);
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .metric .value {
        font-size: 1.05rem;
        font-weight: 700;
      }
      .control-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 8px;
      }
      .tesla-input {
        width: 100%;
        padding: 10px;
        border: 1px solid #394450;
        border-radius: 10px;
        background: #10161e;
        color: #f2f6fa;
        font-family: var(--ui-font);
        font-size: 0.86rem;
        box-sizing: border-box;
      }
      .tesla-input:focus {
        outline: none;
        border-color: #637285;
        box-shadow: 0 0 0 2px rgba(91, 108, 130, 0.28);
      }
      .button-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 10px;
      }
      .tesla-btn {
        border: 1px solid #36414d;
        background: #0f151d;
        color: #e6ecf4;
        padding: 8px 13px;
        border-radius: 999px;
        cursor: pointer;
        font-weight: 600;
        letter-spacing: 0.02em;
      }
      .tesla-btn:hover {
        border-color: #596677;
      }
      .btn-primary {
        background: linear-gradient(180deg, #ff454a 0%, #d81f24 100%);
        border-color: #b1191d;
        color: white;
      }
      .btn-danger {
        background: linear-gradient(180deg, #2a1113 0%, #17090a 100%);
        border-color: #6d2224;
        color: #ffb4b8;
      }
      .tesla-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .action-msg {
        align-self: center;
        color: #bdc8d6;
      }
      .workspace-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
        gap: 10px;
      }
      .panel-title {
        margin: 0 0 8px 0;
        font-size: 1.06rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }
      .status-text {
        font-size: 1.02rem;
        font-weight: 700;
      }
      .status-detail {
        color: #b4c0cf;
      }
      .progress-track {
        height: 12px;
        margin-top: 6px;
        background: #161c24;
        border: 1px solid #2f3945;
        border-radius: 999px;
        overflow: hidden;
      }
      .progress-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #ab1418, #ff4248);
      }
      .progress-text {
        margin-top: 4px;
        font-weight: 700;
      }
      .status-subtitle {
        margin: 12px 0 4px 0;
        font-size: 0.86rem;
        color: #c4cfdb;
        letter-spacing: 0.07em;
        text-transform: uppercase;
      }
      .mono-box {
        margin: 0;
        padding: 9px;
        border: 1px solid #2a3440;
        border-radius: 10px;
        background: #0c1117;
        color: #d2dbe6;
        white-space: pre-wrap;
        overflow-y: auto;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.77rem;
      }
      .summary-box { max-height: 170px; }
      .log-box { max-height: 250px; }
      .graph-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
        gap: 10px;
      }
      .menu-drawer {
        border: 1px solid #27313d;
        border-radius: 12px;
        margin-top: 10px;
        padding: 10px;
        background: rgba(10, 13, 18, 0.92);
      }
      .menu-dropdown {
        width: 100%;
        margin-bottom: 10px;
      }
      .menu-label {
        margin-top: 0;
        margin-bottom: 6px;
        color: #a9b7c9;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      .slider-readout {
        margin-top: 8px;
        color: #d5dee8;
        font-size: 0.88rem;
      }
      .rc-slider-rail {
        background-color: #3a4657 !important;
      }
      .rc-slider-track {
        background-color: #ff4d5a !important;
      }
      .rc-slider-handle {
        border-color: #ff9aa5 !important;
        background-color: #fff4f5 !important;
      }
      .rc-slider-dot {
        border-color: #59657a !important;
      }
      .rc-slider-dot-active {
        border-color: #ff6b77 !important;
      }
      .rc-slider-mark-text {
        color: #c7d2e0 !important;
      }
      .menu-checklist label {
        display: block;
        color: #d4dce8;
        margin-bottom: 4px;
      }
      @media (max-width: 900px) {
        .hero-panel {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""
app.layout = html.Div(
    id="app-shell",
    className="tesla-shell",
    style={
        "--ui-font": FONT_OPTIONS["source"],
        "--bg-0": COLOR_SCHEMES["tesla_dark"]["bg0"],
        "--bg-1": COLOR_SCHEMES["tesla_dark"]["bg1"],
        "--panel-a": COLOR_SCHEMES["tesla_dark"]["panel_a"],
        "--panel-b": COLOR_SCHEMES["tesla_dark"]["panel_b"],
        "--line": COLOR_SCHEMES["tesla_dark"]["line"],
        "--line-soft": COLOR_SCHEMES["tesla_dark"]["line_soft"],
        "--text-main": COLOR_SCHEMES["tesla_dark"]["text_main"],
        "--text-soft": COLOR_SCHEMES["tesla_dark"]["text_soft"],
        "--accent": COLOR_SCHEMES["tesla_dark"]["accent"],
    },
    children=[
        html.Div(
            className="panel hero-panel",
            children=[
                html.Div(
                    children=[
                        html.Div("Thermal Simulation Dashboard", className="hero-kicker"),
                        html.H1("Reactive Process Cockpit", className="hero-title"),
                        html.P("Tesla-inspired control surface for run management and diagnostics.", className="hero-sub"),
                    ]
                ),
                html.Div(
                    className="hero-metrics",
                    children=[
                        html.Button(
                            [html.Span(className="bar"), html.Span(className="bar"), html.Span(className="bar")],
                            id="menu-toggle-btn",
                            className="tesla-btn menu-launch",
                            title="Menu",
                        ),
                        html.Div([html.Div("Mode", className="label"), html.Div("Manual", className="value")], className="metric"),
                        html.Div([html.Div("Drive", className="label"), html.Div("Dual", className="value")], className="metric"),
                        html.Div([html.Div("Stack", className="label"), html.Div("Dash + Plotly", className="value")], className="metric"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="panel",
            children=[
                html.Div(
                    className="control-grid",
                    children=[
                        dcc.Input(id="config-path", type="text", value="default_config.json", className="tesla-input"),
                        dcc.Input(id="output-dir", type="text", value="outputs/latest", className="tesla-input"),
                    ],
                ),
                html.Div(
                    className="button-row",
                    children=[
                        html.Button("Run Simulation", id="btn-run", className="tesla-btn btn-primary"),
                        html.Button("Cancel Run", id="btn-cancel", className="tesla-btn btn-danger"),
                        html.Button("Refresh Plots", id="btn-refresh", className="tesla-btn"),
                        html.Button("Choose Save Path", id="btn-choose-save-path", className="tesla-btn"),
                    ],
                ),
                html.Div(
                    className="button-row",
                    children=[
                        html.Button("Load Config", id="btn-load", className="tesla-btn"),
                        html.Button("Save Config", id="btn-save", className="tesla-btn"),
                        html.Div(id="action-msg", className="action-msg"),
                    ],
                ),
                html.Div(
                    id="menu-drawer",
                    className="menu-drawer",
                    children=[
                        html.Div("Font", className="menu-label"),
                        dcc.Dropdown(
                            id="font-selector",
                            className="menu-dropdown",
                            clearable=False,
                            value="source",
                            options=[
                                {"label": "Source Sans 3", "value": "source"},
                                {"label": "Inter", "value": "inter"},
                                {"label": "IBM Plex Sans", "value": "plex"},
                                {"label": "Roboto", "value": "roboto"},
                            ],
                        ),
                        html.Div("Color Scheme", className="menu-label"),
                        dcc.Dropdown(
                            id="color-scheme-selector",
                            className="menu-dropdown",
                            clearable=False,
                            value="tesla_dark",
                            options=[
                                {"label": "Tesla Dark", "value": "tesla_dark"},
                                {"label": "Carbon Blue", "value": "carbon_blue"},
                                {"label": "Graphite Slate", "value": "graphite_slate"},
                            ],
                        ),
                        html.Div("Display", className="menu-label"),
                        dcc.Slider(
                            id="plot-size-slider",
                            min=MIN_PLOT_CARD_WIDTH_PX,
                            max=MAX_PLOT_CARD_WIDTH_PX,
                            step=20,
                            value=DEFAULT_PLOT_CARD_WIDTH_PX,
                            marks={
                                MIN_PLOT_CARD_WIDTH_PX: str(MIN_PLOT_CARD_WIDTH_PX),
                                480: "480",
                                640: "640",
                                800: "800",
                                960: "960",
                                MAX_PLOT_CARD_WIDTH_PX: str(MAX_PLOT_CARD_WIDTH_PX),
                            },
                            tooltip={"always_visible": False, "placement": "bottom"},
                        ),
                        html.Div(id="plot-size-label", className="slider-readout"),
                        html.Div("Refresh Interval (ms)", className="menu-label"),
                        dcc.Slider(
                            id="refresh-interval-slider",
                            min=300,
                            max=3000,
                            step=100,
                            value=800,
                            marks={300: "300", 800: "800", 1500: "1500", 2200: "2200", 3000: "3000"},
                            tooltip={"always_visible": False, "placement": "bottom"},
                        ),
                        html.Div(id="refresh-interval-label", className="slider-readout"),
                        html.Div("Log Lines", className="menu-label"),
                        dcc.Slider(
                            id="log-lines-slider",
                            min=50,
                            max=400,
                            step=10,
                            value=250,
                            marks={50: "50", 150: "150", 250: "250", 350: "350", 400: "400"},
                            tooltip={"always_visible": False, "placement": "bottom"},
                        ),
                        html.Div(id="log-lines-label", className="slider-readout"),
                        html.Div("Plot Options", className="menu-label"),
                        dcc.Checklist(
                            id="plot-options-checklist",
                            className="menu-checklist",
                            options=[
                                {"label": "Show pulse path overlay", "value": "pulse_path"},
                                {"label": "Show plot legends", "value": "legend"},
                            ],
                            value=["pulse_path", "legend"],
                            inline=False,
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="workspace-grid",
            children=[
                html.Div(
                    className="panel",
                    children=[
                        html.H3("Parameter Matrix", className="panel-title"),
                        dash_table.DataTable(
                            id="cfg-table",
                            columns=[
                                {"name": "Group", "id": "group", "editable": False},
                                {"name": "Parameter", "id": "parameter", "editable": False},
                                {"name": "Value", "id": "value", "editable": True},
                                {"name": "Units", "id": "units", "editable": False},
                            ],
                            data=initial_rows(),
                            editable=True,
                            page_action="none",
                            style_table={"height": "620px", "overflowY": "auto", "border": "1px solid #2b3440", "borderRadius": "10px"},
                            style_header={"backgroundColor": "#141a22", "fontWeight": "bold", "color": "#f1f5f9", "border": "1px solid #2a3440", "fontFamily": "var(--ui-font)"},
                            style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "var(--ui-font)", "fontSize": 14, "backgroundColor": "#0d1218", "color": "#d3dce8", "border": "1px solid #202833"},
                            style_data_conditional=[
                                {"if": {"filter_query": "{group} = 'Domain'"}, "backgroundColor": "#111824"},
                                {"if": {"filter_query": "{group} = 'Thermal'"}, "backgroundColor": "#111b18"},
                                {"if": {"filter_query": "{group} = 'Reaction'"}, "backgroundColor": "#1a1312"},
                                {"if": {"filter_query": "{group} = 'Laser/Scan'"}, "backgroundColor": "#19121a"},
                                {"if": {"filter_query": "{group} = 'Time/Output'"}, "backgroundColor": "#12131d"},
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="panel",
                    children=[
                        html.H3("Drive State", className="panel-title"),
                        html.Div(id="status-text", className="status-text"),
                        html.Div(id="status-detail", className="status-detail"),
                        html.Div(className="progress-track", children=[html.Div(id="progress-fill", className="progress-fill")]),
                        html.Div(id="progress-text", className="progress-text"),
                        html.Div("Summary", className="status-subtitle"),
                        html.Pre(id="summary-box", className="mono-box summary-box"),
                        html.Div("Log", className="status-subtitle"),
                        html.Pre(id="log-box", className="mono-box log-box"),
                    ],
                ),
            ],
        ),
        html.Div(
            id="graph-grid",
            className="graph-grid",
            children=[
                html.Div(className="panel", children=[dcc.Graph(id="fig-final", style={"height": "420px"}, config={"displaylogo": False})]),
                html.Div(className="panel", children=[dcc.Graph(id="fig-peak", style={"height": "420px"}, config={"displaylogo": False})]),
                html.Div(className="panel", children=[dcc.Graph(id="fig-conv", style={"height": "420px"}, config={"displaylogo": False})]),
                html.Div(className="panel", children=[dcc.Graph(id="fig-center", style={"height": "420px"}, config={"displaylogo": False})]),
                html.Div(className="panel", children=[dcc.Graph(id="fig-global", style={"height": "420px"}, config={"displaylogo": False})]),
                html.Div(className="panel", children=[dcc.Graph(id="fig-energy", style={"height": "420px"}, config={"displaylogo": False})]),
            ],
        ),
        dcc.Store(id="menu-open", data=False),
        dcc.Interval(id="poll", interval=800, n_intervals=0),
    ],
)


@app.callback(
    Output("menu-open", "data"),
    Input("menu-toggle-btn", "n_clicks"),
    State("menu-open", "data"),
    prevent_initial_call=True,
)
def toggle_menu(n_clicks: int | None, is_open: bool | None) -> bool:
    if not n_clicks:
        return bool(is_open)
    return not bool(is_open)


@app.callback(
    Output("menu-drawer", "style"),
    Output("plot-size-label", "children"),
    Output("refresh-interval-label", "children"),
    Output("log-lines-label", "children"),
    Output("poll", "interval"),
    Output("app-shell", "style"),
    Output("graph-grid", "style"),
    Output("fig-final", "style"),
    Output("fig-peak", "style"),
    Output("fig-conv", "style"),
    Output("fig-center", "style"),
    Output("fig-global", "style"),
    Output("fig-energy", "style"),
    Input("menu-open", "data"),
    Input("plot-size-slider", "value"),
    Input("refresh-interval-slider", "value"),
    Input("log-lines-slider", "value"),
    Input("font-selector", "value"),
    Input("color-scheme-selector", "value"),
)
def update_display_controls(
    is_open: bool | None,
    width_value: int | None,
    refresh_interval_value: int | None,
    log_lines_value: int | None,
    font_value: str | None,
    scheme_value: str | None,
) -> tuple[
    dict[str, str],
    str,
    str,
    str,
    int,
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    width_px = int(width_value or DEFAULT_PLOT_CARD_WIDTH_PX)
    width_px = max(MIN_PLOT_CARD_WIDTH_PX, min(MAX_PLOT_CARD_WIDTH_PX, width_px))
    height_px = max(320, int(width_px * 0.62))
    refresh_ms = int(refresh_interval_value or 800)
    refresh_ms = max(300, min(3000, refresh_ms))
    log_lines = int(log_lines_value or 250)
    log_lines = max(50, min(400, log_lines))
    font_css = FONT_OPTIONS.get(str(font_value), FONT_OPTIONS["source"])
    scheme = COLOR_SCHEMES.get(str(scheme_value), COLOR_SCHEMES["tesla_dark"])

    drawer_style = {"display": "block"} if bool(is_open) else {"display": "none"}
    label = f"Plot card width: {width_px}px | Plot height: {height_px}px"
    refresh_label = f"Auto-refresh: {refresh_ms} ms"
    log_label = f"Visible log lines: {log_lines}"
    shell_style = {
        "--ui-font": font_css,
        "--bg-0": scheme["bg0"],
        "--bg-1": scheme["bg1"],
        "--panel-a": scheme["panel_a"],
        "--panel-b": scheme["panel_b"],
        "--line": scheme["line"],
        "--line-soft": scheme["line_soft"],
        "--text-main": scheme["text_main"],
        "--text-soft": scheme["text_soft"],
        "--accent": scheme["accent"],
    }
    grid_style = {"gridTemplateColumns": f"repeat(auto-fit, minmax({width_px}px, 1fr))"}
    graph_style = {"height": f"{height_px}px"}
    return (
        drawer_style,
        label,
        refresh_label,
        log_label,
        refresh_ms,
        shell_style,
        grid_style,
        graph_style,
        graph_style,
        graph_style,
        graph_style,
        graph_style,
        graph_style,
    )


@app.callback(
    Output("cfg-table", "data"),
    Output("action-msg", "children"),
    Output("output-dir", "value"),
    Input("btn-load", "n_clicks"),
    Input("btn-save", "n_clicks"),
    Input("btn-run", "n_clicks"),
    Input("btn-cancel", "n_clicks"),
    Input("btn-choose-save-path", "n_clicks"),
    State("config-path", "value"),
    State("output-dir", "value"),
    State("cfg-table", "data"),
    prevent_initial_call=True,
)
def handle_actions(
    _: int | None,
    __: int | None,
    ___: int | None,
    ____: int | None,
    _____: int | None,
    cfg_path_text: str | None,
    out_dir_text: str | None,
    rows: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str, str]:
    trigger = ctx.triggered_id
    table_rows = rows or cfg_to_rows(SimulationConfig())
    cfg_path = resolve_path(cfg_path_text, "default_config.json")
    out_dir = resolve_path(out_dir_text, "outputs/latest")
    out_dir_text_clean = str(out_dir)
    if trigger == "btn-load":
        try:
            cfg = load_or_create(cfg_path)
            msg = f"Loaded config: {cfg_path}"
            RUNTIME.append_log(msg)
            return cfg_to_rows(cfg), msg, out_dir_text_clean
        except Exception as exc:
            msg = f"Load failed: {exc}"
            RUNTIME.append_log(msg)
            return table_rows, msg, out_dir_text_clean
    if trigger == "btn-save":
        try:
            rows_to_cfg(table_rows).to_json(cfg_path)
            msg = f"Saved config: {cfg_path}"
            RUNTIME.append_log(msg)
            return table_rows, msg, out_dir_text_clean
        except Exception as exc:
            msg = f"Save failed: {exc}"
            RUNTIME.append_log(msg)
            return table_rows, msg, out_dir_text_clean
    if trigger == "btn-choose-save-path":
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                title="Choose output save folder",
                initialdir=str(out_dir.parent if out_dir.parent.exists() else Path.cwd()),
                mustexist=False,
            )
            root.destroy()
            if selected:
                selected_path = str(Path(selected))
                msg = f"Selected save path: {selected_path}"
                RUNTIME.append_log(msg)
                return table_rows, msg, selected_path
            return table_rows, "Save path selection canceled.", out_dir_text_clean
        except Exception as exc:
            msg = f"Save path chooser failed: {exc}"
            RUNTIME.append_log(msg)
            return table_rows, msg, out_dir_text_clean
    if trigger == "btn-run":
        if not RUNTIME.start():
            return table_rows, "Simulation already running.", out_dir_text_clean
        try:
            cfg = rows_to_cfg(table_rows)
        except Exception as exc:
            RUNTIME.finish("Solver failed", str(exc))
            return table_rows, f"Config validation failed: {exc}", out_dir_text_clean
        threading.Thread(target=sim_worker, args=(cfg, cfg_path, out_dir), daemon=True).start()
        msg = f"Simulation started -> {out_dir}"
        RUNTIME.append_log(msg)
        return table_rows, msg, out_dir_text_clean
    if trigger == "btn-cancel":
        return table_rows, "Cancel request sent." if RUNTIME.request_cancel() else "No active simulation.", out_dir_text_clean
    return table_rows, "Ready.", out_dir_text_clean


@app.callback(
    Output("status-text", "children"),
    Output("status-detail", "children"),
    Output("progress-fill", "style"),
    Output("progress-text", "children"),
    Output("log-box", "children"),
    Output("btn-run", "disabled"),
    Output("btn-cancel", "disabled"),
    Output("fig-final", "figure"),
    Output("fig-peak", "figure"),
    Output("fig-conv", "figure"),
    Output("fig-center", "figure"),
    Output("fig-global", "figure"),
    Output("fig-energy", "figure"),
    Output("summary-box", "children"),
    Input("poll", "n_intervals"),
    Input("btn-refresh", "n_clicks"),
    Input("font-selector", "value"),
    Input("color-scheme-selector", "value"),
    Input("plot-options-checklist", "value"),
    State("output-dir", "value"),
    State("log-lines-slider", "value"),
)
def refresh(
    _: int,
    __: int | None,
    font_value: str | None,
    scheme_value: str | None,
    plot_options: list[str] | None,
    out_dir_text: str | None,
    log_lines_value: int | None,
) -> tuple[str, str, dict[str, str], str, str, bool, bool, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, str]:
    snap = RUNTIME.snapshot()
    out_dir = resolve_path(out_dir_text, "outputs/latest")
    plot_font = FONT_OPTIONS.get(str(font_value), FONT_OPTIONS["source"])
    scheme = COLOR_SCHEMES.get(str(scheme_value), COLOR_SCHEMES["tesla_dark"])
    opts = set(plot_options or [])
    show_overlay = "pulse_path" in opts
    show_legend = "legend" in opts
    f0, f1, f2, f3, f4, f5, summary = load_figs(out_dir, plot_font, scheme, show_overlay, show_legend)
    log_lines = int(log_lines_value or 250)
    log_lines = max(50, min(400, log_lines))
    visible_logs = snap["logs"][-log_lines:]
    return (
        snap["status"],
        snap["detail"],
        {"width": f"{snap['percent']:.1f}%"},
        f"{snap['percent']:.1f}%",
        "\n".join(visible_logs) if visible_logs else "(no log entries)",
        snap["running"],
        not snap["running"],
        f0,
        f1,
        f2,
        f3,
        f4,
        f5,
        summary,
    )


def main() -> None:
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()
