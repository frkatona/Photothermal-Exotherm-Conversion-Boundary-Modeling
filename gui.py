from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from thermal_sim import SimulationConfig, run_simulation


class ThermalSimGUI(tk.Tk):
    FIELD_TYPES = {
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
    }

    FIELD_LABELS = {
        "width": "Width (m)",
        "height": "Height (m)",
        "nx": "Grid nx",
        "ny": "Grid ny",
        "thickness": "Thickness (m)",
        "rho": "Density rho (kg/m^3)",
        "cp": "Heat capacity cp (J/kg/K)",
        "alpha_th": "Thermal diffusivity alpha_th (m^2/s)",
        "thermal_conductivity": "Thermal conductivity k (W/m/K, blank=auto)",
        "h_loss": "Ambient loss h_loss (W/m^2/K)",
        "emissivity": "Emissivity (0-1)",
        "secondary_emissivity": "Secondary emissivity (reacted)",
        "h_edge": "Edge Robin h_edge (W/m^2/K)",
        "t_amb": "Ambient temperature (K)",
        "A": "Arrhenius A (1/s)",
        "Ea": "Activation energy Ea (J/mol)",
        "gas_constant": "Gas constant R (J/mol/K)",
        "delta_h": "Reaction enthalpy delta_h (J/kg)",
        "pulse_energy": "Pulse energy (J)",
        "absorptivity": "Absorptivity (0-1)",
        "secondary_absorptivity": "Secondary absorptivity (reacted)",
        "use_secondary_absorptivity_for_reacted": "Use secondary absorptivity+emissivity in reacted zones",
        "spot_diameter": "Spot diameter (m)",
        "pulse_width": "Pulse width FWHM (s)",
        "rep_rate": "Rep rate (Hz)",
        "scan_speed": "Scan speed (m/s)",
        "line_pitch": "Line pitch (m)",
        "pass_pattern": "Pass pattern",
        "pass_start_offset": "Pass start offset (line-pitch units)",
        "dt": "Time step dt (s)",
        "t_end": "End time (s)",
        "initial_temperature": "Initial temperature (K)",
        "initial_conversion": "Initial conversion (0-1)",
        "output_interval": "Frame/output interval (s)",
        "save_frames": "Save frame snapshots",
    }

    FIELD_TABS = [
        ("Domain", ["width", "height", "nx", "ny", "thickness"]),
        (
            "Thermal",
            [
                "rho",
                "cp",
                "alpha_th",
                "thermal_conductivity",
                "h_loss",
                "emissivity",
                "secondary_emissivity",
                "h_edge",
                "t_amb",
            ],
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
        ("Time/Output", ["dt", "t_end", "output_interval", "save_frames"]),
    ]
    TAB_COLORS = {
        "Domain": "#f1f6ff",
        "Thermal": "#f2fbf5",
        "Reaction": "#fff6ee",
        "Laser/Scan": "#fff4fa",
        "Time/Output": "#f4f4ff",
        "Postprocess": "#f1fafb",
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("2D Pulsed-Laser Thermal Simulation")
        self.geometry("980x760")

        self.config_path = tk.StringVar(value="default_config.json")
        self.output_dir = tk.StringVar(value="outputs/latest")

        self.animate_fps = tk.StringVar(value="20")
        self.animate_dpi = tk.StringVar(value="140")
        self.animate_mp4 = tk.StringVar(value="")
        self.plot_no_show = tk.BooleanVar(value=False)
        self.status_percent = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="Idle")
        self.status_detail = tk.StringVar(value="")
        self.cancel_event = threading.Event()
        self.operation_lock = threading.Lock()
        self.operation_active = False
        self.operation_name = ""
        self.active_process: subprocess.Popen | None = None

        self.field_vars: dict[str, tk.Variable] = {}
        for key, kind in self.FIELD_TYPES.items():
            if kind == "bool":
                self.field_vars[key] = tk.BooleanVar(value=False)
            else:
                self.field_vars[key] = tk.StringVar(value="")

        self._build_ui()
        self._load_config(initial=True)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Config JSON").grid(row=0, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.config_path, width=76).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(root, text="Browse", command=self._browse_config).grid(row=0, column=2)
        ttk.Button(root, text="Load", command=self._load_config_click).grid(row=0, column=3, padx=(6, 0))
        ttk.Button(root, text="Save", command=self._save_config_click).grid(row=0, column=4, padx=(6, 0))

        ttk.Label(root, text="Output Directory").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(root, textvariable=self.output_dir, width=76).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=6,
            pady=(8, 0),
        )
        ttk.Button(root, text="Browse", command=self._browse_output).grid(row=1, column=2, pady=(8, 0))

        notebook = ttk.Notebook(root)
        notebook.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=(12, 6))

        for tab_name, keys in self.FIELD_TABS:
            tab, body = self._create_colored_tab(notebook, f"{tab_name} Parameters", self.TAB_COLORS[tab_name])
            notebook.add(tab, text=tab_name)
            self._build_field_group(body, keys)

        post_tab, post_body = self._create_colored_tab(
            notebook,
            "Postprocess Parameters",
            self.TAB_COLORS["Postprocess"],
        )
        notebook.add(post_tab, text="Postprocess")
        self._build_post_tab(post_body)

        button_row = ttk.Frame(root)
        button_row.grid(row=3, column=0, columnspan=5, sticky="w", pady=(6, 10))
        ttk.Button(button_row, text="Run Simulation", command=self._run_simulation_click).pack(side="left")
        ttk.Button(button_row, text="Plot 2x2 Figure", command=self._plot_click).pack(side="left", padx=8)
        ttk.Button(button_row, text="Create MP4", command=self._animate_click).pack(side="left")
        self.cancel_button = ttk.Button(
            button_row,
            text="Cancel Active",
            command=self._cancel_active_click,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(8, 0))

        status_row = ttk.Frame(root)
        status_row.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(0, 8))
        ttk.Label(status_row, textvariable=self.status_text, width=28).pack(side="left")
        ttk.Progressbar(
            status_row,
            orient="horizontal",
            mode="determinate",
            length=240,
            maximum=100.0,
            variable=self.status_percent,
        ).pack(side="left", padx=8)
        ttk.Label(status_row, textvariable=self.status_detail).pack(side="left")

        ttk.Label(root, text="Log").grid(row=5, column=0, sticky="w")
        self.log_box = tk.Text(root, height=12, wrap="word")
        self.log_box.grid(row=6, column=0, columnspan=5, sticky="nsew")

        root.columnconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)
        root.rowconfigure(6, weight=1)

    def _create_colored_tab(self, notebook: ttk.Notebook, title: str, color: str) -> tuple[tk.Frame, ttk.Frame]:
        tab = tk.Frame(notebook, background=color, highlightthickness=0, bd=0)
        banner = tk.Frame(tab, background=color, height=24)
        banner.pack(fill="x", side="top")
        banner.pack_propagate(False)
        tk.Label(
            banner,
            text=title,
            background=color,
            foreground="#475569",
            anchor="w",
            padx=10,
        ).pack(fill="x")
        body = ttk.Frame(tab, padding=(10, 10, 10, 10))
        body.pack(fill="both", expand=True)
        return tab, body

    def _build_field_group(self, tab: ttk.Frame, keys: list[str]) -> None:
        for row, key in enumerate(keys):
            label = self.FIELD_LABELS.get(key, key)
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", pady=4)
            if key == "pass_pattern":
                widget = ttk.Combobox(
                    tab,
                    textvariable=self.field_vars[key],
                    values=("offset_start", "cross_hatch"),
                    width=28,
                    state="readonly",
                )
                widget.grid(row=row, column=1, sticky="w", padx=8, pady=4)
            elif self.FIELD_TYPES[key] == "bool":
                widget = ttk.Checkbutton(tab, variable=self.field_vars[key])
                widget.grid(row=row, column=1, sticky="w", padx=8, pady=4)
            else:
                widget = ttk.Entry(tab, textvariable=self.field_vars[key], width=34)
                widget.grid(row=row, column=1, sticky="w", padx=8, pady=4)
        tab.columnconfigure(1, weight=1)

    def _build_post_tab(self, tab: ttk.Frame) -> None:
        ttk.Label(tab, text="Animate FPS").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.animate_fps, width=20).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(tab, text="Animate DPI").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.animate_dpi, width=20).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(tab, text="Animate MP4 Path (optional)").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.animate_mp4, width=50).grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        ttk.Checkbutton(tab, text="Plot without interactive window", variable=self.plot_no_show).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
            pady=6,
        )
        tab.columnconfigure(1, weight=1)

    def _log(self, text: str) -> None:
        def append() -> None:
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")

        self.after(0, append)

    def _set_status(self, text: str, percent: float | None = None, detail: str = "") -> None:
        def apply() -> None:
            self.status_text.set(text)
            if percent is not None:
                self.status_percent.set(max(0.0, min(100.0, percent)))
            self.status_detail.set(detail)

        self.after(0, apply)

    def _begin_operation(self, name: str) -> bool:
        with self.operation_lock:
            if self.operation_active:
                active_name = self.operation_name
                self._log(f"Cannot start {name}; '{active_name}' is already running.")
                return False
            self.operation_active = True
            self.operation_name = name
            self.active_process = None
            self.cancel_event.clear()

        def apply() -> None:
            self.cancel_button.state(["!disabled"])

        self.after(0, apply)
        return True

    def _end_operation(self) -> None:
        with self.operation_lock:
            self.operation_active = False
            self.operation_name = ""
            self.active_process = None
            self.cancel_event.clear()

        def apply() -> None:
            self.cancel_button.state(["disabled"])

        self.after(0, apply)

    def _is_cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def _set_active_process(self, process: subprocess.Popen | None) -> None:
        with self.operation_lock:
            self.active_process = process

    def _cancel_active_click(self) -> None:
        with self.operation_lock:
            active = self.operation_active
            name = self.operation_name
            process = self.active_process
            self.cancel_event.set()

        if not active:
            self._log("No active operation to cancel.")
            return

        self._log(f"Cancellation requested for {name}.")
        self._set_status(f"Cancelling {name}...", None, "waiting for graceful stop")

        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()

    def _browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select config JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.config_path.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(title="Select output directory")
        if selected:
            self.output_dir.set(selected)

    def _set_form_from_config(self, cfg: SimulationConfig) -> None:
        cfg_dict = cfg.to_dict()
        for key, kind in self.FIELD_TYPES.items():
            value = cfg_dict[key]
            if kind == "bool":
                self.field_vars[key].set(bool(value))
            elif kind == "optional_float":
                self.field_vars[key].set("" if value is None else str(value))
            else:
                self.field_vars[key].set(str(value))

    def _build_config_from_form(self) -> SimulationConfig:
        values: dict[str, object] = {}
        for key, kind in self.FIELD_TYPES.items():
            var = self.field_vars[key]
            if kind == "bool":
                values[key] = bool(var.get())
                continue

            text = str(var.get()).strip()
            if kind == "int":
                values[key] = int(text)
            elif kind == "float":
                values[key] = float(text)
            elif kind == "optional_float":
                values[key] = None if text == "" else float(text)
            else:
                values[key] = text

        cfg = SimulationConfig(**values)
        cfg.validate()
        return cfg

    def _load_config(self, initial: bool = False) -> None:
        config_file = Path(self.config_path.get())
        if config_file.exists():
            cfg = SimulationConfig.from_json(config_file)
            self._set_form_from_config(cfg)
            if not initial:
                self._log(f"Loaded config: {config_file}")
            return

        cfg = SimulationConfig()
        cfg.to_json(config_file)
        self._set_form_from_config(cfg)
        self._log(f"Config not found. Wrote defaults: {config_file}")

    def _load_config_click(self) -> None:
        try:
            self._load_config(initial=False)
        except Exception as exc:
            self._log(f"Failed to load config: {exc}")

    def _save_config_click(self) -> None:
        try:
            cfg = self._build_config_from_form()
            config_file = Path(self.config_path.get())
            cfg.to_json(config_file)
            self._log(f"Saved config: {config_file}")
        except Exception as exc:
            self._log(f"Failed to save config: {exc}")

    def _run_simulation_click(self) -> None:
        if not self._begin_operation("simulation"):
            return
        threading.Thread(target=self._run_simulation_worker, daemon=True).start()

    def _run_simulation_worker(self) -> None:
        try:
            cfg = self._build_config_from_form()
            config_file = Path(self.config_path.get())
            cfg.to_json(config_file)
            out = Path(self.output_dir.get())
            self._log(f"Running simulation -> {out}")
            start_time = time.perf_counter()
            n_steps = max(1, int(round(cfg.t_end / cfg.dt)))
            self._set_status("Solver 0.0%", 0.0, f"step 0/{n_steps}")

            def on_progress(data: dict[str, float]) -> None:
                step = int(data.get("step", 0.0))
                total_steps = max(1, int(data.get("total_steps", float(n_steps))))
                percent = float(data.get("percent", 0.0))
                max_temp = float(data.get("max_temp", float("nan")))
                mean_conversion = float(data.get("mean_conversion", float("nan")))
                steps_left = max(0, total_steps - step)

                elapsed = time.perf_counter() - start_time
                eta = math.inf
                if percent > 0.0:
                    eta = elapsed * (100.0 - percent) / percent

                detail = f"step {step}/{total_steps}, remaining {steps_left}, Tmax {max_temp:.2f} K, mean a {mean_conversion:.4g}"
                if math.isfinite(eta):
                    detail += f", ETA {eta:.1f}s"
                self._set_status(f"Solver {percent:.1f}%", percent, detail)

            result = run_simulation(
                cfg,
                out,
                progress_callback=on_progress,
                cancel_check=self._is_cancel_requested,
            )
            self._log("Simulation complete.")
            self._log(f"Peak temperature: {result.summary['t_max_peak_K']:.3f} K")
            self._log(f"Center conversion: {result.summary['alpha_center_final']:.6f}")
            self._log(f"Total laser energy: {result.summary['laser_energy_total_J']:.6e} J")
            self._log(f"Total reaction energy: {result.summary['reaction_energy_total_J']:.6e} J")
            self._log(f"Total radiative loss: {result.summary['radiative_energy_loss_total_J']:.6e} J")
            self._set_status("Solver complete", 100.0, "ready for plotting")
        except InterruptedError as exc:
            self._log(f"Simulation canceled: {exc}")
            self._set_status("Solver canceled", None, "operation canceled by user")
        except Exception as exc:
            self._log(f"Simulation failed: {exc}")
            self._set_status("Solver failed", None, str(exc))
        finally:
            self._end_operation()

    def _plot_click(self) -> None:
        if not self._begin_operation("plotting"):
            return
        threading.Thread(target=self._plot_worker, daemon=True).start()

    def _animate_click(self) -> None:
        if not self._begin_operation("animation"):
            return
        threading.Thread(target=self._animate_worker, daemon=True).start()

    def _plot_worker(self) -> None:
        try:
            from plot_results import create_static_plot

            out = Path(self.output_dir.get())
            no_show = self.plot_no_show.get()
            self._set_status("Plotter 0.0%", 0.0, "starting static figure")

            if not no_show:
                script = Path(__file__).resolve().parent / "plot_results.py"
                cmd = [sys.executable, str(script), "--output", str(out)]
                self._set_status("Plotter 20.0%", 20.0, "running external plot window")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._set_active_process(proc)
                try:
                    while proc.poll() is None:
                        if self._is_cancel_requested():
                            proc.terminate()
                            try:
                                proc.wait(timeout=1.0)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            raise InterruptedError("External plot process canceled.")
                        time.sleep(0.1)

                    stdout, stderr = proc.communicate()
                finally:
                    self._set_active_process(None)

                if stdout and stdout.strip():
                    self._log(stdout.strip())
                if stderr and stderr.strip():
                    self._log(stderr.strip())
                if proc.returncode != 0:
                    raise RuntimeError(f"plot command failed with exit code {proc.returncode}")
                self._set_status("Plotter complete", 100.0, "static figure done")
                return

            def on_progress(percent: float, message: str) -> None:
                self._set_status(f"Plotter {percent:.1f}%", percent, message)

            saved = create_static_plot(
                output_dir=out,
                save_path=None,
                no_show=no_show,
                progress_callback=on_progress,
                cancel_check=self._is_cancel_requested,
            )
            self._log(f"Saved static figure: {saved}")
            self._set_status("Plotter complete", 100.0, "static figure done")
        except InterruptedError as exc:
            self._log(f"Plotter canceled: {exc}")
            self._set_status("Plotter canceled", None, "operation canceled by user")
        except Exception as exc:
            self._log(f"Plotter failed: {exc}")
            self._set_status("Plotter failed", None, str(exc))
        finally:
            self._end_operation()

    def _animate_worker(self) -> None:
        try:
            from animate_results import create_animation

            out = Path(self.output_dir.get())
            fps = int(self.animate_fps.get().strip())
            dpi = int(self.animate_dpi.get().strip())
            mp4_text = self.animate_mp4.get().strip()
            mp4_target = Path(mp4_text) if mp4_text else None

            self._set_status("Plotter 0.0%", 0.0, "starting animation")

            def on_progress(percent: float, message: str) -> None:
                self._set_status(f"Plotter {percent:.1f}%", percent, message)

            saved = create_animation(
                output_dir=out,
                mp4_path=mp4_target,
                fps=fps,
                dpi=dpi,
                progress_callback=on_progress,
                cancel_check=self._is_cancel_requested,
            )
            self._log(f"Saved animation: {saved}")
            self._set_status("Plotter complete", 100.0, "animation done")
        except InterruptedError as exc:
            self._log(f"Animation canceled: {exc}")
            self._set_status("Plotter canceled", None, "operation canceled by user")
        except Exception as exc:
            self._log(f"Animation failed: {exc}")
            self._set_status("Plotter failed", None, str(exc))
        finally:
            self._end_operation()


def main() -> None:
    app = ThermalSimGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
