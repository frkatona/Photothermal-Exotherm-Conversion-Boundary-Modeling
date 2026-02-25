# Conversion boundary model for a photothermal exotherm with dynamic emissivity

## Summary

### Current Browser Dashboard

![GUI](screenshots/GUI-example.png)

### Static summary plot for high thermal diffusivity, high exotherm case

![static summary 1](outputs/latest/static_summary_1.png)

### Gif animation

![sim 1](outputs/latest/sim_1.gif)

2D pulsed-laser raster simulation over a reactive thin layer with:
- explicit finite-difference solver for temperature and conversion,
- Arrhenius reaction + exotherm coupling,
- Gaussian pulse deposition in space/time,
- raster path models (`offset_start`, `cross_hatch`),
- outputs, static plotting, ffmpeg MP4 animation, and GUI launcher.

Animation panels include:
- temperature field,
- conversion field (no scan path overlay),
- laser pulse power-density field (`W/m^2`),
- max-temperature + total-conversion line plot,
- cumulative laser-energy vs cumulative reaction-energy line plot.

## Install

```bash
python -m pip install numpy matplotlib dash plotly numba
```

`ffmpeg` is required for MP4 animation.

## Run Simulation

```bash
python run_simulation.py --config default_config.json --output outputs/latest
```

## Static Plot (2x2)

```bash
python plot_results.py --output outputs/latest
```

## MP4 Animation (streamed to ffmpeg)

```bash
python animate_results.py --output outputs/latest --fps 20
```

## Stability Preflight

Use this before long runs to estimate explicit-solver stability limits and expected substepping:

```bash
python check_stability.py --config default_config.json
```

## GUI

```bash
python gui.py
```

Then open `http://127.0.0.1:8050` in your browser.
- Use the top-right menu button to adjust plot card dimensions.

Performance note:
- Set `compute_backend` to `auto`, `numpy`, or `numba` in config.
- `auto` will use JIT (`numba`) when available and fall back to NumPy otherwise.

## NaN Troubleshooting

- The solver is explicit in time. Smaller `width/height` at fixed `nx/ny` reduces `dx/dy`, which tightens the stable `dt` limit.
- If requested `dt` is too large, the solver now automatically uses internal substeps and reports:
  - `dt_internal_s`
  - `substeps_per_step`
  - `dt_cfl_limit_s`
  - `dt_recommended_s`
- Check `summary.txt` or run `check_stability.py` to anticipate runtime and stability before long jobs.

## Simulation methods, math, pitfalls, alternatives, and roadmap

### Governing model (as implemented)

State variables:
- `T(x, y, t)` = temperature (K)
- `a(x, y, t)` = conversion fraction (`0 <= a <= 1`)

Thin-layer energy balance (2D in-plane, thickness-lumped losses/sources):

```text
dT/dt = alpha_th * Laplacian(T)
      - h_loss/(rho*cp*thickness) * (T - T_amb)
      - eps(a)*sigma/(rho*cp*thickness) * (T^4 - T_amb^4)
      + q_laser/(rho*cp*thickness)
      + (delta_h/cp) * da/dt
```

Reaction model:

```text
da/dt = A * exp(-Ea / (R*T)) * (1 - a)
```

Implementation notes:
- `a` is clipped each step to `[0, 1]`.
- The Arrhenius update is explicit and evaluated from `T_trial` (post diffusion/cooling/laser, pre reaction-heat release).
- Edge boundaries are Robin (convective) using ghost cells:
  - `-k * dT/dn = h_edge * (T - T_amb)`
  - `k = alpha_th * rho * cp` unless `thermal_conductivity` is explicitly provided.
- Optional reacted-property coupling uses linear interpolation by conversion:
  - `absorptivity(a) = absorptivity + (secondary_absorptivity - absorptivity)*a`
  - `emissivity(a) = emissivity + (secondary_emissivity - emissivity)*a`

Laser pulse model:
- Pulse times: `t_n = n / rep_rate`.
- Pulse center follows raster path (`offset_start` or `cross_hatch`).
- Spatial profile for one pulse:

```text
q(r) = (2*E_pulse/(pi*w0^2)) * absorptivity(a) * exp(-2*r^2/w0^2),  w0 = spot_diameter/2
```

- Temporal profile is Gaussian with `sigma_t = pulse_width/(2*sqrt(2*ln(2)))`.
- Each substep deposits a pulse-energy fraction from nearby pulses only (for speed).

Numerics:
- Uniform Cartesian grid (`nx x ny`), central-difference Laplacian.
- Forward-Euler explicit time integration with automatic internal substepping.
- Internal stability check combines:
  - diffusion CFL limit, and
  - a linearized cooling/radiation limit.
- Reported in `summary.txt`: `dt_cfl_limit_s`, `dt_cooling_limit_s`, `dt_stable_limit_s`, `dt_recommended_s`, `dt_internal_s`, `substeps_per_step`.

### Most common pitfalls when building this from scratch

1. Unit inconsistency across area-based and mass-based terms.
2. Hidden explicit instability when refining grid (`dx`, `dy`) without shrinking `dt`.
3. Stiff thermal-runaway behavior from Arrhenius plus exotherm at high `T`.
4. Confusing `alpha_th` vs `thermal_conductivity` consistency (`k = alpha*rho*cp` unless intentionally overridden).
5. Parameter non-identifiability: `A`, `Ea`, `delta_h`, absorptivity, emissivity, and loss coefficients can compensate each other.
6. 2D thin-layer assumption can miss through-thickness gradients, volumetric absorption, and gas/mass-loss coupling.
7. Pulse/discretization aliasing: `rep_rate`, `scan_speed`, `dt`, and grid spacing can produce misleading overlap behavior if not jointly checked.

### Alternatives that were feasible, and why current choices were used

| Design area | Feasible alternatives | Why this code currently uses the existing approach |
| --- | --- | --- |
| Time integration | Implicit Euler, Crank-Nicolson, IMEX | Explicit + auto-substep is simpler to reason about, easier to debug, and fast enough for current grid sizes. |
| Spatial discretization | Finite volume, finite element, adaptive mesh | Uniform FD keeps implementation compact and transparent, with predictable memory and runtime. |
| Chemistry | Multi-step kinetics, autocatalytic models, transport-limited kinetics | Single-step Arrhenius is a tractable first model with few parameters and straightforward calibration. |
| Laser absorption | 3D Beer-Lambert, ray tracing, Monte Carlo optics | Surface/thin-layer Gaussian deposition matches the current abstraction and avoids heavy optical solvers. |
| Boundary model | Pure Neumann, pure Dirichlet, full conjugate heat transfer | Robin edges plus top-surface loss gives a practical compromise between realism and complexity. |
| Property changes with conversion | Constant properties, nonlinear lookup tables, phase-field coupling | Linear conversion blending is stable, cheap, and captures first-order "reacted vs unreacted" contrast. |

### Where this sits in thermal simulation history (short version)

- Early era: analytic heat-equation solutions for simple geometries gave foundational intuition.
- Mid 20th century: finite-difference methods made transient conduction practical on computers.
- Later: finite-element/finite-volume methods enabled complex geometries and multiphysics coupling.
- HPC era: parallel solvers supported 3D nonlinear, radiative, and reactive process models.
- Current era: digital-twin workflows add calibration, uncertainty quantification, and surrogate/ML acceleration.

### Directions for future work

More accurate:
- Add temperature-dependent `rho`, `cp`, `k`, absorptivity, and emissivity from measured data.
- Replace one-step kinetics with multi-step pathways (dehydrogenation, pyrolysis, clustering) and optional transport limits.
- Move from 2D thin-layer to multilayer or full 3D with through-thickness absorption and conduction.
- Add gas evolution/mass loss and latent/phase-change terms where applicable.
- Upgrade radiation from gray-body local loss to view-factor or participating-media treatment.

More insightful:
- Add built-in sensitivity analysis and parameter ranking.
- Add energy-balance diagnostics and residual checks at every output interval.
- Add automated "regime maps" (pulse overlap, runaway threshold, conversion completeness).
- Add calibration utilities against experimental thermography and conversion data.

Faster:
- Introduce IMEX/implicit options to reduce required substeps for stiff cases.
- JIT or GPU backends (Numba/CuPy/JAX) for large sweeps.
- Adaptive time stepping and optional adaptive mesh refinement near hot spots.
- Event-driven pulse handling to avoid scanning pulses that cannot contribute at a given time.

Easier to use:
- Preset material/process libraries with unit-checked templates.
- Rich config validation (schema + warnings for likely unstable/nonphysical settings).
- Expanded GUI guidance: tooltips, constraints, default profiles, and run comparison views.
- Single-command pipeline for run + plots + animation + report.

More modern:
- Package structure with tests, CI, and reproducible benchmark cases.
- Standardized output datasets (`xarray`/`zarr`) for large studies.
- Browser-based dashboard for interactive exploration and batch management.
- Optional surrogate model workflow for rapid inverse design and optimization.

---

it's interesting note that "large initial conversion exotherm" and "drastic emissivity change in reacted material + pulse overlap" are similar "delayed impact" justifications that generate the necessary temperature rise to trigger the full reaction after the initial pulse(s) have deposited energy but before the reaction has fully run to completion
it's interesting note that "large initial conversion exotherm" and "drastic emissivity change in reacted material + pulse overlap" are similar "delayed impact" justifications that generate the necessary temperature rise to trigger the full reaction after the initial pulse(s) have deposited energy but before the reaction has fully run to completion

## To-do

- implement drastic increase in absorptivity for reacted regions

  - evaluate consequence of pulse overlap

  - evaluate consequence of scan path (e.g. `cross_hatch` vs `offset_start`)

- include unit change in the GUI

- status bar in GUI with steps remaining, max temp, etc.

- normalize conversion

- isothermal contours

- better GUI (browser?)

- cancel run from GUI

---

## Parameter considerations

### Asma thermodynamic estimates
*abridged; see email for slideshow and details*
Ea = 167 kJ/mol 
 - (40 kcal/mol)
 - for low-barrier dimerization-initiation of B10H14

dH = 3340 kJ/mol
 - (800 kcal/mol)
 - 3340 kJ/mol * 5.2 mol/kg = 1.7e7 J/kg
 - B10H8 forming large B clusters

---

*some insights from (2018) Li et al:* [here](<literature/(2018) Li et al - Synthesis and ceramic conversion of a new organodecaborane preceramic polymer with high ceramic yield.pdf>)

---
 
### process simplification
full thermo consideration seemingly would many processes:
1) dehydrogenation of B10H14 to B10H8
2) backbone scission
3) graphitization
4) dimerization of B10H8 to B20H16
5) cluster growth to large (100+) B clusters (**primary exotherm**)
6) B-C formation
7) pyrolysis (**mass loss -> heat loss**)
8) densification?

insofar as the backbone scission/pyrolysis can be simplified to **polyethylene**, [(2023) Mastalski et al](<literature/(2023) Mastalski et al - Intrinsic millisecond kinetics of polyethylene pyrolysis via pulse-heated analysis of solid reactions.pdf>) reports PE pyrolysis activation energy (Ea) and pre-exponential factor (A) as:

$Ea = 180 +/- 2.5 \space kJ \cdot mol^{-1}$  

 - will deviate depending on whether the transition state is stabilized or destabilized by the boron pendants/clusters

$A = 10^{11} \space s^{-1}$

- unimolecular bond-scission seems to appear around 2 orders of magnitude of here
- will go down if observed rate is transport-limited

(just starting to build rough estimates, I don't necessarily have a good reason to believe the kinetics are rate-limited here)

---

For the H2 evolution in the dehydrogenation step, (2015) Barm et al. reported [here](<literature/(2015) Barm et al - A new homogeneous catalyst fo the dehydrogenation of dimethylamine borane starting with ruthenium(III) acetylacetonate.pdf>) on an amine-borane dehydrogenation catalyst with Ea = 85 kJ/mol.  It's not clear if there is any catalytic effect from the boron clusters, but this is perhaps useful as a lower bound for dehydrogenation.

- solid mass transfer constraint probably prevents H2 diffusion and depresses effective pre-factor 
---

Other than laser absorption, reaction enthalpy ($\Delta H$) stands to play a substantial role in the transfer of thermal energy, but so does phase change and mass loss (pyrolysis). 

B10H14 heat capacity

---

### misc. scratch calculations
density of the poly-decaborane monomer
B10H14 MW = 122 g/mol
C5H10 = 70 g/mol
total density = 192 g/mol
1000 g/kg / 192 g/mol = **5.2 mol/kg**


---

## notes (2026-02-21)

- lowered thickness to 6 um

- changed Ea and dH to reflect estimates

- estimated density, heat capacity, and thermal diffusivity from polymer materials (should ask Rob/Ben if there are better estimates from their papers)

- shot in the dark at Arrhenius = 1e3

- picked ambient loss of 10 W/m2K (partway between bounds for free convection), but probably bigger radiative losses at the expected temperatures

  - do radiative losses overwhelm convective losses before 1000K? it's probably worth modeling both at least between 500 K and 1000 K
  - what considerations are there for radiative losses from a bulk materials (scattering/reabsorption before it escapes?)

---

it's interesting note that "large initial conversion exotherm" and "drastic emissivity change in reacted material + pulse overlap" are similar "delayed impact" justifications that generate the necessary temperature rise to trigger the full reaction after the initial pulse(s) have deposited energy but before the reaction has fully run to completion
