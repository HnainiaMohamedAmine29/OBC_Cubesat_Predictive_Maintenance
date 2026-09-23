# Technical Documentation — CubeSat Battery Model (Nominal Conditions)

**Document scope:** `02_battery_model_Normal_conditions`
**Files analyzed:** `cubesat_params.m`, `cubesat_ode_dch.m`, `cubesat_ode_ch.m`, `cubesat_events.m`, `simulate_cycle.m`, `cubesat_run.m`, `MATLAB_CONFIG.m`

---

## 1. Context and purpose of the model

The code models the electro-thermal behavior and aging of a **Li-ion battery pack** onboard a CubeSat in low Earth orbit (LEO), on a sun-synchronous orbit (`p.mission_label = 'LEO Earth-observation satellite, sun-synchronous orbit'`). The goal is to generate, through simulation, a **synthetic dataset** (a `.csv` file) describing the pack's evolution over thousands of successive orbital cycles, until end of life (EOL) is reached.

Each orbital cycle reproduces the physical sequence experienced by the satellite:

1. **Eclipse phase**: the satellite is in the Earth's shadow, the solar panels produce nothing, the battery **discharges** to power the payload and the platform.
2. **Sunlight phase**: the solar panels recharge the battery and power the bus.

This model is described as "normal conditions": no fault is injected (`p.gamma = 1.0`, an aging multiplier coefficient frozen at its nominal value). It likely serves as a **reference (baseline) case** before a failure scenario is introduced in a separate module (the comment `overwritten by inject_fault.m` in `cubesat_params.m` indicates the existence of such a module in a variant of the project).

The model rests on three coupled physical pillars:

- an **electrical model** (open-circuit voltage + internal resistance),
- a **thermal model** (thermal balance in a space vacuum environment),
- an **aging model** (capacity fade, State of Health).

---

## 2. Software architecture

```
cubesat_params.m   → defines all physical parameters (structure p)
cubesat_ode_dch.m  → differential equations during discharge (eclipse)
cubesat_ode_ch.m   → differential equations during charge (sunlight)
cubesat_events.m   → generic event functions (low/high SOC stop)
simulate_cycle.m   → simulates ONE full cycle (discharge then charge), extracts metrics
cubesat_run.m      → main script: loop over N cycles, EOL handling, CSV export
MATLAB_CONFIG.m    → "mirror" configuration file listing the same parameters
```

The execution flow is as follows:

```
cubesat_run.m
   │
   ├─► cubesat_params()                     (loading physical constants)
   │
   └─► for loop, cycle = 1:N_max
          │
          └─► simulate_cycle(x, p_cycle, cycle)
                 │
                 ├─► ode45(cubesat_ode_dch, ...)   [discharge phase, with cubesat_events]
                 │        → integrates SOC, T, SOH over the eclipse duration
                 │
                 ├─► ode45(cubesat_ode_ch,  ...)   [charge phase, with cubesat_events]
                 │        → integrates SOC, T, SOH over the sunlight duration
                 │
                 └─► extraction of the cycle's characteristics → row
          │
          └─► x ← row.x_next   (final state passed to the next cycle)
```

The state vector passed from one cycle to the next ensures the **physical continuity** of the simulation: the battery does not "start from scratch" at each orbit — it inherits its SOC, its temperature, and above all its SOH (cumulative aging).

`MATLAB_CONFIG.m` is not, strictly speaking, an executable MATLAB script (it starts with a `#`, non-standard syntax): it is a **reference configuration / parameter documentation file**, redundant with `cubesat_params.m`, probably intended to be consulted or to serve as a template for a future configurable configuration system (JSON/YAML) without touching the MATLAB code.

---

## 3. State vector and modeling assumptions

### 3.1 State vector

The dynamic system is described by a 3-component state vector:

$$
x(t) = \begin{bmatrix} SOC(t) \\ T(t) \\ SOH(t) \end{bmatrix}
$$

| Variable | Meaning | Unit | Physical domain |
|---|---|---|---|
| `SOC` | State of Charge | dimensionless [0–1] | bounded to `[SOC_min, SOC_max] = [0.50, 0.99]` |
| `T` | Pack temperature | K | bounded to `[T_operating_min, T_operating_max] = [263.15, 323.15]` K, i.e. [−10 °C, +50 °C] |
| `SOH` | State of Health (residual capacity) | dimensionless [0–1] | bounded to `[SOH_min, 1.05] = [0.70, 1.05]` |

These three variables are **coupled**: internal resistance depends on `T` and `SOH`, thermal dynamics depend on current (and thus indirectly on SOC via the bounds), and aging depends on both `T` and SOC (via the stress factor).

### 3.2 Structuring assumptions

- **0D ("lumped") model**: the battery is treated as a single material point in temperature (no internal spatial gradient). Justified for a small CubeSat pack.
- **No convection**: the code comment is explicit — `THERMAL MODEL (VACUUM: conduction + radiation, NO convection)`. This is a correct assumption in a space vacuum environment, where convective exchange is physically zero.
- **SOC bounded at 50%**: unlike a "classic" battery model running from 0 to 100%, here `SOC_min = 0.50`. The comment specifies: *"absolute safety floor (normal DoD ~27%, floor well below)"*. In other words, the typical operational depth of discharge (DoD) is about 27%, and the 50% safety floor is normally never reached in nominal operation — it acts as a numerical/safety safeguard rather than a real operational limit.
- **SOH as a capacity scaling factor**: `SOH` directly multiplies `Q_nom` to give the actual available capacity (`QD = SOH * Q_nom`), and also enters the internal resistance calculation (electrical aging) — a classic coupled capacity/impedance modeling approach in battery management (BMS).
- **No current variable as a state**: the charge/discharge currents (`I_ch`, `I_dch`) are **imposed parameters (constant-current control)**, not model outputs. The model is therefore a "current-driven" model, representative of a regulated bus (constant-current charge/discharge), not a "constant power" model or one driven by the payload load.

---

## 4. Electrical model

### 4.1 Open Circuit Voltage (OCV)

The pack's open-circuit voltage is modeled by a cubic polynomial as a function of SOC:

$$
OCV(SOC) = V_{min} + (V_{max}-V_{min}) \cdot \left(0.02 + 0.98\,SOC - 0.50\,SOC^2 + 0.50\,SOC^3\right)
$$

with `V_pack_min = 6.20 V`, `V_pack_max = 8.40 V`.

This is a **classic polynomial approximation of the OCV-SOC curve** of a Li-ion cell/pack, intended to qualitatively reproduce:
- a steep slope zone at the start of charge (0–20%),
- a flatter plateau in the intermediate zone,
- an upswing at the end of charge.

This polynomial is not calibrated against specific test data in the provided code (no associated characterization file) — it is a **generic analytical parameterization**, to be validated/recalibrated against real cell data before use in mission engineering.

### 4.2 Internal resistance (Arrhenius law + aging-related growth)

$$
R_{int}(T, SOH) = \frac{R_0}{SOH}\Big(1 + k_R\,(1-SOH)\Big)\, \exp\!\left[\frac{E_a}{R_{gaz}}\left(\frac{1}{T}-\frac{1}{T_{ref}}\right)\right]
$$

with:
- `R0 = 0.015 Ω`: the pack's series resistance at beginning of life (BOL) at 25 °C,
- `k_R = 1.899`: resistance growth coefficient with capacity loss,
- `Ea = 32,000 J/mol`: activation energy (Arrhenius law),
- `R_gaz = 8.314 J/(mol·K)`: ideal gas constant,
- `T_ref = 298.15 K` (25 °C): reference temperature.

This expression combines **two well-identified physical effects in battery electrochemistry**:

1. **Thermally activated term (Arrhenius)**: $\exp[(E_a/R_{gaz})(1/T - 1/T_{ref})]$. When `T < T_ref`, the exponent is positive → resistance **increases** at low temperatures (reduced ionic mobility in the electrolyte). When `T > T_ref`, it decreases. This is the expected behavior.
2. **Aging-related growth term**: $(1/SOH)\,(1 + k_R(1-SOH))$. A double effect: division by `SOH` already increases resistance as `SOH` decreases, and the factor $(1+k_R(1-SOH))$ further amplifies this growth non-linearly — consistent with experimental observations of accelerated impedance growth at end of life (impedance "knee point").

**Numerical sanity check:** at `SOH=1`, `T=T_ref`, we do indeed recover `R_int = R0 = 15 mΩ`. The comment `p.R0 = 0.015; % Ω BOL pack resistance (55.20 mΩ)` contains an apparent inconsistency between the numerical value (15 mΩ) and the comment (55.20 mΩ) — **a point to check/correct in the source code**, since the value actually used by the solver is indeed 15 mΩ.

### 4.3 Terminal voltage

- On discharge: $V_{term} = OCV(SOC) - R_{int}\cdot I_{dch}$ (ohmic drop that **reduces** the available voltage).
- On charge: $V_{term} = OCV(SOC) + R_{int}\cdot I_{ch}$ (overvoltage needed to **force** the charge current through the internal resistance).

In both cases, the voltage is then clamped to the interval `[V_pack_min, V_pack_max]` — consistent with the pack's physical/safety limits (BMS protection).

---

## 5. SOC dynamics

### 5.1 Discharge (eclipse)

$$
\frac{dSOC}{dt} = -\frac{\eta \cdot I_{dch}}{Q_{nom}\cdot SOH \cdot 3600}
$$

### 5.2 Charge (sunlight)

$$
\frac{dSOC}{dt} = +\frac{\eta \cdot I_{ch}}{Q_{nom}\cdot SOH \cdot 3600}
$$

This is the classic integration of **Coulomb counting**: the SOC variation per second is the current divided by the actual available capacity (`Q_nom · SOH`, in Ah converted to As via the factor 3600), weighted by the coulombic efficiency `η = 0.98` (2% faradaic losses, typical of a healthy Li-ion cell).

Notable point: efficiency `η` is applied only to the **current**, in both directions (charge and discharge). Strictly speaking, coulombic efficiency generally applies primarily to charging (part of the injected charge current does not translate into useful charge accumulation, due to side reactions). Here, the model applies it symmetrically to both phases, which is a simplification worth noting but remains conservative (slightly pessimistic on discharge).

### 5.3 Operating currents and sizing logic

$$
I_{ch} = I_{dch}\cdot\frac{t_{eclipse}}{t_{sunlight}} = 1.91 \times \frac{35}{55} = 1.2155\ \text{A}
$$

This relationship is an interesting design choice: it aims to keep the **actual charging time close to the available sunlight duration (~55 min)**, regardless of SOH. Indeed, since `SOH` appears as a common factor in the numerator (via `Q_nom·SOH`) in the calculation of the time needed to charge (ΔSOC / (dSOC/dt)), and since `I_ch` is proportional to `I_dch`, fixing the ratio `I_ch/I_dch` to the ratio of the orbital durations means that **the charging duration remains stable over time, independent of aging** (SOH does indeed cancel out in the ratio, as the code comment indicates). This reproduces operational reality: a CubeSat's charging strategy is generally governed by the available time window (sunlight), not directly by a fixed current setpoint independent of context.

---

## 6. Thermal model

The thermal balance is a **first-order energy balance** (generalized Newton/Fourier law), where the temperature variation results from the sum of incoming and outgoing heat flows, divided by the system's thermal capacity:

$$
m \cdot C_p \cdot \frac{dT}{dt} = \sum \dot{Q}
$$

### 6.1 Balance terms

| Term | Expression | Physical meaning |
|---|---|---|
| Joule heating | $\dot Q_{joule} = R_{int}\cdot I^2$ | internal resistive dissipation, always positive, depends on the phase (I_dch or I_ch) |
| Radiation | $\dot Q_{rad} = \sigma\,\varepsilon\,A_{rad}\,(T^4 - T_{space}^4)$ | radiative loss toward space (Stefan-Boltzmann law), with `T_space = 3 K` (cosmic microwave background) |
| Structural conduction (eclipse) | $\dot Q_{cond} = G_{structure}\,(T - T_{amb,eclipse})$ | flow **outward** toward a cold structure (`T_amb_eclipse` ≈ −10 °C or −2 °C depending on cycle parity) |
| Structural conduction (sunlight) | $\dot Q_{cond} = G_{structure}\,(T_{amb,sun} - T)$ | flow **inward** from a structure heated by the Sun (`T_amb_sun` = +35 °C) |
| Active heating | $\dot Q_{heater} = P_{heater}\cdot \mathbb{1}[T<T_{heater,on}]$ | heat input from the BMS/platform resistive heater |

### 6.2 Full equations

**Discharge (eclipse):**
$$
\frac{dT}{dt} = \frac{\dot Q_{joule} + \dot Q_{heater} - \dot Q_{rad} - \dot Q_{cond}}{m\,C_p}
$$

**Charge (sunlight):**
$$
\frac{dT}{dt} = \frac{\dot Q_{joule} + \dot Q_{cond} + \dot Q_{heater} - \dot Q_{rad}}{m\,C_p}
$$

Note the sign inversion of the conduction term between the two phases (consistent with the direction of the physical flow described above), and the fact that the radiative term is **always subtracted** (the battery always radiates toward the absolute cold of space).

### 6.3 Thermal regulation (heating)

$$
P_{heater} =
\begin{cases}
2.0\ \text{W} & \text{if } T < T_{heater,on} = -10\,°C \\
0\ \text{W} & \text{otherwise}
\end{cases}
$$

This is an **on/off logic with a single threshold** (the code only models a trigger, via `T_heater_on`, without explicitly using `T_heater_off` in the differential equations). Strictly speaking, a correctly modeled thermostat would require **two-threshold hysteresis** (turn-on at −10 °C, turn-off at +20 °C) to avoid high-frequency cycling of the heater around the low threshold. The parameter `T_heater_off = 293.15 K` (+20 °C) is indeed defined in `cubesat_params.m`, but **is used nowhere in `cubesat_ode_ch.m` or `cubesat_ode_dch.m`** — a potential inconsistency to fix (dead parameter, or missing hysteresis logic in the current implementation).

### 6.4 Pack thermal parameters

| Parameter | Value | Meaning |
|---|---|---|
| `m` | 0.350 kg | pack mass + mounting bracket |
| `Cp` | 1050 J/(kg·K) | specific heat capacity |
| `σ` | 5.67 × 10⁻⁸ W/(m²·K⁴) | Stefan-Boltzmann constant |
| `ε` | 0.85 | surface emissivity |
| `A_rad` | 0.004 m² | effective radiative surface |
| `G_structure` | 0.50 W/K | thermal conductance to the structure |

---

## 7. Aging model (State of Health)

$$
\frac{dSOH}{dt} = -\alpha \cdot I \cdot \exp\!\left(-\frac{E_a}{R_{gaz}\,T}\right)\cdot \text{stress}(SOC)\cdot \gamma
$$

with `α = 0.02085` (base rate) and `γ = 1.0` (fault severity multiplier, neutralized under normal conditions).

### 7.1 SOC-dependent stress factor ("U" shape)

$$
\text{stress}(SOC) = 0.10 + 0.50\,e^{-14\,SOC} + 0.40\,e^{-14(1-SOC)}
$$

This function has a **minimum around intermediate SOC** and grows strongly at both extremes (very low or very high SOC). This is a well-established qualitative representation in Li-ion aging: storage/cycling at high SOC (close to 100%) accelerates electrolyte degradation and SEI-interface degradation, while very low SOC favors other degrading mechanisms (dissolution of the negative current collector's copper, local over-discharge of cells). The "bathtub-shaped" U form is consistent with the Li-ion calendar/cycling aging literature.

Numerically, at the model's operational bounds (`SOC ∈ [0.50, 0.99]`), the term `e^{-14·SOC}` is negligible (high SOC), while `e^{-14(1-SOC)}` remains moderate — the stress is therefore dominated by the constant term and the second exponential term near `SOC_max`.

### 7.2 Thermal dependence of aging

The term $\exp(-E_a/(R_{gaz}T))$ expresses the **classic acceleration of aging with temperature** (Arrhenius law applied this time to degradation kinetics rather than resistance). The higher `T` is, the larger the exponential (since the negative exponent tends toward 0), and so the faster the SOH loss per unit of time — behavior consistent with the well-known empirical rule in battery engineering ("+10 °C ≈ lifetime cut in half").

### 7.3 Current dependence

The term `α · I` makes aging **proportional to the applied current** (I_dch on discharge, I_ch on charge): the higher the current, the greater the mechanical/electrochemical stress (electrode expansion/contraction, concentration gradient), which is consistent with cycling aging, distinct from calendar aging (not modeled here since SOH only decreases during active phases, not at rest).

### 7.4 The `γ` (gamma) factor

This multiplicative factor, frozen at 1.0 in this "normal conditions" module, is clearly the entry point intended for an external `inject_fault.m` module (mentioned in a comment), allowing accelerated aging to be simulated (`γ > 1`) in the event of a fault (for example, an unbalanced cell, a local thermal fault, etc.). It **confirms that this set of files constitutes the healthy reference case**, to be contrasted with a degraded scenario simulated elsewhere in the project.

---

## 8. Numerical integration

### 8.1 Solver

Each phase (discharge, charge) is integrated with `ode45` (an explicit Runge-Kutta 4(5) method, Dormand-Prince), suitable for non-stiff systems such as this one.

Options used (`odeset`):

```matlab
RelTol   = 1e-5
AbsTol   = 1e-8
MaxStep  = 10          % s — imposed maximum step
NonNegative = [1,2,3]  % SOC, T, SOH forced positive
Events   = @(t,x) evt_dch/evt_ch(...)
```

- The maximum step of **10 seconds** guarantees fine time resolution over phases lasting several tens of minutes (typically 2,100 s of eclipse), essential for properly capturing thermal dynamics and the approach of SOC thresholds.
- The `NonNegative` constraint avoids numerical artifacts (negative SOC or SOH) that the adaptive solver could otherwise produce in the event of a transient overshoot.
- The strict physical bounds (`SOC_min/max`, `T_min/max`, `SOH_min`) are actually applied **inside the ODE functions themselves** (`max(...,min(...,x))`), in addition to the solver's `NonNegative` constraint — a double safeguard.

### 8.2 Event functions (phase termination)

Each phase ends either by expiration of the allotted time (end of eclipse or sun window), or by an **SOC-based event** detected by `ode45`:

- **Discharge**: stops if `SOC` crosses `SOC_min` from above (`direction = -1`). Models over-discharge protection (the pack must never cross its safety floor, even if the eclipse window lasted longer).
- **Charge**: stops if `SOC` crosses `0.99` from below (`direction = +1`). Models the end of charge (the BMS cuts/reduces the current as full charge approaches to avoid overcharging).

Two implementations of these events coexist in the code: `cubesat_events.m` (a generic function parameterized by `phase`) and local subfunctions `evt_dch`/`evt_ch` in `simulate_cycle.m`. It is the **latter that are actually used** by `simulate_cycle.m` (the file `cubesat_events.m` appears to be an alternative/earlier version not called in the observed execution chain) — a point of caution for code maintainability (redundancy to factor out).

### 8.3 Continuity between phases and between cycles

The final state of discharge (`X_d(end,:)`) serves as the initial condition for charge. The final state of charge (`X_c(end,:)`, or the end-of-discharge state if charge did not converge) becomes `x_next`, passed as the initial condition for the next orbital cycle in `cubesat_run.m`. This **continuous state chain** is what makes it possible to simulate realistic cumulative aging over thousands of cycles.

A safeguard exists in case the charge phase produces fewer than 2 solution points (`length(t_c) < 2`): in that case, the end-of-discharge state is kept as-is as the next state, and a `warning` is issued — an edge case designed to avoid crashing the main loop in the event of a degenerate numerical configuration.

---

## 9. Per-cycle feature extraction (`simulate_cycle.m`)

At the end of each full orbital cycle, a data row (`row`) is produced, containing in particular:

| Field | Computation / origin | Interpretation |
|---|---|---|
| `SOC_start`, `SOC_end` | initial/final states of **discharge** | the end-of-cycle SOC is the one **after discharge** (discharged state), not after the following charge — a convention worth noting for using the dataset |
| `DoD` | `SOC_start − SOC_end` | actual depth of discharge for the orbit |
| `IR_ohm` | `mean(IR_d)` | average internal resistance during discharge (classic BMS electrical-health indicator) |
| `QD_Ah` | `SOH_now · Q_nom` | available discharge capacity, re-evaluated with the end-of-cycle SOH |
| `QC_Ah` | `SOH_now · Q_nom · η` | corresponding charge capacity (incorporating coulombic efficiency) |
| `V_mean/min/max_V` | statistics over the whole voltage series (discharge + charge concatenated) | voltage envelope observed over the orbit |
| `Tavg/min/max_C` | temperature statistics (discharge + charge), clamped to [−10, +50 °C] | thermal envelope of the orbit |
| `chargetime_min`, `discharge_time_min` | **actual** durations from the solver (can be < nominal duration if an SOC event cut the phase short) | actually required times, useful for checking the margin against the orbital windows |
| `x_next` | final state `[SOC; T; SOH]` | passed to the next cycle |
| `SOH_end` | `x_next(3)` | end-of-cycle SOH |
| `T_amb_K` | eclipse ambient temperature used this cycle | traceability of the applied orbital variability |

A **light internal validation** is performed (`if ~isfinite(QD) ...`) to flag, via `warning`, any non-physical capacity value (NaN, Inf, or out of expected bounds), without interrupting the simulation.

---

## 10. Multi-cycle simulation loop (`cubesat_run.m`)

### 10.1 Cycle-to-cycle orbital variability

To avoid a purely deterministic/repetitive model, each cycle randomly draws (with a fixed seed `rng(batt_idx*42)` for reproducibility):

$$
t_{eclipse} \sim \mathcal{N}(35,\ 1.0)\ \text{min, truncated to } [32, 38]
$$
$$
t_{sunlight} \sim \mathcal{N}(55,\ 0.8)\ \text{min, truncated to } [53, 57]
$$

This represents the **realistic variation of orbital geometry** (beta angle, seasons, eccentricity) of a satellite in sun-synchronous LEO, where the eclipse duration is never perfectly constant from one orbit to the next.

In addition, the structural eclipse temperature alternates deterministically according to cycle parity:

$$
T_{amb,eclipse} =
\begin{cases}
-10\,°C & \text{if odd cycle} \\
-2\,°C & \text{if even cycle}
\end{cases}
$$

This alternation introduces **additional, controlled environmental thermal variability**, potentially intended to enrich the diversity of the training dataset (if it is used downstream for machine learning — a hypothesis consistent with the script's "dataset generator" nature).

### 10.2 Simulation initial condition

$$
x_0 = [SOC=0.99;\ T = T_{operating,min}+5.0\ \text{K} = -5\,°C;\ SOH=1.0]
$$

The pack starts full, slightly cold, and in perfect health (BOL — Beginning of Life).

### 10.3 Stopping criterion and end of life (EOL)

The loop runs up to `N_max = 7000` cycles, but stops **prematurely** as soon as:

$$
SOH < SOH_{eol} = 0.70
$$

This 70% threshold is justified in the code as a widely recognized standard in Li-ion engineering (`IEC 62660-1`), beyond which:
- the voltage drop under discharge becomes too large to meet the mission's power budget,
- the risk of thermal runaway increases.

Two **cycle-life indicators** are also computed afterward:
- `cycle_life_80`: first cycle where `SOH < 0.80`,
- `cycle_life_70`: first cycle where `SOH < 0.70` (generally identical to the loop's stopping cycle).

### 10.4 Consistency checks (validation gates)

Before export, the script prints several physical plausibility checks over the entire run:
- finiteness of the quantities `QD`, `QC`, `IR` (absence of NaN/Inf),
- `QD` within `[Q_nom·SOH_min, Q_nom]`,
- `QC` within `[3.50, 4.00]` Ah,
- strict consistency `DoD = SOC_start − SOC_end` (to within 10⁻⁵),
- average temperatures within `[−10, +50] °C`,
- average voltage within `[V_pack_min, V_pack_max]`,
- internal resistance within `[0, 200] mΩ`.

These checks do not block execution: they are **printed as a summary** (counters "x/n" meeting each criterion), in the style of a quality-acceptance report for the generated dataset, allowing the user to visually judge the overall consistency of the run before using it.

### 10.5 Exporting the results

The script exports a `battery_dataset_normal_<timestamp>.csv` file (a MATLAB `writetable` table) containing, for each simulated cycle, the following 18 columns:

```
battery_id, cycle, SOC_start, SOC_end, DoD, IR_ohm, QD_Ah, QC_Ah,
V_mean_V, V_min_V, V_max_V, Tavg_C, Tmin_C, Tmax_C,
chargetime_min, discharge_time_min, SOH, T_amb_K
```

This "one row = one orbital cycle" format is directly usable for:
- statistical analysis of degradation over time (SOH/IR curves vs. cycle),
- training machine-learning models for Remaining Useful Life (RUL) prediction,
- later comparison with fault scenarios (a dataset generated by a "fault" module), with the present dataset serving as the healthy reference.

---

## 11. Summary of model parameters

| Category | Parameter | Value | Unit |
|---|---|---|---|
| **Pack** | Q_nom | 4.035 | Ah |
| | V_pack_min / max / mid | 6.20 / 8.40 / 7.40 | V |
| **Currents** | I_dch | 1.91 | A |
| | I_ch | 1.2155 | A |
| | η (coulombic efficiency) | 0.98 | — |
| **Internal resistance** | R0 | 0.015 (15) | Ω (mΩ) |
| | k_R | 1.899 | — |
| | Ea | 32,000 | J/mol |
| | T_ref | 298.15 (25) | K (°C) |
| **Aging** | α | 0.02085 | — |
| | γ | 1.0 (frozen, nominal) | — |
| | SOH_eol | 0.70 | — |
| **Thermal** | m | 0.350 | kg |
| | Cp | 1050 | J/(kg·K) |
| | ε | 0.85 | — |
| | A_rad | 0.004 | m² |
| | G_structure | 0.50 | W/K |
| | T_heater_on / off | −10 / +20 | °C |
| | P_heater | 2.0 | W |
| **Environment** | T_amb_sun | +35 | °C |
| | T_amb_eclipse | −10 (or −2, alternating) | °C |
| | Orbit | 90 | min |
| | Nominal eclipse / sunlight | 35 / 55 | min |
| **State bounds** | SOC_min / max | 0.50 / 0.99 | — |
| | T_operating min/max | −10 / +50 | °C |
| | SOH_min | 0.70 | — |

---

## 12. Model limitations and possible improvements

1. **OCV(SOC) not experimentally calibrated**: the cubic polynomial is a plausible but generic form; test-based characterization (low-current discharge curve, GITT/pseudo-OCV technique) would be needed for use in real mission sizing.
2. **Comment inconsistency on R0**: `0.015 Ω` commented as *"55.20 mΩ"* — to be corrected/clarified in the source code to avoid any confusion during review.
3. **Heater hysteresis not implemented**: `T_heater_off` is defined but never used in the heater control logic, which remains a simple low threshold — a risk of unrepresented high-frequency cycling (potential underestimation of the heater's power consumption and duty cycle).
4. **`cubesat_events.m` / local functions redundancy**: two implementations of the event logic coexist, with only the version local to `simulate_cycle.m` actually used in the main script's execution chain — to be factored out for maintainability.
5. **No calendar aging**: SOH only decreases during the active phases (charge/discharge) integrated by the solver; no rest-period (storage) aging term is modeled, even though it exists in reality even outside cycling.
6. **0D model only**: no cell-to-cell distinction within the pack (no inter-cell imbalance, no internal thermal gradient) — a reasonable assumption for a CubeSat but one that limits representativeness for larger packs.
7. **`MATLAB_CONFIG.m` not executable as-is**: the `#` syntax on the first line is not valid MATLAB; this file appears intended as reference documentation/configuration rather than a functional script integrated into the computation chain.

---

## 13. Conclusion

This model constitutes a **physically motivated, cycle-by-cycle simulation chain** for a CubeSat Li-ion battery pack, coupling:
- an electrical model with OCV + thermally- and aging-dependent internal resistance (Arrhenius),
- a complete thermal balance in a space vacuum environment (radiation + structural conduction + Joule dissipation + active heating),
- a capacity-fade-type aging model, dependent on current, temperature, and SOC (U-shaped stress factor),

all numerically integrated by an adaptive Runge-Kutta solver with event detection, over potentially several thousand orbital cycles, until a standard end-of-life stopping criterion (SOH < 70%) is reached. The result is a multi-cycle tabular dataset intended to represent the **nominal (fault-free)** behavior of the battery, likely serving as a reference for comparison with degraded scenarios simulated in a separate module.
