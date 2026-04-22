# CubeSat Battery Cycle Dataset Generator — U3 (Updated)

A MATLAB-based simulation framework that generates realistic, physics-informed battery aging datasets for CubeSat missions in Low Earth Orbit (LEO). The model captures the coupled electro‑thermal‑aging dynamics of a 2S Li‑ion battery pack under cyclic charge/discharge operation, including vacuum thermal environment and variable fault scenarios.

**This version (v3‑updated) features a modified internal resistance baseline (`R₀ = 15 mΩ`) and targets a State of Health (SOH) degradation from 99% down to approximately 58% over 3000 cycles.**

## Table of Contents
- [Overview](#overview)
- [Physical Models](#physical-models)
  - [Electrical Submodel](#electrical-submodel)
  - [Thermal Submodel (Vacuum)](#thermal-submodel-vacuum)
  - [Aging Submodel](#aging-submodel)
- [Orbit and Operational Parameters](#orbit-and-operational-parameters)
- [Fault Injection](#fault-injection)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Output Dataset](#output-dataset)
- [Validation Targets](#validation-targets)
- [References](#references)

---

## Overview
This framework simulates the long‑term health degradation of a CubeSat battery pack over thousands of orbital cycles. It generates a tabular dataset suitable for:

- Battery health monitoring algorithm development
- Remaining useful life (RUL) prediction
- Fault diagnosis and prognosis studies
- Mission planning and power budget validation

The simulation uses coupled ordinary differential equations (ODEs) for **State of Charge (SOC)**, **temperature (T)**, and **State of Health (SOH)**. Thermal effects are modeled for a vacuum environment (no convection), and aging follows a stress‑factor approach with Arrhenius temperature dependence.

**Updated calibration targets:**
- SOH degrades from **99%** to approximately **58%** over 3000 nominal cycles.
- Internal resistance growth modeled as `IR = (R₀/SOH)·(1 + k_R·(1‑SOH))·exp(Ea/R·(1/T‑1/T_ref))` with `R₀ = 0.015 Ω` (15 mΩ).
- Capacity features `QD` and `QC` represent **rated full‑discharge/charge capacity** at current SOH (not per‑orbit Ah).

---

## Physical Models

### Electrical Submodel
The battery pack is modelled as a 2‑cell series Li‑ion system.

**Open‑Circuit Voltage (OCV):** OCV(SOC) = V_min + (V_max - V_min) · (0.02 + 0.98·SOC - 0.50·SOC² + 0.50·SOC³)

with `V_min = 7.4 V`, `V_max = 8.4 V`.

**Internal Resistance (IR):** IR(T, SOH) = (R₀ / SOH) · (1 + k_R·(1 - SOH)) · exp( (Ea/R_gaz) · (1/T - 1/T_ref) )

- `R₀` = **0.015 Ω** (15 mΩ) — BOL resistance at T_ref = 25 °C, SOH = 1
- `k_R` = 1.899 (resistance growth factor with capacity fade)
- `Ea` = 32 000 J/mol (activation energy)
- `R_gaz` = 8.314 J/(mol·K)
- `T_ref` = 298.15 K

**State of Charge Dynamics:**
- Discharge: `dSOC/dt = -η·I_dch / (Q_nom·SOH·3600)`
- Charge:    `dSOC/dt =  η·I_ch  / (Q_nom·SOH·3600)`

### Thermal Submodel (Vacuum)
Heat transfer occurs via **radiation** and **conduction** only (no convection).
dT/dt = ( Q_joule + Q_cond + P_heater - Q_rad ) / (m·Cp) 

where:
- `Q_joule = I²·IR`
- `Q_rad = σ·ε·A_rad·(T⁴ - T_space⁴)`,  `T_space = 3 K`
- `Q_cond = G_structure·(T_amb - T)` (sign depends on phase)
- `P_heater` activates when `T < T_heater_on` (2 W)

During **sunlight (charge)**, heat flows **in** from the warm structure (`T_amb_sun ≈ 35 °C`).  
During **eclipse (discharge)**, heat flows **out** to the cold structure (`T_amb_eclipse ≈ -10 °C`).

### Aging Submodel
Capacity fade is driven by current, temperature, and SOC stress:
 dSOH/dt = -α · I · exp(-Ea/(R_gaz·T)) · stress(SOC) · γ 

 with stress factor:stress(SOC) = 0.10 + 0.50·exp(-14·SOC) + 0.40·exp(-14·(1-SOC))

 - `α` = 0.02085 (base aging rate)
- `γ` = fault multiplier (≥1, modified by fault injection)

**Note:** With the reduced `R₀`, joule heating is lower, which affects temperature and consequently aging rate. The 3000‑cycle SOH drop to ~58% reflects this updated parameter set.

---

## Orbit and Operational Parameters
The mission is a sun‑synchronous LEO with a 90‑minute period.

| Parameter              | Value                     |
|------------------------|---------------------------|
| Eclipse duration       | ~N(35,1) min, ∈ [32,38]   |
| Sunlight duration      | ∈ [55,54]                 |
| Discharge current      | 1.91 A                    |
| Charge current         | 1.2155 A                  |
| Nominal capacity (BOL) | 4.035 Ah                  |
| Coulombic efficiency η | 0.98                      |

The discharge current is derived from an average eclipse load of 14.71 W at ~7.7 V. Charge current is chosen so that the average charge time equals 55 minutes (independent of SOH).

---

## Fault Injection
Faults are persistent, with a 0.5% per‑cycle transition probability. Severity increases by 0.001 per cycle once a fault starts.

| Fault Type          | Weight | Effect                                                         |
|---------------------|--------|----------------------------------------------------------------|
| `normal`            | 60%    | Baseline aging, γ = 1.0                                        |
| `high_temperature`  | 15%    | Cooling failure → warmer structure, γ↑, R₀↑, α↑                |
| `low_temperature`   | 5%     | Heater failure → higher R₀, slower aging (γ↓), P_heater = 0    |
| `capacity_fade`     | 15%    | Accelerated material degradation → γ↑, α↑, R₀↑                 |
| `thermal_runaway`   | 5%     | Catastrophic, irreversible → γ=50, α=0.1, R₀×10, conduction halved |

Fault parameters are applied per‑cycle via `inject_fault.m`.

---

## File Structure

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `cubesat_params.m`    | Defines all physical constants, battery parameters, and validation targets.  |
| `cubesat_ode_dch.m`   | ODE function for the discharge (eclipse) phase.                              |
| `cubesat_ode_ch.m`    | ODE function for the charge (sunlight) phase.                                |
| `cubesat_events.m`    | Event function for SOC cutoffs (provided for reference; actual events are inline in `simulate_cycle`). |
| `simulate_cycle.m`    | Simulates a single orbit (discharge → charge) and extracts cycle features.   |
| `inject_fault.m`      | Applies fault‑specific parameter multipliers and updates fault severity.     |
| `cubesat_run.m`       | Main script – runs the full simulation for N cycles, exports dataset.        |

---

## Usage
1. Ensure all `.m` files are in the MATLAB path.
2. Run the main script:
   ```matlab
   cubesat_run

3. The simulation will:

Execute 3000 cycles (default).

Display progress every 200 cycles.

Validate feature ranges.

Export a timestamped CSV file (battery_dataset_v3_YYYYMMDD_HHMMSS.csv).

## To modify simulation parameters:

Edit cubesat_params.m for physical constants (e.g., R0, alpha).

Change N_max or n_batt in cubesat_run.m for number of cycles or batteries. 

## Validation Targets
The model is calibrated to meet the following reference targets for a normal battery over 3000 cycles (with updated R₀ = 0.015 Ω):

Metric	Expected Value
Initial SOH	99.0 %
Final SOH (3000 cycles)	≈ 58 %
Initial IR 	≈ 71 mOhm
Final IR ≈ 140 mOhm
Initial QD	~4.00 Ah
Final QD	~2.34 Ah
QC range	~[2.29, 3.92] Ah
Average charge time	≈ 55 min
Average discharge time	≈ 35 min
Validation checks are performed automatically and printed to the console.