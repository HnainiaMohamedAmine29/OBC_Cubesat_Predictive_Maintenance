# CubeSat Battery Aging Simulator – Panasonic NCR18650B (NCA) Model

A high‑fidelity MATLAB simulation of Li‑ion battery degradation for Low Earth Orbit (LEO) CubeSats.  
Calibrated against published experimental data for **Panasonic NCR18650B** cells in a **2S1P** pack configuration.

---

## Overview

This simulator models the coupled **electrical–thermal–aging** dynamics of a CubeSat battery over thousands of orbital cycles.  
Each cycle consists of:

- **Discharge (eclipse)** – constant current load  
- **Charge (sunlight)** – constant current charge to 99% SOC  

The model captures realistic **non‑linear capacity fade** caused by SEI growth, mid‑life plateau, and end‑of‑life acceleration.  
It outputs a per‑cycle dataset suitable for lifetime prediction, machine learning, or mission planning.

---

## Features

- **Realistic NCR18650B parameters** – capacity, internal resistance, thermal mass.
- **Non‑linear aging model** – three‑stage SOH evolution validated against literature.
- **Arrhenius temperature dependence** for both resistance growth and aging rate.
- **Vacuum thermal environment** – radiation, conduction, and optional survival heater.
- **Automatic end‑of‑life detection** – stops when SOH falls below 70% (IEC 62660‑1).
- **CSV export** – clean, per‑cycle data ready for analysis.

---

## Requirements

- MATLAB R2020b or later (requires `ode45` and table support).
- No additional toolboxes needed.

---

## File Structure

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `cubesat_params.m`    | Defines pack parameters, thermal properties, and aging coefficients.    |
| `cubesat_ode_dch.m`   | ODE function for discharge (eclipse) phase.                             |
| `cubesat_ode_ch.m`    | ODE function for charge (sunlight) phase.                               |
| `cubesat_events.m`    | Event function to stop integration at SOC limits (standalone utility).  |
| `simulate_cycle.m`    | Simulates one full orbit and extracts per‑cycle metrics.                |
| `cubesat_run.m`       | Main script – runs multi‑cycle simulation and saves results.            |

---

## How to Run

1. Place all `.m` files in the same folder or add them to the MATLAB path.
2. Open `cubesat_run.m` and adjust the simulation parameters if desired:
   - `N_max` – maximum number of cycles to simulate.
   - `n_batt` – number of independent battery realizations (for Monte Carlo).
3. Run the script:
   ```matlab
   cubesat_run 


   