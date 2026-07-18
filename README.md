# OBC CubeSat Predictive Maintenance

A comprehensive framework for **battery State of Health (SOH) prediction** for CubeSats in Low Earth Orbit (LEO). This project combines physics-informed simulation, machine learning data pipelines, and LSTM neural networks to predict battery SOH degradation in normal orbital conditions.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Components](#components)
- [Workflow](#workflow)
- [Dataset](#dataset)
- [Results & Validation](#results--validation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project addresses the critical challenge of **predicting battery State of Health (SOH) in CubeSats**, where accurate health monitoring is essential for mission safety and operational planning. The system uses:

1. **Physics-based simulation** (`battery_model_Normal_conditions`) to generate synthetic battery aging datasets under normal operational conditions
2. **Data processing pipelines** to extract features and prepare time-series data for machine learning
3. **LSTM neural networks** to predict SOH trends in real-time

### Why This Matters

- CubeSats operate in harsh LEO environments with extreme thermal cycling
- Battery failures are often the limiting factor in mission lifespan
- Accurate SOH prediction enables proactive power management
- Real-time SOH monitoring allows onboard decision-making for mission-critical tasks

---

## Project Structure

```
OBC_Cubesat_Predictive_Maintenance/
├── battery_model_Normal_conditions/       # PRIMARY: Normal battery aging simulation (MATLAB)
│   ├── cubesat_params.m                  # Battery & thermal parameters
│   ├── cubesat_ode_dch.m                 # Discharge phase ODE
│   ├── cubesat_ode_ch.m                  # Charge phase ODE
│   ├── simulate_cycle.m                  # Single-orbit simulator
│   ├── cubesat_run.m                     # Main simulation driver (generates dataset)
│   └── README.md                         # Detailed technical docs
│
├── battery_cubesat_model/                # Alternative: Multi-fault model (MATLAB)
│   ├── cubesat_params.m                  # Battery & thermal parameters
│   ├── cubesat_ode_dch.m                 # Discharge phase ODE
│   ├── cubesat_ode_ch.m                  # Charge phase ODE
│   ├── simulate_cycle.m                  # Single-orbit simulator
│   ├── inject_fault.m                    # Fault injection system
│   ├── cubesat_run.m                     # Main simulation driver
│   └── README.md                         # Detailed technical docs
│
├── simulation_with_NCR1865B_Model/       # Alternative: NCR18650B cell model (MATLAB)
│   ├── cubesat_params.m                  # NCR18650B parameters
│   ├── cubesat_ode_dch.m                 # Discharge ODE
│   ├── cubesat_ode_ch.m                  # Charge ODE
│   ├── simulate_cycle.m                  # Orbit simulator
│   ├── cubesat_run.m                     # Main driver
│   └── README.md                         # Model documentation
│
├── pipline_data_preparing/               # Data processing pipeline (Python)
│   ├── config.py                         # Configuration hub (edit for each mission)
│   ├── feature_engineering.py            # Extract features from raw battery data
│   ├── split.py                          # Train/test splitting (temporal or random)
│   ├── normalize.py                      # Feature normalization & scaler
│   ├── sequences_construction.py         # LSTM sequence generation (sliding window)
│   └── README.md                         # Pipeline documentation
│
├── notebooks/                            # Jupyter analysis notebooks (81.9% of repo)
│   ├── exploration_*.ipynb               # Data exploration & visualization
│   ├── model_training_*.ipynb            # LSTM training & hyperparameter tuning
│   ├── evaluation_*.ipynb                # Performance analysis & SOH prediction curves
│   └── [Additional analysis notebooks]
│
└── visualization/                        # EDA and analysis scripts
    └── data_analysis_normal_state/       # Temporal patterns, distributions, etc.
```

---

## Key Features

### 🔧 Physics-Based Simulation (battery_model_Normal_conditions)

**Primary Data Source** — Generates synthetic battery aging under normal LEO conditions:

- **Coupled ODE system**: State of Charge (SOC) → Temperature (T) → State of Health (SOH)
- **Arrhenius temperature dependence**: Realistic aging rate acceleration with thermal stress
- **Vacuum thermal model**: Radiation, conduction, and heater control
- **Normal operation only**: No fault injection — pure baseline degradation
- **Validated parameters**: 
  - Q_nom = 4.035 Ah (BOL capacity)
  - R₀ = 15 mΩ (pack resistance at 25°C)
  - α = 0.02085 (base aging rate)
  - Discharge current: 1.91 A (eclipse)
  - Charge current: 1.2155 A (sunlight)

### 📊 Data Processing Pipeline

- **Feature extraction**: Raw battery metrics → 20 engineered features per cycle
- **Temporal or random splitting**: Flexible train/test strategies
- **Automatic scaling**: StandardScaler or MinMaxScaler (fit on training set only)
- **LSTM-ready sequences**: Sliding window (30-cycle windows) for time-series SOH prediction
- **Mission-agnostic config**: Single `config.py` file adapts pipeline to any dataset

### 🧠 Machine Learning Models

- **LSTM architectures** for sequence-to-sequence **SOH prediction** (not RUL)
- **Jupyter notebooks** for experimentation and analysis
- **Metrics**: RMSE, MAE, R² for validation
- **Embedded deployment**: Lightweight scaler & model for onboard SOH monitoring

---

## Tech Stack

| Component | Technology | Percent |
|-----------|------------|---------|
| **Analysis & ML** | Jupyter Notebooks | 81.9% |
| **Simulation** | MATLAB | 10.6% |
| **Data Processing** | Python + scikit-learn | 7.5% |

**Requirements:**
- MATLAB R2020b+ (for simulation)
- Python 3.8+ with pandas, numpy, scikit-learn, joblib
- PyTorch or TensorFlow (for LSTM training in notebooks)

---

## Quick Start

### 1. Generate Synthetic Battery Data (Normal Conditions)

```bash
cd battery_model_Normal_conditions
matlab -batch "cubesat_run"
```

**Output**: `battery_dataset_normal_YYYYMMDD_HHMMSS.csv`
- **Cycles**: 7000 (until SOH < 70%)
- **End-of-Life**: SOH = 70% (IEC 62660-1 standard)
- **Columns**: cycle, SOC_start, SOC_end, DoD, IR_ohm, QD_Ah, QC_Ah, V_mean_V, V_min_V, V_max_V, Tavg_C, Tmin_C, Tmax_C, chargetime_min, discharge_time_min, **SOH**, T_amb_K

### 2. Process Data for Machine Learning

```bash
cd pipline_data_preparing
# Edit config.py if needed for custom parameters
python feature_engineering.py      # Extract features
python split.py                    # Split into train/test
python normalize.py                # Normalize features
python sequences_construction.py   # Build sliding windows
```

**Output**: 
- `train_normalized.npy`, `test_normalized.npy` (LSTM-ready sequences)
- `scaler.pkl` (for inference)
- Train/test CSV files with features

### 3. Train & Evaluate SOH Prediction Models

Open Jupyter notebooks in `notebooks/`:
```bash
jupyter notebook
# Open model_training_*.ipynb for LSTM training
# Open evaluation_*.ipynb for SOH prediction performance
```

---

## Components

### Primary Data Source: battery_model_Normal_conditions

Simulates realistic battery aging in LEO under **normal operating conditions** (no faults).

**Simulation Parameters:**
- **Orbital period**: 90 minutes (sun-synchronous LEO)
- **Eclipse duration**: ~35 min (N(35, 1) with bounds [32, 38])
- **Sunlight duration**: ~55 min
- **Ambient structure temps**:
  - Sunlight: +35°C
  - Eclipse: -10°C (alternates: -10°C and -2°C)
- **Initial SOH**: 99% (99.0)
- **End-of-Life threshold**: 70% (SOH_eol = 0.70)
- **Coulombic efficiency**: η = 0.98

**Physics Model:**
- **Electrical**: Open-circuit voltage, internal resistance with Arrhenius temperature dependence
- **Thermal**: Vacuum environment (radiation + conduction, no convection)
- **Aging**: Capacity fade driven by current, temperature, and SOC stress

**Output**: Per-cycle dataset with SOH, temperature, voltage, current, and internal resistance

### Alternative Models

**battery_cubesat_model**: Multi-fault injection scenario (5 fault types, 0.5% per-cycle transition)

**simulation_with_NCR1865B_Model**: Panasonic NCR18650B cell parameters with non-linear aging

*(Note: Main project uses data from `battery_model_Normal_conditions` exclusively)*

### Data Pipeline

**Feature Engineering**: 20 features extracted per cycle
- Electrical: mean/std current, voltage swing, resistance growth, power dissipation
- Thermal: temperature range, eclipse/sunlight temps, thermal stress index
- Capacity: discharge/charge capacity, nominal capacity ratios
- Degradation: SOH value, SOH trend, cycle count

**Splitting**: Temporal or random (default: 70/30 split)
- Temporal split: Earlier cycles → training, later cycles → testing (recommended for time-series)

**Normalization**: StandardScaler fitted on training set only
- Prevents data leakage
- Scaler saved for onboard inference

**Sequences**: 30-cycle sliding windows
- ~2 days of orbital history per sample (1 day ≈ 15 orbits)
- Captures short-term SOH degradation trends
- 600 features per LSTM input (30 cycles × 20 features)
- HORIZON = 1: Predict SOH at next cycle (t+1)

### Machine Learning (Notebooks)

- **Architecture**: LSTM with dropout and early stopping
- **Task**: Sequence-to-sequence **SOH prediction** (next-cycle SOH forecast)
- **Validation**: Time-series cross-validation, RMSE/MAE/R²
- **Deployment**: Model quantization and lightweight scaler for onboard use

---

## Workflow

```
Normal Battery Simulation (MATLAB)
↓
battery_dataset_normal_YYYYMMDD_HHMMSS.csv
↓
Feature Engineering & Normalization (Python)
↓
Train/Test Sequence Construction (Sliding Window)
↓
LSTM Training & Hyperparameter Tuning (Jupyter)
↓
SOH Prediction Evaluation & Performance Analysis
↓
Embedded Model Deployment for Real-Time SOH Monitoring
```

---

## Dataset

### Simulation Output

Each simulation generates a CSV with **per-cycle metrics** over 7000 cycles (≈200 days):

| Feature | Description | Unit |
|---------|-------------|------|
| `battery_id` | Battery identifier | — |
| `cycle` | Orbital cycle number | — |
| `SOC_start` | Initial State of Charge | % |
| `SOC_end` | Final State of Charge | % |
| `DoD` | Depth of Discharge | % |
| `T_amb_K` | Ambient structure temperature | K |
| `Tavg_C` | Mean battery temperature during cycle | °C |
| `Tmin_C` | Min battery temperature | °C |
| `Tmax_C` | Max battery temperature | °C |
| `I_discharge` | Discharge current | A |
| `I_charge` | Charge current | A |
| `V_mean_V` | Mean voltage | V |
| `V_min_V` | Min voltage | V |
| `V_max_V` | Max voltage | V |
| `IR_ohm` | Internal resistance | Ω |
| `QD_Ah` | Discharge capacity | Ah |
| `QC_Ah` | Charge capacity | Ah |
| `chargetime_min` | Charge duration | min |
| `discharge_time_min` | Discharge duration | min |
| `SOH` | **State of Health** | % |

### Example Sizes

- **Single battery, normal operation**: ~7000 rows (until SOH < 70%)
- **After feature engineering**: ~7000 rows × 20 features
- **After sequence construction** (WINDOW_SIZE=30, STEP_SIZE=1): ~6970 samples × (30 cycles × 20 features)

---

## Results & Validation

### Simulation Targets (battery_model_Normal_conditions)

| Metric | Expected Value | Notes |
|--------|---|---|
| Initial SOH | 99.0% | Beginning of life |
| **Final SOH** | **~70.0%** | End-of-life threshold (IEC 62660-1) |
| Cycles to 70% SOH | ~7000 | Normal operation |
| Initial IR | ~15 mΩ | Room temperature (25°C) |
| Final IR | ~100+ mΩ | Degraded state |
| Initial QD | ~4.0 Ah | BOL discharge capacity |
| Final QD | ~2.8 Ah | At EOL (70% SOH) |
| Average charge time | ~55 min | Per orbit |
| Average discharge time | ~35 min | Per eclipse |

### Model Performance Targets (SOH Prediction)

- **RMSE (SOH prediction)**: < 2% absolute error
- **MAE**: < 1.5% absolute error
- **R²**: > 0.95 on validation set
- **Inference latency**: < 10 ms (embedded system)

---

## Contributing

Contributions are welcome! Some ideas:

- Add alternative battery chemistries (LFP, NCA, etc.)
- Implement ensemble methods for robust SOH prediction
- Extend to multi-battery packs with cell balancing monitoring
- Validate against real CubeSat flight data
- Optimize models for embedded deployment on flight computers
- Add uncertainty quantification (Bayesian LSTM)

---

## License

MIT License – see LICENSE file for details

---

## References

- CubeSat EPS Design Guidelines (Cal Poly CubeSat Program)
- IEC 62660-1: Li-ion battery cyclic life test and safety requirements
- Arrhenius aging models for Li-ion batteries
- LSTM for time-series SOH prediction (literature)

---

## Acknowledgments

- Battery parameters validated against Panasonic NCR18650B datasheet
- LEO orbital mechanics from CubeSat design standards
- Battery degradation literature (SEI growth, lithium plating, thermal effects)
- Dataset generation in normal conditions for realistic mission scenarios
