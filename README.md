# OBC CubeSat Predictive Maintenance

A comprehensive framework for **battery health monitoring** and **Remaining Useful Life (RUL) prediction** for CubeSats in Low Earth Orbit (LEO). This project combines physics-informed simulation, machine learning, and data processing pipelines to predict battery State of Health (SOH) degradation.

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

This project addresses the critical challenge of **predicting battery degradation in CubeSats**, where power budget failures can compromise entire missions. The system uses:

1. **Physics-based simulation** to generate synthetic battery aging datasets
2. **Data processing pipelines** to prepare time-series data for machine learning
3. **LSTM neural networks** to predict SOH trends and RUL

### Why This Matters

- CubeSats operate in harsh LEO environments with extreme thermal cycling
- Battery failures are often the limiting factor in mission lifespan
- Predictive models enable proactive power management and extended operations
- Real-time SOH prediction allows onboard decision-making for mission-critical tasks

---

## Project Structure

```
OBC_Cubesat_Predictive_Maintenance/
├── battery_cubesat_model/              # Physics-based simulation (MATLAB)
│   ├── cubesat_params.m               # Battery & thermal parameters
│   ├── cubesat_ode_dch.m              # Discharge phase ODE
│   ├── cubesat_ode_ch.m               # Charge phase ODE
│   ├── simulate_cycle.m               # Single-orbit simulator
│   ├── inject_fault.m                 # Fault injection system
│   ├── cubesat_run.m                  # Main simulation driver
│   └── README.md                      # Detailed technical docs
│
├── simulation_with_NCR1865B_Model/    # Alternative NCR18650B model (MATLAB)
│   ├── cubesat_params.m               # NCR18650B parameters
│   ├── cubesat_ode_dch.m              # Discharge ODE
│   ├── cubesat_ode_ch.m               # Charge ODE
│   ├── simulate_cycle.m               # Orbit simulator
│   ├── cubesat_run.m                  # Main driver
│   └── README.md                      # Model documentation
│
├── pipline_data_preparing/            # Data processing pipeline (Python)
│   ├── config.py                      # Configuration hub (edit for each mission)
│   ├── feature_engineering.py         # Extract 20 battery features
│   ├── split.py                       # Train/test splitting (temporal or random)
│   ├── normalize.py                   # Feature normalization & scaler
│   ├── sequences_construction.py      # LSTM sequence generation (sliding window)
│   └── README.md                      # Pipeline documentation
│
└── notebooks/                         # Jupyter analysis notebooks (81.9% of repo)
    ├── exploration_*.ipynb            # Data exploration & visualization
    ├── model_training_*.ipynb         # LSTM training & tuning
    ├── evaluation_*.ipynb             # Performance analysis & RUL curves
    └── [Additional analysis notebooks]
```

---

## Key Features

### 🔧 Physics-Based Simulation
- **Coupled ODE system**: State of Charge (SOC) → Temperature (T) → State of Health (SOH)
- **Arrhenius temperature dependence**: Realistic aging acceleration with thermal stress
- **Vacuum thermal model**: Radiation, conduction, and heater control
- **Fault injection**: 5 fault modes (high temp, low temp, capacity fade, thermal runaway, normal)
- **Calibrated models**: Validated against Panasonic NCR18650B and custom cell parameters

### 📊 Data Processing Pipeline
- **20 engineered features**: Current, voltage, temperature, resistance, capacity metrics
- **Temporal or random splitting**: Flexible train/test strategies
- **Automatic scaling**: StandardScaler or MinMaxScaler (fit on training set only)
- **LSTM-ready sequences**: Sliding window (30-cycle windows) for time-series prediction
- **Mission-agnostic config**: Single `config.py` file adapts pipeline to any mission

### 🧠 Machine Learning
- **LSTM architectures** for sequence-to-sequence SOH prediction
- **Jupyter notebooks** for experimentation and analysis
- **Metrics**: RMSE, MAE, R² for validation
- **Embedded deployment**: Lightweight scaler & model for onboard prediction

---

## Tech Stack

| Component | Technology | Percent |
|-----------|------------|---------|
| **Simulation** | MATLAB | 10.6% |
| **Data Processing** | Python + scikit-learn | 7.5% |
| **Analysis & ML** | Jupyter Notebooks + PyTorch/TensorFlow | 81.9% |

**Requirements:**
- MATLAB R2020b+ (for simulation)
- Python 3.8+ with pandas, numpy, scikit-learn, joblib
- PyTorch or TensorFlow (for LSTM training in notebooks)

---

## Quick Start

### 1. Generate Synthetic Battery Data

```bash
cd battery_cubesat_model
matlab -batch "cubesat_run"
```

Output: `battery_dataset_v3_YYYYMMDD_HHMMSS.csv` (3000 cycles × N batteries)

Or use the alternative model:
```bash
cd simulation_with_NCR1865B_Model
matlab -batch "cubesat_run"
```

### 2. Process Data for Machine Learning

```bash
cd pipline_data_preparing
# Edit config.py if needed for custom parameters
python feature_engineering.py
python split.py
python normalize.py
python sequences_construction.py
```

Output: Normalized train/test sets + `.npy` files for LSTM

### 3. Train & Evaluate Models

Open Jupyter notebooks in `notebooks/`:
```bash
jupyter notebook
# Open model_training_*.ipynb for LSTM training
# Open evaluation_*.ipynb for RUL curves and performance
```

---

## Components

### Battery Simulation Models

Both models simulate realistic battery aging in LEO conditions with **3000 cycles** (≈200 days of orbital operation).

**Model 1: U3 Updated (Default)**
- Modified internal resistance baseline (R₀ = 15 mΩ)
- SOH degradation: 99% → ~58% over 3000 cycles
- Orbital parameters: 90-minute LEO orbit, 2S Li-ion pack, 4.035 Ah

**Model 2: NCR18650B**
- Panasonic NCR18650B cell parameters
- Non-linear aging validation against literature
- End-of-life detection (SOH < 70%)

Both generate realistic datasets with:
- Electrical features (voltage, current, resistance)
- Thermal dynamics (temperature, heater state)
- Degradation markers (capacity fade, SOH trend)
- Fault injection for anomaly detection

### Data Pipeline

**Feature Engineering**: 20 features extracted per cycle
- Electrical: mean/std current, voltage swing, resistance growth
- Thermal: temperature range, eclipse/sunlight temps, thermal stress
- Capacity: discharge/charge capacity, nominal capacity
- Degradation: SOH, trend, cycle count

**Splitting**: Temporal or random (default: 70/30 split)

**Normalization**: StandardScaler fitted on training set only
- Prevents data leakage
- Scaler saved for onboard inference

**Sequences**: 30-cycle sliding windows
- ~2 days of orbital history per sample
- Captures short-term degradation trends
- 600 features per LSTM input (30 cycles × 20 features)

### Machine Learning (Notebooks)

- **Training**: LSTM with dropout, early stopping, Adam optimizer
- **Validation**: Time-series cross-validation
- **Evaluation**: RMSE, MAE, R², RUL prediction accuracy
- **Deployment**: Model quantization and lightweight scaler export

---

## Workflow

```
Physical Battery Models (MATLAB)
    ↓
Synthetic Aging Datasets
    ↓
Feature Engineering & Normalization (Python)
    ↓
Train/Test Sequence Construction
    ↓
LSTM Training & Hyperparameter Tuning (Notebooks)
    ↓
Performance Evaluation & RUL Curves
    ↓
Embedded Model Deployment
```

---

## Dataset

### Simulation Output

Each simulation generates a CSV with **per-cycle metrics**:

| Feature | Description |
|---------|-------------|
| `battery_id` | Battery identifier |
| `cycle` | Orbital cycle number |
| `SOC_start` | Initial State of Charge |
| `SOC_end` | Final State of Charge |
| `T_mean` | Mean temperature during cycle |
| `I_discharge` | Discharge current (A) |
| `I_charge` | Charge current (A) |
| `V_min` | Min voltage (V) |
| `V_max` | Max voltage (V) |
| `IR` | Internal resistance (Ω) |
| `SOH` | State of Health (%) |
| `fault_type` | Injected fault (if any) |
| ... | 10+ additional features |

### Example Sizes

- **Single battery, 3000 cycles**: ~3000 rows
- **Batch of 100 batteries**: ~300K rows
- **Training dataset** (after sequences): ~8K–10K samples

---

## Results & Validation

### Simulation Targets (U3 Model)

| Metric | Expected | ± Notes |
|--------|----------|---------|
| Initial SOH | 99.0% | — |
| Final SOH (3000 cycles) | ~58% | Reflects updated R₀ |
| Initial IR | ~71 mΩ | Room temperature |
| Final IR | ~140 mΩ | Doubled with degradation |
| Charge time (avg) | ~55 min | Per orbit |
| Discharge time (avg) | ~35 min | Per eclipse |

### Model Performance Targets

- **RMSE (SOH prediction)**: < 2% absolute
- **MAE (RUL estimate)**: < 50 cycles
- **Inference latency**: < 10 ms (embedded system)

---

## Contributing

Contributions are welcome! Some ideas:

- Add alternative battery chemistries (LFP, NCA, etc.)
- Implement advanced RUL techniques (particle filters, kriging)
- Optimize models for embedded deployment
- Validate against real flight data
- Extend to multi-battery packs with cell balancing

---

## License

MIT License – see LICENSE file for details

---

## References

- CubeSat EPS Design Guidelines (Cal Poly)
- IEC 62660-1: Li-ion battery cyclic life test
- Arrhenius aging models for Li-ion batteries
- LSTM for time-series RUL prediction (literature)

---

## Acknowledgments

- Panasonic NCR18650B datasheet for model validation
- LEO orbital mechanics from CubeSat design standards
- Battery degradation literature (SEI growth, lithium plating models)
