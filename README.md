# 🚗 Driver Input Smoothness Analyzer (DISA)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DISA** is a production-grade, mathematically rigorous software system designed to automatically evaluate and score time-series driving behavior. 

By ingesting high-frequency telemetry data (such as steering bounds, throttle/brake pressure, acceleration, and yaw rate), DISA applies robust signal preprocessing, extracts deterministic physical features (like jerk and signal variance), and grades the driver on a standardized **0–100 Smoothness Scale**.

---

## 🌟 Key Features

- **Robust Signal Processing Pipeline:** Automatically handles missing values and applies a zero-phase 4th-order Butterworth low-pass filter (5 Hz cutoff) to remove high-frequency noise and sensor artifacts.
- **Explainable Feature Engineering:** Computes four explicit mathematical pillars of driving behavior: Rate of Change ($S$), Jerk ($J$), Signal Variance ($V$), and Spike Frequency ($P$).
- **Data-Driven Weight Optimization:** Includes a training module relying on constrained `SLSQP` optimization to tune the scoring algorithm's layer weights against labeled ground-truth datasets to minimize MSE.
- **Physics-Guided Synthetic Simulator:** Replaces random noise generation with temporally continuous, physically plausible *Ornstein–Uhlenbeck* processes and event injections (e.g., hard braking, swerving) to model *Smooth*, *Aggressive*, and *Mixed* drivers.
- **Interactive Research Dashboard:** A sleek, dark-mode Streamlit UI equipped with Plotly visualizations, live anomaly highlighting, radar metric breakdowns, and session ranking algorithms.
- **Exportable PDF Reports:** Generates professional driving analysis reports utilizing radar plots and event distribution markers.

---

## 🏗️ System Architecture

The project is structured entirely as a modular Python package:

```text
DISA/
├── backend/            # Processing Pipeline & FastAPI
│   ├── main.py         # API Endpoints
│   ├── preprocessing.py# Butterworth filtering & physical clipping
│   ├── features.py     # Feature extraction (S, J, V, P) math
│   ├── scoring.py      # Base metrics and Smoothness Formula
│   ├── optimizer.py    # SLSQP / Grid Search weight fitting
│   ├── anomaly.py      # Spike detection and event extraction
│   └── database.py     # SQLite persistence layer
├── frontend/           # Presentation Layer
│   └── app.py          # 5-Tab Interactive Streamlit Dashboard
├── simulator/          # Synthetic Data Engineering
│   ├── simulator.py    # Generates O-U process driven data sets
│   └── noise_model.py  # Layered sensor delays/noise framework
├── validation/         # Testing & Metrics
│   └── validate.py     # Generates Pearson Correlation, MSE, MAE
├── data/               # Persistent storage for generated CSVs/db
├── requirements.txt    # Project dependencies
└── README.md           # You are here
```

---

## ⚙️ Installation & Setup

**1. Clone the project and navigate to the root directory:**
```bash
cd "DISA"
```

**2. Install dependencies via pip:**
```bash
pip install -r requirements.txt
```
*(Dependencies include: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `streamlit`, `plotly`, `fpdf2`, `python-multipart`)*

---

## 🚀 Usage Guide

### 1. Launching the Interactive Dashboard
The easiest way to explore DISA is through the frontend web application.
```bash
python -m streamlit run frontend/app.py
```
*(Navigates to `http://localhost:8501`. View the mathematical radar charts, analyze anomalies, generate synthetic CSV files using the Simulator tab, or tune model parameters via the Weight Optimizer).*

### 2. Launching the API Backend
For system integrations, ping DISA through its REST API architecture. 
```bash
uvicorn backend.main:app --reload
```
*(FastAPI interactive docs will automatically be served at `http://localhost:8000/docs`. Provides endpoints like `/analyze` and `/report`).*

### 3. Running the Validation Suite
Ensure the mathematical tuning models are properly calibrated and your `FEATURE_REF_MAX` limits natively adhere to physical ranges.
```bash
python validation/validate.py
```
*(This triggers a hierarchy compliance test directly ensuring `Smooth > Mixed > Aggressive` scores, alongside printing the resulting MSE & Pearson $r$ correlations).*

---

## 🧠 The Scoring Math

Driving smoothness is calculated directly against four physical parameters extracted from un-normalized telemetry boundaries. 

$$ \text{Overall Score} = 100 - \left( w_1 S + w_2 J + w_3 V + w_4 P \right) $$

- **Rate of Change ($S$):** The average magnitude of $dx/dt$ across steering, throttle, and braking, measuring twitchiness.
- **Jerk ($J$):** The average magnitude of $d^2x/dt^2$ (Acceleration derivatives), capturing sudden, un-smooth physical forces applied to the driver.
- **Variance ($V$):** The moving-window variance profile indicating signal instability.
- **Spike Frequency ($P$):** A percentage computation of events breaking an upper-bound intensity threshold, representing anomaly occurrences.

All components are bounded to physical extrema (`FEATURE_REF_MAX`) calibrated from known aggressive synthetic samples, avoiding non-linear suppression when dealing with heavily filtered parameters.

---

## 🛡️ License

This project is licensed under the MIT License. You are free to utilize, build upon, or distribute this tool for open research and commercial safety metrics.
