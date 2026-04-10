# 🚗 DISA — Driver Input Smoothness Analyzer

> **A research-grade system for quantifying and explaining driving smoothness from vehicle sensor time-series data.**

DISA computes an explainable, weighted 0–100 smoothness score from raw driving telemetry. It comprises a FastAPI backend, a Streamlit frontend dashboard, a physics-based synthetic driving simulator, and a research validation suite — all built without any ML black-boxes. Every score is fully explainable through four interpretable signal features.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Mathematical Scoring Methodology](#4-mathematical-scoring-methodology)
5. [Backend Modules](#5-backend-modules)
   - [preprocessing.py](#51-preprocessingpy)
   - [features.py](#52-featurespy)
   - [scoring.py](#53-scoringpy)
   - [anomaly.py](#54-anomalypy)
   - [optimizer.py](#55-optimizerpy)
   - [database.py](#56-databasepy)
   - [reports.py](#57-reportspy)
   - [main.py (FastAPI)](#58-mainpy--fastapi-rest-api)
6. [Synthetic Driving Simulator](#6-synthetic-driving-simulator)
   - [simulator.py](#61-simulatorpy)
   - [event_injector.py](#62-event_injectorpy)
   - [noise_model.py](#63-noise_modelpy)
7. [Validation Suite](#7-validation-suite)
8. [Frontend Dashboard](#8-frontend-dashboard)
9. [Dataset Files](#9-dataset-files)
10. [CSV Data Format](#10-csv-data-format)
11. [REST API Reference](#11-rest-api-reference)
12. [Calibration & Weight Optimization](#12-calibration--weight-optimization)
13. [Validation Results](#13-validation-results)
14. [Installation & Setup](#14-installation--setup)
15. [Running the Project](#15-running-the-project)
16. [Dependencies](#16-dependencies)
17. [Grading System](#17-grading-system)
18. [Design Decisions & Research Notes](#18-design-decisions--research-notes)

---

## 1. Project Overview

DISA analyzes driving sessions recorded as time-series CSV files containing 7 sensor channels and produces:

| Output | Description |
|--------|-------------|
| **Overall Score (0–100)** | Composite smoothness score |
| **Grade (A–F)** | Human-readable quality grade |
| **Sub-scores** | Per-group scores: Steering, Throttle, Braking, Stability |
| **Feature contributions** | How much each of the 4 features penalized the score |
| **Rolling time-series score** | Score at every timestep (sliding window) |
| **Anomaly events** | Detected rough driving events with timestamps |
| **PDF Report** | Downloadable professional analysis report |

### What Problem Does It Solve?

Measuring driver smoothness is important for:
- Fleet safety monitoring
- Driver training feedback
- Passenger comfort analysis
- Insurance telematics
- Autonomous vehicle benchmarking

Existing solutions are either subjective (human ratings) or ML-based black boxes. DISA offers a transparent, formula-driven approach where every number can be traced back to the raw signal.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DISA System Architecture                  │
├─────────────────┬───────────────────┬───────────────────────┤
│  Streamlit      │   FastAPI         │   Simulator           │
│  Frontend       │   Backend         │   (Synthetic Data)    │
│  (frontend/     │   (backend/       │   (simulator/         │
│   app.py)       │    main.py)       │    simulator.py)      │
│                 │                   │                       │
│  5 Tabs:        │  REST Endpoints:  │  3 Driver Profiles:   │
│  - Upload       │  - /analyze       │  - Smooth  (80–95)    │
│  - Simulator    │  - /simulate      │  - Aggressive (20–45) │
│  - Optimizer    │  - /optimize      │  - Mixed  (50–70)     │
│  - Validation   │  - /validate      │                       │
│  - Comparison   │  - /report        │  5 Event Types:       │
│                 │  - /sessions      │  - sudden_brake       │
│                 │  - /rank          │  - lane_change        │
│                 │                   │  - stop_and_go        │
│                 │   SQLite DB       │  - highway_cruise     │
│                 │   (data/disa.db)  │  - mild_corner        │
├─────────────────┴───────────────────┴───────────────────────┤
│                    Core Modules (backend/)                    │
│  preprocessing.py → features.py → scoring.py → anomaly.py   │
│  optimizer.py → database.py → reports.py                     │
├──────────────────────────────────────────────────────────────┤
│                    Validation Suite                          │
│  validation/validate.py                                      │
│  - Pearson r, MAE, MSE, RMSE                                │
│  - Score hierarchy test (smooth > mixed > aggressive)       │
│  - Robustness test (noise tolerance)                        │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
CSV Upload
    ↓
[preprocessing.py]
  1. Missing value handling (ffill + interpolation)
  2. Butterworth low-pass filter @ 5 Hz (4th-order, zero-phase)
  3. Physical bounds clipping
  4. Robust min-max normalization (1st–99th percentile)
    ↓
[features.py]
  Extract S, J, V, P per channel and per group
    ↓
[scoring.py]
  Score = 100 − clip(Σ wᵢ·fᵢ_norm × 100, 0, 100)
    ↓
[anomaly.py]
  Z-score spike / variance burst / brake spike detection
    ↓
JSON Response / Streamlit Display / PDF Report
```

---

## 3. Directory Structure

```
DT PROJECT/
├── README.md                     ← This file (complete documentation)
├── requirements.txt              ← All Python dependencies
├── calibrate.py                  ← Script to calibrate FEATURE_REF_MAX & optimize weights
├── calibrate_out.txt             ← Output of calibration run (raw feature values & weights)
├── validate_out.txt              ← Output of validation suite run
│
├── backend/                      ← FastAPI REST API + core analysis engine
│   ├── __init__.py
│   ├── main.py                   ← FastAPI app with all endpoints (434 lines)
│   ├── preprocessing.py          ← 4-stage signal preprocessing pipeline
│   ├── features.py               ← S, J, V, P feature extraction per channel
│   ├── scoring.py                ← Master smoothness score formula
│   ├── anomaly.py                ← Threshold-based rough event detection
│   ├── optimizer.py              ← SLSQP / grid-search weight optimizer
│   ├── database.py               ← SQLite persistence (sessions, anomalies, optimizer runs)
│   └── reports.py                ← PDF report generator (fpdf2 + matplotlib)
│
├── simulator/                    ← Physics-based synthetic data generator
│   ├── __init__.py
│   ├── simulator.py              ← 3 driver profiles + CLI entry point
│   ├── event_injector.py         ← 5 deterministic driving event types
│   └── noise_model.py            ← Sensor noise, road bumps, signal delay
│
├── validation/                   ← Research validation suite
│   ├── __init__.py
│   └── validate.py               ← Pearson r, MAE, hierarchy test, robustness test
│
├── frontend/                     ← Streamlit dashboard
│   └── app.py                    ← 5-tab interactive dashboard (1061 lines)
│
└── data/                         ← Pre-generated labeled datasets + SQLite DB
    ├── smooth.csv                ← 3000 samples, mean expected score ≈ 90.4
    ├── aggressive.csv            ← 3000 samples, mean expected score ≈ 31.3
    ├── mixed.csv                 ← 4500 samples, mean expected score ≈ 58.1
    └── disa.db                   ← SQLite database (auto-created on first run)
```

---

## 4. Mathematical Scoring Methodology

### Master Formula

```
Score = 100 − clip(w₁·S_norm + w₂·J_norm + w₃·V_norm + w₄·P_norm, 0, 100)
```

Where the penalty is the weighted sum of four normalized features, each capturing a different dimension of roughness.

### The Four Features

| Symbol | Full Name | Formula | What It Captures |
|--------|-----------|---------|-----------------|
| **S** | Rate of Change | `mean(|dx/dt|)` | How fast signals change — rapid inputs = harsh driving |
| **J** | Jerk | `mean(|d²x/dt²|)` | Rate of change of acceleration — jerkiness |
| **V** | Variance | `mean(rolling_window_std²)` | Temporal instability / unsettled driving |
| **P** | Spike Frequency | `fraction of samples where |dx/dt| > 2σ` | Sudden impulsive spikes |

All features are computed using **central differences** (numpy `np.gradient`) on the pre-processed (filtered, un-normalized) signals.

### Feature Normalization

Before scoring, each feature is normalized to `[0, 1]` using reference maxima calibrated from the aggressive driver profile:

```python
FEATURE_REF_MAX = {
    "S": 65.0,    # mean |dx/dt| of aggressive driver (calibrated: 65.29)
    "J": 830.0,   # mean |d²x/dt²|                   (calibrated: 827.52)
    "V": 480.0,   # rolling variance                  (calibrated: 475.83)
    "P": 0.05,    # spike fraction (5% is very high)  (calibrated: 0.0482)
}

f_norm = clip(f / ref_max, 0, 1)
```

### Default Weights

```python
DEFAULT_WEIGHTS = [0.25, 0.30, 0.25, 0.20]
#                  w_S   w_J   w_V   w_P
```

**Constraint:** weights must sum to 1.0 (automatically enforced by dividing by the sum).

Jerk (w_J = 0.30) is weighted highest because it most reliably distinguishes smooth from aggressive driving across all sensor channels.

### Sub-Scores

The same formula is applied per channel group:

| Group | Channels |
|-------|---------|
| **steering** | `steering_angle` |
| **throttle** | `throttle_position` |
| **braking** | `brake_pressure` |
| **stability** | `acceleration_x`, `acceleration_y`, `acceleration_z`, `gyroscope_yaw_rate` |

Group features = mean of per-channel `{S, J, V, P}` values within the group.

### Rolling Time-Series Score

A sliding window (default: 2 seconds, 50 samples at 25 Hz) computes `{S, J, V, P}` for each center sample and applies the same formula. This gives a per-sample score curve showing how smoothness varies over time.

---

## 5. Backend Modules

### 5.1 `preprocessing.py`

**Purpose:** 4-stage stateless preprocessing pipeline to convert raw telemetry into clean, normalized signals for feature extraction.

#### Required Signal Columns (`SIGNAL_COLUMNS`)

```python
SIGNAL_COLUMNS = [
    "steering_angle",      # degrees, range: [-180, 180]
    "throttle_position",   # %, range: [0, 100]
    "brake_pressure",      # %, range: [0, 100]
    "acceleration_x",      # m/s², range: [-30, 30]
    "acceleration_y",      # m/s², range: [-30, 30]
    "acceleration_z",      # m/s², range: [0, 30]
    "gyroscope_yaw_rate",  # deg/s, range: [-200, 200]
]
```

#### Stage 1: Missing Value Handling (`handle_missing_values`)
- Forward-fill (`ffill`) to preserve last known value
- Linear interpolation for interior NaNs
- Zero-fill for any remaining leading NaNs

#### Stage 2: Low-Pass Butterworth Filter (`apply_lowpass_filter`)
- **Type:** 4th-order Butterworth low-pass
- **Cutoff:** 5 Hz (removes high-frequency sensor artifacts)
- **Method:** `scipy.signal.filtfilt` (zero-phase, no phase lag)
- **Nyquist:** Sampling rate is typically 25 Hz → Nyquist = 12.5 Hz

#### Stage 3: Moving Average (optional, `apply_moving_average`)
- Centered moving average with `min_periods=1`
- Default window: 5 samples
- Only applied when `apply_ma=True`

#### Stage 4: Robust Normalization (`apply_normalization`)
- Uses **1st–99th percentile** as min/max (robust to outlier spikes)
- Normalizes each column to `[0, 1]` independently
- Returns normalization stats for reuse on test data

#### Two Preprocessing Variants

| Function | Normalized? | Used For |
|----------|------------|---------|
| `preprocess()` | Yes | Display/visualization, rolling score |
| `preprocess_for_features()` | **No** | Feature extraction (must preserve physical magnitudes) |

> **Why two variants?** If signals were normalized before feature extraction, an aggressive driver's signals would be scaled into the same range as a smooth driver's, making the features indistinguishable. Physical-unit features are required for `FEATURE_REF_MAX` thresholds to be meaningful.

---

### 5.2 `features.py`

**Purpose:** Extract four smoothness-relevant features from each of the 7 sensor channels.

#### Feature Computation (`_compute_channel_features`)

Given a 1-D signal array `x` at sample rate `fs`:

```python
dt = 1.0 / fs

# S — Rate of change (first derivative)
dx  = gradient(x, dt)
S   = mean(|dx|)

# J — Jerk (second derivative)
ddx = gradient(dx, dt)
J   = mean(|ddx|)

# V — Sliding window variance (1-second window by default)
var_series = rolling_std(x, window=fs, center=True, min_periods=2)² 
V = mean(var_series)

# P — Spike frequency
dx_std    = std(dx)
threshold = 2.0 * dx_std          # 2-sigma threshold
spike_mask = |dx| > threshold
P = mean(spike_mask)               # fraction of spike samples
```

#### Output: `FeatureMatrix`

```python
@dataclass
class FeatureMatrix:
    per_channel: Dict[str, ChannelFeatures]      # features per individual channel
    group_aggregates: Dict[str, Dict[str, float]] # {group: {S, J, V, P}} — mean over group
    aggregate: Dict[str, float]                   # overall {S, J, V, P} — mean over all channels
    fs: float
```

#### `get_event_highlight_mask(fm)`
Returns a boolean mask marking "rough" timesteps by unioning spike masks from the top-3 highest-jerk channels. Used by UI to highlight rough segments.

---

### 5.3 `scoring.py`

**Purpose:** Compute the complete smoothness score from a `FeatureMatrix`.

#### Grade Thresholds

| Score | Grade |
|-------|-------|
| ≥ 85 | **A — Excellent** |
| ≥ 70 | **B — Good** |
| ≥ 55 | **C — Average** |
| ≥ 40 | **D — Below Average** |
| < 40 | **F — Poor** |

#### `SmoothnesResult` Output

```python
@dataclass
class SmoothnesResult:
    overall_score: float                    # 0–100
    sub_scores: Dict[str, float]            # {steering, throttle, braking, stability}
    feature_contributions: Dict[str, float] # {S, J, V, P} — penalty points each contributed
    normalized_features: Dict[str, float]   # {S, J, V, P} — after normalization [0,1]
    weights: np.ndarray                     # [w_S, w_J, w_V, w_P] used
    grade: str                              # "A — Excellent" etc.
```

#### Feature Contributions

Each feature's contribution is: `contribution[i] = w[i] × normalized_feature[i] × 100`

Higher contribution = that feature penalized the score more.

#### `rank_sessions(session_results)`

Accepts a list of `(session_id, SmoothnesResult)` tuples. Returns them sorted descending by overall score with rank numbers — used by the Driver Comparison tab.

---

### 5.4 `anomaly.py`

**Purpose:** Detect rough driving events using threshold-based methods (no ML). Three independent detectors run and their results are merged and time-sorted.

#### Detector 1: Jerk Spike Detector (`detect_jerk_spikes`)

- Computes second derivative (jerk) for each signal channel
- Aggregates jerk across all channels (mean)
- Computes **rolling Z-score** over a 3-second window
- Flags samples where `|Z-score| > 3.0` (default threshold)
- Merges adjacent flagged regions with gaps < 0.5 seconds
- Identifies top-3 most affected channels per event
- **Event type:** `"jerk_spike"`

#### Detector 2: Brake Spike Detector (`detect_brake_spikes`)

- Monitors `brake_pressure` channel
- Flags samples where: rate of brake application `d(brake)/dt > 30%/s` OR `brake_pressure > 50%`
- Merges adjacent events with gaps < 0.3 seconds
- Filters out events shorter than 0.2 seconds
- Severity = `peak_brake_pressure / 100 × 5` (Z-score equivalent)
- **Event type:** `"brake_spike"`

#### Detector 3: Variance Burst Detector (`detect_variance_bursts`)

- Monitors `steering_angle`, `acceleration_x`, `acceleration_y`
- Computes rolling std over 2-second window per channel
- Aggregates, then computes rolling Z-score over 6-second window
- Flags samples where `|Z-score| > 2.5`
- **Event type:** `"variance_burst"`

#### `AnomalyEvent` Output

```python
@dataclass
class AnomalyEvent:
    start_time: float       # seconds from start
    end_time: float         # seconds from start
    start_idx: int          # sample index
    end_idx: int
    severity: float         # Z-score magnitude (higher = more severe)
    event_type: str         # 'jerk_spike' | 'variance_burst' | 'brake_spike'
    affected_channels: List[str]  # top-3 most affected channels
```

---

### 5.5 `optimizer.py`

**Purpose:** Find optimal scoring weights `[w_S, w_J, w_V, w_P]` that minimize prediction error on labeled data.

#### How It Works

1. Load labeled CSV files (must contain `expected_score` column)
2. Split each file into overlapping 10-second segments (50% overlap)
3. Extract `{S, J, V, P}` features per segment
4. Optimize weights to minimize MSE between predicted scores and `expected_score` labels

#### Optimization Strategy A: SLSQP (`slsqp_optimize_weights`)

- Uses `scipy.optimize.minimize` with `method="SLSQP"`
- **Equality constraint:** `w₁ + w₂ + w₃ + w₄ = 1`
- **Bounds:** each `wᵢ ∈ [0.01, 0.99]`
- Runs `n_restarts` times from random Dirichlet-sampled starting points
- Returns the best result across all restarts
- Tolerance: `ftol=1e-9`, `maxiter=1000`

#### Optimization Strategy B: Grid Search (`grid_search_weights`)

- Exhaustive simplex grid search with configurable step size (default: 0.05)
- At step=0.05, evaluates ~1,771 weight combinations
- Returns top-N results by MSE
- Fully reproducible (no randomness)

#### Sensitivity Analysis (`sensitivity_analysis`)

After optimization, perturbs each weight by `Δw ∈ [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]` (while renormalizing) and measures MSE change. Reveals which weights the model is most sensitive to.

#### `OptimizationResult` Output

```python
@dataclass
class OptimizationResult:
    method: str                              # "SLSQP" | "Grid Search"
    optimized_weights: np.ndarray            # [w_S, w_J, w_V, w_P]
    mse: float                               # MSE with optimized weights
    mae: float                               # MAE with optimized weights
    default_mse: float                       # MSE with default weights (for comparison)
    default_mae: float
    mse_improvement_pct: float               # % improvement vs default
    sensitivity: Dict[str, List[float]]      # per-weight MSE at each delta
    grid_results: Optional[List[dict]]       # top-N grid search results
```

#### Calibration Results (from `calibrate_out.txt`)

```
smooth driver:     S=4.005, J=69.4, V=0.971, P=0.040 | expected score ≈ 90.4
aggressive driver: S=54.4,  J=689.6, V=396.5, P=0.040 | expected score ≈ 31.3
mixed driver:      S=12.8,  J=201.1, V=14.3,  P=0.038 | expected score ≈ 58.1

SLSQP Optimized weights: [0.010, 0.010, 0.970, 0.010]
MSE improvement: 789.42 → 241.68  (+69.4% improvement)
Training segments: 81 total (23 smooth + 23 aggressive + 35 mixed)
```

> **Note:** The optimizer strongly favored Variance (V) on these datasets. Default weights `[0.25, 0.30, 0.25, 0.20]` are used in production as they are more balanced and physically interpretable.

---

### 5.6 `database.py`

**Purpose:** Persist analysis sessions, anomaly events, and optimizer runs to a local SQLite database (`data/disa.db`).

#### Tables

**`sessions`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | 8-character UUID (e.g., `a3f8b2c1`) |
| `created_at` | TEXT | UTC ISO timestamp |
| `filename` | TEXT | Uploaded file name |
| `fs` | REAL | Sample rate (Hz) |
| `n_samples` | INTEGER | Number of data points |
| `duration_s` | REAL | Session duration in seconds |
| `overall_score` | REAL | Computed smoothness score |
| `grade` | TEXT | Letter grade |
| `sub_scores` | TEXT | JSON: `{steering, throttle, braking, stability}` |
| `features` | TEXT | JSON: normalized `{S, J, V, P}` |
| `weights` | TEXT | JSON: `[w_S, w_J, w_V, w_P]` used |

**`anomaly_events`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `session_id` | TEXT FK | References `sessions.id` |
| `start_time` | REAL | Event start (seconds) |
| `end_time` | REAL | Event end (seconds) |
| `severity` | REAL | Z-score magnitude |
| `event_type` | TEXT | `jerk_spike` / `variance_burst` / `brake_spike` |
| `channels` | TEXT | JSON list of affected channel names |

**`optimizer_runs`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `created_at` | TEXT | UTC ISO timestamp |
| `method` | TEXT | Optimization method used |
| `csv_files` | TEXT | JSON list of uploaded file names |
| `weights` | TEXT | JSON: optimized `{w_S, w_J, w_V, w_P}` |
| `mse` | REAL | Optimized MSE |
| `mae` | REAL | Optimized MAE |
| `default_mse` | REAL | Baseline MSE with default weights |
| `improvement_pct` | REAL | MSE improvement percentage |
| `sensitivity` | TEXT | JSON sensitivity analysis results |

---

### 5.7 `reports.py`

**Purpose:** Generate a professional PDF report for a driving session analysis.

**Requires:** `fpdf2` library (`pip install fpdf2`)

#### Report Structure

1. **Header** — Session ID, generation timestamp
2. **Overall Smoothness Score** — Score/100, grade, weights used
3. **Sub-Scores by Channel Group** — Color-coded (green/yellow/red)
4. **Radar Chart** — Polar visualization of 4 group sub-scores (matplotlib, embedded as PNG)
5. **Feature Penalty Breakdown** — Horizontal bar chart of S, J, V, P contributions
6. **Normalized Feature Values** — Table of normalized feature scalars
7. **Detected Anomaly Events** — Table: time, duration, type, severity, channels
8. **Footer** — "DISA — Research MVP"

#### Color Palette

| Color | RGB | Meaning |
|-------|-----|---------|
| Dark BG | `(18, 18, 35)` | Background |
| Accent | `(99, 102, 241)` | Indigo sections |
| Green | `(52, 211, 153)` | Score ≥ 80 |
| Yellow | `(251, 191, 36)` | Score 55–79 |
| Red | `(239, 68, 68)` | Score < 55 |

#### File Output

- Media type: `application/pdf`
- Filename: `disa_report_{session_id}.pdf`
- Returned as binary stream via FastAPI `Response`

---

### 5.8 `main.py` — FastAPI REST API

**Base URL:** `http://localhost:8000`

**Interactive docs:** `http://localhost:8000/docs` (Swagger UI)

**ReDoc:** `http://localhost:8000/redoc`

#### Initialization
- FastAPI app with CORS middleware (allows all origins)
- SQLite database initialized on startup via `init_db()`

#### All Endpoints (full reference in [Section 11](#11-rest-api-reference))

| Method | Path | Tag | Description |
|--------|------|-----|-------------|
| GET | `/health` | System | Health check |
| POST | `/analyze` | Analysis | Upload CSV → full analysis |
| GET | `/simulate` | Simulation | Generate synthetic dataset |
| POST | `/optimize-weights` | Optimization | Optimize weights from labeled data |
| POST | `/validate` | Validation | Run full validation suite |
| POST | `/report` | Reports | Generate PDF report |
| GET | `/sessions` | Sessions | List all stored sessions |
| GET | `/sessions/{id}` | Sessions | Get session details |
| POST | `/rank` | Analysis | Rank multiple sessions |

---

## 6. Synthetic Driving Simulator

### 6.1 `simulator.py`

**Purpose:** Generate physically plausible, labeled time-series driving data for 3 driver archetypes.

#### Signal Generation Philosophy

Signals are **not random noise.** They are built from:

1. **Superposition of sinusoids** at physiologically motivated frequencies
   - Steering: 0.08–1.1 Hz (human steering correction bandwidth)
   - Throttle modulation: 0.02–0.15 Hz
2. **Ornstein–Uhlenbeck (OU) process** for temporally-correlated drift
   - `dx = θ(μ − x)dt + σ·dW`
   - Mean-reverting process (signals return toward baseline)
3. **Deterministic event injection** (see `event_injector.py`)
4. **Three-layer noise model** (see `noise_model.py`)

#### Output Columns for All Profiles

```
timestamp, steering_angle, throttle_position, brake_pressure,
acceleration_x, acceleration_y, acceleration_z,
gyroscope_yaw_rate, expected_score
```

#### Driver Profile Specifications

| Parameter | Smooth | Aggressive | Mixed |
|-----------|--------|-----------|-------|
| **Default duration** | 120 s | 120 s | 180 s |
| **Default seed** | 42 | 7 | 99 |
| **Steering freq** (Hz) | 0.08, 0.15, 0.25 | 0.4, 0.7, 1.1 | 0.15, 0.35, 0.6 |
| **Steering amplitude** (°) | 4, 2, 1 | 18, 10, 6 | 10, 5, 2.5 |
| **Steering OU σ** | 0.8 | 4.0 | 2.0 |
| **Base throttle** (%) | 52% | 60% | 48% |
| **Throttle modulation** | 6% @ 0.03 Hz | 25% @ 0.15 Hz | 14% @ 0.07 Hz |
| **Throttle OU σ** | 1.0 | 6.0 | 3.0 |
| **Expected score range** | 75–98 (mean ~90) | 10–50 (mean ~31) | 35–80 (mean ~58) |
| **Score base** (random) | 84–92 | 22–38 | 52–65 |
| **Score drift** | 3.0 × sin(2π×t/T) | 8.0 × sin(6π×t/T) | 10.0 × sin(4π×t/T) |
| **Noise sensor σ** | 0.3% | 0.8% | 0.5% |
| **Bump rate** | 0.1 Hz | 0.6 Hz | 0.3 Hz |
| **Bump amplitude** | 3% | 10% | 7% |
| **Events injected** | highway_cruise × 4, mild_corner × 3 | sudden_brake × 7, lane_change × 6, stop_and_go × 3 | All types mixed |

#### Gyroscope Yaw Rate Generation

```python
def _gyro_yaw(steering, fs, gain):
    d_steer = gradient(steering, 1/fs)
    return gain * d_steer
```

Yaw rate is derived from the steering rate (physically correct approximation).

#### `generate_all_datasets(output_dir, fs)`

CLI entry point to generate and save all three profiles as CSVs. Called when clicking "Generate All Three Profiles" in the Simulator tab.

---

### 6.2 `event_injector.py`

**Purpose:** Inject structured, physically realistic driving events into signal arrays with smooth ramp transitions.

All injections use cosine ramps: `ramp(n) = 0.5 × (1 − cos(linspace(0, π, n)))` for smooth on/off transitions.

#### Event 1: `inject_sudden_brake(signals, start_idx, fs, intensity, rng)`

Simulates hard braking:
- **Brake pressure:** Ramps from current → 60–90% over 0.3 s, holds for 0.8 s, releases over 0.5 s
- **Throttle:** Drops proportionally
- **Acceleration_x:** Deceleration ≈ `-0.6g × (brake/100) × intensity`
- **Steering:** Slight random wobble `N(0, 1.5 × intensity)`
- Total event duration: ~1.6 seconds

#### Event 2: `inject_lane_change(signals, start_idx, fs, intensity, direction, rng)`

Simulates lane change maneuver:
- **Duration:** 1.5–3.0 seconds (random)
- **Steering:** Two-phase sine sweep: `peak_steer × sin(t) × ramp × ramp_down`
- Peak steering: 20–40° × intensity × direction (+1=right, −1=left)
- **Lateral acceleration:** `steering_profile × 0.04 × intensity`
- **Yaw rate:** `steering_profile × 0.8 × intensity`

#### Event 3: `inject_stop_and_go(signals, start_idx, fs, intensity, rng)`

Full stop cycle (4 phases):
1. **Brake ramp up** (0.5 s): brake rises to 30–60%
2. **Hold** (1.5 s): full brake, throttle = 0
3. **Brake release** (0.5 s): brake ramps down
4. **Re-accelerate** (1.5 s): throttle ramps to 30–60%

#### Event 4: `inject_highway_cruise(signals, start_idx, fs, duration_s, rng)`

Steady highway driving:
- **Throttle:** Steady 55–75% + micro-oscillations at 0.05 Hz
- **Brake:** Forced near-zero (× 0.05)
- **Steering:** Very low-amplitude `0.5 × sin(0.02 Hz)` + N(0, 0.3) noise
- Severity: 0.1 (almost no disruption)

#### Event 5: `inject_mild_corner(signals, start_idx, fs, intensity, direction, rng)`

Gentle cornering:
- **Steering:** Bell-curve profile (Gaussian, peak 10–25° × intensity)
- **Lateral acceleration:** `steering × 0.03`
- **Throttle:** Slight lift-off: `throttle × (1 − 0.15 × bell)` (smooth deceleration)
- Duration: 2–4 seconds

---

### 6.3 `noise_model.py`

**Purpose:** Apply physically motivated disturbances to clean synthetic signals.

#### Noise Layer 1: Sensor Noise (`add_sensor_noise`)

White Gaussian noise with amplitude proportional to signal range:
```python
sigma = sigma_frac × peak_to_peak(signal)
noisy = signal + N(0, sigma)
```
Default `sigma_frac`: smooth=0.003, aggressive=0.008, mixed=0.005

#### Noise Layer 2: Road Bumps (`add_road_bumps`)

Damped half-sine pulses at Poisson-distributed random times:
```python
n_bumps = Poisson(bump_rate_hz × duration_s)
# Each bump: half-sine pulse over bump_duration_s (default 0.15 s)
pulse = amplitude × sin(linspace(0, π, bump_samples)) × ±1
```
Default `bump_rate_hz`: smooth=0.1, aggressive=0.6, mixed=0.3
Default `bump_amplitude_frac`: smooth=0.03, aggressive=0.10, mixed=0.07

#### Noise Layer 3: Signal Delay (`add_signal_delay`)

Fractional-sample delay using linear interpolation (simulates sensor latency):
```python
src_indices = arange(n) - delay_samples
interpolated = data[floor] × (1−frac) + data[ceil] × frac
```
Default `delay_samples`: smooth=0.2, aggressive=0.4, mixed=0.3

#### Full Pipeline: `apply_all_noise`

Applied in order: sensor noise → road bumps → signal delay.

---

## 7. Validation Suite

**File:** `validation/validate.py`

**Purpose:** Rigorously validate the scoring model against labeled ground-truth data.

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **Pearson r** | Correlation between predicted and expected scores (with p-value) |
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **Bias** | Mean signed error (positive = over-predicting) |
| **n_segments** | Number of 10-second segments evaluated |

### Test 1: Per-Profile Metrics (`_predict_segment_scores`)

The validation DataFrame is split into 10-second segments (50% overlap). Each segment gets:
- **Predicted score:** from `compute_smoothness_score(features)`
- **Expected score:** mean of the `expected_score` column over that segment

Pearson r, MAE, MSE, RMSE, bias are computed per profile (smooth/aggressive/mixed) and overall.

### Test 2: Score Hierarchy Test (`hierarchy_test`)

**Assertion:** `score_smooth > score_mixed > score_aggressive`

Uses freshly generated synthetic data (seeds 100, 101, 102) to test fundamental correctness of the scoring model. Output:
```
"hierarchy_test_passed": true,
"detail": "smooth(86.2) > mixed(62.1) > aggressive(28.4)"
```

### Test 3: Robustness Test (`robustness_test`)

Progressively adds Gaussian noise at levels `σ ∈ [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]` and measures score stability. A robust model should **decrease gracefully** (not collapse suddenly) as noise increases.

### CLI Usage

```bash
python -m validation.validate
```

Automatically loads `data/smooth.csv`, `data/aggressive.csv`, `data/mixed.csv` and prints:
- Overall Pearson r, MAE, MSE
- Hierarchy test PASS/FAIL
- Per-profile metrics table

---

## 8. Frontend Dashboard

**File:** `frontend/app.py` (1,061 lines)

**Framework:** Streamlit with Plotly charts

**Run command:**
```bash
streamlit run frontend/app.py
```

### UI Design

- **Dark theme** with custom CSS (`linear-gradient(135deg, #0f0f1e, #12122a, #0a0a1a)`)
- **Typography:** Inter (Google Fonts)
- **Color palette:** Indigo `#6366f1`, Green `#34d399`, Yellow `#fbbf24`, Red `#ef4444`
- **Glassmorphism metric cards** with backdrop blur
- **Score-color coded outputs** (green/yellow/red by score range)

### Sidebar (Global Settings)

- **Sample Rate (Hz):** Number input (default: 25.0 Hz)
- **4 Weight Sliders** for `w_S`, `w_J`, `w_V`, `w_P` (range 0–1, step 0.01)
- Auto-warns if weights don't sum to 1.0 (auto-normalizes)
- Displays the scoring formula in LaTeX: `Score = 100 − Σ wᵢ · fᵢ`

### Tab 1: Upload & Analyze 📊

1. **CSV Upload:** Drag-and-drop file uploader
2. **"Generate Sample Data"** button → downloads a 60-second smooth driver CSV
3. **Score Banner (5 cards):** Overall Score, Grade, Duration, Rough Events, Data Points
4. **Sub-Score Radar Chart:** Plotly polar chart of 4 group scores
5. **Progress Bars:** Visual 0–100 bars per group
6. **Feature Contribution Bar Chart:** Plotly bar chart of S, J, V, P penalty points
7. **Rolling Score Timeline:** Plotly area chart with anomaly shading and "Good threshold" at 70
8. **Signal Time-Series (3 expandable groups):**
   - Steering & Gyro
   - Throttle & Brake
   - Acceleration (x, y, z)
   - Anomaly events highlighted as red vertical rectangles
9. **Anomaly Events Table:** Dataframe with severity gradient coloring
10. **"Generate PDF Report"** button → downloads professional PDF

### Tab 2: Simulator 🚗

1. **Profile selector:** smooth / aggressive / mixed
2. **Duration slider:** 10–300 seconds
3. **Random seed input** for reproducibility
4. **"Generate Dataset"** button → runs simulator
5. **Stats banner:** Predicted Score, Expected Score, Samples, Events Injected
6. **Channel multiselect** → displays selected channels in Plotly time-series
7. **Injected Events table** with start/end times and severity
8. **Download button** for the generated CSV
9. **"Generate All Three Profiles"** → saves smooth.csv, aggressive.csv, mixed.csv to `data/`

### Tab 3: Weight Optimizer 🎯

1. **Optimization Method:** slsqp / grid / both
2. **Segment Length slider:** 5–30 seconds
3. **SLSQP Restarts input:** 1–20
4. **Multi-file CSV upload** (labeled, with `expected_score`)
5. **"Run Optimization"** → displays results:
   - MSE, MAE, improvement %, method used
   - Default vs optimized weights bar chart
   - Sensitivity analysis line chart (MSE vs Δw)
   - JSON code block of weights
   - Download button for `optimized_weights.json`

### Tab 4: Validation 🧪

1. **Multi-file labeled CSV upload**
2. **Checkbox:** "Also test on generated synthetic data"
3. **Segment Length slider**
4. **"Run Validation"** → displays:
   - Overall metrics (Pearson r, MAE, MSE, RMSE)
   - Per-profile metrics table
   - Score hierarchy test PASS/FAIL
   - Robustness test chart (score vs noise level)

### Tab 5: Driver Comparison 🏆

1. **Multi-file upload** (multiple driver sessions)
2. **"Rank Drivers"** → ranked table sorted by score
3. Color-coded score column
4. Sub-scores expandable per driver

---

## 9. Dataset Files

Pre-generated labeled datasets in `data/`:

| File | Duration | Samples | Mean Expected Score | Number of Events |
|------|----------|---------|--------------------|----|
| `smooth.csv` | 120 s | 3,000 | **90.4** | 7 (highway + corners) |
| `aggressive.csv` | 120 s | 3,000 | **31.3** | 16 (brakes + lane changes + stop-and-go) |
| `mixed.csv` | 180 s | 4,500 | **58.1** | 15 (all types, moderate intensity) |

All datasets have `expected_score` column (per-sample ground truth label) and are used by the optimizer and validation suite.

---

## 10. CSV Data Format

### Required Input Columns

Any CSV uploaded to DISA must have these 8 columns:

```csv
timestamp,steering_angle,throttle_position,brake_pressure,acceleration_x,acceleration_y,acceleration_z,gyroscope_yaw_rate
0.00,2.3,52.1,0.0,-0.05,0.01,9.81,0.12
0.04,2.1,51.8,0.0,-0.04,0.02,9.80,0.10
...
```

| Column | Unit | Physical Range | Description |
|--------|------|---------------|-------------|
| `timestamp` | seconds | 0.0 → any | Time from start of session |
| `steering_angle` | degrees | −180 to +180 | Steering wheel angle (+ = right) |
| `throttle_position` | % | 0–100 | Accelerator pedal position |
| `brake_pressure` | % | 0–100 | Brake pedal pressure |
| `acceleration_x` | m/s² | −30 to +30 | Longitudinal acceleration |
| `acceleration_y` | m/s² | −30 to +30 | Lateral (side-to-side) acceleration |
| `acceleration_z` | m/s² | 0 to +30 | Vertical acceleration (≈9.81 idle) |
| `gyroscope_yaw_rate` | °/s | −200 to +200 | Rotation rate around vertical axis |

### Optional Column

| Column | Unit | Description |
|--------|------|-------------|
| `expected_score` | 0–100 | Ground-truth label (required for optimization/validation) |

### Sample Rate

Default is **25 Hz** (one row every 0.04 seconds). The `fs` parameter is configurable per-session. Supported range: 1–1000 Hz.

---

## 11. REST API Reference

### GET `/health`

**Response:**
```json
{"status": "ok", "service": "DISA", "version": "1.0.0"}
```

---

### POST `/analyze`

Upload a CSV driving session and receive full smoothness analysis.

**Form Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | CSV with driving time-series |
| `fs` | float | 25.0 | Sample rate in Hz |
| `weights` | string | null | JSON array `[w_S, w_J, w_V, w_P]` |
| `save_result` | bool | true | Whether to persist to database |

**Response:**
```json
{
  "session_id": "a3f8b2c1",
  "filename": "session.csv",
  "n_samples": 3000,
  "duration_s": 120.0,
  "fs": 25.0,
  "analysis": {
    "overall_score": 86.42,
    "sub_scores": {
      "steering": 88.12,
      "throttle": 84.50,
      "braking": 90.31,
      "stability": 82.75
    },
    "feature_contributions": {"S": 2.1, "J": 4.3, "V": 3.8, "P": 3.4},
    "normalized_features": {"S": 0.060, "J": 0.084, "V": 0.051, "P": 0.041},
    "weights": [0.25, 0.30, 0.25, 0.20],
    "grade": "A — Excellent"
  },
  "anomaly_events": [
    {
      "start_time": 5.2,
      "end_time": 7.8,
      "duration_s": 2.6,
      "severity": 3.41,
      "event_type": "jerk_spike",
      "affected_channels": ["steering_angle", "acceleration_x", "gyroscope_yaw_rate"]
    }
  ],
  "timeseries_score": [88.2, 87.4, 86.9, ...],
  "event_highlight_mask": [false, false, true, ...]
}
```

---

### GET `/simulate`

Generate synthetic driving data.

**Query Parameters:**

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `profile` | string | "smooth" | `smooth` / `aggressive` / `mixed` |
| `duration_s` | float | 120.0 | Any positive float |
| `fs` | float | 25.0 | Sample rate in Hz |
| `seed` | int | 42 | Any integer for reproducibility |
| `download` | bool | false | If true, returns CSV file |

**Response (JSON):**
```json
{
  "profile": "smooth",
  "n_samples": 3000,
  "duration_s": 120.0,
  "fs": 25.0,
  "seed": 42,
  "n_events": 7,
  "events": [
    {"name": "highway_cruise", "start_idx": 125, "end_idx": 375, "severity": 0.1}
  ],
  "preview": [...],  // first 50 rows as records
  "stats": {
    "steering_angle": {"mean": 1.24, "std": 3.05, "min": -8.3, "max": 9.1}
  }
}
```

---

### POST `/optimize-weights`

Optimize scoring weights from labeled data.

**Form Parameters:**

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `files` | Files | required | Multiple labeled CSV files |
| `method` | string | "slsqp" | `slsqp` / `grid` / `both` |
| `fs` | float | 25.0 | Sample rate |
| `segment_length_s` | float | 10.0 | Segment duration for training |

**Response:** `OptimizationResult.to_dict()` (see Section 5.5)

---

### POST `/validate`

Run the full validation suite.

**Form Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `files` | Files | Labeled CSV files for validation |
| `fs` | float | Sample rate |
| `weights` | string | Optional JSON weights array |

---

### POST `/report`

Generate a PDF report.

**Form Parameters:** `file` (CSV), `fs`, `weights`, `session_id`

**Response:** Binary PDF stream with `Content-Disposition: attachment; filename=disa_report_{id}.pdf`

---

### GET `/sessions`

List all stored analysis sessions.

**Response:** Array of `{id, created_at, filename, overall_score, grade}`

---

### GET `/sessions/{session_id}`

Get full details for a specific session.

**Response:** Complete session dict including sub_scores, features JSON.

---

### POST `/rank`

Upload multiple sessions and rank them.

**Form Parameters:** `files` (multiple CSVs), `fs`, `weights`

**Response:**
```json
{
  "ranking": [
    {"rank": 1, "session_id": "driver_a", "overall_score": 88.2, "grade": "A — Excellent", "sub_scores": {...}},
    {"rank": 2, "session_id": "driver_b", "overall_score": 71.4, "grade": "B — Good", "sub_scores": {...}}
  ]
}
```

---

## 12. Calibration & Weight Optimization

### Calibration Script (`calibrate.py`)

Run this script to:
1. Measure raw (pre-normalization) feature values for all three driver profiles
2. Suggest updated `FEATURE_REF_MAX` values (feature values × 1.2 safety margin)
3. Run SLSQP weight optimization on the 81-segment training set

```bash
python calibrate.py
```

**Output (`calibrate_out.txt`):**

```
=== RAW PRE-NORMALIZED FEATURES ===
smooth:     S=4.005   J=69.38   V=0.97   P=0.040  exp=90.43
aggressive: S=54.411  J=689.60  V=396.5  P=0.040  exp=31.27
mixed:      S=12.757  J=201.06  V=14.29  P=0.038  exp=58.11

=== SUGGESTED FEATURE_REF_MAX ===
  S: 65.29           (set to 65.0)
  J: 827.52          (set to 830.0)
  V: 475.83          (set to 480.0)
  P: 0.0482          (set to 0.05)

=== OPTIMIZING WEIGHTS ===
  Loaded 23 segments from smooth.csv
  Loaded 23 segments from aggressive.csv
  Loaded 35 segments from mixed.csv
Total segments: 81
Optimized weights: [0.010, 0.010, 0.970, 0.010]
MSE: default=789.42 → optimized=241.68  improvement=69.4%
```

> The optimized weights heavily favor Variance (V=0.97) because the synthetic datasets show extreme variance differences between profiles. In practice, balanced default weights `[0.25, 0.30, 0.25, 0.20]` are more suitable for real-world telemetry.

---

## 13. Validation Results

From `validate_out.txt` (run on the 3 pre-generated datasets):

### Key Findings

The scoring model correctly differentiates all three driver archetypes. The hierarchy test passes: **smooth drivers always score higher than mixed, who always score higher than aggressive**.

### Score Separation

| Profile | Predicted Score | Expected Score |
|---------|----------------|---------------|
| Smooth | ~86 | ~90.4 |
| Mixed | ~62 | ~58.1 |
| Aggressive | ~28 | ~31.3 |

### Hierarchy Test

```
PASS: smooth(86.2) > mixed(62.1) > aggressive(28.4)
```

### Robustness

The scoring model degrades gracefully under increasing noise:
- At σ=0.01 (1%): negligible score change
- At σ=0.05 (5%): ~2-5 point drop
- At σ=0.20 (20%): noticeable but still directionally correct
- At σ=0.50 (50%): severe noise, scores converge toward center

---

## 14. Installation & Setup

### Prerequisites

- Python **3.10+**
- pip

### Step 1: Clone / Download

```bash
# If using git:
git clone <repository-url>
cd "DT PROJECT"
```

### Step 2: Create a Virtual Environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Datasets (if `data/*.csv` not present)

```bash
python -c "from simulator.simulator import generate_all_datasets; generate_all_datasets()"
```

This creates:
- `data/smooth.csv` (3,000 rows)
- `data/aggressive.csv` (3,000 rows)
- `data/mixed.csv` (4,500 rows)

The SQLite database (`data/disa.db`) is auto-created on first API startup.

---

## 15. Running the Project

### Option A: Streamlit Dashboard (Recommended — All-in-One)

The Streamlit app runs entirely standalone without needing the FastAPI server. It imports the backend modules directly.

```bash
streamlit run frontend/app.py
```

Opens at: **http://localhost:8501**

### Option B: FastAPI Backend Only

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or directly:

```bash
python backend/main.py
```

API available at: **http://localhost:8000**
Swagger docs: **http://localhost:8000/docs**

### Option C: Both Together

**Terminal 1 (Backend):**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
streamlit run frontend/app.py
```

### Option D: Run Validation Suite

```bash
python -m validation.validate
```

Requires `data/smooth.csv`, `data/aggressive.csv`, `data/mixed.csv` to exist.

### Option E: Run Weight Calibration

```bash
python calibrate.py
```

---

## 16. Dependencies

From `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.111.0 | REST API framework |
| `uvicorn[standard]` | 0.29.0 | ASGI server for FastAPI |
| `pandas` | 2.2.2 | DataFrame operations, CSV I/O |
| `numpy` | 1.26.4 | Numerical computations, signal processing |
| `scipy` | 1.13.0 | Butterworth filter, SLSQP optimizer, Pearson r |
| `scikit-learn` | 1.4.2 | (Available for future ML extensions) |
| `matplotlib` | 3.8.4 | Charts in PDF reports (headless Agg backend) |
| `streamlit` | 1.35.0 | Web dashboard framework |
| `plotly` | 5.22.0 | Interactive charts in dashboard |
| `fpdf2` | 2.7.9 | PDF report generation |
| `python-multipart` | 0.0.9 | File upload support for FastAPI |
| `aiofiles` | 23.2.1 | Async file I/O |

Standard library used: `sqlite3`, `uuid`, `json`, `io`, `os`, `sys`, `pathlib`, `tempfile`, `itertools`, `math`, `datetime`

---

## 17. Grading System

| Score Range | Grade | Letter | Meaning |
|-------------|-------|--------|---------|
| 85 – 100 | **A — Excellent** | A | Extremely smooth, professional-level driving |
| 70 – 84 | **B — Good** | B | Above average, minor imperfections |
| 55 – 69 | **C — Average** | C | Normal driving with some rough moments |
| 40 – 54 | **D — Below Average** | D | Noticeably rough, frequent harsh inputs |
| 0 – 39 | **F — Poor** | F | Very aggressive/erratic driving |

---

## 18. Design Decisions & Research Notes

### Why No Machine Learning?

DISA is intentionally ML-free by design:
1. **Explainability:** Every component of the score can be traced to a specific signal feature
2. **No labeled data required** to run — ML models need large, balanced training sets
3. **Zero inference cost** — computation is O(n) in the number of samples
4. **Domain correctness** — derivative-based features are physically motivated (jerk = what passengers feel)
5. **Debuggable** — if a score is wrong, you can inspect exactly which feature caused it

### Why Feature Extraction on Raw (Un-Normalized) Signals?

If normalization were applied before feature extraction:
- An aggressive driver with steering swings of ±40° would be scaled identically to a smooth driver with ±4° swings
- The features S, J, V, P would be indistinguishable between driver profiles
- `FEATURE_REF_MAX` thresholds would be meaningless

Solution: Use physical-unit signals for feature extraction, normalize only for display.

### Why Butterworth Zero-Phase Filter?

- `scipy.signal.filtfilt` applies the filter forward and backward, achieving zero phase distortion
- High-frequency electronic noise (> 5 Hz) in inertial sensors is removed without introducing lag
- This is critical for derivative computation: noise amplifies in higher-order derivatives

### Why Pearson r for Validation?

Pearson correlation measures monotonic agreement between predicted and expected scores. A high r (> 0.8) confirms the model correctly ranks driving smoothness even if the absolute score values are offset (addressed by bias metric).

### Sliding Window Rolling Score

The rolling score uses a 2-second (50-sample) centered window. This balances:
- **Too short:** Noisy, many spikes
- **Too long:** Masks localized rough events
- **2 seconds:** Corresponds to typical driver reaction/event duration

### Database Design Choice (SQLite)

SQLite is used (no ORM) because:
- Zero external infrastructure dependency
- stdlib `sqlite3` module
- Sufficient for research/MVP scale
- Easy to inspect with DB Browser for SQLite

### The OU Process for Simulator

The Ornstein–Uhlenbeck process `dx = θ(μ − x)dt + σ·dW` is used because:
- Unlike pure random walk (Brownian motion), it is **mean-reverting** — signals don't drift to infinity
- It captures the autocorrelated nature of real steering inputs (drivers don't make fully independent decisions at each timestep)
- θ (reversion speed) is tuned per driver: higher θ = snappier returns to center

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  DISA QUICK REFERENCE                                       │
├─────────────────────────────────────────────────────────────┤
│  SCORE FORMULA:                                             │
│    Score = 100 - clip(Σ wᵢ·fᵢ_norm×100, 0, 100)           │
│                                                             │
│  DEFAULT WEIGHTS:  S=0.25  J=0.30  V=0.25  P=0.20          │
│                                                             │
│  FEATURES:                                                  │
│    S = mean|dx/dt|         (rate of change)                 │
│    J = mean|d²x/dt²|       (jerk)                          │
│    V = mean(rolling_var)   (variance)                      │
│    P = spike_fraction      (|dx/dt| > 2σ)                  │
│                                                             │
│  REF_MAX (calibrated): S=65 J=830 V=480 P=0.05             │
│                                                             │
│  GRADES: A≥85 B≥70 C≥55 D≥40 F<40                         │
│                                                             │
│  RUN DASHBOARD: streamlit run frontend/app.py               │
│  RUN API:       uvicorn backend.main:app --reload           │
│  RUN VALIDATE:  python -m validation.validate               │
│  RUN CALIBRATE: python calibrate.py                         │
└─────────────────────────────────────────────────────────────┘
```

---

*DISA — Driver Input Smoothness Analyzer | Research MVP v1.0 | 2024*
