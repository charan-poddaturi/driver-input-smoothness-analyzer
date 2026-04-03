"""
DISA — Data Preprocessing Pipeline
====================================
Stages:
  1. Schema validation and missing value handling
  2. Low-pass Butterworth filter (removes sensor artifacts above 5 Hz)
  3. Optional moving-average smoothing
  4. Per-signal robust min-max normalization

All stages are stateless functions that operate on DataFrames,
allowing individual stages to be tested in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_COLUMNS = [
    "steering_angle",
    "throttle_position",
    "brake_pressure",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_yaw_rate",
]

# Physical plausibility bounds (for clipping after preprocessing)
SIGNAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "steering_angle":     (-180.0, 180.0),
    "throttle_position":  (0.0,    100.0),
    "brake_pressure":     (0.0,    100.0),
    "acceleration_x":     (-30.0,  30.0),
    "acceleration_y":     (-30.0,  30.0),
    "acceleration_z":     (0.0,    30.0),
    "gyroscope_yaw_rate": (-200.0, 200.0),
}


# ---------------------------------------------------------------------------
# Stage 1: Missing value handling
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in signal columns:
      - Forward-fill first (preserves last known value)
      - Linear interpolation for interior NaNs
      - Zero-fill any remaining (e.g., leading NaNs)
    """
    df = df.copy()
    for col in SIGNAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            continue
        df[col] = df[col].ffill()
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
        df[col] = df[col].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Stage 2: Low-pass Butterworth filter
# ---------------------------------------------------------------------------

def _design_butterworth(cutoff_hz: float, fs: float, order: int = 4) -> Tuple:
    """Return IIR filter coefficients (b, a) for a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    # Clamp to valid range
    normal_cutoff = np.clip(normal_cutoff, 1e-4, 0.999)
    b, a = sp_signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_lowpass_filter(
    df: pd.DataFrame,
    fs: float = 25.0,
    cutoff_hz: float = 5.0,
    order: int = 4,
) -> pd.DataFrame:
    """
    Apply zero-phase Butterworth low-pass filter to all signal columns.
    Uses filtfilt for zero phase distortion.
    """
    df = df.copy()
    b, a = _design_butterworth(cutoff_hz=cutoff_hz, fs=fs, order=order)
    for col in SIGNAL_COLUMNS:
        if col in df.columns and len(df[col]) > order * 3:
            df[col] = sp_signal.filtfilt(b, a, df[col].values)
    return df


# ---------------------------------------------------------------------------
# Stage 3: Moving average (optional fallback / extra smoothing)
# ---------------------------------------------------------------------------

def apply_moving_average(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Apply a centered moving-average with min_periods=1 to each signal column."""
    df = df.copy()
    for col in SIGNAL_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )
    return df


# ---------------------------------------------------------------------------
# Stage 4: Normalization
# ---------------------------------------------------------------------------

def compute_normalization_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute robust normalization stats (1st / 99th percentile as min/max)
    to be immune to outlier spikes.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in SIGNAL_COLUMNS:
        if col in df.columns:
            p1 = float(np.percentile(df[col].dropna(), 1))
            p99 = float(np.percentile(df[col].dropna(), 99))
            if p99 == p1:
                p99 = p1 + 1.0   # avoid division by zero
            stats[col] = {"min": p1, "max": p99}
    return stats


def apply_normalization(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Apply per-column robust min-max normalization to [0, 1].

    Parameters
    ----------
    df    : input DataFrame
    stats : optional pre-computed stats (for applying training stats to test data)

    Returns
    -------
    (normalized_df, stats)
    """
    df = df.copy()
    if stats is None:
        stats = compute_normalization_stats(df)

    for col, s in stats.items():
        if col in df.columns:
            rng = s["max"] - s["min"]
            df[col] = (df[col] - s["min"]) / rng
            df[col] = df[col].clip(0.0, 1.0)

    return df, stats


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    fs: float = 25.0,
    cutoff_hz: float = 5.0,
    apply_ma: bool = False,
    ma_window: int = 5,
    normalize: bool = True,
    norm_stats: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    df          : raw input DataFrame (must have timestamp column)
    fs          : sample rate (Hz)
    cutoff_hz   : Butterworth cutoff frequency
    apply_ma    : whether to also apply moving average after filter
    ma_window   : moving average window size
    normalize   : whether to normalize signals to [0, 1]
    norm_stats  : pre-computed stats; if None, compute from data

    Returns
    -------
    (processed_df, norm_stats)
    """
    df = handle_missing_values(df)
    df = apply_lowpass_filter(df, fs=fs, cutoff_hz=cutoff_hz)
    if apply_ma:
        df = apply_moving_average(df, window=ma_window)

    # Physical clipping after filtering
    for col, (lo, hi) in SIGNAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    stats = {}
    if normalize:
        df, stats = apply_normalization(df, stats=norm_stats)

    return df, stats


def preprocess_for_features(
    df: pd.DataFrame,
    fs: float = 25.0,
    cutoff_hz: float = 5.0,
) -> pd.DataFrame:
    """
    Return a filtered (but NOT normalized) DataFrame for feature extraction.

    Feature magnitudes must be preserved in physical units so that:
      - aggressive driver has S >> smooth driver (not squashed to same range)
      - FEATURE_REF_MAX thresholds are meaningful

    Use this for extract_features(); use preprocess() for display/visualization.
    """
    df = handle_missing_values(df)
    df = apply_lowpass_filter(df, fs=fs, cutoff_hz=cutoff_hz)
    for col, (lo, hi) in SIGNAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df
