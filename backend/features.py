"""
DISA — Feature Engineering
============================
Extracts four smoothness-relevant features from each signal channel:

  S  — Rate of change (first derivative magnitude)
  J  — Jerk magnitude (second derivative magnitude)
  V  — Sliding window variance (temporal roughness)
  P  — Spike frequency (fraction of samples exceeding 2σ threshold)

All features are computed on pre-processed (normalized) signals and
returned both per-channel and as aggregate scalars for scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Feature extraction constants
# ---------------------------------------------------------------------------

# Channel groups for sub-score computation
CHANNEL_GROUPS = {
    "steering":  ["steering_angle"],
    "throttle":  ["throttle_position"],
    "braking":   ["brake_pressure"],
    "stability": ["acceleration_x", "acceleration_y", "acceleration_z", "gyroscope_yaw_rate"],
}

ALL_SIGNAL_COLS = [
    "steering_angle",
    "throttle_position",
    "brake_pressure",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_yaw_rate",
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChannelFeatures:
    """Feature vector for a single signal channel."""
    channel: str
    S: float       # mean absolute rate of change
    J: float       # mean absolute jerk
    V: float       # mean sliding-window variance
    P: float       # spike frequency (0–1)
    S_series: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    J_series: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    V_series: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    spike_mask: np.ndarray = field(repr=False, default_factory=lambda: np.array([], dtype=bool))


@dataclass
class FeatureMatrix:
    """All features extracted from a driving session."""
    per_channel: Dict[str, ChannelFeatures]
    group_aggregates: Dict[str, Dict[str, float]]   # group → {S, J, V, P}
    aggregate: Dict[str, float]                      # overall {S, J, V, P}
    fs: float


# ---------------------------------------------------------------------------
# Per-channel feature computation
# ---------------------------------------------------------------------------

def _compute_channel_features(
    x: np.ndarray,
    fs: float,
    window_s: float = 1.0,
    spike_sigma_mult: float = 2.0,
) -> tuple:
    """
    Compute S, J, V, P features for a 1-D signal array.

    Parameters
    ----------
    x              : 1-D normalized signal
    fs             : sample rate (Hz)
    window_s       : sliding window duration for variance (seconds)
    spike_sigma_mult : multiplier for spike detection threshold

    Returns
    -------
    (S, J, V, P, dx, ddx, var_series, spike_mask)
    """
    dt = 1.0 / fs
    n = len(x)

    # --- Rate of change (first derivative via central differences) ---
    dx = np.gradient(x, dt)          # shape (n,)
    S = float(np.mean(np.abs(dx)))

    # --- Jerk (second derivative) ---
    ddx = np.gradient(dx, dt)
    J = float(np.mean(np.abs(ddx)))

    # --- Sliding window variance ---
    window_samples = max(int(window_s * fs), 2)
    series = pd.Series(x)
    var_series = series.rolling(window=window_samples, center=True, min_periods=2).std().fillna(0).values ** 2
    V = float(np.mean(var_series))

    # --- Spike detection (|dx/dt| > spike_sigma_mult * σ(dx)) ---
    dx_std = np.std(dx)
    threshold = spike_sigma_mult * dx_std if dx_std > 0 else 1e-6
    spike_mask = np.abs(dx) > threshold
    P = float(np.mean(spike_mask))   # fraction of samples that are spikes

    return S, J, V, P, dx, ddx, var_series, spike_mask


def extract_channel_features(
    x: np.ndarray,
    channel_name: str,
    fs: float,
    window_s: float = 1.0,
    spike_sigma_mult: float = 2.0,
) -> ChannelFeatures:
    """Extract and package all features for a single channel."""
    S, J, V, P, dx, ddx, var_series, spike_mask = _compute_channel_features(
        x, fs, window_s=window_s, spike_sigma_mult=spike_sigma_mult
    )
    return ChannelFeatures(
        channel=channel_name,
        S=S, J=J, V=V, P=P,
        S_series=np.abs(dx),
        J_series=np.abs(ddx),
        V_series=var_series,
        spike_mask=spike_mask,
    )


# ---------------------------------------------------------------------------
# Multi-channel feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    fs: float = 25.0,
    window_s: float = 1.0,
    spike_sigma_mult: float = 2.0,
) -> FeatureMatrix:
    """
    Extract the full FeatureMatrix from a pre-processed DataFrame.

    Parameters
    ----------
    df               : pre-processed, normalized DataFrame
    fs               : sample rate (Hz)
    window_s         : variance window duration
    spike_sigma_mult : spike detection threshold multiplier

    Returns
    -------
    FeatureMatrix with per-channel, group, and aggregate features
    """
    per_channel: Dict[str, ChannelFeatures] = {}

    for col in ALL_SIGNAL_COLS:
        if col in df.columns:
            x = df[col].values.astype(float)
            cf = extract_channel_features(x, col, fs, window_s, spike_sigma_mult)
            per_channel[col] = cf

    # --- Group aggregates (mean of channels in each group) ---
    group_aggregates: Dict[str, Dict[str, float]] = {}
    for group, channels in CHANNEL_GROUPS.items():
        group_cfs = [per_channel[c] for c in channels if c in per_channel]
        if not group_cfs:
            group_aggregates[group] = {"S": 0.0, "J": 0.0, "V": 0.0, "P": 0.0}
        else:
            group_aggregates[group] = {
                "S": float(np.mean([cf.S for cf in group_cfs])),
                "J": float(np.mean([cf.J for cf in group_cfs])),
                "V": float(np.mean([cf.V for cf in group_cfs])),
                "P": float(np.mean([cf.P for cf in group_cfs])),
            }

    # --- Overall aggregate ---
    all_cfs = list(per_channel.values())
    aggregate = {
        "S": float(np.mean([cf.S for cf in all_cfs])),
        "J": float(np.mean([cf.J for cf in all_cfs])),
        "V": float(np.mean([cf.V for cf in all_cfs])),
        "P": float(np.mean([cf.P for cf in all_cfs])),
    }

    return FeatureMatrix(
        per_channel=per_channel,
        group_aggregates=group_aggregates,
        aggregate=aggregate,
        fs=fs,
    )


# ---------------------------------------------------------------------------
# Feature normalization (for scoring)
# ---------------------------------------------------------------------------

def normalize_feature_vector(
    features: Dict[str, float],
    ref_max: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Normalize a feature dict {S, J, V, P} to [0, 1] using reference maxima.
    If ref_max is None, clamp to fixed domain maxima.

    Fixed domain maxima (empirically calibrated from aggressive driver
    raw pre-normalization signals at 25 Hz):
      S_max = 65.0  (mean |dx/dt| °/s or %/s per channel)
      J_max = 830.0 (mean |d²x/dt²|)
      V_max = 480.0 (rolling variance of raw signals)
      P_max = 0.05  (5% spike fraction = very aggressive)
    """
    defaults = {"S": 65.0, "J": 830.0, "V": 480.0, "P": 0.05}
    maxima = ref_max or defaults

    normalized = {}
    for k, v in features.items():
        denom = maxima.get(k, 1.0)
        normalized[k] = float(np.clip(v / denom, 0.0, 1.0)) if denom > 0 else 0.0

    return normalized


def get_event_highlight_mask(fm: FeatureMatrix, top_n_channels: int = 3) -> np.ndarray:
    """
    Return a boolean mask indicating 'rough' timesteps across all channels.
    A timestep is rough if it is a spike in any top-variance channel.
    """
    # Rank channels by mean jerk (most impactful first)
    ranked = sorted(
        [(ch, cf) for ch, cf in fm.per_channel.items()],
        key=lambda x: x[1].J,
        reverse=True,
    )[:top_n_channels]

    if not ranked:
        return np.array([], dtype=bool)

    n = len(ranked[0][1].spike_mask)
    mask = np.zeros(n, dtype=bool)
    for _, cf in ranked:
        if len(cf.spike_mask) == n:
            mask |= cf.spike_mask
    return mask
