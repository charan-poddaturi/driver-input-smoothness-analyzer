"""
DISA — Anomaly Detection
=========================
Detects rough driving events using a sliding-window Z-score approach
on the jerk signal derived from multiple sensor channels.

No ML is used — all detection is threshold-based and interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.features import ALL_SIGNAL_COLS


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyEvent:
    """A detected rough driving event."""
    start_time: float      # seconds
    end_time: float        # seconds
    start_idx: int
    end_idx: int
    severity: float        # Z-score magnitude
    event_type: str        # 'jerk_spike' | 'variance_burst' | 'brake_spike'
    affected_channels: List[str]

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_s": round(self.end_time - self.start_time, 3),
            "severity": round(self.severity, 3),
            "event_type": self.event_type,
            "affected_channels": self.affected_channels,
        }


# ---------------------------------------------------------------------------
# Z-score anomaly detector
# ---------------------------------------------------------------------------

def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling Z-score for each sample in the series.
    Z[i] = (x[i] - rolling_mean[i]) / rolling_std[i]
    """
    s = pd.Series(x)
    roll_mean = s.rolling(window=window, center=True, min_periods=2).mean()
    roll_std = s.rolling(window=window, center=True, min_periods=2).std()
    roll_std = roll_std.replace(0.0, np.nan).fillna(1e-8)
    z = (s - roll_mean) / roll_std
    return z.fillna(0.0).values


def _merge_adjacent_events(
    indices: np.ndarray,
    timestamps: np.ndarray,
    fs: float,
    min_gap_s: float = 0.5,
) -> List[tuple]:
    """Merge adjacent/overlapping anomalous regions into single events."""
    if len(indices) == 0:
        return []

    min_gap = int(min_gap_s * fs)
    events = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx - prev > min_gap:
            events.append((start, prev))
            start = idx
        prev = idx
    events.append((start, prev))
    return events


# ---------------------------------------------------------------------------
# Jerk spike detector
# ---------------------------------------------------------------------------

def detect_jerk_spikes(
    df: pd.DataFrame,
    fs: float = 25.0,
    z_threshold: float = 3.0,
    window_s: float = 3.0,
    channels: Optional[List[str]] = None,
) -> List[AnomalyEvent]:
    """
    Detect timesteps where jerk (second derivative) exceeds Z-score threshold.

    Parameters
    ----------
    df          : processed DataFrame (not necessarily normalized)
    fs          : sample rate
    z_threshold : Z-score cutoff for anomaly
    window_s    : rolling window for Z-score baseline
    channels    : list of channels to check; None = all available

    Returns
    -------
    List of AnomalyEvent
    """
    if channels is None:
        channels = [c for c in ALL_SIGNAL_COLS if c in df.columns]

    window = max(int(window_s * fs), 10)
    dt = 1.0 / fs
    timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df)) / fs

    # Compute jerk for each channel
    channel_jerks: Dict[str, np.ndarray] = {}
    for ch in channels:
        x = df[ch].values.astype(float)
        dx = np.gradient(x, dt)
        ddx = np.gradient(dx, dt)
        channel_jerks[ch] = np.abs(ddx)

    # Aggregate jerk (mean across channels)
    all_jerks = np.column_stack(list(channel_jerks.values()))
    agg_jerk = np.mean(all_jerks, axis=1)

    z_scores = _rolling_zscore(agg_jerk, window)
    anomaly_mask = np.abs(z_scores) > z_threshold
    anomaly_indices = np.where(anomaly_mask)[0]

    raw_events = _merge_adjacent_events(anomaly_indices, timestamps, fs)

    events = []
    for (start, end) in raw_events:
        # Identify which channels are most affected
        event_z = {ch: float(np.mean(np.abs(_rolling_zscore(channel_jerks[ch], window)[start:end + 1])))
                   for ch in channels}
        top_channels = sorted(event_z, key=event_z.get, reverse=True)[:3]
        severity = float(np.mean(np.abs(z_scores[start:end + 1])))

        events.append(AnomalyEvent(
            start_time=float(timestamps[start]),
            end_time=float(timestamps[min(end, len(timestamps) - 1)]),
            start_idx=int(start),
            end_idx=int(end),
            severity=severity,
            event_type="jerk_spike",
            affected_channels=top_channels,
        ))

    return events


# ---------------------------------------------------------------------------
# Brake spike detector
# ---------------------------------------------------------------------------

def detect_brake_spikes(
    df: pd.DataFrame,
    fs: float = 25.0,
    threshold_pct: float = 30.0,
    min_duration_s: float = 0.2,
) -> List[AnomalyEvent]:
    """
    Detect abrupt brake applications (brake_pressure exceeds threshold rapidly).
    """
    if "brake_pressure" not in df.columns:
        return []

    timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df)) / fs
    brake = df["brake_pressure"].values.astype(float)
    dt = 1.0 / fs
    d_brake = np.gradient(brake, dt)

    # Brake spike: rate of brake application > threshold
    min_dur_samples = int(min_duration_s * fs)
    spike_mask = (d_brake > threshold_pct) | (brake > 50.0)

    indices = np.where(spike_mask)[0]
    if len(indices) == 0:
        return []

    raw_events = _merge_adjacent_events(indices, timestamps, fs, min_gap_s=0.3)

    events = []
    for (start, end) in raw_events:
        duration = (end - start) / fs
        if duration < min_duration_s:
            continue
        peak_brake = float(np.max(brake[start:end + 1]))
        severity = min(peak_brake / 100.0 * 5, 5.0)   # scale to Z-score equivalent
        events.append(AnomalyEvent(
            start_time=float(timestamps[start]),
            end_time=float(timestamps[min(end, len(timestamps) - 1)]),
            start_idx=int(start),
            end_idx=int(end),
            severity=round(severity, 2),
            event_type="brake_spike",
            affected_channels=["brake_pressure"],
        ))

    return events


# ---------------------------------------------------------------------------
# Variance burst detector
# ---------------------------------------------------------------------------

def detect_variance_bursts(
    df: pd.DataFrame,
    fs: float = 25.0,
    window_s: float = 2.0,
    z_threshold: float = 2.5,
) -> List[AnomalyEvent]:
    """
    Detect rolling variance bursts in steering and acceleration channels
    (indicates erratic/unsettled driving).
    """
    channels = ["steering_angle", "acceleration_x", "acceleration_y"]
    channels = [c for c in channels if c in df.columns]
    if not channels:
        return []

    timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df)) / fs
    window = max(int(window_s * fs), 5)

    var_signals = []
    for ch in channels:
        x = df[ch].values.astype(float)
        s = pd.Series(x)
        std_series = s.rolling(window=window, center=True, min_periods=2).std().fillna(0).values
        var_signals.append(std_series)

    agg_var = np.mean(var_signals, axis=0)
    z = _rolling_zscore(agg_var, window * 3)
    burst_mask = np.abs(z) > z_threshold

    indices = np.where(burst_mask)[0]
    raw_events = _merge_adjacent_events(indices, timestamps, fs, min_gap_s=1.0)

    events = []
    for (start, end) in raw_events:
        severity = float(np.mean(np.abs(z[start:end + 1])))
        events.append(AnomalyEvent(
            start_time=float(timestamps[start]),
            end_time=float(timestamps[min(end, len(timestamps) - 1)]),
            start_idx=int(start),
            end_idx=int(end),
            severity=round(severity, 2),
            event_type="variance_burst",
            affected_channels=channels,
        ))

    return events


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def detect_all_anomalies(
    df: pd.DataFrame,
    fs: float = 25.0,
    jerk_z_threshold: float = 3.0,
    brake_threshold_pct: float = 30.0,
    variance_z_threshold: float = 2.5,
) -> List[AnomalyEvent]:
    """
    Run all anomaly detectors and return a merged, time-sorted event list.
    """
    events = []
    events.extend(detect_jerk_spikes(df, fs=fs, z_threshold=jerk_z_threshold))
    events.extend(detect_brake_spikes(df, fs=fs, threshold_pct=brake_threshold_pct))
    events.extend(detect_variance_bursts(df, fs=fs, z_threshold=variance_z_threshold))

    # Sort by start time
    events.sort(key=lambda e: e.start_time)
    return events
