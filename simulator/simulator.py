"""
DISA — Synthetic Driving Simulator
===================================
Generates physically plausible, labeled time-series driving data for three
driver archetypes: smooth, aggressive, and mixed.

Signal generation philosophy
-----------------------------
Signals are NOT random noise. They use:
  1. Superposition of sinusoids at physiologically motivated frequencies
     (steering: 0.1–0.5 Hz, throttle modulation: 0.02–0.1 Hz)
  2. Ornstein–Uhlenbeck (OU) process for temporally-correlated drift
  3. Deterministic event injection (see event_injector.py)
  4. Layered noise (see noise_model.py)

Output columns
--------------
timestamp, steering_angle, throttle_position, brake_pressure,
acceleration_x, acceleration_y, acceleration_z,
gyroscope_yaw_rate, expected_score
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

from simulator.event_injector import (
    inject_sudden_brake,
    inject_lane_change,
    inject_stop_and_go,
    inject_highway_cruise,
    inject_mild_corner,
    Event,
)
from simulator.noise_model import apply_all_noise


# ---------------------------------------------------------------------------
# Ornstein–Uhlenbeck process
# ---------------------------------------------------------------------------

def _ou_process(
    n: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate an Ornstein–Uhlenbeck (mean-reverting) process.

    dx = theta*(mu - x)*dt + sigma*dW
    """
    x = np.empty(n)
    x[0] = x0
    sqrt_dt = math.sqrt(dt)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * sqrt_dt * rng.standard_normal()
    return x


# ---------------------------------------------------------------------------
# Base signal generators
# ---------------------------------------------------------------------------

def _steering_signal(
    n: int,
    fs: float,
    frequencies: List[float],
    amplitudes: List[float],
    ou_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Combine low-frequency sinusoids + OU drift for steering."""
    t = np.arange(n) / fs
    signal = np.zeros(n)
    for f, a in zip(frequencies, amplitudes):
        phase = rng.uniform(0, 2 * np.pi)
        signal += a * np.sin(2 * np.pi * f * t + phase)

    drift = _ou_process(n, theta=0.5, mu=0.0, sigma=ou_sigma, x0=0.0, dt=1 / fs, rng=rng)
    return signal + drift


def _throttle_signal(
    n: int,
    fs: float,
    base_throttle: float,
    modulation_amp: float,
    modulation_freq: float,
    ou_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Throttle = slow sinusoidal modulation around a base level + OU noise."""
    t = np.arange(n) / fs
    phase = rng.uniform(0, 2 * np.pi)
    signal = base_throttle + modulation_amp * np.sin(2 * np.pi * modulation_freq * t + phase)
    ou = _ou_process(n, theta=1.0, mu=0.0, sigma=ou_sigma, x0=0.0, dt=1 / fs, rng=rng)
    return np.clip(signal + ou, 0, 100)


def _brake_signal(n: int, rng: np.random.Generator, base_level: float = 0.5) -> np.ndarray:
    """Baseline brake is near zero (only events push it up)."""
    return np.clip(_ou_process(n, theta=2.0, mu=base_level, sigma=0.5, x0=base_level, dt=0.04, rng=rng), 0, 100)


def _acceleration_signal(
    n: int,
    fs: float,
    axis_freq: float,
    axis_amp: float,
    base_decel: float,
    ou_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(n) / fs
    phase = rng.uniform(0, 2 * np.pi)
    base = base_decel + axis_amp * np.sin(2 * np.pi * axis_freq * t + phase)
    ou = _ou_process(n, theta=1.5, mu=0.0, sigma=ou_sigma, x0=0.0, dt=1 / fs, rng=rng)
    return base + ou


def _gyro_yaw(steering: np.ndarray, fs: float, gain: float = 0.8) -> np.ndarray:
    """Yaw rate is approximately proportional to steering rate."""
    d_steer = np.gradient(steering, 1 / fs)
    return gain * d_steer


# ---------------------------------------------------------------------------
# Driver profile factories
# ---------------------------------------------------------------------------

def _build_empty_signals(n: int) -> dict:
    return {
        "steering_angle": np.zeros(n),
        "throttle_position": np.zeros(n),
        "brake_pressure": np.zeros(n),
        "acceleration_x": np.zeros(n),
        "acceleration_y": np.zeros(n),
        "acceleration_z": np.zeros(n),
        "gyroscope_yaw_rate": np.zeros(n),
    }


def generate_smooth_driver(
    duration_s: float = 120.0,
    fs: float = 25.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Event]]:
    """
    Smooth driver profile:
      - Very low-frequency, low-amplitude steering (0.1 Hz, ±8°)
      - Steady throttle 45–65%, slow modulation at 0.03 Hz
      - Almost no braking spikes
      - Injected events: highway cruises, mild corners only
      - Expected score range: 80–95
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    signals = _build_empty_signals(n)

    signals["steering_angle"] = _steering_signal(
        n, fs,
        frequencies=[0.08, 0.15, 0.25],
        amplitudes=[4.0, 2.0, 1.0],
        ou_sigma=0.8,
        rng=rng,
    )
    signals["throttle_position"] = _throttle_signal(
        n, fs, base_throttle=52.0, modulation_amp=6.0,
        modulation_freq=0.03, ou_sigma=1.0, rng=rng,
    )
    signals["brake_pressure"] = _brake_signal(n, rng, base_level=0.3)
    signals["acceleration_x"] = _acceleration_signal(n, fs, 0.05, 0.3, -0.1, 0.2, rng)
    signals["acceleration_y"] = _acceleration_signal(n, fs, 0.08, 0.15, 0.0, 0.15, rng)
    signals["acceleration_z"] = _acceleration_signal(n, fs, 0.12, 0.2, 9.81, 0.1, rng)

    events: List[Event] = []

    # Inject highway cruise segments
    for t_start in [5, 30, 70, 100]:
        idx = int(t_start * fs)
        if idx < n:
            e = inject_highway_cruise(signals, idx, fs, duration_s=10.0, rng=rng)
            events.append(e)

    # Inject a few mild corners
    for t_start in [20, 50, 90]:
        idx = int(t_start * fs)
        if idx < n:
            direction = rng.choice([-1, 1])
            e = inject_mild_corner(signals, idx, fs, intensity=0.4, direction=direction, rng=rng)
            events.append(e)

    # Add noise
    for ch in signals:
        signals[ch] = apply_all_noise(
            signals[ch], fs,
            sensor_sigma_frac=0.003,
            bump_rate_hz=0.1,
            bump_amplitude_frac=0.03,
            delay_samples=0.2,
            rng=rng,
        )

    signals["gyroscope_yaw_rate"] = _gyro_yaw(signals["steering_angle"], fs, gain=0.6)

    # Score label: smooth drivers score 82–92
    timestamp = np.arange(n) / fs
    score_base = rng.uniform(84, 92)
    # Add slight drift over the session
    score_drift = 3.0 * np.sin(2 * np.pi * np.arange(n) / n)
    expected_score = np.clip(score_base + score_drift + rng.normal(0, 0.5, n), 75, 98)

    df = pd.DataFrame(signals)
    df.insert(0, "timestamp", timestamp)
    df["expected_score"] = expected_score

    return df, events


def generate_aggressive_driver(
    duration_s: float = 120.0,
    fs: float = 25.0,
    seed: int = 7,
) -> Tuple[pd.DataFrame, List[Event]]:
    """
    Aggressive driver profile:
      - High-frequency, high-amplitude steering (0.4–1.2 Hz, ±35°)
      - Rapid throttle transients
      - Frequent hard braking events
      - Injected: sudden brakes, lane changes, stop-and-go
      - Expected score range: 20–45
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    signals = _build_empty_signals(n)

    signals["steering_angle"] = _steering_signal(
        n, fs,
        frequencies=[0.4, 0.7, 1.1],
        amplitudes=[18.0, 10.0, 6.0],
        ou_sigma=4.0,
        rng=rng,
    )
    signals["throttle_position"] = _throttle_signal(
        n, fs, base_throttle=60.0, modulation_amp=25.0,
        modulation_freq=0.15, ou_sigma=6.0, rng=rng,
    )
    signals["brake_pressure"] = _brake_signal(n, rng, base_level=2.0)
    signals["acceleration_x"] = _acceleration_signal(n, fs, 0.3, 1.5, -0.3, 0.8, rng)
    signals["acceleration_y"] = _acceleration_signal(n, fs, 0.4, 0.8, 0.0, 0.5, rng)
    signals["acceleration_z"] = _acceleration_signal(n, fs, 0.5, 0.6, 9.81, 0.4, rng)

    events: List[Event] = []

    # Frequent sudden brakes
    for t_start in [8, 20, 35, 50, 70, 90, 110]:
        idx = int(t_start * fs)
        if idx < n:
            e = inject_sudden_brake(signals, idx, fs, intensity=rng.uniform(0.8, 1.0), rng=rng)
            events.append(e)

    # Lane changes
    for t_start in [12, 28, 45, 65, 85, 105]:
        idx = int(t_start * fs)
        if idx < n:
            direction = rng.choice([-1, 1])
            e = inject_lane_change(signals, idx, fs, intensity=rng.uniform(0.8, 1.0),
                                   direction=direction, rng=rng)
            events.append(e)

    # Stop-and-go sequences
    for t_start in [40, 75, 100]:
        idx = int(t_start * fs)
        if idx < n:
            e = inject_stop_and_go(signals, idx, fs, intensity=rng.uniform(0.7, 1.0), rng=rng)
            events.append(e)

    # Heavy noise (aggressive drivers often drive on rough roads)
    for ch in signals:
        signals[ch] = apply_all_noise(
            signals[ch], fs,
            sensor_sigma_frac=0.008,
            bump_rate_hz=0.6,
            bump_amplitude_frac=0.10,
            delay_samples=0.4,
            rng=rng,
        )

    signals["gyroscope_yaw_rate"] = _gyro_yaw(signals["steering_angle"], fs, gain=1.0)

    timestamp = np.arange(n) / fs
    score_base = rng.uniform(22, 38)
    score_drift = 8.0 * np.sin(2 * np.pi * np.arange(n) / n * 3)
    expected_score = np.clip(score_base + score_drift + rng.normal(0, 1.5, n), 10, 50)

    df = pd.DataFrame(signals)
    df.insert(0, "timestamp", timestamp)
    df["expected_score"] = expected_score

    return df, events


def generate_mixed_driver(
    duration_s: float = 180.0,
    fs: float = 25.0,
    seed: int = 99,
) -> Tuple[pd.DataFrame, List[Event]]:
    """
    Normal/mixed driver profile:
      - Moderate steering variability
      - Occasional spirited driving interspersed with calm periods
      - Expected score range: 50–70
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    signals = _build_empty_signals(n)

    signals["steering_angle"] = _steering_signal(
        n, fs,
        frequencies=[0.15, 0.35, 0.6],
        amplitudes=[10.0, 5.0, 2.5],
        ou_sigma=2.0,
        rng=rng,
    )
    signals["throttle_position"] = _throttle_signal(
        n, fs, base_throttle=48.0, modulation_amp=14.0,
        modulation_freq=0.07, ou_sigma=3.0, rng=rng,
    )
    signals["brake_pressure"] = _brake_signal(n, rng, base_level=1.0)
    signals["acceleration_x"] = _acceleration_signal(n, fs, 0.15, 0.7, -0.15, 0.4, rng)
    signals["acceleration_y"] = _acceleration_signal(n, fs, 0.2, 0.4, 0.0, 0.25, rng)
    signals["acceleration_z"] = _acceleration_signal(n, fs, 0.25, 0.35, 9.81, 0.2, rng)

    events: List[Event] = []

    # Mix of mild and moderate events
    for t_start in [15, 45, 90, 140]:
        idx = int(t_start * fs)
        if idx < n:
            e = inject_sudden_brake(signals, idx, fs, intensity=rng.uniform(0.4, 0.7), rng=rng)
            events.append(e)

    for t_start in [25, 60, 100, 155]:
        idx = int(t_start * fs)
        if idx < n:
            direction = rng.choice([-1, 1])
            e = inject_lane_change(signals, idx, fs, intensity=rng.uniform(0.4, 0.7),
                                   direction=direction, rng=rng)
            events.append(e)

    for t_start in [10, 70, 120]:
        idx = int(t_start * fs)
        if idx < n:
            e = inject_highway_cruise(signals, idx, fs, duration_s=8.0, rng=rng)
            events.append(e)

    for t_start in [35, 80, 130]:
        idx = int(t_start * fs)
        if idx < n:
            direction = rng.choice([-1, 1])
            e = inject_mild_corner(signals, idx, fs, intensity=0.6, direction=direction, rng=rng)
            events.append(e)

    for t_start in [50, 110, 160]:
        idx = int(t_start * fs)
        if idx < n:
            e = inject_stop_and_go(signals, idx, fs, intensity=rng.uniform(0.4, 0.6), rng=rng)
            events.append(e)

    for ch in signals:
        signals[ch] = apply_all_noise(
            signals[ch], fs,
            sensor_sigma_frac=0.005,
            bump_rate_hz=0.3,
            bump_amplitude_frac=0.07,
            delay_samples=0.3,
            rng=rng,
        )

    signals["gyroscope_yaw_rate"] = _gyro_yaw(signals["steering_angle"], fs, gain=0.8)

    timestamp = np.arange(n) / fs
    score_base = rng.uniform(52, 65)
    score_drift = 10.0 * np.sin(2 * np.pi * np.arange(n) / n * 2)
    expected_score = np.clip(score_base + score_drift + rng.normal(0, 1.0, n), 35, 80)

    df = pd.DataFrame(signals)
    df.insert(0, "timestamp", timestamp)
    df["expected_score"] = expected_score

    return df, events


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def generate_all_datasets(output_dir: str = "data", fs: float = 25.0):
    """Generate and save all three labeled datasets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating smooth driver dataset...")
    df_smooth, ev_smooth = generate_smooth_driver(duration_s=120, fs=fs, seed=42)
    df_smooth.to_csv(out / "smooth.csv", index=False)
    print(f"  → {len(df_smooth)} samples, {len(ev_smooth)} events | "
          f"mean expected score: {df_smooth['expected_score'].mean():.1f}")

    print("Generating aggressive driver dataset...")
    df_agg, ev_agg = generate_aggressive_driver(duration_s=120, fs=fs, seed=7)
    df_agg.to_csv(out / "aggressive.csv", index=False)
    print(f"  → {len(df_agg)} samples, {len(ev_agg)} events | "
          f"mean expected score: {df_agg['expected_score'].mean():.1f}")

    print("Generating mixed driver dataset...")
    df_mix, ev_mix = generate_mixed_driver(duration_s=180, fs=fs, seed=99)
    df_mix.to_csv(out / "mixed.csv", index=False)
    print(f"  → {len(df_mix)} samples, {len(ev_mix)} events | "
          f"mean expected score: {df_mix['expected_score'].mean():.1f}")

    print(f"\nDatasets saved to: {out.resolve()}")
    return df_smooth, df_agg, df_mix


if __name__ == "__main__":
    generate_all_datasets()
