"""
DISA — Noise Model
==================
Adds physically motivated disturbances to clean synthetic signals:
  - Road bumps: high-frequency burst noise at random intervals
  - Sensor noise: additive Gaussian (σ ≈ 0.5 % of signal range)
  - Signal delay: fractional-sample first-order hold approximation
"""

import numpy as np
from scipy import signal as sp_signal


def add_sensor_noise(data: np.ndarray, sigma_frac: float = 0.005, rng: np.random.Generator = None) -> np.ndarray:
    """
    Add white Gaussian sensor noise.

    Parameters
    ----------
    data        : 1-D signal array
    sigma_frac  : noise σ as fraction of peak-to-peak range
    rng         : numpy random Generator for reproducibility

    Returns
    -------
    Noisy signal (same shape).
    """
    if rng is None:
        rng = np.random.default_rng()
    peak_to_peak = np.ptp(data) if np.ptp(data) > 0 else 1.0
    sigma = sigma_frac * peak_to_peak
    return data + rng.normal(0.0, sigma, size=data.shape)


def add_road_bumps(
    data: np.ndarray,
    fs: float,
    bump_rate_hz: float = 0.3,
    bump_amplitude_frac: float = 0.08,
    bump_duration_s: float = 0.15,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Inject high-frequency road-bump bursts at random times.

    A bump is modelled as a damped half-sine pulse applied primarily to
    the z-acceleration channel but passed through here for any channel.

    Parameters
    ----------
    data              : 1-D signal array
    fs                : sample rate (Hz)
    bump_rate_hz      : average bumps per second (Poisson process)
    bump_amplitude_frac : bump amplitude as fraction of peak-to-peak
    bump_duration_s   : duration of each bump burst (s)
    rng               : random Generator

    Returns
    -------
    Signal with bumps injected.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(data)
    duration_s = n / fs
    peak_to_peak = np.ptp(data) if np.ptp(data) > 0 else 1.0
    amplitude = bump_amplitude_frac * peak_to_peak

    out = data.copy()

    # Expected number of bumps over the recording
    n_bumps = rng.poisson(bump_rate_hz * duration_s)
    bump_samples = int(bump_duration_s * fs)

    for _ in range(n_bumps):
        start_idx = rng.integers(0, max(1, n - bump_samples))
        end_idx = min(start_idx + bump_samples, n)
        t_bump = np.linspace(0, np.pi, end_idx - start_idx)
        pulse = amplitude * np.sin(t_bump) * rng.choice([-1, 1])
        out[start_idx:end_idx] += pulse

    return out


def add_signal_delay(data: np.ndarray, delay_samples: float = 0.5) -> np.ndarray:
    """
    Simulate fractional-sample signal delay using linear interpolation.

    Parameters
    ----------
    data          : 1-D signal
    delay_samples : fractional number of samples to delay
    rng           : random Generator

    Returns
    -------
    Delayed signal (same length, edge padded).
    """
    n = len(data)
    src_indices = np.arange(n) - delay_samples
    # Clamp to valid range
    src_indices_clamped = np.clip(src_indices, 0, n - 1)
    floor_idx = np.floor(src_indices_clamped).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, n - 1)
    frac = src_indices_clamped - floor_idx
    return data[floor_idx] * (1 - frac) + data[ceil_idx] * frac


def apply_all_noise(
    data: np.ndarray,
    fs: float,
    sensor_sigma_frac: float = 0.005,
    bump_rate_hz: float = 0.3,
    bump_amplitude_frac: float = 0.06,
    delay_samples: float = 0.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply the full noise pipeline: sensor noise → bumps → delay.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = add_sensor_noise(data, sigma_frac=sensor_sigma_frac, rng=rng)
    out = add_road_bumps(out, fs=fs, bump_rate_hz=bump_rate_hz,
                         bump_amplitude_frac=bump_amplitude_frac, rng=rng)
    out = add_signal_delay(out, delay_samples=delay_samples)
    return out
