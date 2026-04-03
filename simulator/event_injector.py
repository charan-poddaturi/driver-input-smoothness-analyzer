"""
DISA — Event Injector
=====================
Injects structured driving events into synthetic signal timelines.

Each event modifies one or more signal channels over a defined time window,
maintaining temporal continuity (smooth ramps in/out).

Supported events
----------------
- sudden_brake   : hard brake application + throttle cut
- lane_change    : steering sine burst
- stop_and_go    : full stop cycle (brake → hold → release → accelerate)
- highway_cruise : low-variance steady state with micro-oscillations
- mild_corner    : gentle steering + slight deceleration
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Event:
    """Metadata for an injected event."""
    name: str
    start_idx: int
    end_idx: int
    severity: float  # 0–1


def _ramp(n: int, up: bool = True) -> np.ndarray:
    """Smooth cosine ramp of length n (0→1 if up, 1→0 if down)."""
    r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    return r if up else r[::-1]


def inject_sudden_brake(
    signals: Dict[str, np.ndarray],
    start_idx: int,
    fs: float,
    intensity: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Event:
    """
    Sudden braking event:
      - brake_pressure ramps from current → 60–90% over 0.3 s
      - throttle_position drops to 0
      - accel_x goes negative (deceleration)
      - steering may wobble slightly
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signals["brake_pressure"])
    ramp_up_s, hold_s, ramp_down_s = 0.3, 0.8, 0.5
    ru = int(ramp_up_s * fs)
    hold = int(hold_s * fs)
    rd = int(ramp_down_s * fs)
    total = ru + hold + rd

    end_idx = min(start_idx + total, n)
    actual_total = end_idx - start_idx

    peak_brake = rng.uniform(60, 90) * intensity

    # Brake pressure profile
    if actual_total >= ru + hold:
        profile = np.concatenate([
            _ramp(ru) * peak_brake,
            np.ones(min(hold, actual_total - ru)) * peak_brake,
            _ramp(min(rd, actual_total - ru - hold), up=False) * peak_brake
                if actual_total > ru + hold else np.array([]),
        ])
    else:
        profile = _ramp(actual_total) * peak_brake

    sl = slice(start_idx, start_idx + len(profile))
    signals["brake_pressure"][sl] = np.maximum(signals["brake_pressure"][sl], profile)
    signals["throttle_position"][sl] = np.minimum(
        signals["throttle_position"][sl],
        np.maximum(0.0, signals["throttle_position"][sl] * (1 - profile / 100)),
    )
    # Deceleration ≈ −0.6·g at peak
    decel = -9.81 * 0.6 * (profile / 100) * intensity
    signals["acceleration_x"][sl] += decel
    # Slight steering wobble
    signals["steering_angle"][sl] += rng.normal(0, 1.5 * intensity, size=len(profile))

    return Event("sudden_brake", start_idx, start_idx + len(profile), severity=intensity)


def inject_lane_change(
    signals: Dict[str, np.ndarray],
    start_idx: int,
    fs: float,
    intensity: float = 1.0,
    direction: int = 1,          # +1 = right, -1 = left
    rng: Optional[np.random.Generator] = None,
) -> Event:
    """
    Lane change: two-phase sine sweep on steering_angle + lateral acceleration.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signals["steering_angle"])
    duration_s = rng.uniform(1.5, 3.0)
    total = min(int(duration_s * fs), n - start_idx)
    if total <= 0:
        return Event("lane_change", start_idx, start_idx, severity=0.0)

    t = np.linspace(0, 2 * np.pi, total)
    peak_steer = rng.uniform(20, 40) * intensity * direction
    steer_profile = peak_steer * np.sin(t) * _ramp(total) * _ramp(total, up=False) * 2

    sl = slice(start_idx, start_idx + total)
    signals["steering_angle"][sl] += steer_profile
    signals["acceleration_y"][sl] += steer_profile * 0.04 * intensity   # lateral g
    signals["gyroscope_yaw_rate"][sl] += steer_profile * 0.8 * intensity

    return Event("lane_change", start_idx, start_idx + total, severity=intensity)


def inject_stop_and_go(
    signals: Dict[str, np.ndarray],
    start_idx: int,
    fs: float,
    intensity: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> Event:
    """
    Stop-and-go traffic: throttle release → brake → hold → release → re-accelerate.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signals["brake_pressure"])
    phase_s = [0.5, 1.5, 0.5, 1.5]          # brake-ramp, hold, release, accel
    phases = [int(p * fs) for p in phase_s]
    total = min(sum(phases), n - start_idx)
    if total <= 0:
        return Event("stop_and_go", start_idx, start_idx, severity=0.0)

    brake_peak = rng.uniform(30, 60) * intensity
    cursor = start_idx

    # Phase 1: ramp brake up
    p1 = min(phases[0], total)
    sl1 = slice(cursor, cursor + p1)
    signals["brake_pressure"][sl1] = _ramp(p1) * brake_peak
    signals["throttle_position"][sl1] *= (1 - _ramp(p1) * 0.9)
    cursor += p1

    # Phase 2: hold
    p2 = min(phases[1], total - (cursor - start_idx))
    sl2 = slice(cursor, cursor + p2)
    signals["brake_pressure"][sl2] = brake_peak
    signals["throttle_position"][sl2] = 0.0
    cursor += p2

    # Phase 3: release brake
    p3 = min(phases[2], total - (cursor - start_idx))
    sl3 = slice(cursor, cursor + p3)
    signals["brake_pressure"][sl3] = _ramp(p3, up=False) * brake_peak
    cursor += p3

    # Phase 4: re-accelerate
    p4 = min(phases[3], total - (cursor - start_idx))
    if p4 > 0:
        sl4 = slice(cursor, cursor + p4)
        signals["throttle_position"][sl4] = _ramp(p4) * rng.uniform(30, 60)

    end_idx = start_idx + total
    return Event("stop_and_go", start_idx, end_idx, severity=intensity)


def inject_highway_cruise(
    signals: Dict[str, np.ndarray],
    start_idx: int,
    fs: float,
    duration_s: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> Event:
    """
    Highway cruise: steady throttle with micro-oscillations, minimal steering.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signals["throttle_position"])
    total = min(int(duration_s * fs), n - start_idx)
    if total <= 0:
        return Event("highway_cruise", start_idx, start_idx, severity=0.0)

    t = np.arange(total) / fs
    steady_throttle = rng.uniform(55, 75)
    micro_osc = 2.0 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.5, total)

    sl = slice(start_idx, start_idx + total)
    signals["throttle_position"][sl] = steady_throttle + micro_osc
    signals["brake_pressure"][sl] = np.maximum(0, signals["brake_pressure"][sl] * 0.05)
    signals["steering_angle"][sl] = 0.5 * np.sin(2 * np.pi * 0.02 * t) + rng.normal(0, 0.3, total)

    return Event("highway_cruise", start_idx, start_idx + total, severity=0.1)


def inject_mild_corner(
    signals: Dict[str, np.ndarray],
    start_idx: int,
    fs: float,
    intensity: float = 0.5,
    direction: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Event:
    """Gentle corner: smooth bell-curve steering profile + slight brake."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(signals["steering_angle"])
    duration_s = rng.uniform(2.0, 4.0)
    total = min(int(duration_s * fs), n - start_idx)
    if total <= 0:
        return Event("mild_corner", start_idx, start_idx, severity=0.0)

    # Bell curve profile
    t_norm = np.linspace(-3, 3, total)
    bell = np.exp(-0.5 * t_norm ** 2)
    peak_steer = rng.uniform(10, 25) * intensity * direction

    sl = slice(start_idx, start_idx + total)
    signals["steering_angle"][sl] += peak_steer * bell
    signals["acceleration_y"][sl] += peak_steer * bell * 0.03
    signals["throttle_position"][sl] *= (1.0 - 0.15 * bell)   # slight lift-off

    return Event("mild_corner", start_idx, start_idx + total, severity=intensity * 0.5)
