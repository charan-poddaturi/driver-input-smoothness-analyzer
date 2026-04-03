"""
DISA — Smoothness Scoring Engine
==================================
Implements the master smoothness score formula:

    Score = 100 − clip(w₁·S_norm + w₂·J_norm + w₃·V_norm + w₄·P_norm, 0, 100)

Sub-scores are computed per channel group (steering, throttle, braking, stability)
using the same formula with the same weights applied to group-level features.

All feature inputs must already be normalized to [0, 1].
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from backend.features import FeatureMatrix, normalize_feature_vector, CHANNEL_GROUPS


# ---------------------------------------------------------------------------
# Default weights (S, J, V, P) — overridden after optimization
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = np.array([0.25, 0.30, 0.25, 0.20])

# Reference maxima for feature normalization
# Calibrated from aggressive driver profile (raw, pre-normalization features)
# S: mean |dx/dt| of normalized signals, aggressive driver
# J: mean |d²x/dt²|, V: rollling variance, P: spike fraction
FEATURE_REF_MAX = {"S": 65.0, "J": 830.0, "V": 480.0, "P": 0.05}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SmoothnesResult:
    """Complete scoring output for one driving session."""
    overall_score: float                   # 0–100
    sub_scores: Dict[str, float]           # per group: steering, throttle, etc.
    feature_contributions: Dict[str, float]  # how much each feature penalized
    normalized_features: Dict[str, float]  # S, J, V, P after normalization
    weights: np.ndarray                    # weights used
    grade: str                             # A–F descriptor

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 2),
            "sub_scores": {k: round(v, 2) for k, v in self.sub_scores.items()},
            "feature_contributions": {k: round(v, 4) for k, v in self.feature_contributions.items()},
            "normalized_features": {k: round(v, 4) for k, v in self.normalized_features.items()},
            "weights": self.weights.tolist(),
            "grade": self.grade,
        }


# ---------------------------------------------------------------------------
# Grade classifier
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 85:
        return "A — Excellent"
    if score >= 70:
        return "B — Good"
    if score >= 55:
        return "C — Average"
    if score >= 40:
        return "D — Below Average"
    return "F — Poor"


# ---------------------------------------------------------------------------
# Core score computation
# ---------------------------------------------------------------------------

def compute_score_from_features(
    features: Dict[str, float],
    weights: np.ndarray,
    ref_max: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Compute a score from a {S, J, V, P} feature dict.

    Returns
    -------
    (score, normalized_features, feature_contributions)
    """
    ref_max = ref_max or FEATURE_REF_MAX
    norm = normalize_feature_vector(features, ref_max=ref_max)

    w = np.array([weights[0], weights[1], weights[2], weights[3]])
    f = np.array([norm["S"], norm["J"], norm["V"], norm["P"]])

    penalty = float(np.dot(w, f) * 100.0)   # scale to 0–100
    score = float(np.clip(100.0 - penalty, 0.0, 100.0))

    contributions = {
        "S": float(w[0] * norm["S"] * 100),
        "J": float(w[1] * norm["J"] * 100),
        "V": float(w[2] * norm["V"] * 100),
        "P": float(w[3] * norm["P"] * 100),
    }
    return score, norm, contributions


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_smoothness_score(
    fm: FeatureMatrix,
    weights: Optional[np.ndarray] = None,
    ref_max: Optional[Dict[str, float]] = None,
) -> SmoothnesResult:
    """
    Compute the full smoothness score from a FeatureMatrix.

    Parameters
    ----------
    fm      : FeatureMatrix from features.extract_features()
    weights : [w_S, w_J, w_V, w_P] — must sum to 1; defaults to DEFAULT_WEIGHTS
    ref_max : dict of max reference values for normalization

    Returns
    -------
    SmoothnesResult with all computed scores
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    weights = np.array(weights, dtype=float)
    # Enforce constraint: weights sum to 1
    w_sum = weights.sum()
    if abs(w_sum - 1.0) > 1e-6:
        weights = weights / w_sum

    ref_max = ref_max or FEATURE_REF_MAX

    # --- Overall score ---
    overall_score, norm_feats, contributions = compute_score_from_features(
        fm.aggregate, weights, ref_max=ref_max
    )

    # --- Sub-scores per group ---
    sub_scores: Dict[str, float] = {}
    for group in CHANNEL_GROUPS:
        group_feats = fm.group_aggregates.get(group, {"S": 0.0, "J": 0.0, "V": 0.0, "P": 0.0})
        sub_score, _, _ = compute_score_from_features(group_feats, weights, ref_max=ref_max)
        sub_scores[group] = round(sub_score, 2)

    return SmoothnesResult(
        overall_score=round(overall_score, 2),
        sub_scores=sub_scores,
        feature_contributions=contributions,
        normalized_features=norm_feats,
        weights=weights,
        grade=_grade(overall_score),
    )


# ---------------------------------------------------------------------------
# Time-series score (per-sample rolling score)
# ---------------------------------------------------------------------------

def compute_timeseries_score(
    df_processed,
    fs: float = 25.0,
    weights: Optional[np.ndarray] = None,
    window_s: float = 2.0,
    ref_max: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute a rolling smoothness score over time using a sliding window.
    Useful for highlighting rough vs smooth segments in the UI.

    Returns
    -------
    score_series : np.ndarray of shape (n,) with scores in [0, 100]
    """
    import pandas as pd
    from backend.features import ALL_SIGNAL_COLS

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    ref_max = ref_max or FEATURE_REF_MAX

    window = max(int(window_s * fs), 10)
    n = len(df_processed)
    scores = np.full(n, np.nan)

    # Pre-compute derivatives
    dt = 1.0 / fs
    channel_data = {}
    channel_dx = {}
    channel_ddx = {}
    for col in ALL_SIGNAL_COLS:
        if col in df_processed.columns:
            x = df_processed[col].values.astype(float)
            dx = np.gradient(x, dt)
            ddx = np.gradient(dx, dt)
            channel_data[col] = x
            channel_dx[col] = np.abs(dx)
            channel_ddx[col] = np.abs(ddx)

    if not channel_data:
        return scores

    # Stack into matrices for vectorized rolling computation
    dx_mat = np.column_stack(list(channel_dx.values()))   # (n, c)
    ddx_mat = np.column_stack(list(channel_ddx.values()))

    w0, w1, w2, w3 = weights[0], weights[1], weights[2], weights[3]

    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        S = float(np.mean(dx_mat[lo:hi]))
        J = float(np.mean(ddx_mat[lo:hi]))

        # Variance of raw signal in window
        V_vals = []
        for x in channel_data.values():
            V_vals.append(np.var(x[lo:hi]) if (hi - lo) > 1 else 0.0)
        V = float(np.mean(V_vals))

        # Spike frequency
        dx_std = np.std(dx_mat[lo:hi])
        thresh = 2.0 * dx_std if dx_std > 0 else 1e-6
        P = float(np.mean(dx_mat[lo:hi] > thresh))

        feats = {"S": S, "J": J, "V": V, "P": P}
        score, _, _ = compute_score_from_features(feats, weights, ref_max=ref_max)
        scores[i] = score

    return scores


# ---------------------------------------------------------------------------
# Driver session comparison
# ---------------------------------------------------------------------------

def rank_sessions(session_results: List[Tuple[str, SmoothnesResult]]) -> List[dict]:
    """
    Rank multiple driver sessions by overall smoothness score.

    Parameters
    ----------
    session_results : list of (session_id, SmoothnesResult) tuples

    Returns
    -------
    Sorted list of dicts with rank, session_id, score, grade
    """
    ranked = sorted(session_results, key=lambda x: x[1].overall_score, reverse=True)
    return [
        {
            "rank": i + 1,
            "session_id": sid,
            "overall_score": res.overall_score,
            "grade": res.grade,
            "sub_scores": res.sub_scores,
        }
        for i, (sid, res) in enumerate(ranked)
    ]
