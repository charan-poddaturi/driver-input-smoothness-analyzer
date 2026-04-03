"""
DISA — Weight Optimizer
========================
Finds optimal weights [w_S, w_J, w_V, w_P] by minimizing MSE between
predicted smoothness scores and expected labels in a labeled dataset.

Two optimization strategies:
  A. Grid Search  — exhaustive simplex search (interpretable, reproducible)
  B. SLSQP        — scipy constrained optimization (fast, precise)

Constraint: w₁ + w₂ + w₃ + w₄ = 1,  wᵢ ∈ [0, 1]
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.optimize import minimize

from backend.preprocessing import preprocess, preprocess_for_features
from backend.features import extract_features, FeatureMatrix
from backend.scoring import compute_score_from_features, FEATURE_REF_MAX


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Results of weight optimization."""
    method: str
    optimized_weights: np.ndarray
    mse: float
    mae: float
    default_mse: float
    default_mae: float
    mse_improvement_pct: float
    sensitivity: Dict[str, List[float]]          # {weight_name: [mse at each delta]}
    grid_results: Optional[List[dict]] = None    # top-N grid search results

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "optimized_weights": {
                "w_S": round(float(self.optimized_weights[0]), 4),
                "w_J": round(float(self.optimized_weights[1]), 4),
                "w_V": round(float(self.optimized_weights[2]), 4),
                "w_P": round(float(self.optimized_weights[3]), 4),
            },
            "mse": round(self.mse, 4),
            "mae": round(self.mae, 4),
            "default_mse": round(self.default_mse, 4),
            "default_mae": round(self.default_mae, 4),
            "mse_improvement_pct": round(self.mse_improvement_pct, 2),
            "sensitivity": {k: [round(v, 4) for v in vs] for k, vs in self.sensitivity.items()},
        }


# ---------------------------------------------------------------------------
# Feature extraction from labeled dataset
# ---------------------------------------------------------------------------

def _extract_session_features(
    df: pd.DataFrame,
    fs: float = 25.0,
) -> Tuple[Dict[str, float], float]:
    """
    Extract aggregate {S, J, V, P} features and mean expected score
    from a labeled session DataFrame.
    """
    # Use raw (un-normalized) signals for feature extraction
    feat_df = preprocess_for_features(df, fs=fs)
    fm = extract_features(feat_df, fs=fs)
    expected = float(df["expected_score"].mean()) if "expected_score" in df.columns else 50.0
    return fm.aggregate, expected


def extract_labeled_features(
    df: pd.DataFrame,
    fs: float = 25.0,
    segment_length_s: float = 10.0,
) -> Tuple[List[Dict[str, float]], List[float]]:
    """
    Split a labeled DataFrame into segments and extract features per segment.
    
    This gives us many training points from one CSV file for robust optimization.

    Parameters
    ----------
    df               : labeled DataFrame with expected_score column
    fs               : sample rate
    segment_length_s : segment duration in seconds

    Returns
    -------
    (feature_list, expected_scores)
    """
    seg_len = int(segment_length_s * fs)
    n = len(df)

    feature_list = []
    score_list = []

    for start in range(0, n - seg_len + 1, seg_len // 2):   # 50% overlap
        end = start + seg_len
        seg = df.iloc[start:end].copy().reset_index(drop=True)
        try:
            feats, expected = _extract_session_features(seg, fs=fs)
            feature_list.append(feats)
            score_list.append(expected)
        except Exception:
            continue

    return feature_list, score_list


def prepare_training_data(
    csv_paths: List[str],
    fs: float = 25.0,
    segment_length_s: float = 10.0,
) -> Tuple[List[Dict[str, float]], List[float]]:
    """Load multiple labeled CSVs and prepare training data."""
    all_features = []
    all_scores = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if "expected_score" not in df.columns:
                print(f"  [WARN] {path} has no expected_score column, skipping.")
                continue
            feats, scores = extract_labeled_features(df, fs=fs, segment_length_s=segment_length_s)
            all_features.extend(feats)
            all_scores.extend(scores)
            print(f"  Loaded {len(feats)} training segments from {path}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {path}: {e}")

    return all_features, all_scores


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _mse_loss(
    weights: np.ndarray,
    feature_list: List[Dict[str, float]],
    expected_scores: List[float],
    ref_max: Optional[Dict[str, float]] = None,
) -> float:
    """Compute MSE between predicted scores and expected labels."""
    ref_max = ref_max or FEATURE_REF_MAX
    predicted = []
    for feats in feature_list:
        score, _, _ = compute_score_from_features(feats, weights, ref_max=ref_max)
        predicted.append(score)

    residuals = np.array(predicted) - np.array(expected_scores)
    return float(np.mean(residuals ** 2))


def _mae_loss(
    weights: np.ndarray,
    feature_list: List[Dict[str, float]],
    expected_scores: List[float],
    ref_max: Optional[Dict[str, float]] = None,
) -> float:
    ref_max = ref_max or FEATURE_REF_MAX
    predicted = []
    for feats in feature_list:
        score, _, _ = compute_score_from_features(feats, weights, ref_max=ref_max)
        predicted.append(score)
    return float(np.mean(np.abs(np.array(predicted) - np.array(expected_scores))))


# ---------------------------------------------------------------------------
# Strategy A: Grid Search
# ---------------------------------------------------------------------------

def grid_search_weights(
    feature_list: List[Dict[str, float]],
    expected_scores: List[float],
    step: float = 0.05,
    ref_max: Optional[Dict[str, float]] = None,
    top_n: int = 10,
) -> Tuple[np.ndarray, float, List[dict]]:
    """
    Exhaustive simplex grid search over weight combinations.

    Parameters
    ----------
    feature_list    : list of {S, J, V, P} dicts
    expected_scores : corresponding expected scores
    step            : grid step (0.05 → ~1771 combinations for 4 weights)
    ref_max         : feature normalization maxima
    top_n           : number of top results to return

    Returns
    -------
    (best_weights, best_mse, top_results)
    """
    ref_max = ref_max or FEATURE_REF_MAX
    ticks = np.arange(0.0, 1.0 + step, step)

    best_weights = np.array([0.25, 0.30, 0.25, 0.20])
    best_mse = float("inf")
    results = []

    # Generate all combinations on the (w1+w2+w3+w4=1) simplex
    for w1, w2, w3 in itertools.product(ticks, repeat=3):
        w4 = 1.0 - w1 - w2 - w3
        if not (0.0 <= w4 <= 1.0):
            continue
        w = np.array([w1, w2, w3, w4])
        mse = _mse_loss(w, feature_list, expected_scores, ref_max=ref_max)
        results.append({"weights": w.tolist(), "mse": mse})
        if mse < best_mse:
            best_mse = mse
            best_weights = w.copy()

    results.sort(key=lambda x: x["mse"])
    return best_weights, best_mse, results[:top_n]


# ---------------------------------------------------------------------------
# Strategy B: SLSQP (scipy constrained optimization)
# ---------------------------------------------------------------------------

def slsqp_optimize_weights(
    feature_list: List[Dict[str, float]],
    expected_scores: List[float],
    ref_max: Optional[Dict[str, float]] = None,
    n_restarts: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Scipy SLSQP optimization with equality constraint (Σwᵢ = 1).
    Run multiple restarts from random starting points to avoid local minima.
    """
    ref_max = ref_max or FEATURE_REF_MAX
    rng = np.random.default_rng(seed)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.01, 0.99)] * 4

    best_weights = None
    best_mse = float("inf")

    for restart in range(n_restarts):
        # Random starting point on simplex
        x0_raw = rng.dirichlet(np.ones(4))   # sums to 1 by construction
        x0 = np.clip(x0_raw, 0.01, 0.99)
        x0 = x0 / x0.sum()

        result = minimize(
            fun=_mse_loss,
            x0=x0,
            args=(feature_list, expected_scores, ref_max),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if result.success and result.fun < best_mse:
            best_mse = result.fun
            best_weights = result.x.copy()

    if best_weights is None:
        best_weights = np.array([0.25, 0.30, 0.25, 0.20])
        best_mse = _mse_loss(best_weights, feature_list, expected_scores, ref_max)

    return best_weights, best_mse


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    base_weights: np.ndarray,
    feature_list: List[Dict[str, float]],
    expected_scores: List[float],
    deltas: Optional[List[float]] = None,
    ref_max: Optional[Dict[str, float]] = None,
) -> Dict[str, List[float]]:
    """
    Vary each weight by ±delta (while renormalizing) and measure MSE change.
    
    Returns dict: {weight_name: [mse for each delta]}
    """
    ref_max = ref_max or FEATURE_REF_MAX
    if deltas is None:
        deltas = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

    weight_names = ["w_S", "w_J", "w_V", "w_P"]
    sensitivity: Dict[str, List[float]] = {n: [] for n in weight_names}

    for wi, name in enumerate(weight_names):
        for delta in deltas:
            w_perturbed = base_weights.copy()
            w_perturbed[wi] = np.clip(base_weights[wi] + delta, 0.0, 1.0)
            # Renormalize
            w_perturbed = w_perturbed / w_perturbed.sum()
            mse = _mse_loss(w_perturbed, feature_list, expected_scores, ref_max)
            sensitivity[name].append(mse)

    return sensitivity


# ---------------------------------------------------------------------------
# Main optimizer interface
# ---------------------------------------------------------------------------

def optimize_weights(
    feature_list: List[Dict[str, float]],
    expected_scores: List[float],
    method: str = "slsqp",
    ref_max: Optional[Dict[str, float]] = None,
    grid_step: float = 0.05,
    n_restarts: int = 5,
) -> OptimizationResult:
    """
    Run weight optimization and return a complete OptimizationResult.

    Parameters
    ----------
    method  : "slsqp" | "grid" | "both"
    """
    ref_max = ref_max or FEATURE_REF_MAX
    default_weights = np.array([0.25, 0.30, 0.25, 0.20])
    default_mse = _mse_loss(default_weights, feature_list, expected_scores, ref_max)
    default_mae = _mae_loss(default_weights, feature_list, expected_scores, ref_max)

    grid_results = None

    if method in ("grid", "both"):
        print("  Running grid search...")
        best_w_grid, best_mse_grid, grid_results = grid_search_weights(
            feature_list, expected_scores, step=grid_step, ref_max=ref_max
        )

    if method in ("slsqp", "both"):
        print("  Running SLSQP optimization...")
        best_w_slsqp, best_mse_slsqp = slsqp_optimize_weights(
            feature_list, expected_scores, ref_max=ref_max, n_restarts=n_restarts
        )

    # Pick best result
    if method == "grid":
        opt_weights, opt_mse = best_w_grid, best_mse_grid
        used_method = "Grid Search"
    elif method == "slsqp":
        opt_weights, opt_mse = best_w_slsqp, best_mse_slsqp
        used_method = "SLSQP"
    else:
        if best_mse_slsqp <= best_mse_grid:
            opt_weights, opt_mse = best_w_slsqp, best_mse_slsqp
            used_method = "SLSQP (best of both)"
        else:
            opt_weights, opt_mse = best_w_grid, best_mse_grid
            used_method = "Grid Search (best of both)"

    opt_mae = _mae_loss(opt_weights, feature_list, expected_scores, ref_max)
    improvement = 100.0 * (default_mse - opt_mse) / default_mse if default_mse > 0 else 0.0

    sensitivity = sensitivity_analysis(opt_weights, feature_list, expected_scores, ref_max=ref_max)

    return OptimizationResult(
        method=used_method,
        optimized_weights=opt_weights,
        mse=opt_mse,
        mae=opt_mae,
        default_mse=default_mse,
        default_mae=default_mae,
        mse_improvement_pct=improvement,
        sensitivity=sensitivity,
        grid_results=grid_results,
    )
