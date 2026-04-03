"""
DISA — Research Validation Suite
==================================
Validates the scoring model against labeled datasets by computing:

  1. Pearson correlation coefficient (r)
  2. MAE / MSE / RMSE between predicted and expected scores
  3. Score hierarchy test: smooth > mixed > aggressive
  4. Robustness test: score stability under increasing noise levels

All results returned as a structured dict suitable for API JSON response
or direct display in the Streamlit dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, List, Optional, Tuple

from backend.preprocessing import preprocess, preprocess_for_features
from backend.features import extract_features
from backend.scoring import compute_smoothness_score, DEFAULT_WEIGHTS
from backend.optimizer import extract_labeled_features
from simulator.noise_model import add_sensor_noise


# ---------------------------------------------------------------------------
# Core validation helpers
# ---------------------------------------------------------------------------

def _predict_segment_scores(
    df: pd.DataFrame,
    fs: float = 25.0,
    weights: Optional[np.ndarray] = None,
    segment_length_s: float = 10.0,
) -> Tuple[List[float], List[float]]:
    """
    Split a labeled DataFrame into segments and compute predicted vs expected scores.
    Returns (predicted_list, expected_list).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    seg_len = int(segment_length_s * fs)
    n = len(df)
    predicted = []
    expected = []

    for start in range(0, n - seg_len + 1, seg_len // 2):
        end = start + seg_len
        seg = df.iloc[start:end].copy().reset_index(drop=True)
        try:
            feat_df = preprocess_for_features(seg, fs=fs)
            fm = extract_features(feat_df, fs=fs)
            result = compute_smoothness_score(fm, weights=weights)
            expected_score = float(df["expected_score"].iloc[start:end].mean()) \
                if "expected_score" in df.columns else 50.0
            predicted.append(result.overall_score)
            expected.append(expected_score)
        except Exception:
            continue

    return predicted, expected


def _compute_metrics(predicted: List[float], expected: List[float]) -> dict:
    """Compute regression metrics between predicted and expected lists."""
    p = np.array(predicted)
    e = np.array(expected)

    if len(p) < 2:
        return {"error": "Not enough data points for metric computation"}

    residuals = p - e
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))
    r, p_value = sp_stats.pearsonr(p, e)

    bias = float(np.mean(residuals))
    return {
        "n_segments": len(p),
        "pearson_r": round(float(r), 4),
        "pearson_p_value": round(float(p_value), 6),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "predicted_mean": round(float(np.mean(p)), 2),
        "expected_mean": round(float(np.mean(e)), 2),
        "predicted_std": round(float(np.std(p)), 2),
        "expected_std": round(float(np.std(e)), 2),
    }


# ---------------------------------------------------------------------------
# Robustness test
# ---------------------------------------------------------------------------

def robustness_test(
    df: pd.DataFrame,
    fs: float = 25.0,
    weights: Optional[np.ndarray] = None,
    noise_levels: Optional[List[float]] = None,
    seed: int = 42,
) -> dict:
    """
    Measure score stability as Gaussian noise is progressively added.

    Parameters
    ----------
    df           : a single driving session DataFrame
    noise_levels : list of sigma_frac values (fraction of signal range)

    Returns dict with noise_level → mean score mapping.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

    from backend.features import ALL_SIGNAL_COLS

    rng = np.random.default_rng(seed)
    results = {}

    for sigma in noise_levels:
        df_noisy = df.copy()
        if sigma > 0:
            for col in ALL_SIGNAL_COLS:
                if col in df_noisy.columns:
                    df_noisy[col] = add_sensor_noise(
                        df_noisy[col].values.astype(float),
                        sigma_frac=sigma,
                        rng=rng,
                    )

        try:
            processed, _ = preprocess(df_noisy, fs=fs)
            fm = extract_features(processed, fs=fs)
            res = compute_smoothness_score(fm, weights=weights)
            results[str(sigma)] = round(res.overall_score, 2)
        except Exception:
            results[str(sigma)] = None

    return results


# ---------------------------------------------------------------------------
# Score hierarchy test
# ---------------------------------------------------------------------------

def hierarchy_test(
    csv_paths: Optional[List[str]] = None,
    fs: float = 25.0,
    weights: Optional[np.ndarray] = None,
    use_generated: bool = True,
) -> dict:
    """
    Verify that: score_smooth > score_mixed > score_aggressive.

    If csv_paths are provided, load from disk.
    If use_generated is True, generate fresh synthetic data for the test.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    profiles = {}
    if use_generated:
        from simulator.simulator import generate_smooth_driver, generate_aggressive_driver, generate_mixed_driver
        dfs = {
            "smooth": generate_smooth_driver(duration_s=60, fs=fs, seed=100)[0],
            "aggressive": generate_aggressive_driver(duration_s=60, fs=fs, seed=101)[0],
            "mixed": generate_mixed_driver(duration_s=60, fs=fs, seed=102)[0],
        }
    else:
        dfs = {}
        if csv_paths:
            for path in csv_paths:
                profile = Path(path).stem
                dfs[profile] = pd.read_csv(path)

    for name, df_prof in dfs.items():
        try:
            processed = preprocess_for_features(df_prof, fs=fs)
            fm = extract_features(processed, fs=fs)
            res = compute_smoothness_score(fm, weights=weights)
            profiles[name] = {
                "score": res.overall_score,
                "grade": res.grade,
                "sub_scores": res.sub_scores,
            }
        except Exception as e:
            profiles[name] = {"error": str(e)}

    # Hierarchy assertion
    passed = False
    detail = "N/A"
    if all(k in profiles and "score" in profiles[k] for k in ("smooth", "aggressive", "mixed")):
        s_smooth = profiles["smooth"]["score"]
        s_mix = profiles["mixed"]["score"]
        s_agg = profiles["aggressive"]["score"]
        passed = s_smooth > s_mix > s_agg
        detail = (f"smooth({s_smooth:.1f}) > mixed({s_mix:.1f}) > aggressive({s_agg:.1f})"
                  if passed else
                  f"FAILED: smooth={s_smooth:.1f}, mixed={s_mix:.1f}, aggressive={s_agg:.1f}")

    return {
        "hierarchy_test_passed": passed,
        "detail": detail,
        "profiles": profiles,
    }


# ---------------------------------------------------------------------------
# Full validation runner
# ---------------------------------------------------------------------------

def run_validation(
    csv_paths: List[Tuple[str, str]],   # list of (name, filepath) tuples
    fs: float = 25.0,
    weights: Optional[np.ndarray] = None,
    segment_length_s: float = 10.0,
) -> dict:
    """
    Run the complete validation suite on provided labeled CSV files.

    Parameters
    ----------
    csv_paths : list of (profile_name, filepath) tuples
    fs        : sample rate
    weights   : scoring weights (None = default)

    Returns
    -------
    Structured result dict
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    per_profile: Dict[str, dict] = {}
    all_predicted = []
    all_expected = []

    for name, path in csv_paths:
        try:
            df = pd.read_csv(path)
            pred, exp = _predict_segment_scores(df, fs=fs, weights=weights,
                                                segment_length_s=segment_length_s)
            metrics = _compute_metrics(pred, exp)
            per_profile[name] = {
                "metrics": metrics,
                "predicted_samples": [round(p, 2) for p in pred[:100]],  # cap for API response
                "expected_samples": [round(e, 2) for e in exp[:100]],
            }
            all_predicted.extend(pred)
            all_expected.extend(exp)
        except Exception as e:
            per_profile[name] = {"error": str(e)}

    # Overall metrics
    overall = _compute_metrics(all_predicted, all_expected)

    # Hierarchy test
    hier = hierarchy_test(fs=fs, weights=weights, use_generated=True)

    # Robustness test (use first valid file)
    robustness = {}
    for name, path in csv_paths:
        try:
            df = pd.read_csv(path)
            robustness[name] = robustness_test(df, fs=fs, weights=weights)
            break   # just test the first file
        except Exception:
            pass

    return {
        "weights_used": weights.tolist(),
        "overall_metrics": overall,
        "per_profile": per_profile,
        "hierarchy_test": hier,
        "robustness_test": robustness,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    data_dir = ROOT / "data"
    csv_paths = [
        ("smooth", str(data_dir / "smooth.csv")),
        ("aggressive", str(data_dir / "aggressive.csv")),
        ("mixed", str(data_dir / "mixed.csv")),
    ]

    print("\n=== DISA Validation Suite ===\n")
    results = run_validation(csv_paths, fs=25.0)

    print(f"Overall Pearson r : {results['overall_metrics'].get('pearson_r', 'N/A')}")
    print(f"Overall MAE       : {results['overall_metrics'].get('mae', 'N/A')}")
    print(f"Overall MSE       : {results['overall_metrics'].get('mse', 'N/A')}")
    print(f"\nHierarchy test    : {'PASS' if results['hierarchy_test']['hierarchy_test_passed'] else 'FAIL'}")
    print(f"  {results['hierarchy_test']['detail']}")

    print("\nPer-profile metrics:")
    for profile, data in results["per_profile"].items():
        if "error" in data:
            print(f"  {profile}: ERROR — {data['error']}")
        else:
            m = data["metrics"]
            print(f"  {profile}: r={m.get('pearson_r', '?')}  MAE={m.get('mae', '?')}  "
                  f"predicted_mean={m.get('predicted_mean', '?')}  expected_mean={m.get('expected_mean', '?')}")
