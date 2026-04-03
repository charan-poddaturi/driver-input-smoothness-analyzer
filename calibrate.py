import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from backend.preprocessing import preprocess
from backend.features import extract_features
from backend.optimizer import prepare_training_data, optimize_weights

# Step 1: Get raw feature values before normalization
print("=== RAW PRE-NORMALIZED FEATURES ===")
raw_results = {}
for profile in ['smooth', 'aggressive', 'mixed']:
    df = pd.read_csv(f'data/{profile}.csv')
    # Don't normalize for this calibration pass
    from backend.preprocessing import handle_missing_values, apply_lowpass_filter, SIGNAL_BOUNDS
    df2 = handle_missing_values(df)
    df2 = apply_lowpass_filter(df2, fs=25.0)
    for col, (lo, hi) in SIGNAL_BOUNDS.items():
        if col in df2.columns:
            df2[col] = df2[col].clip(lo, hi)
    fm = extract_features(df2, fs=25.0)
    ag = fm.aggregate
    exp = df['expected_score'].mean()
    raw_results[profile] = ag
    print(f"{profile}: S={ag['S']:.6f}  J={ag['J']:.4f}  V={ag['V']:.6f}  P={ag['P']:.6f}  exp={exp:.2f}")

print()
print("=== SUGGESTED FEATURE_REF_MAX ===")
for k in ['S', 'J', 'V', 'P']:
    max_val = max(v[k] for v in raw_results.values())
    print(f"  {k}: {max_val * 1.2:.6f}")

# Step 2: Optimize weights
print()
print("=== OPTIMIZING WEIGHTS ===")
feat_list, exp_scores = prepare_training_data(
    ['data/smooth.csv', 'data/aggressive.csv', 'data/mixed.csv'],
    fs=25.0, segment_length_s=10.0
)
print(f"Total segments: {len(feat_list)}")
opt = optimize_weights(feat_list, exp_scores, method='slsqp', n_restarts=5)
print(f"Optimized weights: {opt.optimized_weights.tolist()}")
print(f"MSE: default={opt.default_mse:.2f} -> optimized={opt.mse:.2f}  improvement={opt.mse_improvement_pct:.1f}%")
