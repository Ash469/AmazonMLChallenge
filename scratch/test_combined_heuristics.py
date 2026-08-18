# Test script to evaluate combined heuristics: Explicit Overrides + Dummy Scaling
import os, numpy as np, pandas as pd

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))

# Load data
DATA_DIR = "dataset/sampled/experiment"
df_train_all = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_idx = np.load("processed_features/train_indices.npy")
val_idx   = np.load("processed_features/val_indices.npy")

df_train = df_train_all.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_all.iloc[val_idx].reset_index(drop=True)

y_train = df_train['PRODUCT_LENGTH'].values.astype(float)
y_val   = df_val['PRODUCT_LENGTH'].values.astype(float)

X_va_df = pd.read_parquet("processed_features/X_val_features.parquet")

lgb_va = np.load("processed_features/lgb_val.npy")
cb_va  = np.load("processed_features/cb_val.npy")
blend_va = 0.8 * lgb_va + 0.2 * cb_va

# Compute category dummy fractions
df_train['is_dummy'] = (y_train <= 100).astype(int)
dummy_fractions = df_train.groupby('PRODUCT_TYPE_ID')['is_dummy'].mean().to_dict()
type_medians = df_train.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].median().to_dict()

# Base validation MAPE
print(f"Base Validation MAPE: {mape(y_val, blend_va)*100:.4f}%")

# Grid search combined parameters
best_m = float('inf')
best_params = None

# Cache mapped values for speed
val_pids = df_val['PRODUCT_TYPE_ID'].values
val_fracs = np.array([dummy_fractions.get(pid, 0.0) for pid in val_pids])
val_medians = np.array([type_medians.get(pid, 665.35) for pid in val_pids])

eu   = X_va_df['explicit_length_u'].fillna(0).values
has_e = X_va_df['has_explicit_length'].fillna(0).values
meas  = X_va_df['measurement_count'].fillna(0).values

# High confidence override mask: explicit length found, is sensible (>10), and >=2 measurements found in text
override_mask = (has_e == 1) & (eu > 10.0) & (meas >= 2)

for frac_thresh in [0.03, 0.05, 0.08, 0.10, 0.15]:
    for scale_factor in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        for override_weight in [0.5, 0.7, 0.8, 0.9, 1.0]:
            processed = blend_va.copy()
            
            # 1. Apply category scaling first
            scale_mask = val_fracs > frac_thresh
            processed[scale_mask] = processed[scale_mask] * scale_factor
            
            # 2. Apply explicit override (override trusted values)
            processed[override_mask] = override_weight * eu[override_mask] + (1.0 - override_weight) * processed[override_mask]
            
            m = mape(y_val, processed)
            if m < best_m:
                best_m = m
                best_params = (frac_thresh, scale_factor, override_weight)

print(f"\nBest Combined Post-processing Params:")
print(f"  frac_thresh:     {best_params[0]}")
print(f"  scale_factor:    {best_params[1]}")
print(f"  override_weight: {best_params[2]}")
print(f"Optimized Validation MAPE: {best_m*100:.4f}%")
