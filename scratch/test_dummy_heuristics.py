# Test script to evaluate post-processing using category-level dummy fractions on the experiment dataset.
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

# Load validation predictions (blend of LGB + CatBoost)
lgb_va = np.load("processed_features/lgb_val.npy")
cb_va  = np.load("processed_features/cb_val.npy")
blend_va = 0.8 * lgb_va + 0.2 * cb_va

# Compute category dummy fractions from training data
# Let's define a target <= 100 as a potential dummy/small value
df_train['is_dummy'] = (y_train <= 100).astype(int)
dummy_fractions = df_train.groupby('PRODUCT_TYPE_ID')['is_dummy'].mean().to_dict()
type_medians = df_train.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].median().to_dict()

# Base validation MAPE
print(f"Base Validation MAPE: {mape(y_val, blend_va)*100:.4f}%")

# Grid search post-processing thresholds
best_m = float('inf')
best_params = None

for frac_thresh in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    for scale_factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        processed = blend_va.copy()
        
        # Apply scaling based on category dummy fraction
        for i in range(len(processed)):
            pid = df_val.loc[i, 'PRODUCT_TYPE_ID']
            frac = dummy_fractions.get(pid, 0.0)
            
            if frac > frac_thresh:
                # This category has a significant fraction of dummy/small labels.
                # Scale the prediction down to prevent catastrophic MAPE errors.
                t_median = type_medians.get(pid, 665.35)
                processed[i] = processed[i] * scale_factor
                
        m = mape(y_val, processed)
        if m < best_m:
            best_m = m
            best_params = (frac_thresh, scale_factor)

print(f"\nBest Post-processing Params:")
print(f"  frac_thresh:  {best_params[0]}")
print(f"  scale_factor: {best_params[1]}")
print(f"Optimized Validation MAPE: {best_m*100:.4f}%")
