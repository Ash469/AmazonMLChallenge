# Test continuous/progressive scaling functions based on category dummy fraction
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

lgb_va = np.load("processed_features/lgb_val.npy")
cb_va  = np.load("processed_features/cb_val.npy")
blend_va = 0.8 * lgb_va + 0.2 * cb_va

# Compute category dummy fractions
df_train['is_dummy'] = (y_train <= 100).astype(int)
dummy_fractions = df_train.groupby('PRODUCT_TYPE_ID')['is_dummy'].mean().to_dict()

# Base validation MAPE
print(f"Base Validation MAPE: {mape(y_val, blend_va)*100:.4f}%")

val_pids = df_val['PRODUCT_TYPE_ID'].values
val_fracs = np.array([dummy_fractions.get(pid, 0.0) for pid in val_pids])

# Grid search parameters for: scale = 1.0 - alpha * (frac ** beta)
best_m = float('inf')
best_params = None

for alpha in np.arange(0.1, 1.05, 0.05):
    for beta in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        processed = blend_va.copy()
        
        # Continuous scale factor
        scale = 1.0 - alpha * (val_fracs ** beta)
        scale = np.clip(scale, 0.1, 1.0)
        
        processed = processed * scale
        
        m = mape(y_val, processed)
        if m < best_m:
            best_m = m
            best_params = (alpha, beta)

print(f"\nBest Continuous Post-processing Params:")
print(f"  alpha: {best_params[0]:.2f}")
print(f"  beta:  {best_params[1]:.2f}")
print(f"Optimized Validation MAPE: {best_m*100:.4f}%")
