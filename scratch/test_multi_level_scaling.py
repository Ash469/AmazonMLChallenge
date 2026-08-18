# Test multi-level step-wise scaling based on category dummy fraction
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

val_pids = df_val['PRODUCT_TYPE_ID'].values
val_fracs = np.array([dummy_fractions.get(pid, 0.0) for pid in val_pids])

# Grid search multi-level thresholds
best_m = float('inf')
best_cfg = None

# Let's test a set of progressive scaling rules
# We define levels of dummy fractions: [0.02, 0.05, 0.10, 0.20]
# and search for the multiplier factor at each level
for f1 in [0.7, 0.8, 0.9, 1.0]:      # factor for frac > 0.02
    for f2 in [0.4, 0.5, 0.6, 0.7]:  # factor for frac > 0.05
        for f3 in [0.2, 0.3, 0.4, 0.5]:  # factor for frac > 0.10
            for f4 in [0.1, 0.2, 0.3]:  # factor for frac > 0.20
                processed = blend_va.copy()
                
                # Apply multi-level scaling
                scale = np.ones(len(processed))
                scale[val_fracs > 0.02] = f1
                scale[val_fracs > 0.05] = f2
                scale[val_fracs > 0.10] = f3
                scale[val_fracs > 0.20] = f4
                
                processed = processed * scale
                m = mape(y_val, processed)
                if m < best_m:
                    best_m = m
                    best_cfg = (f1, f2, f3, f4)

print(f"\nBest Multi-Level Scaling:")
print(f"  frac > 0.02: scale by {best_cfg[0]:.2f}")
print(f"  frac > 0.05: scale by {best_cfg[1]:.2f}")
print(f"  frac > 0.10: scale by {best_cfg[2]:.2f}")
print(f"  frac > 0.20: scale by {best_cfg[3]:.2f}")
print(f"Optimized Validation MAPE: {best_m*100:.4f}%")
