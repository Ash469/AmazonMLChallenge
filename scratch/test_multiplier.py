# Script to test multiplicative bias correction (multiplying predictions by a factor < 1.0)
import numpy as np
import pandas as pd

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))

# Load validation ground truth and predictions
df_train_all = pd.read_csv("d:/AmazonML/dataset/sampled/debug/train.csv")
val_idx      = np.load("processed_features/val_indices.npy")
y_val        = df_train_all.iloc[val_idx]['PRODUCT_LENGTH'].values.astype(float)

lgb_va = np.load("processed_features/lgb_val.npy")
cb_va  = np.load("processed_features/cb_val.npy")

blend_va = 0.8 * lgb_va + 0.2 * cb_va

print(f"Base Blend MAPE: {mape(y_val, blend_va)*100:.4f}%")

# Let's search for the optimal multiplicative factor
best_factor = 1.0
best_m = float('inf')

for factor in np.arange(0.5, 1.2, 0.01):
    m = mape(y_val, blend_va * factor)
    if m < best_m:
        best_m = m
        best_factor = factor

print(f"Best multiplicative factor: {best_factor:.2f}")
print(f"Optimized Blend MAPE: {best_m*100:.4f}%")
