# Analyze the MAPE error breakdown by target range on the experiment validation set.
import os, numpy as np, pandas as pd

# Load validation ground truth and LightGBM predictions
df_train_all = pd.read_csv("d:/AmazonML/dataset/sampled/experiment/train.csv")
val_idx      = np.load("processed_features/val_indices.npy")
y_val        = df_train_all.iloc[val_idx]['PRODUCT_LENGTH'].values.astype(float)

lgb_va = np.load("processed_features/lgb_val.npy")
cb_va  = np.load("processed_features/cb_val.npy")

blend_va = 0.8 * lgb_va + 0.2 * cb_va

# Compute per-row MAPE errors
errors = np.abs(y_val - blend_va) / np.maximum(y_val, 1e-5)

print(f"Overall Validation MAPE: {np.mean(errors)*100:.4f}%")

df_err = pd.DataFrame({
    'y_true': y_val,
    'y_pred': blend_va,
    'error': errors
})

# Define ranges
ranges = [0, 50, 100, 200, 500, 1000, 5000, 10000, 1000000000]
df_err['range'] = pd.cut(df_err['y_true'], ranges)

print("\n=== Error Breakdown by Target Range ===")
summary = df_err.groupby('range').agg(
    count=('error', 'count'),
    mean_error_pct=('error', lambda x: np.mean(x)*100),
    median_error_pct=('error', lambda x: np.median(x)*100),
    total_contrib_pct=('error', lambda x: np.sum(x)/len(df_err)*100)
)
print(summary.to_string())

# Compute MAPE excluding targets <= 100 (which are mostly placeholders)
mask_real = y_val > 100
print(f"\nValidation MAPE on real products (y_true > 100): {np.mean(errors[mask_real])*100:.4f}%  ({mask_real.sum()} / {len(y_val)} samples)")

# Compute MAPE excluding targets <= 200
mask_real_200 = y_val > 200
print(f"Validation MAPE on real products (y_true > 200): {np.mean(errors[mask_real_200])*100:.4f}%  ({mask_real_200.sum()} / {len(y_val)} samples)")
