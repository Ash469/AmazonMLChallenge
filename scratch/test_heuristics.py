# Script to test smart post-processing heuristics to fix the small-product MAPE problem.
import os, numpy as np, pandas as pd
import lightgbm as lgb

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))

# Load data
DATA_DIR = "dataset/sampled/debug"
df_train_all = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_idx = np.load("processed_features/train_indices.npy")
val_idx   = np.load("processed_features/val_indices.npy")

df_val = df_train_all.iloc[val_idx].reset_index(drop=True)
y_val  = df_val['PRODUCT_LENGTH'].values.astype(float)

X_va_df = pd.read_parquet("processed_features/X_val_features.parquet")
nlp_va  = np.load("processed_features/nlp_val.npy")
lgb_va  = np.load("processed_features/lgb_val.npy")
cb_va   = np.load("processed_features/cb_val.npy")

# Let's blend LGB and CatBoost first
blend_preds = 0.8 * lgb_va + 0.2 * cb_va

# Look at target values vs predictions
print(f"Base Blend MAPE: {mape(y_val, blend_preds)*100:.4f}%")

# POST-PROCESSING HEURISTICS:
# Heuristic 1: If we have an extracted explicit length or dimensions, and it's small, clip the prediction.
# Since the target is in 0.01 inches:
# - explicit_length_u, max_dim_u, dim_1_u are in 0.01 inches.
max_dim = X_va_df['max_dim_u'].fillna(0).values
min_dim = X_va_df['min_dim_u'].fillna(0).values
expl_len = X_va_df['explicit_length_u'].fillna(0).values

# Let's try different clipping rules
processed_preds = blend_preds.copy()

for i in range(len(processed_preds)):
    pred = processed_preds[i]
    md = max_dim[i]
    el = expl_len[i]
    
    # Rule A: If explicit length is present and it is small, trust it more
    if el > 0 and el < 500: # less than 5 inches
        # Blend prediction down towards the explicit length
        processed_preds[i] = 0.8 * el + 0.2 * pred
        
    # Rule B: If max parsed dimension is small (and not 0), cap the prediction
    # E.g., if the text only mentions dimensions under 2 inches (200 units),
    # the product cannot be 15 inches (1500 units).
    elif md > 0 and md < 300:
        # Cap prediction at 1.5x of the maximum parsed dimension
        processed_preds[i] = min(pred, md * 1.5)

print(f"Heuristics Val MAPE: {mape(y_val, processed_preds)*100:.4f}%")

# Let's grid search the threshold and multipliers
best_m = float('inf')
best_params = None

for md_thresh in [150, 200, 250, 300, 350, 400, 500]:
    for mult in [1.0, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]:
        for el_weight in [0.5, 0.7, 0.8, 0.9, 1.0]:
            temp = blend_preds.copy()
            for i in range(len(temp)):
                md = max_dim[i]
                el = expl_len[i]
                if el > 0 and el < md_thresh:
                    temp[i] = el_weight * el + (1.0 - el_weight) * temp[i]
                elif md > 0 and md < md_thresh:
                    temp[i] = min(temp[i], md * mult)
            m = mape(y_val, temp)
            if m < best_m:
                best_m = m
                best_params = (md_thresh, mult, el_weight)

print(f"\nBest params: md_thresh={best_params[0]}, mult={best_params[1]}, el_weight={best_params[2]}")
print(f"Optimized Heuristics Val MAPE: {best_m*100:.4f}%")
