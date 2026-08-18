# Diagnostic script to check feature statistics on the experiment dataset
import os, numpy as np, pandas as pd

DATA_DIR = "dataset/sampled/experiment"
df_train_all = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_idx = np.load("processed_features/train_indices.npy")
val_idx   = np.load("processed_features/val_indices.npy")

df_train = df_train_all.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_all.iloc[val_idx].reset_index(drop=True)

y_train = df_train['PRODUCT_LENGTH'].values.astype(float)
y_val   = df_val['PRODUCT_LENGTH'].values.astype(float)

X_tr_df = pd.read_parquet("processed_features/X_train_features.parquet")
X_va_df = pd.read_parquet("processed_features/X_val_features.parquet")

nlp_tr = np.load("processed_features/nlp_train.npy")
nlp_va = np.load("processed_features/nlp_val.npy")

print(f"y_train size: {len(y_train)} | y_val size: {len(y_val)}")
print(f"X_train shape: {X_tr_df.shape} | X_val shape: {X_va_df.shape}")

# Check explicit length coverage
has_expl_tr = X_tr_df['has_explicit_length'].fillna(0).values
expl_len_tr = X_tr_df['explicit_length_u'].fillna(0).values
mask_tr = (has_expl_tr == 1) & (expl_len_tr > 0)
print(f"\nExplicit length coverage in Train: {mask_tr.sum()} / {len(X_tr_df)} ({mask_tr.mean()*100:.2f}%)")

# Calculate direct MAPE error for explicit matches in train
if mask_tr.sum() > 0:
    mape_tr = np.mean(np.abs(y_train[mask_tr] - expl_len_tr[mask_tr]) / np.maximum(y_train[mask_tr], 1e-5))
    print(f"Direct parsed dimension MAPE (Train): {mape_tr*100:.2f}%")

# Same for val
has_expl_va = X_va_df['has_explicit_length'].fillna(0).values
expl_len_va = X_va_df['explicit_length_u'].fillna(0).values
mask_va = (has_expl_va == 1) & (expl_len_va > 0)
print(f"Explicit length coverage in Val: {mask_va.sum()} / {len(X_va_df)} ({mask_va.mean()*100:.2f}%)")
if mask_va.sum() > 0:
    mape_va = np.mean(np.abs(y_val[mask_va] - expl_len_va[mask_va]) / np.maximum(y_val[mask_va], 1e-5))
    print(f"Direct parsed dimension MAPE (Val): {mape_va*100:.2f}%")

# Check distribution of target
print("\n=== PRODUCT_LENGTH Distribution ===")
for q in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f"  p{q:3d}: {np.percentile(y_train, q):.2f}")
