# Experiment script to optimize LightGBM hyperparameters and feature engineering.
import os, re, numpy as np, pandas as pd
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

df_train = df_train_all.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_all.iloc[val_idx].reset_index(drop=True)

y_train = df_train['PRODUCT_LENGTH'].values.astype(float)
y_val   = df_val['PRODUCT_LENGTH'].values.astype(float)

X_tr_df = pd.read_parquet("processed_features/X_train_features.parquet")
X_va_df = pd.read_parquet("processed_features/X_val_features.parquet")

nlp_tr = np.load("processed_features/nlp_train.npy")
nlp_va = np.load("processed_features/nlp_val.npy")

# Base stats
type_medians = df_train.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].median().to_dict()
global_median = float(np.median(y_train))

# Build features
DROP = {'PRODUCT_ID', 'PRODUCT_LENGTH'}
DIM_COLS = [c for c in X_tr_df.columns if c not in DROP]

def build_features(X_df, df_src, nlp_preds):
    feat = X_df[DIM_COLS].fillna(0).copy()
    pid = df_src['PRODUCT_TYPE_ID'].reset_index(drop=True)
    feat['type_median'] = pid.map(type_medians).fillna(global_median).values
    feat['nlp_ridge_pred'] = nlp_preds
    
    # Let's add extra dimension-based features
    # Try to find the dimension closest to the type median or NLP prediction
    for col in ['dim_1_u', 'dim_2_u', 'dim_3_u', 'max_dim_u', 'min_dim_u', 'mid_dim_u', 'explicit_length_u']:
        if col in feat.columns:
            feat[f'{col}_to_type'] = feat[col] / np.maximum(feat['type_median'], 1.0)
            feat[f'{col}_to_nlp']  = feat[col] / np.maximum(feat['nlp_ridge_pred'], 1.0)
            
    # Category categorical feature
    feat['PRODUCT_TYPE_ID'] = pid.astype(int).values
    return feat

Xf_tr = build_features(X_tr_df, df_train, nlp_tr)
Xf_va = build_features(X_va_df, df_val, nlp_va)

# Filter training outliers: target values that are clearly erroneous
# e.g., length > 100,000 units (1000 inches = 83 feet)
outlier_mask = (y_train < 100000) & (y_train > 1.0)
Xf_tr_clean = Xf_tr[outlier_mask].reset_index(drop=True)
y_train_clean = y_train[outlier_mask]

print(f"Original train size: {len(y_train)} | Cleaned: {len(y_train_clean)} (removed {len(y_train)-len(y_train_clean)} outliers)")

# Train LightGBM
dtrain = lgb.Dataset(Xf_tr_clean, label=y_train_clean, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
dval   = lgb.Dataset(Xf_va, label=y_val, reference=dtrain, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)

params = {
    'objective':        'mape',
    'learning_rate':    0.03,
    'max_depth':        8,
    'num_leaves':       127,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_samples':15,
    'reg_alpha':        0.5,
    'reg_lambda':       2.0,
    'seed':             42,
    'verbose':          -1
}

model = lgb.train(
    params, dtrain, num_boost_round=3000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)]
)

preds = np.clip(model.predict(Xf_va), 0.5, None)
print(f"\nValidation MAPE: {mape(y_val, preds)*100:.4f}%")
