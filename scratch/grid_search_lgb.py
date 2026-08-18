# Grid search script to optimize LightGBM hyperparameters and feature sets
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

df_train = df_train_all.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_all.iloc[val_idx].reset_index(drop=True)

y_train = df_train['PRODUCT_LENGTH'].values.astype(float)
y_val   = df_val['PRODUCT_LENGTH'].values.astype(float)

X_tr_df = pd.read_parquet("processed_features/X_train_features.parquet")
X_va_df = pd.read_parquet("processed_features/X_val_features.parquet")

nlp_tr = np.load("processed_features/nlp_train.npy")
nlp_va = np.load("processed_features/nlp_val.npy")

type_medians = df_train.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].median().to_dict()
global_median = float(np.median(y_train))

DROP = {'PRODUCT_ID', 'PRODUCT_LENGTH'}
DIM_COLS = [c for c in X_tr_df.columns if c not in DROP]

# Pre-extract standard features
def get_base_features(X_df, df_src, nlp_preds):
    feat = X_df[DIM_COLS].fillna(0).copy()
    pid = df_src['PRODUCT_TYPE_ID'].reset_index(drop=True)
    feat['type_median'] = pid.map(type_medians).fillna(global_median).values
    feat['nlp_ridge_pred'] = nlp_preds
    feat['PRODUCT_TYPE_ID'] = pid.astype(int).values
    return feat

Xf_tr_base = get_base_features(X_tr_df, df_train, nlp_tr)
Xf_va_base = get_base_features(X_va_df, df_val, nlp_va)

# Add ratio features to create the advanced feature set
def add_ratios(df):
    feat = df.copy()
    for col in ['dim_1_u', 'dim_2_u', 'dim_3_u', 'max_dim_u', 'min_dim_u', 'mid_dim_u', 'explicit_length_u']:
        if col in feat.columns:
            feat[f'{col}_to_type'] = feat[col] / np.maximum(feat['type_median'], 1.0)
            feat[f'{col}_to_nlp']  = feat[col] / np.maximum(feat['nlp_ridge_pred'], 1.0)
    return feat

Xf_tr_adv = add_ratios(Xf_tr_base)
Xf_va_adv = add_ratios(Xf_va_base)

# Parameter Grid
experiments = [
    # (max_depth, num_leaves, min_child_samples, learning_rate, outlier_cap, use_adv_features)
    (7, 63, 10, 0.03, None, False),    # The user's baseline (which gave ~84.52%)
    (7, 63, 10, 0.03, 100000, False),  # Baseline + outlier filtering
    (6, 31, 20, 0.03, 100000, False),  # Simpler model + outlier filtering
    (7, 63, 15, 0.03, 100000, True),   # Advanced features + medium complexity
    (6, 31, 15, 0.03, 100000, True),   # Advanced features + simpler model
    (5, 15, 10, 0.05, 100000, False),  # Very simple model
    (8, 127, 20, 0.02, 100000, False), # Complex, lower learning rate
]

for idx, (depth, leaves, min_child, lr, cap, use_adv) in enumerate(experiments):
    # Choose feature set
    X_tr = Xf_tr_adv if use_adv else Xf_tr_base
    X_va = Xf_va_adv if use_adv else Xf_va_base
    
    # Filter training outliers
    if cap is not None:
        mask = (y_train < cap) & (y_train > 1.0)
        X_tr_fit = X_tr[mask].reset_index(drop=True)
        y_tr_fit = y_train[mask]
    else:
        X_tr_fit = X_tr
        y_tr_fit = y_train
        
    dtrain = lgb.Dataset(X_tr_fit, label=y_tr_fit, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_val, reference=dtrain, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
    
    params = {
        'objective':        'mape',
        'learning_rate':    lr,
        'max_depth':        depth,
        'num_leaves':       leaves,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': min_child,
        'seed':             42,
        'verbose':          -1
    }
    
    model = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    preds = np.clip(model.predict(X_va), 0.5, None)
    m_val = mape(y_val, preds)
    print(f"Exp {idx}: depth={depth}, leaves={leaves}, min_child={min_child}, lr={lr}, cap={cap}, adv={use_adv} -> Val MAPE: {m_val*100:.4f}%")
