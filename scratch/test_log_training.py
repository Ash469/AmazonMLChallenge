# Test script to evaluate LightGBM on log1p target with MAE/MSE objective to combat MAPE under-prediction bias.
import os, numpy as np, pandas as pd
import lightgbm as lgb

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

X_tr_df = pd.read_parquet("processed_features/X_train_features.parquet")
X_va_df = pd.read_parquet("processed_features/X_val_features.parquet")

nlp_tr = np.load("processed_features/nlp_train.npy")
nlp_va = np.load("processed_features/nlp_val.npy")

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
    feat['PRODUCT_TYPE_ID'] = pid.astype(int).values
    return feat

Xf_tr = build_features(X_tr_df, df_train, nlp_tr)
Xf_va = build_features(X_va_df, df_val, nlp_va)

# We will train on log1p target
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)

# Test L1 (L1 loss = MAE) and L2 (MSE) objectives
for obj in ['regression_l1', 'regression']:
    dtrain = lgb.Dataset(Xf_tr, label=y_train_log, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
    dval   = lgb.Dataset(Xf_va, label=y_val_log, reference=dtrain, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
    
    params = {
        'objective':        obj,
        'learning_rate':    0.05,
        'max_depth':        7,
        'num_leaves':       63,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_samples':15,
        'seed':             42,
        'verbose':          -1
    }
    
    model = lgb.train(
        params, dtrain, num_boost_round=1500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Exponentiate predictions back to raw target
    preds_log = model.predict(Xf_va)
    preds = np.clip(np.expm1(preds_log), 0.5, None)
    
    # Evaluate MAPE
    m_val = mape(y_val, preds)
    print(f"\nObjective: {obj} -> Raw Val MAPE: {m_val*100:.4f}%")
    
    # Grid search a multiplier to optimize MAPE (since MAE/MSE doesn't optimize MAPE directly)
    best_mult, best_m = 1.0, float('inf')
    for mult in np.arange(0.5, 1.2, 0.02):
        m = mape(y_val, preds * mult)
        if m < best_m:
            best_m = m
            best_mult = mult
    print(f"  Optimized with multiplier {best_mult:.2f} -> Val MAPE: {best_m*100:.4f}%")
