# Script to test LightGBM validation MAPE when filtering out dummy targets and outliers from the training set.
import os, re, numpy as np, pandas as pd
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

# Train LightGBM with different target filtering thresholds
for min_y in [1.0, 10.0, 30.0, 50.0]:
    for max_y in [100000, 200000, None]:
        # Build filter mask
        mask = (y_train >= min_y)
        if max_y is not None:
            mask = mask & (y_train <= max_y)
            
        X_tr_fit = Xf_tr[mask].reset_index(drop=True)
        y_tr_fit = y_train[mask]
        
        dtrain = lgb.Dataset(X_tr_fit, label=y_tr_fit, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
        dval   = lgb.Dataset(Xf_va, label=y_val, reference=dtrain, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
        
        params = {
            'objective':        'mape',
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
        
        preds = np.clip(model.predict(Xf_va), 0.5, None)
        m_val = mape(y_val, preds)
        print(f"Filter: min_y={min_y}, max_y={max_y} ({len(y_tr_fit)} samples) -> Val MAPE: {m_val*100:.4f}%")
