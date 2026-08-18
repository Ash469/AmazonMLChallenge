# Test script to evaluate LightGBM with the new 'closest_dim_u' features.
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

type_medians = df_train.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].median().to_dict()
global_median = float(np.median(y_train))

# Build features
DROP = {'PRODUCT_ID', 'PRODUCT_LENGTH'}
DIM_COLS = [c for c in X_tr_df.columns if c not in DROP]

def build_advanced_features(X_df, df_src, nlp_preds):
    feat = X_df[DIM_COLS].fillna(0).copy()
    pid = df_src['PRODUCT_TYPE_ID'].reset_index(drop=True)
    feat['type_median'] = pid.map(type_medians).fillna(global_median).values
    feat['nlp_ridge_pred'] = nlp_preds
    
    # Advanced Heuristics
    closest_dims = []
    closest_errors = []
    
    # Calculate for each row
    for i in range(len(feat)):
        pred = nlp_preds[i]
        dims = [feat.loc[i, 'dim_1_u'], feat.loc[i, 'dim_2_u'], feat.loc[i, 'dim_3_u']]
        valid_dims = [d for d in dims if d > 0]
        
        if valid_dims:
            # Find the dimension closest to the NLP prediction
            errors = [abs(d - pred)/max(pred, 1.0) for d in valid_dims]
            best_idx = np.argmin(errors)
            best_dim = valid_dims[best_idx]
            best_err = errors[best_idx]
        else:
            best_dim = pred  # fallback to NLP prediction
            best_err = 1e5
            
        closest_dims.append(best_dim)
        closest_errors.append(best_err)
        
    feat['closest_dim_u'] = closest_dims
    feat['closest_dim_error'] = closest_errors
    
    # Ratio features
    feat['closest_to_nlp_ratio'] = feat['closest_dim_u'] / np.maximum(feat['nlp_ridge_pred'], 1.0)
    feat['closest_to_type_ratio'] = feat['closest_dim_u'] / np.maximum(feat['type_median'], 1.0)
    
    feat['PRODUCT_TYPE_ID'] = pid.astype(int).values
    return feat

Xf_tr = build_advanced_features(X_tr_df, df_train, nlp_tr)
Xf_va = build_advanced_features(X_va_df, df_val, nlp_va)

# Train LightGBM
dtrain = lgb.Dataset(Xf_tr, label=y_train, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
dval   = lgb.Dataset(Xf_va, label=y_val, reference=dtrain, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)

params = {
    'objective':        'mape',
    'learning_rate':    0.03,
    'max_depth':        7,
    'num_leaves':       63,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_samples':10,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'seed':             42,
    'verbose':          -1
}

model = lgb.train(
    params, dtrain, num_boost_round=3000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)]
)

preds = np.clip(model.predict(Xf_va), 0.5, None)
print(f"\nAdvanced Features Validation MAPE: {mape(y_val, preds)*100:.4f}%")
