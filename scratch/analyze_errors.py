# Diagnostic script to check per-row errors and identify MAPE bottlenecks.
import os, numpy as np, pandas as pd
import lightgbm as lgb

def mape(y_true, y_pred):
    return np.abs(y_true - y_pred) / np.maximum(y_true, 1e-5)

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

feat_tr = X_tr_df[DIM_COLS].fillna(0).copy()
feat_va = X_va_df[DIM_COLS].fillna(0).copy()
pid_tr = df_train['PRODUCT_TYPE_ID'].reset_index(drop=True)
pid_va = df_val['PRODUCT_TYPE_ID'].reset_index(drop=True)

feat_tr['type_median'] = pid_tr.map(type_medians).fillna(global_median).values
feat_va['type_median'] = pid_va.map(type_medians).fillna(global_median).values
feat_tr['nlp_ridge_pred'] = nlp_tr
feat_va['nlp_ridge_pred'] = nlp_va
feat_tr['PRODUCT_TYPE_ID'] = pid_tr.astype(int).values
feat_va['PRODUCT_TYPE_ID'] = pid_va.astype(int).values

dtrain = lgb.Dataset(feat_tr, label=y_train, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)
dval   = lgb.Dataset(feat_va, label=y_val, reference=dtrain, categorical_feature=['PRODUCT_TYPE_ID'], free_raw_data=False)

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

model = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])

preds = np.clip(model.predict(feat_va), 0.5, None)
errors = mape(y_val, preds)

print(f"Overall Validation MAPE: {np.mean(errors)*100:.4f}%")

# Create a DataFrame of errors to analyze
df_err = pd.DataFrame({
    'TITLE': df_val['TITLE'],
    'PRODUCT_TYPE_ID': df_val['PRODUCT_TYPE_ID'],
    'y_true': y_val,
    'y_pred': preds,
    'error': errors
})

print("\n=== TOP 20 LARGEST INDIVIDUAL ROW ERRORS ===")
print(df_err.sort_values(by='error', ascending=False).head(20)[['TITLE', 'y_true', 'y_pred', 'error']])

print("\n=== ERROR BY TARGET VALUE RANGES ===")
ranges = [0, 100, 500, 1000, 5000, 10000, 1000000000]
df_err['range'] = pd.cut(df_err['y_true'], ranges)
print(df_err.groupby('range').agg(
    count=('error', 'count'),
    mean_error=('error', 'mean'),
    median_error=('error', 'median'),
    max_error=('error', 'max')
))
