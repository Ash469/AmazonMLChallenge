import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

sys.path.append(os.path.abspath("."))
from src.features.dimensions import parse_exact_dimensions
from src.features.text import clean_text
from src.models.evaluate import calculate_mape

print("==================================================")
print("RUNNING EXACT STEP-BY-STEP PIPELINE SEQUENCE")
print("==================================================")

DATA_DIR = "d:/AmazonML/dataset/sampled/debug"
PROCESSED_DIR = "d:/AmazonML/processed_features"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. Load Data
df_train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

np.random.seed(42)
n_samples = len(df_train_raw)
indices = np.arange(n_samples)
np.random.shuffle(indices)

split_idx = int(n_samples * 0.8)
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

df_train = df_train_raw.iloc[train_idx].reset_index(drop=True)
df_val = df_train_raw.iloc[val_idx].reset_index(drop=True)
df_test = df_test_raw.copy()

y_train = df_train['PRODUCT_LENGTH'].values
y_val = df_val['PRODUCT_LENGTH'].values

print(f"Train size: {len(df_train):,} | Val size: {len(df_val):,} | Test size: {len(df_test):,}")

# 2. Extract Exact 18 Features
type_counts = df_train['PRODUCT_TYPE_ID'].value_counts().to_dict()

def extract_features(df):
    t_combined = (df['TITLE'].fillna('') + ' ' + df['BULLET_POINTS'].fillna('') + ' ' + df['DESCRIPTION'].fillna('')).apply(clean_text)
    p_df = pd.DataFrame(list(t_combined.apply(parse_exact_dimensions)))
    p_df['product_type_frequency'] = df['PRODUCT_TYPE_ID'].map(type_counts).fillna(0).astype(float)
    return p_df

X_tr_feat = extract_features(df_train)
X_va_feat = extract_features(df_val)
X_te_feat = extract_features(df_test)

EXACT_FEATURE_COLUMNS = [
    "explicit_length_cm", "dim_1_cm", "dim_2_cm", "dim_3_cm", "max_dim_cm", "min_dim_cm",
    "mid_dim_cm", "volume_cm3", "measurement_count", "number_count", "has_explicit_length",
    "has_lxw", "has_lxwxh", "has_inch", "has_cm", "has_mm", "has_ft", "product_type_frequency"
]

X_tr_num = StandardScaler().fit_transform(X_tr_feat[EXACT_FEATURE_COLUMNS].fillna(0))
X_va_num = StandardScaler().fit(X_tr_feat[EXACT_FEATURE_COLUMNS].fillna(0)).transform(X_va_feat[EXACT_FEATURE_COLUMNS].fillna(0))
X_te_num = StandardScaler().fit(X_tr_feat[EXACT_FEATURE_COLUMNS].fillna(0)).transform(X_te_feat[EXACT_FEATURE_COLUMNS].fillna(0))

# 3. Field-Weighted + Char TF-IDF
t_tr = df_train['TITLE'].fillna('').apply(clean_text)
t_va = df_val['TITLE'].fillna('').apply(clean_text)
t_te = df_test['TITLE'].fillna('').apply(clean_text)

b_tr = df_train['BULLET_POINTS'].fillna('').apply(clean_text)
b_va = df_val['BULLET_POINTS'].fillna('').apply(clean_text)
b_te = df_test['BULLET_POINTS'].fillna('').apply(clean_text)

d_tr = df_train['DESCRIPTION'].fillna('').apply(clean_text)
d_va = df_val['DESCRIPTION'].fillna('').apply(clean_text)
d_te = df_test['DESCRIPTION'].fillna('').apply(clean_text)

print("\nExtracting Field & Char TF-IDF...")
t_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True)
X_tr_t = t_vec.fit_transform(t_tr)
X_va_t = t_vec.transform(t_va)
X_te_t = t_vec.transform(t_te)

b_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
X_tr_b = b_vec.fit_transform(b_tr)
X_va_b = b_vec.transform(b_va)
X_te_b = b_vec.transform(b_te)

d_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=250000, sublinear_tf=True)
X_tr_d = d_vec.fit_transform(d_tr)
X_va_d = d_vec.transform(d_va)
X_te_d = d_vec.transform(d_te)

char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=150000, sublinear_tf=True, dtype=np.float32)
c_tr = (t_tr + " " + b_tr).str.strip()
c_va = (t_va + " " + b_va).str.strip()
c_te = (t_te + " " + b_te).str.strip()

X_tr_c = char_vec.fit_transform(c_tr)
X_va_c = char_vec.transform(c_va)
X_te_c = char_vec.transform(c_te)

X_train_final = hstack([X_tr_t * 2.0, X_tr_b * 1.2, X_tr_d * 0.7, X_tr_c * 0.5, csr_matrix(X_tr_num)]).tocsr()
X_val_final = hstack([X_va_t * 2.0, X_va_b * 1.2, X_va_d * 0.7, X_va_c * 0.5, csr_matrix(X_va_num)]).tocsr()
X_test_final = hstack([X_te_t * 2.0, X_te_b * 1.2, X_te_d * 0.7, X_te_c * 0.5, csr_matrix(X_te_num)]).tocsr()

# 4. Dual Ridge Models
print("\nFitting Dual Ridge Models...")
ridge_log = Ridge(alpha=0.1, random_state=42)
ridge_log.fit(X_train_final, np.log1p(y_train))

log_val_preds = np.expm1(ridge_log.predict(X_val_final))
log_train_preds = np.expm1(ridge_log.predict(X_train_final))
log_test_preds = np.expm1(ridge_log.predict(X_test_final))

weights = 1.0 / np.maximum(y_train, 10.0)
weights /= weights.mean()

ridge_raw = Ridge(alpha=1.0, random_state=42)
ridge_raw.fit(X_train_final, y_train, sample_weight=weights)

raw_val_preds = ridge_raw.predict(X_val_final)
raw_train_preds = ridge_raw.predict(X_train_final)
raw_test_preds = ridge_raw.predict(X_test_final)

print("\nTesting Ridge Blend Ratios:")
for w_log, w_raw in [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]:
    blend_v = w_log * log_val_preds + w_raw * raw_val_preds
    m = calculate_mape(y_val, np.clip(blend_v, 0.5, None))
    print(f"Ratio {w_log:.1f}/{w_raw:.1f} -> Validation MAPE: {m * 100:.2f}% ({m:.4f})")

w_log, w_raw = 0.7, 0.3
ridge_val_blend = np.clip(w_log * log_val_preds + w_raw * raw_val_preds, 0.5, None)
ridge_train_blend = np.clip(w_log * log_train_preds + w_raw * raw_train_preds, 0.5, None)
ridge_test_blend = np.clip(w_log * log_test_preds + w_raw * raw_test_preds, 0.5, None)

ridge_mape = calculate_mape(y_val, ridge_val_blend)
print(f"\n---> STEP 6 RIDGE BLEND MAPE: {ridge_mape * 100:.2f}%")

# 5. CatBoost Residual Correction Model
print("\nFitting CatBoost Residual Correction Model...")
SPECIFIED_CATBOOST_FEATURES = [
    "ridge_prediction", "explicit_length_cm", "dim_1_cm", "dim_2_cm", "dim_3_cm",
    "max_dim_cm", "min_dim_cm", "volume_cm3", "measurement_count", "product_type_frequency"
]

def prepare_catboost_df(df_feat, ridge_preds):
    cdf = pd.DataFrame()
    cdf['ridge_prediction'] = ridge_preds
    for col in SPECIFIED_CATBOOST_FEATURES:
        if col != 'ridge_prediction':
            cdf[col] = df_feat[col].values if col in df_feat.columns else 0.0
    return cdf

X_tr_cat = prepare_catboost_df(X_tr_feat, ridge_train_blend)
X_va_cat = prepare_catboost_df(X_va_feat, ridge_val_blend)
X_te_cat = prepare_catboost_df(X_te_feat, ridge_test_blend)

train_residual = np.log1p(y_train) - np.log1p(ridge_train_blend)
val_residual = np.log1p(y_val) - np.log1p(ridge_val_blend)

cat = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
cat.fit(X_tr_cat, train_residual, eval_set=(X_va_cat, val_residual), early_stopping_rounds=40)

val_corr = cat.predict(X_va_cat)
test_corr = cat.predict(X_te_cat)

cat_val_corrected = np.clip(np.expm1(np.log1p(ridge_val_blend) + val_corr), 0.5, None)
cat_test_corrected = np.clip(np.expm1(np.log1p(ridge_test_blend) + test_corr), 0.5, None)

cat_mape = calculate_mape(y_val, cat_val_corrected)
print(f"---> STEP 7 CATBOOST RESIDUAL CORRECTION MAPE: {cat_mape * 100:.2f}%")

# 6. Explicit Length Calibrated Blend
print("\nTesting Explicit-Length Calibrated Overrides...")
mask_val = (X_va_feat['has_explicit_length'] == 1) & (X_va_feat['explicit_length_cm'] > 0.5) & (X_va_feat['explicit_length_cm'] < 1000.0)
exp_val_cm = X_va_feat['explicit_length_cm'].values

best_final_w = 1.00
best_final_mape = float('inf')

for w_mod in [1.00, 0.75, 0.50, 0.25, 0.00]:
    w_exp = 1.0 - w_mod
    c_val = cat_val_corrected.copy()
    c_val[mask_val] = w_mod * cat_val_corrected[mask_val] + w_exp * exp_val_cm[mask_val]
    c_val = np.clip(c_val, 0.5, None)
    m = calculate_mape(y_val, c_val)
    print(f"Model {w_mod:.2f} + Explicit {w_exp:.2f} -> Validation MAPE: {m * 100:.2f}%")
    if m < best_final_mape:
        best_final_mape = m
        best_final_w = w_mod

print(f"\n==================================================")
print(f"FINAL PIPELINE SEQUENCE VALIDATION MAPE: {best_final_mape * 100:.2f}% ({best_final_mape:.4f})")
print(f"==================================================")
