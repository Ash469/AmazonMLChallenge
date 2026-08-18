"""
DIAGNOSTIC: Find what is actually causing MAPE to jump from 83% → 141%
Test each component in isolation, in order.
"""
import os, sys, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.append(os.path.abspath("."))
from src.features.dimensions import parse_exact_dimensions
from src.features.text import clean_text
from src.models.evaluate import calculate_mape

DATA_DIR = "d:/AmazonML/dataset/sampled/debug"
df_train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test_raw  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

np.random.seed(42)
idx = np.random.permutation(len(df_train_raw))
cut = int(len(idx) * 0.8)
df_train = df_train_raw.iloc[idx[:cut]].reset_index(drop=True)
df_val   = df_train_raw.iloc[idx[cut:]].reset_index(drop=True)

y_train = df_train['PRODUCT_LENGTH'].values
y_val   = df_val['PRODUCT_LENGTH'].values

print(f"Train: {len(df_train):,}  Val: {len(df_val):,}")
print(f"y_train: min={y_train.min():.1f} max={y_train.max():.1f} median={np.median(y_train):.1f}")
print(f"y_val:   min={y_val.min():.1f} max={y_val.max():.1f} median={np.median(y_val):.1f}\n")

# ── text helpers ──────────────────────────────────────────────
t_tr = df_train['TITLE'].fillna('').apply(clean_text)
t_va = df_val['TITLE'].fillna('').apply(clean_text)

b_tr = df_train['BULLET_POINTS'].fillna('').apply(clean_text)
b_va = df_val['BULLET_POINTS'].fillna('').apply(clean_text)

d_tr = df_train['DESCRIPTION'].fillna('').apply(clean_text)
d_va = df_val['DESCRIPTION'].fillna('').apply(clean_text)

comb_tr = (t_tr + " " + b_tr + " " + d_tr).str.strip()
comb_va = (t_va + " " + b_va + " " + d_va).str.strip()

def quick_ridge(X_tr, X_va, alpha=0.1):
    r = Ridge(alpha=alpha, random_state=42)
    r.fit(X_tr, np.log1p(y_train))
    preds = np.clip(np.expm1(r.predict(X_va)), 0.5, None)
    return preds, calculate_mape(y_val, preds)

# ── STEP 1: plain combined TF-IDF (reproduce the 83% result) ─
print("==== STEP 1: Plain combined TF-IDF (should ~83%) ====")
v1 = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50_000, sublinear_tf=True)
X1_tr = v1.fit_transform(comb_tr)
X1_va = v1.transform(comb_va)
p1, m1 = quick_ridge(X1_tr, X1_va)
print(f"Combined TF-IDF (50k) MAPE: {m1*100:.2f}%  | pred median: {np.median(p1):.1f}")

# ── STEP 2: title-only TF-IDF ─────────────────────────────────
print("\n==== STEP 2: Title-only TF-IDF ====")
v2 = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=150_000, sublinear_tf=True)
X2_tr = v2.fit_transform(t_tr)
X2_va = v2.transform(t_va)
p2, m2 = quick_ridge(X2_tr, X2_va)
print(f"Title TF-IDF MAPE: {m2*100:.2f}%  | pred median: {np.median(p2):.1f}")

# ── STEP 3: field-weighted word TF-IDF (title + bullets + desc) ─
print("\n==== STEP 3: Field-weighted word TF-IDF ONLY (no char, no numeric) ====")
b_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200_000, sublinear_tf=True)
X3_tr_b = b_vec.fit_transform(b_tr)
X3_va_b = b_vec.transform(b_va)

d_vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=250_000, sublinear_tf=True)
X3_tr_d = d_vec.fit_transform(d_tr)
X3_va_d = d_vec.transform(d_va)

X3_tr = hstack([X2_tr * 2.0, X3_tr_b * 1.2, X3_tr_d * 0.7]).tocsr()
X3_va = hstack([X2_va * 2.0, X3_va_b * 1.2, X3_va_d * 0.7]).tocsr()
p3, m3 = quick_ridge(X3_tr, X3_va)
print(f"Field-weighted word TF-IDF MAPE: {m3*100:.2f}%  | pred median: {np.median(p3):.1f}")

# ── STEP 4: add char TF-IDF ────────────────────────────────────
print("\n==== STEP 4: Field-weighted + char TF-IDF ====")
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2,
                           max_features=150_000, sublinear_tf=True, dtype=np.float32)
c_tr = (t_tr + " " + b_tr).str.strip()
c_va = (t_va + " " + b_va).str.strip()
X4_tr_c = char_vec.fit_transform(c_tr)
X4_va_c = char_vec.transform(c_va)

X4_tr = hstack([X3_tr, X4_tr_c * 0.5]).tocsr()
X4_va = hstack([X3_va, X4_va_c * 0.5]).tocsr()
p4, m4 = quick_ridge(X4_tr, X4_va)
print(f"+ Char TF-IDF MAPE: {m4*100:.2f}%  | pred median: {np.median(p4):.1f}")

# ── STEP 5: add numeric features ──────────────────────────────
print("\n==== STEP 5: + Numeric features ====")
type_counts = df_train['PRODUCT_TYPE_ID'].value_counts().to_dict()

def extract_features(df):
    comb = (df['TITLE'].fillna('') + ' ' + df['BULLET_POINTS'].fillna('') + ' ' + df['DESCRIPTION'].fillna('')).apply(clean_text)
    p = pd.DataFrame(list(comb.apply(parse_exact_dimensions)))
    p['product_type_frequency'] = df['PRODUCT_TYPE_ID'].map(type_counts).fillna(0).astype(float)
    return p

X_tr_feat = extract_features(df_train)
X_va_feat = extract_features(df_val)

FEAT_COLS = ["explicit_length_cm","dim_1_cm","dim_2_cm","dim_3_cm","max_dim_cm","min_dim_cm",
             "mid_dim_cm","volume_cm3","measurement_count","number_count","has_explicit_length",
             "has_lxw","has_lxwxh","has_inch","has_cm","has_mm","has_ft","product_type_frequency"]

scaler = StandardScaler()
X5_tr_n = scaler.fit_transform(X_tr_feat[FEAT_COLS].fillna(0).values)
X5_va_n = scaler.transform(X_va_feat[FEAT_COLS].fillna(0).values)

X5_tr = hstack([X4_tr, csr_matrix(X5_tr_n)]).tocsr()
X5_va = hstack([X4_va, csr_matrix(X5_va_n)]).tocsr()
p5, m5 = quick_ridge(X5_tr, X5_va)
print(f"+ Numeric features MAPE: {m5*100:.2f}%  | pred median: {np.median(p5):.1f}")

# ── SUMMARY ────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Step 1 Plain combined TF-IDF:      {m1*100:.2f}%  (pred median: {np.median(p1):.1f})")
print(f"Step 2 Title only TF-IDF:          {m2*100:.2f}%  (pred median: {np.median(p2):.1f})")
print(f"Step 3 Field-weighted word TF-IDF: {m3*100:.2f}%  (pred median: {np.median(p3):.1f})")
print(f"Step 4 + Char TF-IDF:              {m4*100:.2f}%  (pred median: {np.median(p4):.1f})")
print(f"Step 5 + Numeric features:         {m5*100:.2f}%  (pred median: {np.median(p5):.1f})")
print(f"\nActual y_val median: {np.median(y_val):.1f}")
