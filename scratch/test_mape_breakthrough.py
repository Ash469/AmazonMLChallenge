import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.optimize import minimize

sys.path.append(os.path.abspath("."))
from src.features.text import get_combined_text
from src.models.evaluate import calculate_mape

DATA_DIR = "d:/AmazonML/dataset/sampled/debug"
train_path = os.path.join(DATA_DIR, "train.csv")
df_train_raw = pd.read_csv(train_path)

# Train/Val Split
np.random.seed(42)
n_samples = len(df_train_raw)
indices = np.arange(n_samples)
np.random.shuffle(indices)

split_idx = int(n_samples * 0.8)
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

df_train = df_train_raw.iloc[train_idx].reset_index(drop=True)
df_val = df_train_raw.iloc[val_idx].reset_index(drop=True)

y_train = df_train['PRODUCT_LENGTH'].values
y_val = df_val['PRODUCT_LENGTH'].values

# Fit log1p
y_train_log = np.log1p(y_train)

# Weight by 1 / (y^1.5) to protect small items
weights_train = 1.0 / (np.clip(y_train, 1.0, None) ** 1.5)

df_train['text'] = get_combined_text(df_train)
df_val['text'] = get_combined_text(df_val)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
X_tr = tfidf.fit_transform(df_train['text'])
X_va = tfidf.transform(df_val['text'])

# Ridge alpha=0.1 with 1/(y^1.5) weight
ridge = Ridge(alpha=0.1, random_state=42)
ridge.fit(X_tr, y_train_log, sample_weight=weights_train)

raw_preds = np.expm1(ridge.predict(X_va))
raw_preds = np.clip(raw_preds, 0.5, None)

raw_mape = calculate_mape(y_val, raw_preds)
print(f"Raw Ridge Preds MAPE: {raw_mape * 100:.2f}%")

# Multiplicative Calibration Search k * raw_preds
res = minimize(lambda k: calculate_mape(y_val, k[0] * raw_preds), x0=[0.75], bounds=[(0.4, 1.1)], method='Nelder-Mead')
opt_k = res.x[0]
scaled_preds = opt_k * raw_preds

scaled_mape = calculate_mape(y_val, scaled_preds)
print(f"Optimal Multiplicative Factor k: {opt_k:.4f}")
print(f"Scaled Preds MAPE: {scaled_mape * 100:.2f}%")
