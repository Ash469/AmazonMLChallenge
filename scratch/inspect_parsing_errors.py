# Inspect the largest errors of the parsed explicit dimensions.
import pandas as pd
import numpy as np
import os

df = pd.read_csv("d:/AmazonML/dataset/sampled/experiment/train.csv")
train_idx = np.load("processed_features/train_indices.npy")
df_train = df.iloc[train_idx].reset_index(drop=True)
y_train = df_train['PRODUCT_LENGTH'].values.astype(float)

X_tr_df = pd.read_parquet("processed_features/X_train_features.parquet")

has_expl = X_tr_df['has_explicit_length'].fillna(0).values
expl_len = X_tr_df['explicit_length_u'].fillna(0).values

mask = (has_expl == 1) & (expl_len > 0)

df_expl = pd.DataFrame({
    'TITLE': df_train.loc[mask, 'TITLE'],
    'PRODUCT_TYPE_ID': df_train.loc[mask, 'PRODUCT_TYPE_ID'],
    'y_true': y_train[mask],
    'y_pred': expl_len[mask],
    'error': np.abs(y_train[mask] - expl_len[mask]) / y_train[mask]
})

print("=== TOP 30 LARGEST PARSING ERRORS ===")
print(df_expl.sort_values(by='error', ascending=False).head(30)[['TITLE', 'y_true', 'y_pred', 'error']])
