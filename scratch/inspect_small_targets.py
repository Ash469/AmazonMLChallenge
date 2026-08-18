# Check details of target values in the small range (0, 100]
import pandas as pd
import numpy as np
import os

df = pd.read_csv("d:/AmazonML/dataset/sampled/debug/train.csv")
y = df['PRODUCT_LENGTH'].values.astype(float)

print("=== Overall distribution of PRODUCT_LENGTH ===")
for q in [0, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]:
    print(f"  p{q:5.1f}: {np.percentile(y, q):.4f}")

# Count of values <= 100
small_mask = y <= 100
n_small = small_mask.sum()
print(f"\nNumber of samples with PRODUCT_LENGTH <= 100: {n_small} / {len(df)} ({n_small/len(df)*100:.2f}%)")

print("\n=== Small target samples (first 30) ===")
print(df[small_mask][['TITLE', 'PRODUCT_TYPE_ID', 'PRODUCT_LENGTH']].head(30))
