# Analyze if parsed dimensions (dim_1_u, dim_2_u, max_dim_u, etc.) match the target in the (500, 5000] range.
import os, numpy as np, pandas as pd

# Load data
DATA_DIR = "dataset/sampled/debug"
df_train_all = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_idx = np.load("processed_features/train_indices.npy")
df_train = df_train_all.iloc[train_idx].reset_index(drop=True)
y_train = df_train['PRODUCT_LENGTH'].values.astype(float)

X_tr_df = pd.read_parquet("processed_features/X_train_features.parquet")

# Filter to the (500, 5000] range
mask_range = (y_train > 500) & (y_train <= 5000)
df_sub = df_train[mask_range].reset_index(drop=True)
X_sub = X_tr_df[mask_range].reset_index(drop=True)
y_sub = y_train[mask_range]

print(f"Total products in (500, 5000]: {len(y_sub)}")

# Count how many have parsed dimensions
has_dims = (X_sub['measurement_count'] >= 1).values
print(f"  Products with >=1 parsed dimensions: {has_dims.sum()} ({has_dims.sum()/len(y_sub)*100:.1f}%)")

# Check how close different dimensions are to the target
cols = ['dim_1_u', 'dim_2_u', 'dim_3_u', 'max_dim_u', 'min_dim_u', 'mid_dim_u']
close_counts = {c: 0 for c in cols}
any_close = 0

for i in range(len(y_sub)):
    if not has_dims[i]:
        continue
    t = y_sub[i]
    found_any = False
    for col in cols:
        val = X_sub.loc[i, col]
        if val > 0:
            error = abs(t - val) / t
            if error <= 0.10: # within 10% error
                close_counts[col] += 1
                found_any = True
    if found_any:
        any_close += 1

print("\n=== Count of products where parsed dimension is within 10% of Target ===")
for col in cols:
    print(f"  {col:12s}: {close_counts[col]} ({close_counts[col]/len(y_sub)*100:.1f}%)")
print(f"  Any parsed dim within 10% of target: {any_close} ({any_close/len(y_sub)*100:.1f}%)")

# Check a sample of cases where the dimensions are close vs where they are far
print("\n=== Sample where a dimension matches target within 10% ===")
matched = []
for i in range(len(y_sub)):
    t = y_sub[i]
    if has_dims[i]:
        for col in cols:
            val = X_sub.loc[i, col]
            if val > 0 and abs(t - val) / t <= 0.10:
                matched.append((df_sub.loc[i, 'TITLE'], t, val, col))
                break
        if len(matched) >= 10:
            break
for title, t, val, col in matched:
    print(f"Target: {t:6.1f} | Parsed ({col}): {val:6.1f} | Title: {title[:80]}...")
