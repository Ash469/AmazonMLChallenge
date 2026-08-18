# Verify the exact unit matching and find optimal conversion factors/heuristics.
import pandas as pd
import numpy as np
import re

df = pd.read_csv("d:/AmazonML/dataset/sampled/debug/train.csv")

UNIT_TO_INCH = {
    'in': 1.0, 'inch': 1.0, 'inches': 1.0, '"': 1.0,
    'ft': 12.0, 'foot': 12.0, 'feet': 12.0,
    'cm': 1.0/2.54, 'centimeter': 1.0/2.54, 'centimeters': 1.0/2.54,
    'mm': 1.0/25.4, 'millimeter': 1.0/25.4, 'millimeters': 1.0/25.4,
    'm': 100.0/2.54, 'meter': 100.0/2.54, 'meters': 100.0/2.54,
}

# Regex to find any dimension-like numbers in title/bullets
RE_DIM = re.compile(r'(\d+(?:\.\d+)?)\s*(inch|inches|in|"|cm|mm|ft|feet|m|meter|meters)\b', re.I)

results = []
for idx, row in df.iterrows():
    title = str(row['TITLE'])
    target = float(row['PRODUCT_LENGTH'])
    
    # find all measurements in title
    matches = RE_DIM.findall(title)
    if not matches:
        continue
        
    for val_str, unit in matches:
        val = float(val_str)
        val_in_inches = val * UNIT_TO_INCH.get(unit.lower(), 0.0)
        if val_in_inches <= 0:
            continue
            
        # Target is either in hundredths of an inch, or inches, or cm, or mm?
        # Let's check ratio: target / val_in_inches
        ratio = target / val_in_inches
        results.append({
            'val': val,
            'unit': unit,
            'val_inches': val_in_inches,
            'target': target,
            'ratio': ratio
        })

df_res = pd.DataFrame(results)
print("=== Common Ratios (Target / Val in Inches) ===")
# Bin the ratios to find peaks
bins = [0, 1, 10, 50, 90, 100, 110, 200, 1000, 10000]
print(df_res['ratio'].groupby(pd.cut(df_res['ratio'], bins)).count())

print("\n=== Sample matches near ratio ~100 (Hundredths of an inch) ===")
print(df_res[(df_res['ratio'] >= 95) & (df_res['ratio'] <= 105)].head(15))

print("\n=== Sample matches near ratio ~39.37 (Centimeters) ===")
print(df_res[(df_res['ratio'] >= 35) & (df_res['ratio'] <= 45)].head(15))
