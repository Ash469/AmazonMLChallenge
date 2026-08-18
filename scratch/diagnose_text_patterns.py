# Diagnostic script to check text and dimension parsing coverage on debug dataset.
import pandas as pd
import re

df = pd.read_csv("d:/AmazonML/dataset/sampled/debug/train.csv")

# Sample some rows to look at the text
pd.set_option('display.max_colwidth', None)
print("=== Sample Titles ===")
print(df[['TITLE', 'PRODUCT_LENGTH']].head(20))

# Count how many have numbers followed by some unit-like string in title
title_has_num = df['TITLE'].fillna('').str.contains(r'\d', regex=True).sum()
print(f"\nTitles containing any digit: {title_has_num} / {len(df)} ({title_has_num/len(df)*100:.1f}%)")

# Check common units in title
for unit in ['inch', 'in', 'cm', 'mm', 'ft', 'foot', 'feet', 'yard', 'meter', 'm']:
    count = df['TITLE'].fillna('').str.contains(rf'\b\d+\s*{unit}\b|\b\d+{unit}\b', case=False, regex=True).sum()
    print(f"  Titles containing digits + {unit}: {count}")

# Check description/bullets
desc_has_num = df['DESCRIPTION'].fillna('').str.contains(r'\d', regex=True).sum()
bullets_has_num = df['BULLET_POINTS'].fillna('').str.contains(r'\d', regex=True).sum()
print(f"Descriptions containing digits: {desc_has_num}")
print(f"Bullets containing digits: {bullets_has_num}")
