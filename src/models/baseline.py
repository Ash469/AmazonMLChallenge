import pandas as pd
import numpy as np

class GroupMedianBaseline:
    """
    Computes baseline predictions using the median target value per PRODUCT_TYPE_ID.
    Falls back to global median if category is unseen.
    """
    def __init__(self):
        self.global_median = 0.0
        self.group_medians = {}
        
    def fit(self, df: pd.DataFrame):
        self.global_median = df['PRODUCT_LENGTH'].median()

        # Compute median grouped by product type ID
        grouped = df.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].median()
        self.group_medians = grouped.to_dict()
        return self
        
    def predict(self, df: pd.DataFrame):
        predictions = df['PRODUCT_TYPE_ID'].map(self.group_medians)
        
        # Fill unseen categories with global median
        predictions = predictions.fillna(self.global_median)
        return predictions.values
