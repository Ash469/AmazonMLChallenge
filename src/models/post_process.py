import pandas as pd
import numpy as np
import joblib
import os
import yaml

def post_process_predictions(preds_path: str, config_path: str):
    """
    Rounds model predictions to the nearest common target value in the training dataset
    to align predictions with quantized target sizing labels (improving MAPE).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    sub = pd.read_csv(preds_path)

    # Load the top 100 most frequent unique training target values
    df_train = pd.read_csv(config['raw_train_path'], usecols=['PRODUCT_LENGTH'])
    common_targets = df_train['PRODUCT_LENGTH'].value_counts().head(200).index.values
    
    print("Post-processing: Rounding predictions to closest common training target...")
    
    def round_to_nearest(val):
        idx = np.abs(common_targets - val).argmin()
        return common_targets[idx]
        
    # Round predictions only if they are relatively close to a standard common size
    rounded_lengths = []
    for val in sub['PRODUCT_LENGTH']:
        nearest = round_to_nearest(val)
        if abs(val - nearest) / max(val, 1e-5) < 0.15:
            rounded_lengths.append(nearest)
        else:
            rounded_lengths.append(val)
            
    sub['PRODUCT_LENGTH'] = rounded_lengths
    sub.to_csv(preds_path, index=False)
    print(f"Post-processing complete. Saved to {preds_path}")

if __name__ == '__main__':
    post_process_predictions('d:/AmazonML/dataset/submission.csv', 'd:/AmazonML/configs/base_config.yaml')
