import os
import yaml
import joblib
import pandas as pd
import numpy as np

from src.utils_logger import setup_logger
from src.data.loader import load_dataset
from src.data.validator import validate_schema
from src.features.text import get_combined_text

logger = setup_logger("predict_pipeline")

def run_predictions(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info("Starting test inference execution...")
    
    # 1. Load test data
    df_test = load_dataset(config['raw_test_path'])
    validate_schema(df_test, is_train=False)
    
    # 2. Load trained models/builders
    baseline = joblib.load(os.path.join(config['models_dir'], 'baseline.pkl'))
    tfidf_builder = joblib.load(os.path.join(config['models_dir'], 'tfidf_builder.pkl'))
    ridge = joblib.load(os.path.join(config['models_dir'], 'ridge.pkl'))
    lgbm = joblib.load(os.path.join(config['models_dir'], 'lgbm.pkl'))
    mappings = joblib.load(os.path.join(config['models_dir'], 'category_mappings.pkl'))
    
    # 3. Clean test text sequences
    text_test = get_combined_text(df_test)
    
    # 4. TF-IDF transformations
    X_test_sparse = tfidf_builder.transform(text_test)
    
    # 5. Extract dimension features
    from src.features.dimensions import extract_dimension_features
    feats_dim_test = extract_dimension_features(text_test)
    
    # 6. Ridge Predictions
    ridge_test_preds_log = ridge.predict(X_test_sparse)
    ridge_test_preds = np.expm1(ridge_test_preds_log)
    
    # 7. LightGBM Predictions
    X_tab_test = feats_dim_test.copy()
    X_tab_test['ridge_pred'] = ridge_test_preds
    X_tab_test['cat_median'] = df_test['PRODUCT_TYPE_ID'].map(mappings['train_cat_medians']).fillna(mappings['global_median']).values
    X_tab_test['product_type_freq'] = df_test['PRODUCT_TYPE_ID'].map(mappings['type_counts']).fillna(0).values
    
    lgb_test_preds_log = lgbm.predict(X_tab_test)
    lgb_test_preds = np.expm1(lgb_test_preds_log)
    
    # 8. Ensemble Blend (0.4 Ridge + 0.6 LightGBM)
    final_preds = 0.4 * ridge_test_preds + 0.6 * lgb_test_preds
    
    # Post-processing: predictions must not be zero or negative
    final_preds = np.clip(final_preds, 1.0, None)
    
    # Create submission file
    submission = pd.DataFrame({
        'PRODUCT_ID': df_test['PRODUCT_ID'],
        'PRODUCT_LENGTH': final_preds
    })
    
    submission_path = 'd:/AmazonML/dataset/submission.csv'
    submission.to_csv(submission_path, index=False)
    logger.info(f"Test inference complete. Submission file saved to: {submission_path}")

if __name__ == '__main__':
    run_predictions('d:/AmazonML/configs/base_config.yaml')
