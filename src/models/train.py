import os
import yaml
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
import lightgbm as lgb

from src.utils_logger import setup_logger
from src.data.loader import load_dataset
from src.data.validator import validate_schema
from src.data.splitter import split_train_val
from src.features.text import get_combined_text
from src.features.tfidf import SparseTFIDFBuilder
from src.features.dimensions import extract_dimension_features
from src.models.evaluate import calculate_mape
from src.models.baseline import GroupMedianBaseline

logger = setup_logger("train_pipeline")

def run_experiment(config_path: str, sample_size: int = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info("Starting pipeline execution...")
    

    df_all = load_dataset(config['raw_train_path'], nrows=sample_size)
    validate_schema(df_all, is_train=True)
    
    df_train, df_val = split_train_val(
        df_all, 
        train_ratio=config['train_split_ratio'], 
        seed=config['random_seed']
    )
    
    # Clip training target outliers to handle skewness
    # (99th percentile is 9600, clipping at 20000.0 preserves 99.5%+ of distribution safely)
    df_train['PRODUCT_LENGTH'] = df_train['PRODUCT_LENGTH'].clip(upper=20000.0)
    
    # Target value transformation:
    # Log-scaling is recommended because of high target variance and right skew.
    y_train = df_train['PRODUCT_LENGTH'].values
    y_val = df_val['PRODUCT_LENGTH'].values
    
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    baseline = GroupMedianBaseline()
    baseline.fit(df_train)
    val_baseline_preds = baseline.predict(df_val)
    baseline_mape = calculate_mape(y_val, val_baseline_preds)
    logger.info(f"Group Median Baseline Validation MAPE: {baseline_mape:.5f}")
    
    logger.info("Extracting dimensional parser features...")
    text_train = get_combined_text(df_train)
    text_val = get_combined_text(df_val)
    
    feats_dim_train = extract_dimension_features(text_train)
    feats_dim_val = extract_dimension_features(text_val)
    
    # TF-IDF builders (leakage-safe)
    logger.info("Building sparse word TF-IDF features...")
    tfidf_builder = SparseTFIDFBuilder(
        word_ngram=tuple(config['word_ngram_range']),
        word_max_features=config['word_max_features']
    )
    X_train_sparse = tfidf_builder.fit_transform(text_train)
    X_val_sparse = tfidf_builder.transform(text_val)
    
    # Ridge Regression on TF-IDF
    logger.info("Fitting Ridge model on sparse TF-IDF...")
    ridge = Ridge(alpha=config['ridge_alpha'], random_state=config['random_seed'])
    ridge.fit(X_train_sparse, y_train_log)
    
    ridge_val_preds_log = ridge.predict(X_val_sparse)
    ridge_val_preds = np.expm1(ridge_val_preds_log)
    ridge_mape = calculate_mape(y_val, ridge_val_preds)
    logger.info(f"Ridge TF-IDF Validation MAPE: {ridge_mape:.5f}")
    
    ridge_train_preds_log = ridge.predict(X_train_sparse)
    ridge_train_preds = np.expm1(ridge_train_preds_log)
    
    # Hybrid LightGBM using Dimension Features + Ridge Predictions
    logger.info("Fitting Hybrid LightGBM model...")
    
    # Generate category-specific median features to guide LightGBM
    train_cat_medians = df_train.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].median().to_dict()
    global_median = df_train['PRODUCT_LENGTH'].median()
    
    X_tab_train = feats_dim_train.copy()
    X_tab_train['ridge_pred'] = ridge_train_preds
    X_tab_train['cat_median'] = df_train['PRODUCT_TYPE_ID'].map(train_cat_medians).fillna(global_median).values
    
    # Frequency encode product type ID to keep it structured
    type_counts = df_train['PRODUCT_TYPE_ID'].value_counts().to_dict()
    X_tab_train['product_type_freq'] = df_train['PRODUCT_TYPE_ID'].map(type_counts).fillna(0).values
    
    X_tab_val = feats_dim_val.copy()
    X_tab_val['ridge_pred'] = ridge_val_preds
    X_tab_val['cat_median'] = df_val['PRODUCT_TYPE_ID'].map(train_cat_medians).fillna(global_median).values
    X_tab_val['product_type_freq'] = df_val['PRODUCT_TYPE_ID'].map(type_counts).fillna(0).values
    
    lgb_train = lgb.Dataset(X_tab_train, label=y_train_log)
    lgb_val = lgb.Dataset(X_tab_val, label=y_val_log, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'learning_rate': config['lgbm_learning_rate'],
        'num_leaves': config['lgbm_num_leaves'],
        'seed': config['random_seed'],
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=config['lgbm_n_estimators'],
        valid_sets=[lgb_val]
    )
    
    lgb_val_preds_log = model.predict(X_tab_val)
    lgb_val_preds = np.expm1(lgb_val_preds_log)
    lgb_mape = calculate_mape(y_val, lgb_val_preds)
    logger.info(f"LightGBM Hybrid Validation MAPE: {lgb_mape:.5f}")
    
    # Ensembling: Blend Ridge and LightGBM
    ensemble_preds = 0.4 * ridge_val_preds + 0.6 * lgb_val_preds
    ensemble_mape = calculate_mape(y_val, ensemble_preds)
    logger.info(f"Ensemble (0.4 Ridge + 0.6 LightGBM) Validation MAPE: {ensemble_mape:.5f}")
    
    # Save artifacts
    os.makedirs(config['models_dir'], exist_ok=True)
    joblib.dump(baseline, os.path.join(config['models_dir'], 'baseline.pkl'))
    joblib.dump(tfidf_builder, os.path.join(config['models_dir'], 'tfidf_builder.pkl'))
    joblib.dump(ridge, os.path.join(config['models_dir'], 'ridge.pkl'))
    joblib.dump(model, os.path.join(config['models_dir'], 'lgbm.pkl'))
    # Save target mapping for prediction pipeline
    joblib.dump({
        'train_cat_medians': train_cat_medians,
        'global_median': global_median,
        'type_counts': type_counts
    }, os.path.join(config['models_dir'], 'category_mappings.pkl'))
    logger.info("Saved trained artifacts to disk.")
    
if __name__ == '__main__':
    run_experiment('d:/AmazonML/configs/base_config.yaml', sample_size=None)
