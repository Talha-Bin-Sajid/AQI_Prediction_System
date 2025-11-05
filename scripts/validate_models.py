"""Prove models are not overfitted"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from loguru import logger

from src.config import config, RAW_DATA_DIR
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer


def validate_no_overfitting():
    logger.info("="*60)
    logger.info("OVERFITTING VALIDATION - 5-FOLD TIME SERIES CV")
    logger.info("="*60)
    
    # Load data
    historical_file = RAW_DATA_DIR / f"historical_aqi_{config.location.city_name.lower()}.csv"
    df = pd.read_csv(historical_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    feature_cols = engineer.get_feature_columns(df_features)
    X = df_features[feature_cols]
    y = df_features['aqi_target_24h']
    
    # Load best model
    trainer = ModelTrainer()
    model = trainer.load_model()
    metadata = trainer.load_metadata()
    
    logger.info(f"\nValidating model: {metadata.get('model_name', 'Unknown')}")
    logger.info(f"Model R²: {metadata.get('metrics', {}).get('r2', 0):.4f}")
    
    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_train_scores = []
    cv_val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_cv = X.iloc[train_idx]
        y_train_cv = y.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]
        y_val_cv = y.iloc[val_idx]
        
        # Clone and fit
        from sklearn.base import clone
        model_cv = clone(model)
        model_cv.fit(X_train_cv, y_train_cv)
        
        # Calculate R²
        train_r2 = r2_score(y_train_cv, model_cv.predict(X_train_cv))
        val_r2 = r2_score(y_val_cv, model_cv.predict(X_val_cv))
        
        cv_train_scores.append(train_r2)
        cv_val_scores.append(val_r2)
        
        gap = train_r2 - val_r2
        status = "✅" if gap < 0.1 else "⚠️"
        
        logger.info(f"Fold {fold}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, Gap={gap:.4f} {status}")
    
    # Final verdict
    avg_train = np.mean(cv_train_scores)
    avg_val = np.mean(cv_val_scores)
    avg_gap = avg_train - avg_val
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Average Train R²: {avg_train:.4f} ± {np.std(cv_train_scores):.4f}")
    logger.info(f"Average Val R²:   {avg_val:.4f} ± {np.std(cv_val_scores):.4f}")
    logger.info(f"Average Gap:      {avg_gap:.4f}")
    
    if avg_gap < 0.1:
        logger.info("\n✅ VERDICT: Model is NOT OVERFITTED!")
        logger.info("   Small gap between train and validation (< 0.1)")
    elif avg_gap < 0.15:
        logger.info("\n⚠️  VERDICT: Slight overfitting (gap < 0.15)")
        logger.info("   Model is acceptable but could be improved")
    else:
        logger.info("\n❌ VERDICT: Model is OVERFITTED!")
        logger.info("   Large gap between train and validation (> 0.15)")
    
    logger.info("="*60)


if __name__ == "__main__":
    validate_no_overfitting()