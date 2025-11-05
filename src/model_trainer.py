"""
Model training module for AQI prediction - FIXED
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, List
from loguru import logger

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from src.config import config
from src.feature_engineering import FeatureEngineer


class ModelTrainer:
    """Train and evaluate ML models for AQI prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
        self.metrics = {}
        self.feature_importance = {}
        
    def train_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        params: Dict = None
    ) -> RandomForestRegressor:
        """Train Random Forest model"""
        params = params or config.model.random_forest_params
        
        logger.info("Training Random Forest model...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        logger.info("Random Forest training complete")
        return model
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        params: Dict = None
    ) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        params = params or config.model.xgboost_params.copy()
        
        logger.info("Training XGBoost model...")
        # Remove verbose from params if present (will set separately)
        params.pop('verbose', None)
        
        model = xgb.XGBRegressor(**params, verbose=0)
        model.fit(X_train, y_train, verbose=False)
        
        logger.info("XGBoost training complete")
        return model
    
    def train_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        params: Dict = None
    ) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        params = params or config.model.lightgbm_params.copy()
        
        logger.info("Training LightGBM model...")
        # FIXED: Remove verbose from params (it's being passed twice)
        params.pop('verbose', None)
        
        model = lgb.LGBMRegressor(**params)
        # FIXED: Don't pass verbose to fit() - not supported
        model.fit(X_train, y_train)
        
        logger.info("LightGBM training complete")
        return model
    
    def train_ridge(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        params: Dict = None
    ) -> Ridge:
        """Train Ridge Regression model"""
        params = params or config.model.ridge_params
        
        logger.info("Training Ridge Regression model...")
        model = Ridge(**params)
        model.fit(X_train, y_train)
        
        logger.info("Ridge Regression training complete")
        return model
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            "model_name": model_name,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "test_samples": len(y_test)
        }
        
        logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def get_feature_importance(
        self, 
        model: Any, 
        feature_names: list,
        model_name: str
    ) -> pd.DataFrame:
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                logger.warning(f"Cannot extract feature importance from {model_name}")
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            return pd.DataFrame()
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Train all configured models and select the best one - PREFER RIDGE FOR STABILITY"""
        results = {
            'models': {},
            'metrics': {},
            'feature_importance': {}
        }
        
        models_to_train = config.model.models_to_train
        
        # Track all valid models
        valid_models = []
        
        for model_name in models_to_train:
            try:
                # Train model
                if model_name == 'random_forest':
                    model = self.train_random_forest(X_train, y_train)
                elif model_name == 'xgboost':
                    model = self.train_xgboost(X_train, y_train)
                elif model_name == 'lightgbm':
                    model = self.train_lightgbm(X_train, y_train)
                elif model_name == 'ridge_regression':
                    model = self.train_ridge(X_train, y_train)
                else:
                    continue
                
                # Evaluate model
                predictions = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                metrics = {
                    "rmse": float(rmse),
                    "mae": float(mae), 
                    "r2": float(r2),
                }
                
                # Store results
                results['models'][model_name] = model
                results['metrics'][model_name] = metrics
                
                # PREFER RIDGE REGRESSION for stability (less overfitting)
                if model_name == 'ridge_regression' and r2 > 0.5:
                    # Give ridge regression priority
                    valid_models.insert(0, (model_name, r2, rmse, model))
                    logger.info(f"✅ PRIORITY MODEL: {model_name} (R²: {r2:.4f}, RMSE: {rmse:.4f})")
                elif r2 > 0.5:
                    valid_models.append((model_name, r2, rmse, model))
                    logger.info(f"✅ VALID MODEL: {model_name} (R²: {r2:.4f}, RMSE: {rmse:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Select the best valid model
        if valid_models:
            # If Ridge is available and good, use it (more stable)
            ridge_models = [m for m in valid_models if m[0] == 'ridge_regression']
            if ridge_models and ridge_models[0][1] > 0.6:  # If ridge R² > 0.6
                best_model_name, best_r2, best_rmse, best_model = ridge_models[0]
                logger.info(f"🏆 SELECTED RIDGE (STABLE): {best_model_name} (R²: {best_r2:.4f}, RMSE: {best_rmse:.4f})")
            else:
                # Otherwise use best overall
                valid_models.sort(key=lambda x: x[1], reverse=True)
                best_model_name, best_r2, best_rmse, best_model = valid_models[0]
                logger.info(f"🏆 SELECTED BEST MODEL: {best_model_name} (R²: {best_r2:.4f}, RMSE: {best_rmse:.4f})")
            
            self.best_model = best_model
            self.best_model_name = best_model_name
            self.best_score = best_rmse
        else:
            logger.warning("⚠️ No stable models found with R² > 0.5")
        
        self.models = results['models']
        self.metrics = results['metrics']
        
        return results
    
    def save_model(
        self, 
        model: Any = None,
        model_name: str = None,
        metadata: Dict = None
    ):
        """
        Save model to disk
        
        Args:
            model: Model to save (default: best model)
            model_name: Name of the model (default: best model name)
            metadata: Additional metadata to save
        """
        model = model or self.best_model
        model_name = model_name or self.best_model_name
        
        if model is None:
            logger.error("No model to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = config.MODELS_DIR / model_filename
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'model_name': model_name,
            'timestamp': timestamp,
            'version': config.model.model_version,
            'metrics': self.metrics.get(model_name, {}),
        })
        
        metadata_path = config.MODELS_DIR / f"{model_name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Save as latest model
        latest_model_path = config.MODELS_DIR / f"latest_model.pkl"
        with open(latest_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        latest_metadata_path = config.MODELS_DIR / f"latest_model_metadata.json"
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info("Saved as latest model")
    
    def load_model(self, model_path: Path = None) -> Any:
        """Load model from disk"""
        model_path = model_path or (config.MODELS_DIR / "latest_model.pkl")
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def load_metadata(self, metadata_path: Path = None) -> Dict:
        """Load model metadata"""
        metadata_path = metadata_path or (config.MODELS_DIR / "latest_model_metadata.json")
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found at {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def select_top_features(self, X_train: pd.DataFrame, y_train: pd.Series, n_features: int = 15) -> List[str]:
        """Select top n features using Random Forest importance"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Train a simple RF to get feature importance
        selector = RandomForestRegressor(
            n_estimators=20,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        selector.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        logger.info(f"Selected top {len(top_features)} features")
        for idx, row in importance_df.head(n_features).iterrows():
            logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        return top_features
    
    def validate_predictions_quality(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """Validate that predictions are realistic and meaningful"""
        predictions = model.predict(X_test)
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Quality checks
        prediction_std = np.std(predictions)
        actual_std = np.std(y_test)
        
        # Check if predictions have reasonable variance
        variance_ratio = prediction_std / (actual_std + 1e-6)
        
        # Check prediction realism (shouldn't be constant)
        unique_predictions = len(np.unique(predictions.round(1)))
        
        quality_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "variance_ratio": float(variance_ratio),
            "unique_predictions": int(unique_predictions),
            "prediction_std": float(prediction_std),
            "actual_std": float(actual_std),
        }
        
        logger.info(f"{model_name} Quality Check:")
        logger.info(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}")
        logger.info(f"  Variance Ratio: {variance_ratio:.3f} (should be close to 1)")
        logger.info(f"  Unique Predictions: {unique_predictions}")
        
        # Apply penalty if predictions are unrealistic
        if variance_ratio < 0.3 or unique_predictions < 10:
            quality_metrics["rmse"] *= 2.0  # Heavy penalty for bad predictions
            logger.warning(f"{model_name} failed quality checks - applying penalty")
        
        return quality_metrics


def main():
    """Test model training"""
    from src.data_fetcher import OpenWeatherFetcher
    from datetime import timedelta
    
    # Fetch and prepare data
    logger.info("Fetching data...")
    fetcher = OpenWeatherFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    df = fetcher.fetch_air_pollution_history(start_date, end_date)
    
    if df is not None and not df.empty:
        # Engineer features
        logger.info("Engineering features...")
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # Prepare train-test split
        X_train, X_test, y_train, y_test = engineer.prepare_train_test_split(
            df_features, test_size=0.2
        )
        
        # Train models
        logger.info("Training models...")
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save best model
        trainer.save_model()
        
        # Display results
        logger.info("\nModel Performance Summary:")
        for model_name, metrics in results['metrics'].items():
            logger.info(f"{model_name}: RMSE={metrics['rmse']:.4f}, "
                       f"MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")


if __name__ == "__main__":
    main()