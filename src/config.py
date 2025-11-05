"""
Configuration module for AQI Prediction System
FINAL OPTIMIZED - Target R² > 0.6 for all models
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models" / "saved_models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class OpenWeatherConfig(BaseModel):
    """OpenWeather API configuration"""
    api_key: str = os.getenv("OPENWEATHER_API_KEY", "")
    base_url: str = "http://api.openweathermap.org/data/2.5"
    air_pollution_endpoint: str = "/air_pollution"
    air_pollution_history_endpoint: str = "/air_pollution/history"
    air_pollution_forecast_endpoint: str = "/air_pollution/forecast"
    weather_endpoint: str = "/weather"
    forecast_endpoint: str = "/forecast"


class LocationConfig(BaseModel):
    """Location configuration"""
    latitude: float = float(os.getenv("LATITUDE", "24.8607"))
    longitude: float = float(os.getenv("LONGITUDE", "67.0011"))
    city_name: str = os.getenv("CITY_NAME", "Karachi")


class DatabaseConfig(BaseModel):
    """Database configuration"""
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    mongodb_db: str = os.getenv("MONGODB_DB", "aqi_database")
    use_mongodb: bool = os.getenv("USE_MONGODB", "false").lower() == "true"


class HopsworksConfig(BaseModel):
    """Hopsworks feature store configuration"""
    api_key: str = os.getenv("HOPSWORKS_API_KEY", "")
    project_name: str = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_prediction")
    use_hopsworks: bool = os.getenv("USE_HOPSWORKS", "false").lower() == "true"


# class ModelConfig(BaseModel):
#     """Model configuration - SIMPLIFIED to prevent overfitting"""
#     model_version: str = os.getenv("MODEL_VERSION", "v3_simple")
#     prediction_days: int = int(os.getenv("PREDICTION_DAYS", "3"))
#     models_to_train: list = [
#         "ridge_regression",  # Add Ridge back - it was best!
#         "random_forest",
#         "xgboost",
#         "lightgbm"
#     ]
    
#     # DRASTICALLY REDUCED features
#     lag_features: list = [24]  # Only 24h lag (yesterday same time)
#     rolling_windows: list = [24]  # Only 24h window
    
#     # Random Forest - MUCH SIMPLER
#     random_forest_params: dict = {
#         "n_estimators": 50,  # Reduced from 250
#         "max_depth": 6,  # Reduced from 18
#         "min_samples_split": 20,  # Increased
#         "min_samples_leaf": 10,  # Increased
#         "max_features": 0.3,  # Use only 30% of features
#         "bootstrap": True,
#         "random_state": 42,
#         "n_jobs": -1
#     }
    
#     # XGBoost - SIMPLIFIED
#     xgboost_params: dict = {
#         "n_estimators": 100,
#         "max_depth": 4,
#         "learning_rate": 0.1,
#         "subsample": 0.7,
#         "colsample_bytree": 0.7,
#         "min_child_weight": 5,
#         "gamma": 0.5,
#         "reg_alpha": 1.0,
#         "reg_lambda": 2.0,
#         "random_state": 42,
#         "n_jobs": -1
#     }
    
#     # LightGBM - SIMPLIFIED
#     lightgbm_params: dict = {
#         "n_estimators": 100,
#         "max_depth": 5,
#         "learning_rate": 0.1,
#         "num_leaves": 15,  # Reduced from 50
#         "min_child_samples": 20,
#         "subsample": 0.7,
#         "colsample_bytree": 0.7,
#         "reg_alpha": 0.5,
#         "reg_lambda": 1.0,
#         "random_state": 42,
#         "n_jobs": -1,
#         "verbose": -1
#     }
    
#     # Ridge - Keep strong
#     ridge_params: dict = {
#         "alpha": 50.0,  # Strong regularization
#         "random_state": 42,
#         "max_iter": 10000
#     }

class ModelConfig(BaseModel):
    """ULTRA-SIMPLE model configuration"""
    model_version: str = os.getenv("MODEL_VERSION", "v4_simple")
    prediction_days: int = int(os.getenv("PREDICTION_DAYS", "3"))
    models_to_train: list = ["random_forest", "ridge_regression", "xgboost"]  # Only 3 models
    
    # Minimal features
    lag_features: list = [24]  # Only 24h lag
    rolling_windows: list = [24]  # Only 24h window
    
    # Random Forest - OPTIMIZED for temporal patterns
    random_forest_params: dict = {
        "n_estimators": 100,  # Increased for better learning
        "max_depth": 8,       # Balanced depth
        "min_samples_split": 15,
        "min_samples_leaf": 8,
        "max_features": 0.6,  # Use more features for variety
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # XGBoost - Good for temporal data
    xgboost_params: dict = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Ridge - Strong regularization
    ridge_params: dict = {
        "alpha": 150.0,  # Even stronger
        "random_state": 42,
        "max_iter": 10000
    }

class PipelineConfig(BaseModel):
    """Pipeline configuration - UPDATED FOR MORE DATA"""
    feature_pipeline_schedule: str = os.getenv("FEATURE_PIPELINE_SCHEDULE", "hourly")
    training_pipeline_schedule: str = os.getenv("TRAINING_PIPELINE_SCHEDULE", "daily")
    
    # Data collection parameters - INCREASED
    backfill_days: int = 180  # 6 months for better seasonal patterns
    batch_size_days: int = 30
    
    # Training parameters
    test_size: float = 0.15
    validation_size: float = 0.1
    
    # Minimum data requirements - ADJUSTED
    min_training_samples: int = 500  # Reduced since we're getting more temporal data


class Config:
    """Main configuration class"""
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    RAW_DATA_DIR: Path = RAW_DATA_DIR
    PROCESSED_DATA_DIR: Path = PROCESSED_DATA_DIR
    FEATURES_DIR: Path = FEATURES_DIR
    MODELS_DIR: Path = MODELS_DIR
    openweather = OpenWeatherConfig()
    location = LocationConfig()
    database = DatabaseConfig()
    hopsworks = HopsworksConfig()
    model = ModelConfig()
    pipeline = PipelineConfig()
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Dashboard Configuration
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8501"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # AQI Thresholds for alerts
    aqi_thresholds = {
        "good": (0, 50),
        "moderate": (51, 100),
        "unhealthy_sensitive": (101, 150),
        "unhealthy": (151, 200),
        "very_unhealthy": (201, 300),
        "hazardous": (301, 500)
    }
    
    @staticmethod
    def validate_config():
        """Validate critical configuration"""
        config = Config()
        
        if not config.openweather.api_key:
            raise ValueError("OPENWEATHER_API_KEY not set in environment variables")
        
        return True


# Create global config instance
config = Config()

if __name__ == "__main__":
    # Test configuration
    try:
        config.validate_config()
        print("✅ Configuration validated successfully")
        print(f"📍 Location: {config.location.city_name}")
        print(f"🌐 Latitude: {config.location.latitude}")
        print(f"🌐 Longitude: {config.location.longitude}")
        print(f"📊 Models to train: {', '.join(config.model.models_to_train)}")
        print(f"🎯 Lag features: {config.model.lag_features}")
        print(f"📊 Rolling windows: {config.model.rolling_windows}")
        print(f"🔧 Test size: {config.pipeline.test_size}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")