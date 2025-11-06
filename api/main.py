"""
FastAPI backend for AQI prediction system (IMPROVED VERSION)
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from src.config import config
from src.data_fetcher import OpenWeatherFetcher
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer


# ----------------------------------
# FastAPI Initialization
# ----------------------------------
app = FastAPI(
    title="AQI Prediction API",
    description="API for Air Quality Index prediction",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
feature_engineer = None
fetcher = None
latest_data = None


# ----------------------------------
# Helper utilities
# ----------------------------------
def get_aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


def load_model_and_components():
    """Load model and initialize components"""
    global model, feature_engineer, fetcher
    try:
        trainer = ModelTrainer()
        model = trainer.load_model()

        if model is None:
            logger.warning("No trained model found. Train a model first.")
            return False

        feature_engineer = FeatureEngineer()
        fetcher = OpenWeatherFetcher()
        logger.info("✅ Model + components loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return False


def update_latest_data():
    """Fetch last 7 days & store in memory"""
    global latest_data
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        df = fetcher.fetch_air_pollution_history(start_date, end_date)

        if df is not None and not df.empty:
            latest_data = df
            logger.info(f"Latest AQI records updated → {len(df)} rows")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating latest data: {e}")
        return False


# ----------------------------------
# FastAPI Startup Hook
# ----------------------------------
@app.on_event("startup")
async def startup_event():
    load_model_and_components()
    update_latest_data()


# ----------------------------------
# Pydantic Models
# ----------------------------------
class LocationInfo(BaseModel):
    latitude: float
    longitude: float
    city: str


class PredictionResponse(BaseModel):
    timestamp: datetime
    predicted_aqi: float
    aqi_category: str
    confidence: Optional[float] = None


class CurrentAQIResponse(BaseModel):
    timestamp: datetime
    aqi: float
    aqi_category: str
    pollutants: Dict[str, float]
    location: LocationInfo


class HistoricalDataPoint(BaseModel):
    timestamp: datetime
    aqi: float
    pm2_5: float
    pm10: float
    no2: float
    o3: float


class ModelMetrics(BaseModel):
    model_name: str
    rmse: float
    mae: float
    r2: float
    last_trained: Optional[str] = None


# ----------------------------------
# API ROUTES
# ----------------------------------
@app.get("/")
async def root():
    return {
        "message": "AQI Prediction API",
        "version": "1.1.0",
        "endpoints": {
            "current": "/api/current",
            "predict": "/api/predict",
            "historical": "/api/historical",
            "model_info": "/api/model/info"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "model_loaded": model is not None}


# FIX: current AQI endpoint
@app.get("/api/current", response_model=CurrentAQIResponse)
async def get_current_aqi():
    if fetcher is None:
        raise HTTPException(status_code=500, detail="Fetcher not initialized")

    try:
        current_data = fetcher.fetch_current_air_pollution()
        if not current_data:
            raise HTTPException(status_code=503, detail="Failed to fetch AQI")

        return CurrentAQIResponse(
            timestamp=current_data["timestamp"],
            aqi=current_data["aqi"],
            aqi_category=get_aqi_category(current_data["aqi"]),
            pollutants={k: current_data[k] for k in ["pm2_5", "pm10", "no2", "o3", "so2", "co"]},
            location=LocationInfo(
                latitude=config.location.latitude,
                longitude=config.location.longitude,
                city=config.location.city_name,
            ),
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


# IMPROVED PREDICTION LOGIC
@app.get("/api/predict", response_model=List[PredictionResponse])
async def predict_aqi(days: int = 3):
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if latest_data is None:
        update_latest_data()

    df_features = feature_engineer.engineer_features(latest_data.copy())
    feature_cols = feature_engineer.get_feature_columns(df_features)

    current_features = df_features[feature_cols].iloc[-1:].copy()
    current_aqi = latest_data["aqi"].iloc[-1]
    last_timestamp = latest_data["timestamp"].iloc[-1]

    predictions = []

    for day in range(1, days + 1):
        pred_time = last_timestamp + timedelta(days=day)

        model_pred = get_model_prediction(current_features, pred_time)

        adjusted_pred = apply_realistic_adjustments(model_pred, current_aqi)

        # NEW: update lag features so next day's prediction uses today's output
        current_features["aqi_lag_1"] = adjusted_pred
        current_aqi = adjusted_pred

        predictions.append(
            PredictionResponse(
                timestamp=pred_time,
                predicted_aqi=float(adjusted_pred),
                aqi_category=get_aqi_category(adjusted_pred),
                confidence=calculate_confidence(day, adjusted_pred),
            )
        )

    return predictions


def get_model_prediction(current_features: pd.DataFrame, pred_time: datetime):
    # copy current state
    future = current_features.copy()

    # add temporal effects
    future["hour"] = pred_time.hour
    future["day_of_week"] = pred_time.weekday()
    future["month"] = pred_time.month
    future["hour_sin"] = np.sin(2 * np.pi * pred_time.hour / 24)
    future["hour_cos"] = np.cos(2 * np.pi * pred_time.hour / 24)

    # FILTER FEATURE COLUMNS TO MATCH TRAINING
    try:
        future = future[model.feature_names_in_]
    except Exception as e:
        print(" Feature mismatch:", future.columns, model.feature_names_in_)

    return model.predict(future)[0]




def apply_realistic_adjustments(pred: float, last_aqi: float) -> float:
    """Smooth category transitions instead of hard % bounding"""
    max_jump = 70  # ✅ avoid unnatural spikes
    pred = last_aqi + np.clip(pred - last_aqi, -max_jump, max_jump)
    return float(np.clip(pred, 40, 350))


def calculate_confidence(day: int, predicted_aqi: float) -> float:
    return max(0.5, min(0.95 - (day * 0.1), 0.9))


@app.get("/api/historical", response_model=List[HistoricalDataPoint])
async def get_historical(days: int = 7):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = fetcher.fetch_air_pollution_history(start_date, end_date)

    return [
        HistoricalDataPoint(
            timestamp=row["timestamp"],
            aqi=float(row["aqi"]),
            pm2_5=float(row["pm2_5"]),
            pm10=float(row["pm10"]),
            no2=float(row["no2"]),
            o3=float(row["o3"]),
        )
        for _, row in df.iterrows()
    ]


@app.get("/api/model/info", response_model=ModelMetrics)
async def model_info():
    trainer = ModelTrainer()
    metadata = trainer.load_metadata()

    return ModelMetrics(
        model_name=metadata.get("model_name", "N/A"),
        rmse=metadata["metrics"]["rmse"],
        mae=metadata["metrics"]["mae"],
        r2=metadata["metrics"]["r2"],
        last_trained=metadata["timestamp"],
    )


@app.post("/api/model/reload")
async def reload_model():
    load_model_and_components()
    return {"message": "Model reloaded successfully"}


@app.post("/api/data/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_latest_data)
    return {"message": "Refreshing in background"}


# ----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.api_host, port=config.api_port, reload=True)
