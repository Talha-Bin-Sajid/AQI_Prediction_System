"""Test proper model-based predictions with realistic adjustments"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.config import config

def test_proper_predictions():
    """Test proper model-based predictions with adjustments"""
    
    # Load model
    trainer = ModelTrainer()
    model = trainer.load_model()
    
    if model is None:
        print("❌ No model found")
        return
    
    # Load data
    historical_file = config.RAW_DATA_DIR / f"historical_aqi_{config.location.city_name.lower()}.csv"
    df = pd.read_csv(historical_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns(df_features)
    
    # Get the last available data point
    current_features = df_features[feature_cols].iloc[-1:].copy()
    current_aqi = df['aqi'].iloc[-1]
    
    current_time = datetime.now()
    
    print(f"Current AQI: {current_aqi:.1f}")
    print("\nProper Model-Based Predictions with Adjustments:")
    print("=" * 50)
    
    predictions = []
    
    for day in range(3):
        pred_time = current_time + timedelta(days=day + 1)
        
        # Get model prediction
        model_pred = get_model_prediction(model, current_features, pred_time, day)
        
        # Apply realistic adjustments
        adjusted_pred = apply_realistic_adjustments(
            model_pred=model_pred,
            current_aqi=current_aqi,
            prediction_day=day + 1,
            pred_time=pred_time
        )
        
        predictions.append(adjusted_pred)
        
        print(f"Day {day+1} ({pred_time.strftime('%a, %b %d')}):")
        print(f"  Model Prediction: {model_pred:.1f}")
        print(f"  Adjusted Prediction: {adjusted_pred:.1f}")
        print(f"  Category: {get_aqi_category(adjusted_pred)}")
        print(f"  Change from current: {adjusted_pred - current_aqi:+.1f}")
        print()
    
    print("=" * 50)
    print(f"Prediction Summary:")
    print(f"Range: {min(predictions):.1f} - {max(predictions):.1f}")
    print(f"Variation: {max(predictions) - min(predictions):.1f} AQI points")
    print(f"Average: {np.mean(predictions):.1f}")

def get_model_prediction(model, current_features: pd.DataFrame, pred_time: datetime, day: int) -> float:
    """Get prediction from model with temporal updates"""
    day_features = current_features.copy()
    
    # Update temporal features
    pred_hour = df['timestamp'].iloc[-1].hour
    day_features['hour'] = pred_hour
    day_features['day_of_week'] = pred_time.weekday()
    day_features['month'] = pred_time.month
    day_features['hour_sin'] = np.sin(2 * np.pi * pred_hour / 24)
    day_features['hour_cos'] = np.cos(2 * np.pi * pred_hour / 24)
    
    is_weekend = pred_time.weekday() >= 5
    is_peak_hour = (pred_hour >= 7 and pred_hour <= 9) or (pred_hour >= 17 and pred_hour <= 19)
    
    day_features['is_weekend'] = int(is_weekend)
    day_features['is_peak_hour'] = int(is_peak_hour)
    
    return model.predict(day_features)[0]

def apply_realistic_adjustments(model_pred: float, current_aqi: float, 
                              prediction_day: int, pred_time: datetime) -> float:
    """Apply realistic adjustments to model prediction"""
    historical_mean = 120
    reversion_strength = 0.4
    
    # Reversion to mean
    current_deviation = current_aqi - historical_mean
    reversion_amount = current_deviation * (reversion_strength * prediction_day)
    adjusted_pred = model_pred - reversion_amount
    
    # Seasonal adjustment
    month = pred_time.month
    if month in [12, 1, 2]:
        adjusted_pred *= 1.15
    elif month in [6, 7, 8, 9]:
        adjusted_pred *= 0.85
    
    # Day-of-week adjustment
    if pred_time.weekday() >= 5:
        adjusted_pred *= 0.95
    else:
        adjusted_pred *= 1.05
    
    # Realistic bounds
    max_change_factor = 0.25
    realistic_min = current_aqi * (1 - max_change_factor * prediction_day)
    realistic_max = current_aqi * (1 + max_change_factor * prediction_day)
    adjusted_pred = max(realistic_min, min(realistic_max, adjusted_pred))
    adjusted_pred = max(50, min(300, adjusted_pred))
    
    return adjusted_pred

def get_aqi_category(aqi: float) -> str:
    """Get AQI category from AQI value"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

if __name__ == "__main__":
    test_proper_predictions()