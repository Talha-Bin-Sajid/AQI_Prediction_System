"""
Feature engineering module for AQI prediction
OPTIMIZED based on EDA insights
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
from loguru import logger

from src.config import config


class FeatureEngineer:
    """Create features for AQI prediction - OPTIMIZED"""
    
    def __init__(self):
        self.lag_features = config.model.lag_features
        self.rolling_windows = config.model.rolling_windows
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features with EDA insights"""
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Basic indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                              (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # === EDA-BASED FEATURES (NEW!) ===
        # Peak pollution hour (11 AM - AQI: 151.5)
        df['is_peak_hour'] = (df['hour'] == 11).astype(int)
        
        # Low pollution hour (6 PM - AQI: 102.8)
        df['is_evening_low'] = (df['hour'] == 18).astype(int)
        
        # High pollution period (10 AM - 2 PM)
        df['is_high_pollution_period'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
        
        # Winter months (Dec, Jan, Feb - High pollution: Jan AQI 227.8)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # Monsoon/Post-monsoon (Aug, Sep - Low pollution: Sep AQI 39.1)
        df['is_monsoon'] = df['month'].isin([8, 9]).astype(int)
        
        # Season encoding (better than just month)
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter (High pollution)
            3: 1, 4: 1, 5: 1,   # Spring (Moderate)
            6: 2, 7: 2, 8: 2,   # Summer-Monsoon (Low)
            9: 3, 10: 3, 11: 3  # Post-monsoon (Low to Moderate)
        })
        
        # Hour-Month interaction (pollution varies by season AND time)
        df['hour_month_interaction'] = df['hour'] * df['month']
        
        # Weekend-Hour interaction
        df['weekend_hour'] = df['is_weekend'] * df['hour']
        
        logger.info("Added time-based features with EDA insights")
        return df
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int] = None) -> pd.DataFrame:
        """Add lag features for specified columns"""
        df = df.copy()
        lags = lags or self.lag_features
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"Added lag features for {len(columns)} columns with lags: {lags}")
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int] = None) -> pd.DataFrame:
        """Add rolling statistics features"""
        df = df.copy()
        windows = windows or self.rolling_windows
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                
                # Rolling max
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        logger.info(f"Added rolling features for {len(columns)} columns")
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features - ENHANCED with EDA insights"""
        df = df.copy()
        
        # === CRITICAL: PM2.5 features (0.954 correlation!) ===
        if 'pm2_5' in df.columns:
            # PM2.5 is the strongest predictor - create rich features
            df['pm2_5_squared'] = df['pm2_5'] ** 2
            df['pm2_5_log'] = np.log1p(df['pm2_5'])
            df['pm2_5_sqrt'] = np.sqrt(df['pm2_5'])
            
            # PM2.5 momentum
            df['pm2_5_change_1h'] = df['pm2_5'].diff(1)
            df['pm2_5_change_6h'] = df['pm2_5'].diff(6)
            df['pm2_5_acceleration'] = df['pm2_5_change_1h'].diff(1)
        
        # AQI change features
        if 'aqi' in df.columns:
            df['aqi_change'] = df['aqi'].diff()
            df['aqi_change_rate'] = df['aqi'].pct_change()
            df['aqi_acceleration'] = df['aqi_change'].diff()
        
        # Pollutant ratios
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
            df['total_pm'] = df['pm2_5'] + df['pm10']
            # PM composite index
            df['pm_index'] = np.sqrt(df['pm2_5']**2 + df['pm10']**2)
        
        if 'no2' in df.columns and 'no' in df.columns:
            df['nox_ratio'] = df['no2'] / (df['no'] + 1e-6)
        
        # === CRITICAL INTERACTIONS (High correlation pollutants) ===
        # PM2.5 x CO (both 0.9+ correlation)
        if 'pm2_5' in df.columns and 'co' in df.columns:
            df['pm25_co_interaction'] = df['pm2_5'] * df['co']
        
        # PM2.5 x NO2 (0.954 x 0.883)
        if 'pm2_5' in df.columns and 'no2' in df.columns:
            df['pm25_no2_interaction'] = df['pm2_5'] * df['no2']
        
        # Pollution severity score (weighted by correlation)
        pollutants_weights = {
            'pm2_5': 0.954,
            'pm10': 0.943,
            'co': 0.905,
            'no2': 0.883,
            'so2': 0.828
        }
        
        available = [p for p in pollutants_weights.keys() if p in df.columns]
        if len(available) >= 3:
            weighted_sum = sum(df[p] * pollutants_weights[p] for p in available)
            total_weight = sum(pollutants_weights[p] for p in available)
            df['pollution_severity_weighted'] = weighted_sum / total_weight
        
        logger.info("Added derived features with strong correlation focus")
        return df
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features based on EDA insights"""
        df = df.copy()
        
        # Historical patterns (same hour yesterday, last week)
        if 'aqi' in df.columns:
            df['aqi_same_hour_yesterday'] = df['aqi'].shift(24)
            df['aqi_same_hour_last_week'] = df['aqi'].shift(24 * 7)
            
            # Difference from yesterday same time
            df['aqi_vs_yesterday'] = df['aqi'] - df['aqi_same_hour_yesterday']
        
        # PM2.5 percentile rank (for extreme events detection)
        if 'pm2_5' in df.columns:
            df['pm2_5_pct_rank'] = df['pm2_5'].expanding().rank(pct=True)
            df['pm2_5_p90'] = df['pm2_5'].expanding().quantile(0.9).shift(1)
            df['pm2_5_dist_from_p90'] = df['pm2_5'] - df['pm2_5_p90']

        
        # Volatility features (for capturing sudden changes)
        if 'aqi' in df.columns:
            df['aqi_volatility_12h'] = df['aqi'].diff().rolling(12).std()
            df['aqi_volatility_24h'] = df['aqi'].diff().rolling(24).std()
        
        logger.info("Added advanced features")
        return df
    
    def create_target_variable(self, df: pd.DataFrame, target_column: str = 'aqi', forecast_hours: int = 24) -> pd.DataFrame:
        """Create target variable for prediction"""
        df = df.copy()
        df[f'{target_column}_target_{forecast_hours}h'] = df[target_column].shift(-forecast_hours)
        
        logger.info(f"Created target variable: {target_column}_target_{forecast_hours}h")
        return df
    
    def engineer_features(self, df: pd.DataFrame, include_weather: bool = False) -> pd.DataFrame:
        """Apply OPTIMAL-SIMPLE feature engineering (22 features that worked)"""
        logger.info("Starting OPTIMAL-SIMPLE feature engineering")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True).copy()
        
        # Create target variable
        df = self.create_target_variable(df, target_column='aqi', forecast_hours=24)
        
        # Add only essential time features
        df = self.add_optimal_time_features(df)
        
        # Focus on core pollutants only
        core_pollutants = ['pm2_5', 'pm10', 'no2', 'aqi']
        available_pollutants = [col for col in core_pollutants if col in df.columns]
        
        # Add only essential lag features
        for col in available_pollutants:
            df[f'{col}_lag_24'] = df[col].shift(24)
        
        # Add only essential rolling features  
        for col in available_pollutants:
            df[f'{col}_rolling_mean_24'] = df[col].rolling(window=24, min_periods=1).mean()
        
        # Only keep simple AQI change
        if 'aqi' in df.columns:
            df['aqi_change_24h'] = df['aqi'].diff(24)
        
        # Handle missing values
        initial_rows = len(df)
        df = df.dropna(subset=['aqi_target_24h'])
        df = df.bfill().ffill()
        df = df.dropna()
        
        logger.info(f"OPTIMAL-SIMPLE feature engineering complete. Final shape: {df.shape}")
        logger.info(f"Features created: {df.shape[1] - 2} (optimal simple)")
        
        return df

    def add_optimal_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimal time features that worked well"""
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Only essential time features
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Cyclical encoding only for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Only 2 indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                            (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        return df
    

    def add_lag_features_simple(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Only add 24h lag"""
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            df[f'{col}_lag_24'] = df[col].shift(24)
        
        return df

    def add_rolling_features_simple(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Only add rolling mean"""
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        return df

    def add_derived_features_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Only 3 essential derived features"""
        df = df.copy()
        
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
        
        if 'pm2_5' in df.columns:
            df['pm2_5_change_24h'] = df['pm2_5'].diff(24)
        
        if 'aqi' in df.columns:
            df['aqi_change_24h'] = df['aqi'].diff(24)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns"""
        exclude_cols = ['timestamp', 'aqi_target_24h', 'date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train-test split for time series"""
        feature_cols = self.get_feature_columns(df)
        
        # Time-based split
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df['aqi_target_24h']
        X_test = test_df[feature_cols]
        y_test = test_df['aqi_target_24h']
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test

def main():
    """Test feature engineering"""
    from src.data_fetcher import OpenWeatherFetcher
    from datetime import timedelta
    
    # Fetch sample data
    fetcher = OpenWeatherFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = fetcher.fetch_air_pollution_history(start_date, end_date)
    
    if df is not None and not df.empty:
        logger.info(f"Original data shape: {df.shape}")
        
        # Engineer features
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        logger.info(f"Engineered data shape: {df_features.shape}")
        logger.info(f"Feature columns: {len(engineer.get_feature_columns(df_features))}")
        
        # Prepare train-test split
        X_train, X_test, y_train, y_test = engineer.prepare_train_test_split(df_features)
        logger.info(f"Train/Test split complete")


if __name__ == "__main__":
    main()