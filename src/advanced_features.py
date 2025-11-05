"""
Advanced Feature Engineering for Better Model Performance
Adds domain-specific features based on EDA insights
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import List


class AdvancedFeatureEngineer:
    """Advanced features based on domain knowledge and EDA"""
    
    @staticmethod
    def add_temporal_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal interaction features"""
        df = df.copy()
        
        # Hour-month interactions (seasonal patterns)
        df['hour_month_interaction'] = df['hour'] * df['month']
        
        # Weekend-hour interaction (different patterns on weekends)
        df['weekend_hour'] = df['is_weekend'] * df['hour']
        
        # Rush hour - month interaction (traffic patterns vary by season)
        df['rush_month'] = df['is_rush_hour'] * df['month']
        
        logger.info("Added temporal interaction features")
        return df
    
    @staticmethod
    def add_air_quality_indices(df: pd.DataFrame) -> pd.DataFrame:
        """Add composite air quality indices"""
        df = df.copy()
        
        # Particulate Matter Index (focus on PM2.5 and PM10)
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm_index'] = np.sqrt(df['pm2_5']**2 + df['pm10']**2)
            df['pm_weighted'] = 0.7 * df['pm2_5'] + 0.3 * df['pm10']
        
        # Gaseous Pollutants Index
        gas_cols = ['no2', 'o3', 'so2', 'co']
        available_gas = [col for col in gas_cols if col in df.columns]
        if len(available_gas) >= 2:
            # Normalize and combine
            for col in available_gas:
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
            
            df['gas_index'] = df[[f'{col}_norm' for col in available_gas]].mean(axis=1)
        
        # Overall pollution severity score
        if 'pm2_5' in df.columns and 'no2' in df.columns:
            df['pollution_severity'] = (df['pm2_5'] * 0.4 + 
                                       df['pm10'] * 0.2 + 
                                       df['no2'] * 0.2 + 
                                       df['o3'] * 0.2)
        
        logger.info("Added air quality indices")
        return df
    
    @staticmethod
    def add_change_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """Add change momentum features (rate of change over time)"""
        df = df.copy()
        
        pollutants = ['aqi', 'pm2_5', 'pm10', 'no2', 'o3']
        
        for pollutant in pollutants:
            if pollutant in df.columns:
                # 1-hour change
                df[f'{pollutant}_delta_1h'] = df[pollutant].diff(1)
                
                # 6-hour change
                df[f'{pollutant}_delta_6h'] = df[pollutant].diff(6)
                
                # Acceleration (change in change)
                df[f'{pollutant}_acceleration'] = df[f'{pollutant}_delta_1h'].diff(1)
                
                # Volatility (rolling std of changes)
                df[f'{pollutant}_volatility'] = df[f'{pollutant}_delta_1h'].rolling(
                    window=12, min_periods=1
                ).std()
        
        logger.info("Added change momentum features")
        return df
    
    @staticmethod
    def add_historical_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add historical pattern features"""
        df = df.copy()
        
        # Same hour yesterday
        if 'aqi' in df.columns:
            df['aqi_same_hour_yesterday'] = df['aqi'].shift(24)
            df['aqi_same_hour_last_week'] = df['aqi'].shift(24 * 7)
        
        # Hour-specific averages (expanding mean)
        for hour in range(24):
            mask = df['hour'] == hour
            df.loc[mask, 'hour_avg_expanding'] = df.loc[mask, 'aqi'].expanding().mean().shift(1)
        
        # Day of week specific averages
        for dow in range(7):
            mask = df['day_of_week'] == dow
            df.loc[mask, 'dow_avg_expanding'] = df.loc[mask, 'aqi'].expanding().mean().shift(1)
        
        logger.info("Added historical pattern features")
        return df
    
    @staticmethod
    def add_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add percentile-based features"""
        df = df.copy()
        
        pollutants = ['aqi', 'pm2_5', 'pm10']
        
        for pollutant in pollutants:
            if pollutant in df.columns:
                # Calculate expanding percentiles
                df[f'{pollutant}_pct_rank'] = df[pollutant].expanding().rank(pct=True)
                
                # Distance from 90th percentile (high pollution indicator)
                p90 = df[pollutant].expanding().quantile(0.9)
                df[f'{pollutant}_dist_from_p90'] = df[pollutant] - p90
        
        logger.info("Added percentile features")
        return df
    
    @staticmethod
    def add_all_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all advanced feature engineering"""
        logger.info("Starting advanced feature engineering...")
        
        df = AdvancedFeatureEngineer.add_temporal_interactions(df)
        df = AdvancedFeatureEngineer.add_air_quality_indices(df)
        df = AdvancedFeatureEngineer.add_change_momentum(df)
        df = AdvancedFeatureEngineer.add_historical_patterns(df)
        df = AdvancedFeatureEngineer.add_percentile_features(df)
        
        # Fill NaN values created by new features
        df = df.bfill().ffill()
        
        logger.info(f"Advanced features added. New shape: {df.shape}")
        return df