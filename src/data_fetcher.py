"""
Data fetcher module to collect air quality and weather data from OpenWeather API
ENHANCED FOR 365-DAY DATA COLLECTION
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import json
from pathlib import Path

from src.config import config


class AQICalculator:
    """Calculate AQI from pollutant concentrations"""
    
    @staticmethod
    def calculate_pm25_aqi(pm25: float) -> float:
        """Calculate AQI from PM2.5 concentration (μg/m³)"""
        # PM2.5 AQI breakpoints (US EPA standard)
        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        
        for c_low, c_high, aqi_low, aqi_high in breakpoints:
            if c_low <= pm25 <= c_high:
                aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
                return round(aqi, 2)
        
        # If concentration exceeds all breakpoints
        if pm25 > 500.4:
            return 500.0
        return 0.0
    
    @staticmethod
    def calculate_pm10_aqi(pm10: float) -> float:
        """Calculate AQI from PM10 concentration (μg/m³)"""
        breakpoints = [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500),
        ]
        
        for c_low, c_high, aqi_low, aqi_high in breakpoints:
            if c_low <= pm10 <= c_high:
                aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm10 - c_low) + aqi_low
                return round(aqi, 2)
        
        if pm10 > 604:
            return 500.0
        return 0.0
    
    @staticmethod
    def calculate_o3_aqi(o3: float) -> float:
        """Calculate AQI from O3 concentration (μg/m³)"""
        # Convert μg/m³ to ppb (approximate: divide by 2)
        o3_ppb = o3 / 2.0
        
        breakpoints = [
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
        ]
        
        for c_low, c_high, aqi_low, aqi_high in breakpoints:
            if c_low <= o3_ppb <= c_high:
                aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (o3_ppb - c_low) + aqi_low
                return round(aqi, 2)
        
        if o3_ppb > 200:
            return 300.0
        return 0.0
    
    @staticmethod
    def calculate_no2_aqi(no2: float) -> float:
        """Calculate AQI from NO2 concentration (μg/m³)"""
        # Convert μg/m³ to ppb (approximate: divide by 1.88)
        no2_ppb = no2 / 1.88
        
        breakpoints = [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
        ]
        
        for c_low, c_high, aqi_low, aqi_high in breakpoints:
            if c_low <= no2_ppb <= c_high:
                aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (no2_ppb - c_low) + aqi_low
                return round(aqi, 2)
        
        if no2_ppb > 1249:
            return 300.0
        return 0.0
    
    @staticmethod
    def calculate_overall_aqi(pm25: float, pm10: float, o3: float, no2: float) -> float:
        """Calculate overall AQI (maximum of individual pollutant AQIs)"""
        aqis = [
            AQICalculator.calculate_pm25_aqi(pm25),
            AQICalculator.calculate_pm10_aqi(pm10),
            AQICalculator.calculate_o3_aqi(o3),
            AQICalculator.calculate_no2_aqi(no2),
        ]
        return max(aqis)


class OpenWeatherFetcher:
    """Fetch air quality and weather data from OpenWeather API - ENHANCED FOR 365 DAYS"""
    
    def __init__(self):
        self.api_key = config.openweather.api_key
        self.base_url = config.openweather.base_url
        self.lat = config.location.latitude
        self.lon = config.location.longitude
        self.city_name = config.location.city_name
        self.aqi_calculator = AQICalculator()
        
        if not self.api_key:
            raise ValueError("OpenWeather API key not found. Please set OPENWEATHER_API_KEY in .env")
    
    def fetch_air_pollution_history(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical air pollution data
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            DataFrame with air pollution data
        """
        start_unix = int(start_date.timestamp())
        end_unix = int(end_date.timestamp())
        
        url = f"{self.base_url}{config.openweather.air_pollution_history_endpoint}"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "start": start_unix,
            "end": end_unix,
            "appid": self.api_key
        }
        
        try:
            logger.info(f"📡 Fetching air pollution data from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=60)  # Increased timeout
            response.raise_for_status()
            
            data = response.json()
            
            if "list" not in data:
                logger.warning("❌ No air pollution data found in response")
                return None
            
            records = []
            for item in data["list"]:
                # Get pollutant concentrations
                pm2_5 = item["components"]["pm2_5"]
                pm10 = item["components"]["pm10"]
                o3 = item["components"]["o3"]
                no2 = item["components"]["no2"]
                
                # Calculate proper AQI (0-500 scale)
                calculated_aqi = self.aqi_calculator.calculate_overall_aqi(
                    pm2_5, pm10, o3, no2
                )
                
                record = {
                    "timestamp": datetime.fromtimestamp(item["dt"]),
                    "aqi": calculated_aqi,  # Use calculated AQI instead of API's 1-5 scale
                    "aqi_category": item["main"]["aqi"],  # Keep original for reference
                    "co": item["components"]["co"],
                    "no": item["components"]["no"],
                    "no2": no2,
                    "o3": o3,
                    "so2": item["components"]["so2"],
                    "pm2_5": pm2_5,
                    "pm10": pm10,
                    "nh3": item["components"]["nh3"],
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"✅ Fetched {len(df)} air pollution records")
            logger.info(f"🌈 AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching air pollution data: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    def fetch_weather_forecast(self) -> Optional[pd.DataFrame]:
        """
        Fetch 5-day weather forecast
        
        Returns:
            DataFrame with weather forecast data
        """
        url = f"{self.base_url}{config.openweather.forecast_endpoint}"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            logger.info("🌤️ Fetching weather forecast")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "list" not in data:
                logger.warning("❌ No weather forecast data found")
                return None
            
            records = []
            for item in data["list"]:
                record = {
                    "timestamp": datetime.fromtimestamp(item["dt"]),
                    "temp": item["main"]["temp"],
                    "feels_like": item["main"]["feels_like"],
                    "temp_min": item["main"]["temp_min"],
                    "temp_max": item["main"]["temp_max"],
                    "pressure": item["main"]["pressure"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"],
                    "wind_deg": item["wind"]["deg"],
                    "clouds": item["clouds"]["all"],
                    "weather_main": item["weather"][0]["main"],
                    "weather_description": item["weather"][0]["description"],
                }
                
                # Optional fields
                if "rain" in item:
                    record["rain_3h"] = item["rain"].get("3h", 0)
                else:
                    record["rain_3h"] = 0
                
                if "snow" in item:
                    record["snow_3h"] = item["snow"].get("3h", 0)
                else:
                    record["snow_3h"] = 0
                
                records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"✅ Fetched {len(df)} weather forecast records")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching weather forecast: {e}")
            return None
    
    def fetch_current_air_pollution(self) -> Optional[Dict]:
        """
        Fetch current air pollution data
        
        Returns:
            Dictionary with current air pollution data
        """
        url = f"{self.base_url}{config.openweather.air_pollution_endpoint}"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key
        }
        
        try:
            logger.info("🌫️ Fetching current air pollution")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "list" not in data or len(data["list"]) == 0:
                logger.warning("❌ No current air pollution data found")
                return None
            
            item = data["list"][0]
            
            # Get pollutant concentrations
            pm2_5 = item["components"]["pm2_5"]
            pm10 = item["components"]["pm10"]
            o3 = item["components"]["o3"]
            no2 = item["components"]["no2"]
            
            # Calculate proper AQI
            calculated_aqi = self.aqi_calculator.calculate_overall_aqi(
                pm2_5, pm10, o3, no2
            )
            
            record = {
                "timestamp": datetime.fromtimestamp(item["dt"]),
                "aqi": calculated_aqi,
                "aqi_category": item["main"]["aqi"],
                "co": item["components"]["co"],
                "no": item["components"]["no"],
                "no2": no2,
                "o3": o3,
                "so2": item["components"]["so2"],
                "pm2_5": pm2_5,
                "pm10": pm10,
                "nh3": item["components"]["nh3"],
            }
            
            logger.info(f"✅ Fetched current air pollution data - AQI: {calculated_aqi:.1f}")
            return record
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching current air pollution: {e}")
            return None
    
    def backfill_historical_data(
        self, 
        days: int = 365,  # Default to 365 days
        batch_days: int = 30
    ) -> pd.DataFrame:
        """
        Backfill historical air pollution data for 365 days
        
        Args:
            days: Number of days to backfill (365 for full year)
            batch_days: Days per batch (API limitation)
            
        Returns:
            DataFrame with all historical data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        current_start = start_date
        successful_batches = 0
        total_batches = (days + batch_days - 1) // batch_days  # Ceiling division
        
        logger.info(f"📦 Backfilling {days} days ({total_batches} batches) of data...")
        logger.info(f"📅 From: {start_date.date()} To: {end_date.date()}")
        
        batch_number = 1
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=batch_days), end_date)
            
            try:
                logger.info(f"🔄 Batch {batch_number}/{total_batches}: {current_start.date()} to {current_end.date()}")
                df_batch = self.fetch_air_pollution_history(current_start, current_end)
                
                if df_batch is not None and not df_batch.empty:
                    all_data.append(df_batch)
                    successful_batches += 1
                    logger.info(f"✅ Batch {batch_number} SUCCESS: {len(df_batch)} records")
                else:
                    logger.warning(f"⚠️  Batch {batch_number} EMPTY: {current_start.date()} to {current_end.date()}")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_number} FAILED: {e}")
            
            current_start = current_end
            batch_number += 1
            
            # Rate limiting - be gentle with the API (2 seconds between requests)
            time.sleep(2)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
            
            # Calculate statistics
            total_records = len(combined_df)
            date_range = combined_df['timestamp'].max() - combined_df['timestamp'].min()
            expected_records = days * 24  # 24 records per day (hourly)
            coverage_percentage = (total_records / expected_records) * 100
            
            logger.info(f"🎉 365-DAY BACKFILL COMPLETE!")
            logger.info(f"📊 Summary:")
            logger.info(f"   Successful batches: {successful_batches}/{total_batches}")
            logger.info(f"   Total records: {total_records}")
            logger.info(f"   Data coverage: {coverage_percentage:.1f}%")
            logger.info(f"   Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            logger.info(f"   AQI Statistics:")
            logger.info(f"     Mean: {combined_df['aqi'].mean():.2f}")
            logger.info(f"     Std:  {combined_df['aqi'].std():.2f}")
            logger.info(f"     Min:  {combined_df['aqi'].min():.2f}")
            logger.info(f"     Max:  {combined_df['aqi'].max():.2f}")
            
            # Save to disk
            output_path = config.RAW_DATA_DIR / f"historical_aqi_{self.city_name.lower()}.csv"
            combined_df.to_csv(output_path, index=False)
            logger.info(f"💾 Saved 365-day data to {output_path}")
            
            return combined_df
        else:
            logger.error("💥 No data fetched during 365-day backfill")
            return pd.DataFrame()
    
    def save_latest_data(self, df: pd.DataFrame, data_type: str = "pollution"):
        """Save latest fetched data to disk"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{self.city_name.lower()}_{timestamp_str}.csv"
        output_path = config.RAW_DATA_DIR / filename
        
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved {data_type} data to {output_path}")


def main():
    """Test the enhanced data fetcher"""
    fetcher = OpenWeatherFetcher()
    
    # Test current air pollution
    current = fetcher.fetch_current_air_pollution()
    if current:
        logger.info(f"🌫️ Current AQI: {current['aqi']}")
        logger.info(f"📊 PM2.5: {current['pm2_5']} μg/m³")
    
    # Test 7-day sample
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    sample_data = fetcher.fetch_air_pollution_history(start_date, end_date)
    if sample_data is not None:
        logger.info(f"📈 Sample data: {len(sample_data)} records")
        logger.info(f"🌈 AQI range: {sample_data['aqi'].min():.1f} - {sample_data['aqi'].max():.1f}")


if __name__ == "__main__":
    main()