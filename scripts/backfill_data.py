"""
Script to backfill historical data for 365 days (run once initially)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data_fetcher import OpenWeatherFetcher
from src.config import config


def main():
    """Backfill historical data for 365 days"""
    logger.info("🚀 Starting historical data backfill for 365 DAYS...")
    
    fetcher = OpenWeatherFetcher()
    
    # Backfill data for 365 days (1 year)
    df = fetcher.backfill_historical_data(
        days=365,  # Changed from 180 to 365
        batch_days=30
    )
    
    if df is not None and not df.empty:
        logger.info(f"✅ 365-DAY BACKFILL COMPLETE! Fetched {len(df)} records")
        logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"📊 AQI Statistics for 365 days:")
        logger.info(f"   Mean: {df['aqi'].mean():.2f}")
        logger.info(f"   Std:  {df['aqi'].std():.2f}")
        logger.info(f"   Min:  {df['aqi'].min():.2f}")
        logger.info(f"   Max:  {df['aqi'].max():.2f}")
        
        # Calculate data coverage
        date_range = df['timestamp'].max() - df['timestamp'].min()
        logger.info(f"📈 Data coverage: {date_range.days} days")
        
        return 0
    else:
        logger.error("❌ 365-day backfill failed!")
        return 1


if __name__ == "__main__":
    exit(main())