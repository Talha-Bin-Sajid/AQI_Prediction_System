"""
Feature Pipeline - Incremental data to Hopsworks
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data_fetcher import OpenWeatherFetcher
from src.feature_engineering import FeatureEngineer
from src.config import config
import hopsworks

def get_last_processed_timestamp():
    """Get last timestamp we processed to avoid duplicates"""
    try:
        project = hopsworks.login(
            api_key_value=config.hopsworks.api_key,
            project=config.hopsworks.project_name
        )
        dataset_api = project.get_dataset_api()
        
        # Try to download the latest features to check last timestamp
        try:
            dataset_api.download("Resources/latest_features.csv", "temp_check.csv", overwrite=True)
            existing_df = pd.read_csv("temp_check.csv")
            last_timestamp = pd.to_datetime(existing_df['timestamp']).max()
            Path("temp_check.csv").unlink(missing_ok=True)
            return last_timestamp
        except:
            return None
    except:
        return None

def push_features_to_hopsworks(df_features: pd.DataFrame, incremental: bool = True):
    """Push features to Hopsworks - incremental update - FIXED VERSION"""
    try:
        project = hopsworks.login(
            api_key_value=config.hopsworks.api_key,
            project=config.hopsworks.project_name
        )
        dataset_api = project.get_dataset_api()
        
        # ⚠️ IMPORTANT: Always use the SAME filename for consistency
        target_filename = "incremental_features.csv"
        
        if incremental:
            # Append to existing features
            try:
                # Download existing features
                dataset_api.download(f"Resources/{target_filename}", "temp_existing.csv", overwrite=True)
                existing_df = pd.read_csv("temp_existing.csv")
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Combine with new features, remove duplicates
                combined_df = pd.concat([existing_df, df_features], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                # Upload combined features with SAME filename
                temp_file = "temp_combined_features.csv"
                combined_df.to_csv(temp_file, index=False)
                uploaded_path = dataset_api.upload(temp_file, "Resources", overwrite=True)
                Path(temp_file).unlink()
                Path("temp_existing.csv").unlink()
                
                logger.info(f"✅ Incremental features updated: {len(df_features)} new + {len(existing_df)} existing = {len(combined_df)} total")
                
            except Exception as e:
                # First time - create new file with CONSISTENT filename
                logger.info("No existing features found, creating new file...")
                temp_file = "temp_new_features.csv"
                df_features.to_csv(temp_file, index=False)
                uploaded_path = dataset_api.upload(temp_file, "Resources", overwrite=True)
                Path(temp_file).unlink()
                logger.info(f"✅ Created new incremental features: {len(df_features)} records")
        else:
            # Replace all features (full refresh) with CONSISTENT filename
            temp_file = "temp_refresh_features.csv"
            df_features.to_csv(temp_file, index=False)
            uploaded_path = dataset_api.upload(temp_file, "Resources", overwrite=True)
            Path(temp_file).unlink()
            logger.info(f"✅ Full features refresh: {len(df_features)} records")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to push features to Hopsworks: {e}")
        return False
    
def main():
    """Run incremental feature pipeline"""
    logger.info("=" * 60)
    logger.info("INCREMENTAL FEATURE PIPELINE TO HOPSWORKS")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        fetcher = OpenWeatherFetcher()
        engineer = FeatureEngineer()
        
        # Get last processed timestamp to avoid duplicates
        last_timestamp = get_last_processed_timestamp()
        
        if last_timestamp:
            # Fetch only new data since last processing
            start_date = last_timestamp
            logger.info(f"Fetching new data since: {start_date}")
        else:
            # First run - fetch last 7 days for initial data
            start_date = datetime.now() - timedelta(days=7)
            logger.info(f"Initial run - fetching data since: {start_date}")
        
        end_date = datetime.now()
        
        # Fetch data
        df = fetcher.fetch_air_pollution_history(start_date, end_date)
        
        if df is None or df.empty:
            logger.info("No new data to process")
            return 0
        
        logger.info(f"Fetched {len(df)} new records")
        
        # Engineer features
        df_features = engineer.engineer_features(df)
        
        if df_features.empty:
            logger.error("Feature engineering failed")
            return 1
        
        logger.info(f"Engineered {len(df_features)} feature records")
        
        # Push to Hopsworks (incremental)
        success = push_features_to_hopsworks(df_features, incremental=True)
        
        if success:
            logger.info("✅ Incremental features pushed to Hopsworks")
        else:
            logger.error("❌ Failed to push features to Hopsworks")
            return 1
        
        logger.info("=" * 60)
        logger.info("INCREMENTAL FEATURE PIPELINE COMPLETED")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())