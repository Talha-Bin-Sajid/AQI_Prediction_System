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
    """Get last timestamp from good_features.csv to avoid duplicates"""
    try:
        project = hopsworks.login(
            api_key_value=config.hopsworks.api_key,
            project=config.hopsworks.project_name
        )
        dataset_api = project.get_dataset_api()
        
        # Check good_features.csv for last timestamp
        try:
            dataset_api.download("Resources/good_features.csv", "temp_check.csv", overwrite=True)
            existing_df = pd.read_csv("temp_check.csv")
            last_timestamp = pd.to_datetime(existing_df['timestamp']).max()
            Path("temp_check.csv").unlink(missing_ok=True)
            return last_timestamp
        except:
            return None
    except:
        return None

def push_features_to_hopsworks(df_features: pd.DataFrame, incremental: bool = True):
    """Push features to Hopsworks - APPEND TO GOOD_FEATURES.CSV"""
    try:
        project = hopsworks.login(
            api_key_value=config.hopsworks.api_key,
            project=config.hopsworks.project_name
        )
        dataset_api = project.get_dataset_api()
        
        # Apply the same feature engineering that created good_features.csv
        logger.info("🔧 Applying feature engineering (matching good_features.csv)...")
        engineer = FeatureEngineer()
        df_engineered = engineer.engineer_features(df_features)
        
        # CRITICAL: Use good_features.csv as the target file
        target_filename = "good_features.csv"
        
        if incremental:
            # Download existing good_features.csv
            try:
                logger.info("📥 Downloading existing good_features.csv...")
                downloaded_path = dataset_api.download(
                    f"Resources/{target_filename}", 
                    "existing_good_features.csv",
                    overwrite=True
                )
                existing_df = pd.read_csv(downloaded_path)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Ensure new features match the good_features schema
                df_engineered = df_engineered[existing_df.columns]
                
                # Combine with new features, remove duplicates by timestamp
                combined_df = pd.concat([existing_df, df_engineered], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                # Upload back to SAME good_features.csv file
                temp_file = "temp_updated_good_features.csv"
                combined_df.to_csv(temp_file, index=False)
                uploaded_path = dataset_api.upload(temp_file, "Resources", overwrite=True)
                
                # Cleanup
                Path(temp_file).unlink()
                Path(downloaded_path).unlink()
                Path("existing_good_features.csv").unlink(missing_ok=True)
                
                logger.info(f"✅ GOOD_FEATURES.CSV updated: {len(df_engineered)} new + {len(existing_df)} existing = {len(combined_df)} total")
                
            except Exception as e:
                # First time - create good_features.csv with initial data
                logger.info("🆕 Creating initial good_features.csv...")
                temp_file = "temp_initial_good_features.csv"
                df_engineered.to_csv(temp_file, index=False)
                uploaded_path = dataset_api.upload(temp_file, "Resources", overwrite=True)
                Path(temp_file).unlink()
                logger.info(f"✅ Created initial good_features.csv: {len(df_engineered)} records")
        else:
            # Full refresh - replace entire good_features.csv
            temp_file = "temp_refresh_good_features.csv"
            df_engineered.to_csv(temp_file, index=False)
            uploaded_path = dataset_api.upload(temp_file, "Resources", overwrite=True)
            Path(temp_file).unlink()
            logger.info(f"✅ Full refresh of good_features.csv: {len(df_engineered)} records")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update good_features.csv: {e}")
        return False
    
def main():
    """Run incremental feature pipeline - APPEND TO GOOD_FEATURES"""
    logger.info("=" * 60)
    logger.info("INCREMENTAL FEATURE PIPELINE - APPEND TO GOOD_FEATURES.CSV")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        fetcher = OpenWeatherFetcher()
        engineer = FeatureEngineer()
        
        # Get last processed timestamp from good_features.csv
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
        
        # Push to Hopsworks - APPEND TO GOOD_FEATURES.CSV
        success = push_features_to_hopsworks(df, incremental=True)
        
        if success:
            logger.info("✅ New features appended to good_features.csv")
        else:
            logger.error("❌ Failed to update good_features.csv")
            return 1
        
        logger.info("=" * 60)
        logger.info("FEATURE PIPELINE COMPLETED - GOOD_FEATURES.CSV UPDATED")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
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