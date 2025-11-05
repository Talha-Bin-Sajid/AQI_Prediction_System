"""
Hopsworks Feature Store Integration - INCREMENTAL VERSION
Handles incremental storage and retrieval of processed features
"""
import pandas as pd
import hopsworks
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger

from src.config import config


class HopsworksFeatureStore:
    """Manages incremental feature storage in Hopsworks"""
    
    def __init__(self):
        self.project = None
        self.fs = None
        
        if config.hopsworks.use_hopsworks:
            self.connect()
    
    def connect(self):
        """Connect to Hopsworks project"""
        try:
            logger.info(f"Connecting to Hopsworks project: {config.hopsworks.project_name}")
            
            self.project = hopsworks.login(
                api_key_value=config.hopsworks.api_key,
                project=config.hopsworks.project_name
            )
            
            self.fs = self.project.get_feature_store()
            logger.info("✅ Connected to Hopsworks Feature Store")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Hopsworks: {e}")
            raise

    # ========== INCREMENTAL FEATURE METHODS ==========

    def push_features_incremental(self, df_features: pd.DataFrame):
        """
        Push features incrementally to Hopsworks
        - Downloads existing features
        - Merges with new features
        - Uploads combined features back
        """
        try:
            logger.info("Starting incremental feature push...")
            
            # Download existing features
            existing_df = self.download_features()
            
            if existing_df is not None and not existing_df.empty:
                logger.info(f"Found {len(existing_df)} existing features")
                
                # Ensure timestamp is datetime for both dataframes
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Merge and remove duplicates
                combined_df = pd.concat([existing_df, df_features], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Merged: {len(df_features)} new + {len(existing_df)} existing = {len(combined_df)} total")
            else:
                combined_df = df_features
                logger.info(f"No existing features found, uploading {len(combined_df)} new features")
            
            # Upload merged features
            uploaded_path = self.upload_features(combined_df)
            logger.info(f"✅ Incremental features pushed to: {uploaded_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Incremental push failed: {e}")
            return False

    def download_features(self) -> Optional[pd.DataFrame]:
        """Download all features from Hopsworks"""
        try:
            dataset_api = self.project.get_dataset_api()
            
            # Try to download incremental features
            downloaded_path = dataset_api.download("Resources/incremental_features.csv", overwrite=True)
            
            df = pd.read_csv(downloaded_path)
            logger.info(f"✅ Downloaded {len(df)} features from Hopsworks")
            return df
            
        except Exception as e:
            logger.warning(f"No existing features found in Hopsworks: {e}")
            return None

    def upload_features(self, df: pd.DataFrame) -> str:
        """Upload features to Hopsworks"""
        try:
            dataset_api = self.project.get_dataset_api()
            
            # Save to temporary file
            temp_file = "incremental_features.csv"
            df.to_csv(temp_file, index=False)
            
            # Upload to Hopsworks
            uploaded_path = dataset_api.upload(temp_file, "Resources", overwrite=True)
            
            # Clean up
            Path(temp_file).unlink()
            
            return uploaded_path
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            raise

    def pull_features_for_training(self) -> Optional[pd.DataFrame]:
        """
        Pull ALL features from Hopsworks for training
        This is the main method for the training pipeline
        """
        try:
            logger.info("Pulling features from Hopsworks for training...")
            
            features_df = self.download_features()
            
            if features_df is None:
                logger.error("❌ No features available in Hopsworks for training")
                return None
            
            # Ensure timestamp is datetime
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            logger.info(f"✅ Successfully pulled {len(features_df)} features for training")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Failed to pull features for training: {e}")
            return None

    def get_last_feature_timestamp(self) -> Optional[datetime]:
        """Get timestamp of the most recent feature in Hopsworks"""
        try:
            features_df = self.download_features()
            
            if features_df is None or features_df.empty:
                return None
            
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            last_timestamp = features_df['timestamp'].max()
            
            logger.info(f"Last feature timestamp in Hopsworks: {last_timestamp}")
            return last_timestamp
            
        except Exception as e:
            logger.warning(f"Could not get last timestamp: {e}")
            return None

    # ========== COMPATIBILITY METHODS (for existing code) ==========

    def create_feature_group(
        self, 
        df: pd.DataFrame, 
        name: str = "aqi_features",
        version: int = 1
    ):
        """Legacy method - kept for compatibility"""
        logger.warning("⚠️ Using legacy feature group method. Consider using incremental methods.")
        try:
            # For incremental workflow, we don't use feature groups
            # But we'll implement a basic version for compatibility
            primary_key = ["timestamp"]
            event_time = "timestamp"
            
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.feature_group = self.fs.get_or_create_feature_group(
                name=name,
                version=version,
                description="Legacy feature group",
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=False,
                stream=False
            )
            
            return self.feature_group
            
        except Exception as e:
            logger.error(f"❌ Legacy feature group creation failed: {e}")
            raise

    def insert_features(self, df: pd.DataFrame, overwrite: bool = False):
        """Legacy method - redirects to incremental push"""
        logger.warning("⚠️ Using legacy insert method. Redirecting to incremental push.")
        return self.push_features_incremental(df)

    # ========== FEATURE VIEW METHODS (for training) ==========

    def get_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Retrieve training data from feature store
        Uses incremental features for training
        """
        try:
            logger.info("Fetching training data from Hopsworks incremental features...")
            
            # Pull all features from Hopsworks
            df = self.pull_features_for_training()
            
            if df is None:
                raise ValueError("No features available for training")
            
            # Filter by date if specified
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
            
            logger.info(f"Retrieved {len(df)} records from Hopsworks")
            
            # Separate features and target
            target_col = 'aqi_target_24h'
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in features")
            
            # Time-based split (85/15)
            split_idx = int(len(df) * 0.85)
            
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            # Exclude timestamp and target from features
            feature_cols = [col for col in df.columns 
                          if col not in ['timestamp', target_col, 'date']]
            
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Error getting training data: {e}")
            raise


# ========== CONVENIENCE FUNCTIONS ==========

def push_features_to_hopsworks(df: pd.DataFrame):
    """
    Convenience function to push features to Hopsworks (incremental)
    """
    if not config.hopsworks.use_hopsworks:
        logger.info("Hopsworks not enabled, skipping feature store upload")
        return False
    
    try:
        fs = HopsworksFeatureStore()
        success = fs.push_features_incremental(df)
        
        if success:
            logger.info("✅ Features successfully pushed to Hopsworks (incremental)")
        else:
            logger.error("❌ Failed to push features to Hopsworks")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to push features to Hopsworks: {e}")
        return False

def pull_features_from_hopsworks() -> Optional[pd.DataFrame]:
    """
    Convenience function to pull features from Hopsworks for training
    """
    if not config.hopsworks.use_hopsworks:
        logger.info("Hopsworks not enabled")
        return None
    
    try:
        fs = HopsworksFeatureStore()
        features_df = fs.pull_features_for_training()
        
        if features_df is not None:
            logger.info("✅ Features successfully loaded from Hopsworks")
        else:
            logger.error("❌ No features available from Hopsworks")
        
        return features_df
        
    except Exception as e:
        logger.error(f"❌ Failed to load features from Hopsworks: {e}")
        return None

def load_features_from_hopsworks() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to load training data from Hopsworks
    Maintains compatibility with existing code
    """
    if not config.hopsworks.use_hopsworks:
        raise ValueError("Hopsworks not enabled. Set USE_HOPSWORKS=true in .env")
    
    try:
        fs = HopsworksFeatureStore()
        X_train, X_test, y_train, y_test = fs.get_training_data()
        
        logger.info("✅ Features successfully loaded from Hopsworks")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"❌ Failed to load features from Hopsworks: {e}")
        raise

def get_last_hopsworks_timestamp() -> Optional[datetime]:
    """Get the timestamp of the most recent feature in Hopsworks"""
    if not config.hopsworks.use_hopsworks:
        return None
    
    try:
        fs = HopsworksFeatureStore()
        return fs.get_last_feature_timestamp()
    except Exception as e:
        logger.warning(f"Could not get last Hopsworks timestamp: {e}")
        return None


def main():
    """Test Hopsworks incremental integration"""
    logger.info("Testing Hopsworks incremental integration...")
    
    try:
        fs = HopsworksFeatureStore()
        
        # Test download/upload with sample data
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'aqi': [100],
            'pm2_5': [35.5],
            'aqi_target_24h': [105]
        })
        
        # Test incremental push
        success = fs.push_features_incremental(sample_data)
        logger.info(f"✅ Incremental push test: {success}")
        
        # Test feature pull
        features = fs.pull_features_for_training()
        if features is not None:
            logger.info(f"✅ Feature pull test: {len(features)} records")
        
        logger.info("✅ Hopsworks incremental integration test successful")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()