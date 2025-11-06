"""
Training Pipeline - Pull from Hopsworks for incremental training
FIXED VERSION: Actually uses Hopsworks features
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.config import config
import hopsworks

def pull_all_features_from_hopsworks():
    """Pull features from Hopsworks - ONLY USE GOOD_FEATURES.CSV"""
    try:
        project = hopsworks.login(
            api_key_value=config.hopsworks.api_key,
            project=config.hopsworks.project_name
        )
        dataset_api = project.get_dataset_api()
        
        # ALWAYS use good_features.csv (which now contains both original + incremental data)
        target_filename = "Resources/good_features.csv"
        
        try:
            logger.info(f"Downloading from: {target_filename}")
            downloaded_path = dataset_api.download(
                target_filename,
                "features_download.csv",
                overwrite=True
            )
            
            df = pd.read_csv(downloaded_path)
            logger.info(f"Downloaded {len(df)} features from good_features.csv")
            
            # Clean up
            Path(downloaded_path).unlink()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Log data range
            logger.info(f"Data range in good_features.csv: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download good_features.csv: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to connect to Hopsworks: {e}")
        return None

def main():
    """Training pipeline using ACTUAL Hopsworks features"""
    logger.info("=" * 60)
    logger.info("INCREMENTAL TRAINING FROM HOPSWORKS - FIXED")
    logger.info("=" * 60)
    
    try:
        # Step 1: Pull ALL features from Hopsworks
        logger.info("Pulling ALL features from Hopsworks...")
        hopsworks_data = pull_all_features_from_hopsworks()
        
        if hopsworks_data is None or hopsworks_data.empty:
            logger.error("Cannot train - no features available from Hopsworks")
            return 1

        logger.info(f"Training with {len(hopsworks_data)} ACTUAL records from Hopsworks")
        logger.info(f"Data range: {hopsworks_data['timestamp'].min()} to {hopsworks_data['timestamp'].max()}")

        # Step 2: Prepare training data FROM HOPSWORKS
        engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test = engineer.prepare_train_test_split(
            hopsworks_data, test_size=config.pipeline.test_size
        )
        
        logger.info(f"Training samples from Hopsworks: {len(X_train)}")
        
        # Step 3: Train models WITH HOPSWORKS DATA
        logger.info("Training models with Hopsworks features...")
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("MODEL PERFORMANCE SUMMARY (FROM HOPSWORKS)")
        logger.info("=" * 60)
        
        for model_name, metrics in results['metrics'].items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE:  {metrics['mae']:.4f}")
            logger.info(f"  R²:   {metrics['r2']:.4f}")
        
        # Step 4: Save model
        trainer.save_model(metadata={
            'training_samples': len(X_train),
            'data_source': 'Hopsworks Incremental Features',
            'total_records': len(hopsworks_data),
            'hopsworks_incremental': True,
            'features_count': len(X_train.columns)
        })
        
        logger.info("\n" + "=" * 60)
        logger.info("TRUE INCREMENTAL TRAINING COMPLETED")
        logger.info(f"Used {len(hopsworks_data)} features from Hopsworks")
        logger.info("Real workflow: PUSH → PULL → TRAIN")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())