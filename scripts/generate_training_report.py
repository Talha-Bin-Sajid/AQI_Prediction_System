"""
Generate a training report with visualizations
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import config


def main():
    """Generate training report"""
    logger.info("Generating training report...")
    
    try:
        # Load model metadata
        metadata_file = config.MODELS_DIR / "latest_model_metadata.json"
        
        if not metadata_file.exists():
            logger.warning("No model metadata found")
            return 1
        
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create report directory
        report_dir = config.BASE_DIR / "reports"
        report_dir.mkdir(exist_ok=True)
        
        # Generate report
        report_file = report_dir / f"training_report_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("AQI PREDICTION MODEL - TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL INFORMATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Model Name: {metadata.get('model_name', 'N/A')}\n")
            f.write(f"Version: {metadata.get('version', 'N/A')}\n")
            f.write(f"Training Date: {metadata.get('timestamp', 'N/A')}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 60 + "\n")
            metrics = metadata.get('metrics', {})
            f.write(f"RMSE: {metrics.get('rmse', 0):.4f}\n")
            f.write(f"MAE: {metrics.get('mae', 0):.4f}\n")
            f.write(f"R² Score: {metrics.get('r2', 0):.4f}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Training Samples: {metadata.get('training_samples', 'N/A')}\n")
            f.write(f"Test Samples: {metadata.get('test_samples', 'N/A')}\n")
            f.write(f"Features Count: {metadata.get('features_count', 'N/A')}\n")
            
            data_range = metadata.get('data_date_range', {})
            f.write(f"Data Start Date: {data_range.get('start', 'N/A')}\n")
            f.write(f"Data End Date: {data_range.get('end', 'N/A')}\n\n")
            
            f.write("=" * 60 + "\n")
        
        logger.info(f"Training report saved to {report_file}")
        
        # Generate feature importance plot if available
        importance_file = config.MODELS_DIR / f"feature_importance_{datetime.now().strftime('%Y%m%d')}.csv"
        
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 20 Most Important Features')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            plot_file = report_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {plot_file}")
            plt.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())