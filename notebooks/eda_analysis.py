"""
Comprehensive Exploratory Data Analysis (EDA) for AQI Prediction
This module performs thorough data analysis and visualization
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from loguru import logger

from src.config import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)


class AQIExploratoryAnalysis:
    """Comprehensive EDA for AQI data"""
    
    def __init__(self, data_path: Path = None):
        """Initialize with data path"""
        self.data_path = data_path or (
            config.RAW_DATA_DIR / f"historical_aqi_{config.location.city_name.lower()}.csv"
        )
        self.df = None
        self.report_dir = config.BASE_DIR / "reports" / "eda"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load and prepare data"""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Add time features for analysis
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['date'] = self.df['timestamp'].dt.date
        
        logger.info(f"Data loaded: {self.df.shape}")
        return self.df
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        logger.info("=" * 60)
        logger.info("BASIC STATISTICS")
        logger.info("=" * 60)
        
        print("\n📊 Dataset Overview:")
        print(f"  Shape: {self.df.shape}")
        print(f"  Date Range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"  Duration: {(self.df['timestamp'].max() - self.df['timestamp'].min()).days} days")
        
        print("\n📈 AQI Statistics:")
        print(self.df['aqi'].describe())
        
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values!")
        
        print("\n📊 Data Quality Metrics:")
        print(f"  Duplicate rows: {self.df.duplicated().sum()}")
        print(f"  Unique timestamps: {self.df['timestamp'].nunique()}")
        
        return self.df.describe()
    
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        logger.info("Performing temporal analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Time series plot
        axes[0, 0].plot(self.df['timestamp'], self.df['aqi'], alpha=0.6, linewidth=0.5)
        axes[0, 0].set_title('AQI Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('AQI')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Hourly pattern
        hourly_avg = self.df.groupby('hour')['aqi'].agg(['mean', 'std'])
        axes[0, 1].plot(hourly_avg.index, hourly_avg['mean'], marker='o', linewidth=2)
        axes[0, 1].fill_between(hourly_avg.index, 
                                hourly_avg['mean'] - hourly_avg['std'],
                                hourly_avg['mean'] + hourly_avg['std'],
                                alpha=0.3)
        axes[0, 1].set_title('Average AQI by Hour of Day', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Average AQI')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Day of week pattern
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_avg = self.df.groupby('day_of_week')['aqi'].mean()
        axes[1, 0].bar(range(7), dow_avg, color='steelblue')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels(dow_names)
        axes[1, 0].set_title('Average AQI by Day of Week', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Average AQI')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Monthly pattern
        monthly_avg = self.df.groupby('month')['aqi'].agg(['mean', 'std'])
        axes[1, 1].plot(monthly_avg.index, monthly_avg['mean'], marker='o', linewidth=2)
        axes[1, 1].fill_between(monthly_avg.index,
                                monthly_avg['mean'] - monthly_avg['std'],
                                monthly_avg['mean'] + monthly_avg['std'],
                                alpha=0.3)
        axes[1, 1].set_title('Average AQI by Month', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average AQI')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal analysis to {self.report_dir / 'temporal_patterns.png'}")
        plt.close()
        
        # Print insights
        print("\n⏰ Temporal Insights:")
        print(f"  Peak hour: {hourly_avg['mean'].idxmax()}:00 (AQI: {hourly_avg['mean'].max():.1f})")
        print(f"  Best hour: {hourly_avg['mean'].idxmin()}:00 (AQI: {hourly_avg['mean'].min():.1f})")
        print(f"  Worst month: {monthly_avg['mean'].idxmax()} (AQI: {monthly_avg['mean'].max():.1f})")
        print(f"  Best month: {monthly_avg['mean'].idxmin()} (AQI: {monthly_avg['mean'].min():.1f})")
    
    def pollutant_analysis(self):
        """Analyze pollutant correlations and distributions"""
        logger.info("Analyzing pollutants...")
        
        pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, pollutant in enumerate(pollutants):
            if pollutant in self.df.columns:
                # Scatter plot with AQI
                axes[idx].scatter(self.df[pollutant], self.df['aqi'], 
                                 alpha=0.3, s=10)
                
                # Add trend line
                z = np.polyfit(self.df[pollutant].dropna(), 
                              self.df.loc[self.df[pollutant].notna(), 'aqi'], 1)
                p = np.poly1d(z)
                axes[idx].plot(self.df[pollutant].sort_values(), 
                              p(self.df[pollutant].sort_values()), 
                              "r--", linewidth=2, label='Trend')
                
                # Calculate correlation
                corr = self.df[['aqi', pollutant]].corr().iloc[0, 1]
                
                axes[idx].set_title(f'{pollutant.upper()} vs AQI\n(Correlation: {corr:.3f})',
                                   fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(f'{pollutant.upper()} (μg/m³)')
                axes[idx].set_ylabel('AQI')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'pollutant_correlations.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved pollutant analysis to {self.report_dir / 'pollutant_correlations.png'}")
        plt.close()
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        corr_cols = ['aqi'] + pollutants
        corr_matrix = self.df[corr_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdYlGn_r', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8})
        plt.title('Pollutant Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.report_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation matrix to {self.report_dir / 'correlation_matrix.png'}")
        plt.close()
        
        # Print correlation insights
        print("\n🔬 Pollutant Correlations with AQI:")
        for pollutant in pollutants:
            if pollutant in self.df.columns:
                corr = self.df[['aqi', pollutant]].corr().iloc[0, 1]
                print(f"  {pollutant.upper()}: {corr:.3f}")
    
    def distribution_analysis(self):
        """Analyze AQI distribution and detect outliers"""
        logger.info("Analyzing distributions...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Histogram
        axes[0, 0].hist(self.df['aqi'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.df['aqi'].mean(), color='red', 
                          linestyle='--', linewidth=2, label=f"Mean: {self.df['aqi'].mean():.1f}")
        axes[0, 0].axvline(self.df['aqi'].median(), color='green',
                          linestyle='--', linewidth=2, label=f"Median: {self.df['aqi'].median():.1f}")
        axes[0, 0].set_title('AQI Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('AQI')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Box plot
        axes[0, 1].boxplot(self.df['aqi'], vert=True)
        axes[0, 1].set_title('AQI Box Plot (Outlier Detection)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('AQI')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Q-Q plot
        stats.probplot(self.df['aqi'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. AQI categories distribution
        aqi_categories = pd.cut(self.df['aqi'], 
                               bins=[0, 50, 100, 150, 200, 300, 500],
                               labels=['Good', 'Moderate', 'Unhealthy\nSensitive', 
                                      'Unhealthy', 'Very\nUnhealthy', 'Hazardous'])
        category_counts = aqi_categories.value_counts().sort_index()
        
        colors = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023']
        axes[1, 1].bar(range(len(category_counts)), category_counts.values, color=colors)
        axes[1, 1].set_xticks(range(len(category_counts)))
        axes[1, 1].set_xticklabels(category_counts.index, rotation=45)
        axes[1, 1].set_title('AQI Category Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution analysis to {self.report_dir / 'distribution_analysis.png'}")
        plt.close()
        
        # Statistical tests
        print("\n📊 Distribution Statistics:")
        print(f"  Skewness: {stats.skew(self.df['aqi']):.3f}")
        print(f"  Kurtosis: {stats.kurtosis(self.df['aqi']):.3f}")
        
        # Shapiro-Wilk test for normality (on sample due to size)
        sample = self.df['aqi'].sample(min(5000, len(self.df)))
        _, p_value = stats.shapiro(sample)
        print(f"  Shapiro-Wilk p-value: {p_value:.4f} ({'Normal' if p_value > 0.05 else 'Not Normal'})")
        
        # Outlier detection
        Q1 = self.df['aqi'].quantile(0.25)
        Q3 = self.df['aqi'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.df[(self.df['aqi'] < Q1 - 1.5 * IQR) | 
                          (self.df['aqi'] > Q3 + 1.5 * IQR)]
        print(f"\n⚠️  Outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
    
    def trend_seasonality_analysis(self):
        """Analyze trends and seasonality"""
        logger.info("Analyzing trends and seasonality...")
        
        # Daily aggregation
        daily_aqi = self.df.groupby('date')['aqi'].mean().reset_index()
        daily_aqi['date'] = pd.to_datetime(daily_aqi['date'])
        
        # Calculate moving averages
        daily_aqi['MA_7'] = daily_aqi['aqi'].rolling(window=7, center=True).mean()
        daily_aqi['MA_30'] = daily_aqi['aqi'].rolling(window=30, center=True).mean()
        
        plt.figure(figsize=(20, 8))
        plt.plot(daily_aqi['date'], daily_aqi['aqi'], alpha=0.3, label='Daily AQI')
        plt.plot(daily_aqi['date'], daily_aqi['MA_7'], linewidth=2, label='7-Day MA')
        plt.plot(daily_aqi['date'], daily_aqi['MA_30'], linewidth=2, label='30-Day MA')
        plt.title('AQI Trend Analysis with Moving Averages', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.report_dir / 'trend_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved trend analysis to {self.report_dir / 'trend_analysis.png'}")
        plt.close()
        
        # Calculate trend
        x = np.arange(len(daily_aqi))
        z = np.polyfit(x, daily_aqi['aqi'].dropna(), 1)
        trend_direction = "increasing" if z[0] > 0 else "decreasing"
        
        print(f"\n📈 Trend Analysis:")
        print(f"  Overall trend: {trend_direction}")
        print(f"  Trend coefficient: {z[0]:.4f} AQI units per day")
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        logger.info("=" * 60)
        logger.info("GENERATING COMPREHENSIVE EDA REPORT")
        logger.info("=" * 60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.basic_statistics()
        self.temporal_analysis()
        self.pollutant_analysis()
        self.distribution_analysis()
        self.trend_seasonality_analysis()
        
        # Generate summary report
        report_text = f"""
{'='*60}
AQI EXPLORATORY DATA ANALYSIS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: {config.location.city_name}

DATASET OVERVIEW
{'-'*60}
Total Records: {len(self.df):,}
Date Range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}
Duration: {(self.df['timestamp'].max() - self.df['timestamp'].min()).days} days

AQI STATISTICS
{'-'*60}
Mean AQI: {self.df['aqi'].mean():.2f}
Median AQI: {self.df['aqi'].median():.2f}
Std Dev: {self.df['aqi'].std():.2f}
Min AQI: {self.df['aqi'].min():.2f}
Max AQI: {self.df['aqi'].max():.2f}

AQI DISTRIBUTION
{'-'*60}
Good (0-50): {len(self.df[self.df['aqi'] <= 50]):,} ({len(self.df[self.df['aqi'] <= 50])/len(self.df)*100:.1f}%)
Moderate (51-100): {len(self.df[(self.df['aqi'] > 50) & (self.df['aqi'] <= 100)]):,} ({len(self.df[(self.df['aqi'] > 50) & (self.df['aqi'] <= 100)])/len(self.df)*100:.1f}%)
Unhealthy for Sensitive (101-150): {len(self.df[(self.df['aqi'] > 100) & (self.df['aqi'] <= 150)]):,} ({len(self.df[(self.df['aqi'] > 100) & (self.df['aqi'] <= 150)])/len(self.df)*100:.1f}%)
Unhealthy (151-200): {len(self.df[(self.df['aqi'] > 150) & (self.df['aqi'] <= 200)]):,} ({len(self.df[(self.df['aqi'] > 150) & (self.df['aqi'] <= 200)])/len(self.df)*100:.1f}%)
Very Unhealthy (201-300): {len(self.df[(self.df['aqi'] > 200) & (self.df['aqi'] <= 300)]):,} ({len(self.df[(self.df['aqi'] > 200) & (self.df['aqi'] <= 300)])/len(self.df)*100:.1f}%)
Hazardous (300+): {len(self.df[self.df['aqi'] > 300]):,} ({len(self.df[self.df['aqi'] > 300])/len(self.df)*100:.1f}%)

KEY INSIGHTS
{'-'*60}
1. Temporal Patterns Identified
2. Pollutant Correlations Analyzed  
3. Outliers Detected and Documented
4. Trend and Seasonality Assessed

VISUALIZATIONS GENERATED
{'-'*60}
[OK] temporal_patterns.png
[OK] pollutant_correlations.png
[OK] correlation_matrix.png
[OK] distribution_analysis.png
[OK] trend_analysis.png

{'='*60}
        """
        
        report_file = self.report_dir / 'eda_summary_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✅ EDA Report saved to {report_file}")
        logger.info(f"✅ All visualizations saved to {self.report_dir}")
        
        return report_text


def main():
    """Run EDA analysis"""
    eda = AQIExploratoryAnalysis()
    report = eda.generate_report()
    print(report)


if __name__ == "__main__":
    main()