"""
Streamlit dashboard for AQI prediction system
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"


# Helper functions
def get_aqi_color(aqi: float) -> str:
    """Get color based on AQI value"""
    if aqi <= 50:
        return "#00e400"  # Good - Green
    elif aqi <= 100:
        return "#ffff00"  # Moderate - Yellow
    elif aqi <= 150:
        return "#ff7e00"  # Unhealthy for Sensitive - Orange
    elif aqi <= 200:
        return "#ff0000"  # Unhealthy - Red
    elif aqi <= 300:
        return "#8f3f97"  # Very Unhealthy - Purple
    else:
        return "#7e0023"  # Hazardous - Maroon


def fetch_current_aqi() -> Dict:
    """Fetch current AQI data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/current", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching current AQI: {e}")
        return None


def fetch_predictions(days: int = 3) -> List[Dict]:
    """Fetch AQI predictions from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/predict?days={days}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return []


def fetch_historical_data(days: int = 7) -> List[Dict]:
    """Fetch historical AQI data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/historical?days={days}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return []


def fetch_model_info() -> Dict:
    """Fetch model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/model/info", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def fetch_alerts() -> Dict:
    """Fetch AQI alerts from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/alerts", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-caution {
        background-color: #ff9800;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌍 Air Quality Index Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4047/4047887.png", width=100)
    st.title("Settings")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Prediction settings
    st.subheader("Prediction Settings")
    prediction_days = st.slider("Days to Predict", 1, 7, 3)
    
    st.markdown("---")
    
    # Historical data settings
    st.subheader("Historical Data")
    historical_days = st.slider("Days to Display", 7, 30, 14)
    
    st.markdown("---")
    
    # Model information
    st.subheader("Model Information")
    model_info = fetch_model_info()
    if model_info:
        st.metric("Model", model_info['model_name'])
        st.metric("RMSE", f"{model_info['rmse']:.2f}")
        st.metric("R² Score", f"{model_info['r2']:.3f}")
        if model_info.get('last_trained'):
            st.text(f"Last trained:\n{model_info['last_trained']}")
    else:
        st.warning("Model info not available")

# Main content
# Check for alerts
alerts_data = fetch_alerts()
if alerts_data and alerts_data.get('has_alerts'):
    st.markdown("### ⚠️ Active Alerts")
    for alert in alerts_data['alerts']:
        level = alert['level']
        message = alert['message']
        
        if level == 'critical':
            st.markdown(f'<div class="alert-critical">🚨 <strong>CRITICAL:</strong> {message}</div>', 
                       unsafe_allow_html=True)
        elif level == 'warning':
            st.markdown(f'<div class="alert-warning">⚠️ <strong>WARNING:</strong> {message}</div>', 
                       unsafe_allow_html=True)
        elif level == 'caution':
            st.markdown(f'<div class="alert-caution">⚡ <strong>CAUTION:</strong> {message}</div>', 
                       unsafe_allow_html=True)

# Current AQI Section
st.markdown("### 📊 Current Air Quality")
col1, col2, col3, col4 = st.columns(4)

current_data = fetch_current_aqi()
if current_data:
    with col1:
        aqi_color = get_aqi_color(current_data['aqi'])
        st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: {aqi_color}; 
                        border-radius: 10px; color: white;'>
                <h2 style='margin: 0;'>{current_data['aqi']}</h2>
                <p style='margin: 5px 0;'>AQI</p>
                <p style='margin: 0; font-size: 0.9em;'>{current_data['aqi_category']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    pollutants = current_data['pollutants']
    with col2:
        st.metric("PM2.5", f"{pollutants['pm2_5']:.1f} μg/m³")
        st.metric("PM10", f"{pollutants['pm10']:.1f} μg/m³")
    
    with col3:
        st.metric("NO₂", f"{pollutants['no2']:.1f} μg/m³")
        st.metric("O₃", f"{pollutants['o3']:.1f} μg/m³")
    
    with col4:
        st.metric("SO₂", f"{pollutants['so2']:.1f} μg/m³")
        st.metric("CO", f"{pollutants['co']:.1f} μg/m³")
    
    st.caption(f"📍 Location: {current_data['location']['city']} | "
              f"Last updated: {datetime.fromisoformat(current_data['timestamp'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')}")
else:
    st.error("Unable to fetch current AQI data")

st.markdown("---")

# Predictions Section
st.markdown("### 🔮 AQI Forecast")
predictions = fetch_predictions(prediction_days)

if predictions:
    # Create prediction dataframe
    pred_df = pd.DataFrame([
        {
            'Date': datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')),
            'Predicted AQI': p['predicted_aqi'],
            'Category': p['aqi_category']
        }
        for p in predictions
    ])
    
    # Display prediction cards
    pred_cols = st.columns(min(len(predictions), 3))
    for idx, pred in enumerate(predictions[:3]):
        with pred_cols[idx]:
            pred_date = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
            pred_aqi = pred['predicted_aqi']
            pred_color = get_aqi_color(pred_aqi)
            
            st.markdown(f"""
                <div style='text-align: center; padding: 15px; background-color: {pred_color}; 
                            border-radius: 10px; color: white;'>
                    <p style='margin: 0; font-size: 0.9em;'>{pred_date.strftime('%b %d')}</p>
                    <h3 style='margin: 5px 0;'>{pred_aqi:.0f}</h3>
                    <p style='margin: 0; font-size: 0.85em;'>{pred['aqi_category']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Prediction chart
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Predicted AQI'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    
    fig_pred.update_layout(
        title="3-Day AQI Forecast",
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.warning("Unable to fetch predictions")

st.markdown("---")

# Historical Data Section
st.markdown("### 📈 Historical Trends")
historical_data = fetch_historical_data(historical_days)

if historical_data:
    # Create historical dataframe
    hist_df = pd.DataFrame([
        {
            'Timestamp': datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')),
            'AQI': h['aqi'],
            'PM2.5': h['pm2_5'],
            'PM10': h['pm10'],
            'NO2': h['no2'],
            'O3': h['o3']
        }
        for h in historical_data
    ])
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 AQI Trend", "🔬 Pollutants", "📋 Data Table"])
    
    with tab1:
        # AQI trend chart
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df['Timestamp'],
            y=hist_df['AQI'],
            mode='lines',
            name='AQI',
            fill='tozeroy',
            line=dict(color='#764ba2', width=2)
        ))
        
        # Add AQI threshold lines
        fig_hist.add_hline(y=50, line_dash="dash", line_color="green", 
                          annotation_text="Good")
        fig_hist.add_hline(y=100, line_dash="dash", line_color="yellow", 
                          annotation_text="Moderate")
        fig_hist.add_hline(y=150, line_dash="dash", line_color="orange", 
                          annotation_text="Unhealthy for Sensitive")
        
        fig_hist.update_layout(
            title=f"AQI Trend - Last {historical_days} Days",
            xaxis_title="Date",
            yaxis_title="AQI",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average AQI", f"{hist_df['AQI'].mean():.1f}")
        with col2:
            st.metric("Max AQI", f"{hist_df['AQI'].max():.0f}")
        with col3:
            st.metric("Min AQI", f"{hist_df['AQI'].min():.0f}")
        with col4:
            st.metric("Std Dev", f"{hist_df['AQI'].std():.1f}")
    
    with tab2:
        # Pollutants comparison
        fig_pollutants = go.Figure()
        
        pollutants_to_plot = ['PM2.5', 'PM10', 'NO2', 'O3']
        for pollutant in pollutants_to_plot:
            fig_pollutants.add_trace(go.Scatter(
                x=hist_df['Timestamp'],
                y=hist_df[pollutant],
                mode='lines',
                name=pollutant
            ))
        
        fig_pollutants.update_layout(
            title="Pollutant Levels Over Time",
            xaxis_title="Date",
            yaxis_title="Concentration (μg/m³)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_pollutants, use_container_width=True)
    
    with tab3:
        # Data table
        st.dataframe(
            hist_df.sort_values('Timestamp', ascending=False),
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv = hist_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"aqi_historical_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
else:
    st.warning("Unable to fetch historical data")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌍 AQI Predictor Dashboard | Data powered by OpenWeather API</p>
        <p style='font-size: 0.85em;'>Built with Streamlit, FastAPI, and Machine Learning</p>
    </div>
""", unsafe_allow_html=True)