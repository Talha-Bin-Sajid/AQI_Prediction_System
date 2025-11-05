"""
Utility functions for AQI Prediction System
"""
from datetime import datetime
from typing import Union
import pandas as pd


def get_aqi_category(aqi: Union[int, float]) -> str:
    """
    Get AQI category from AQI value
    
    Args:
        aqi: AQI value
        
    Returns:
        AQI category string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_aqi_color(aqi: Union[int, float]) -> str:
    """
    Get color code for AQI value
    
    Args:
        aqi: AQI value
        
    Returns:
        Hex color code
    """
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


def timestamp_to_unix(dt: datetime) -> int:
    """Convert datetime to unix timestamp"""
    return int(dt.timestamp())


def unix_to_timestamp(unix_time: int) -> datetime:
    """Convert unix timestamp to datetime"""
    return datetime.fromtimestamp(unix_time)