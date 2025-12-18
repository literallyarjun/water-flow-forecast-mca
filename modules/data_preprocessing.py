import pandas as pd
import numpy as np
from datetime import datetime

def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def create_lag_features(df, target_col='consumption', lags=[1, 7, 14, 30]):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    df = df.dropna()
    return df

def extract_date_features(df, date_col='date'):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
        df['month'] = df[date_col].dt.month
        df['weekday'] = df[date_col].dt.dayofweek
        df['day'] = df[date_col].dt.day
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 0
            elif month in [3, 4, 5]:
                return 1
            elif month in [6, 7, 8]:
                return 2
            else:
                return 3
        
        df['season'] = df['month'].apply(get_season)
    return df

def merge_climate_indicators(df, climate_df=None):
    if climate_df is not None and 'date' in df.columns and 'date' in climate_df.columns:
        df = df.merge(climate_df, on='date', how='left')
    return df

def preprocess_data(filepath):
    df = load_csv(filepath)
    df = handle_missing_values(df)
    df = extract_date_features(df)
    df = create_lag_features(df)
    return df

def generate_sample_data(days=365):
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')
    
    base_consumption = 100
    seasonal = 20 * np.sin(2 * np.pi * np.arange(days) / 365)
    weekly = 10 * np.sin(2 * np.pi * np.arange(days) / 7)
    noise = np.random.normal(0, 5, days)
    consumption = base_consumption + seasonal + weekly + noise
    
    temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 3, days)
    humidity = 60 + 20 * np.cos(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 5, days)
    rainfall = np.maximum(0, np.random.exponential(5, days) * (1 + 0.5 * np.sin(2 * np.pi * np.arange(days) / 365)))
    wind = 10 + 5 * np.random.random(days)
    evaporation = 3 + 2 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 0.5, days)
    
    df = pd.DataFrame({
        'date': dates,
        'consumption': np.maximum(consumption, 20),
        'temperature': temperature,
        'humidity': np.clip(humidity, 20, 100),
        'rainfall': rainfall,
        'wind': wind,
        'evaporation': np.maximum(evaporation, 0)
    })
    
    return df
