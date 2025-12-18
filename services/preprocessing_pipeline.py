import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    target_col: str = 'consumption'
    date_col: str = 'date'
    lag_periods: List[int] = None
    test_size: float = 0.2
    lstm_window_size: int = 7
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 7, 14, 30]


class PreprocessingPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.feature_scaler: Optional[MinMaxScaler] = None
        self.target_scaler: Optional[MinMaxScaler] = None
        self.feature_cols: List[str] = []
        self._is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self._handle_missing_values(df)
        df = self._extract_date_features(df)
        df = self._create_lag_features(df)
        
        self.feature_cols = self._identify_feature_columns(df)
        
        X_scaled, y_scaled = self._normalize_features(df, fit=True)
        
        prepared = self._split_and_prepare(X_scaled, y_scaled)
        prepared['processed_data'] = df
        prepared['feature_cols'] = self.feature_cols
        
        self._is_fitted = True
        logger.info(f"Pipeline fitted with {len(self.feature_cols)} features on {len(df)} samples")
        
        return prepared
    
    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        df = self._handle_missing_values(df)
        df = self._extract_date_features(df)
        df = self._create_lag_features(df)
        
        X_scaled, y_scaled = self._normalize_features(df, fit=False)
        
        prepared = self._split_and_prepare(X_scaled, y_scaled)
        prepared['processed_data'] = df
        prepared['feature_cols'] = self.feature_cols
        
        return prepared
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                df[col] = df[col].fillna(fill_val)
        
        return df
    
    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        date_col = self.config.date_col
        
        if date_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            df['month'] = df[date_col].dt.month
            df['weekday'] = df[date_col].dt.dayofweek
            df['day'] = df[date_col].dt.day
            df['day_of_year'] = df[date_col].dt.dayofyear
            df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
            
            df['season'] = df['month'].apply(self._get_season)
            
            df['is_weekend'] = (df['weekday'] >= 5).astype(int)
            
        return df
    
    @staticmethod
    def _get_season(month: int) -> int:
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        target = self.config.target_col
        
        if target in df.columns:
            for lag in self.config.lag_periods:
                df[f'{target}_lag_{lag}'] = df[target].shift(lag)
            
            df[f'{target}_rolling_mean_7'] = df[target].rolling(window=7, min_periods=1).mean()
            df[f'{target}_rolling_std_7'] = df[target].rolling(window=7, min_periods=1).std().fillna(0)
            
            df = df.dropna()
        
        return df
    
    def _identify_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = {self.config.date_col, self.config.target_col}
        
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols 
            and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        if not feature_cols:
            feature_cols = ['month', 'weekday', 'day', 'season']
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
        
        return feature_cols
    
    def _normalize_features(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        target = self.config.target_col
        
        X = df[self.feature_cols].values
        y = df[target].values
        
        if fit:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X_scaled = self.feature_scaler.transform(X)
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled
    
    def _split_and_prepare(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        split_idx = int(len(X) * (1 - self.config.test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train_lstm, y_train_lstm = self._create_sequences(X_train, y_train)
        X_test_lstm, y_test_lstm = self._create_sequences(X_test, y_test)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_lstm': X_train_lstm,
            'X_test_lstm': X_test_lstm,
            'y_train_lstm': y_train_lstm,
            'y_test_lstm': y_test_lstm,
            'scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'split_index': split_idx
        }
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        window = self.config.lstm_window_size
        
        if len(X) <= window:
            return np.array([]), np.array([])
        
        X_seq, y_seq = [], []
        for i in range(len(X) - window):
            X_seq.append(X[i:i+window])
            y_seq.append(y[i+window])
        
        return np.array(X_seq), np.array(y_seq)
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return predictions
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
