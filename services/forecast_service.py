import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from services.preprocessing_pipeline import PreprocessingPipeline, PipelineConfig
from models.random_forest_model import train_random_forest, get_feature_importance, predict_future_rf
from models.xgboost_model import train_xgboost, get_xgb_feature_importance, predict_future_xgb
from models.lstm_model import train_lstm, predict_future

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    mape: float
    rmse: float
    predictions: np.ndarray
    model: Any
    feature_importance: Optional[Dict[str, float]] = None


class ForecastService:
    def __init__(self, min_lstm_samples: int = 20):
        self.min_lstm_samples = min_lstm_samples
        self.pipeline: Optional[PreprocessingPipeline] = None
        self.best_model_name: Optional[str] = None
        self.model_results: Dict[str, ModelMetrics] = {}
    
    def generate_forecast(self, data: pd.DataFrame, forecast_days: int = 30, metric: str = 'accuracy') -> Dict[str, Any]:
        self.pipeline = PreprocessingPipeline(PipelineConfig())
        prepared = self.pipeline.fit_transform(data)
        
        self.model_results = self._train_all_models(prepared)
        
        self.best_model_name = self._select_best_model(metric)
        
        future_predictions = self._generate_future_predictions(prepared, forecast_days)
        
        best_result = self.model_results[self.best_model_name]
        
        last_date = pd.to_datetime(data['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        actual_values = self.pipeline.inverse_transform_predictions(prepared['y_test'])
        predicted_values = self.pipeline.inverse_transform_predictions(best_result.predictions)
        
        feature_importance = best_result.feature_importance or {}
        if not feature_importance and 'random_forest' in self.model_results:
            feature_importance = self.model_results['random_forest'].feature_importance or {}
        
        return {
            'best_model': self.best_model_name,
            'model_results': self._convert_results_for_legacy(),
            'future_dates': future_dates,
            'future_predictions': future_predictions,
            'actual_values': actual_values,
            'predicted_values': predicted_values,
            'feature_importance': feature_importance,
            'data': data,
            'metrics': {
                name: {'accuracy': m.accuracy, 'mape': m.mape, 'rmse': m.rmse}
                for name, m in self.model_results.items()
            }
        }
    
    def _train_all_models(self, prepared: Dict[str, Any]) -> Dict[str, ModelMetrics]:
        results = {}
        
        rf_metrics = self._train_random_forest(prepared)
        if rf_metrics:
            results['random_forest'] = rf_metrics
        
        xgb_metrics = self._train_xgboost(prepared)
        if xgb_metrics:
            results['xgboost'] = xgb_metrics
        
        if len(prepared['X_train_lstm']) >= self.min_lstm_samples:
            lstm_metrics = self._train_lstm(prepared)
            if lstm_metrics:
                results['lstm'] = lstm_metrics
        else:
            logger.info(f"Skipping LSTM: insufficient samples ({len(prepared['X_train_lstm'])} < {self.min_lstm_samples})")
        
        return results
    
    def _train_random_forest(self, prepared: Dict[str, Any]) -> Optional[ModelMetrics]:
        try:
            model, accuracy, predictions = train_random_forest(
                prepared['X_train'], prepared['y_train'],
                prepared['X_test'], prepared['y_test']
            )
            
            feature_importance = get_feature_importance(model, prepared['feature_cols'])
            mape, rmse = self._calculate_metrics(prepared['y_test'], predictions)
            
            return ModelMetrics(
                name='random_forest',
                accuracy=accuracy,
                mape=mape,
                rmse=rmse,
                predictions=predictions,
                model=model,
                feature_importance=feature_importance
            )
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return None
    
    def _train_xgboost(self, prepared: Dict[str, Any]) -> Optional[ModelMetrics]:
        try:
            model, accuracy, predictions = train_xgboost(
                prepared['X_train'], prepared['y_train'],
                prepared['X_test'], prepared['y_test']
            )
            
            feature_importance = get_xgb_feature_importance(model, prepared['feature_cols'])
            mape, rmse = self._calculate_metrics(prepared['y_test'], predictions)
            
            return ModelMetrics(
                name='xgboost',
                accuracy=accuracy,
                mape=mape,
                rmse=rmse,
                predictions=predictions,
                model=model,
                feature_importance=feature_importance
            )
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None
    
    def _train_lstm(self, prepared: Dict[str, Any]) -> Optional[ModelMetrics]:
        try:
            model, accuracy, predictions = train_lstm(
                prepared['X_train_lstm'], prepared['y_train_lstm'],
                prepared['X_test_lstm'], prepared['y_test_lstm']
            )
            
            if model is None:
                return None
            
            mape, rmse = self._calculate_metrics(prepared['y_test_lstm'], predictions)
            
            return ModelMetrics(
                name='lstm',
                accuracy=accuracy,
                mape=mape,
                rmse=rmse,
                predictions=predictions,
                model=model
            )
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return None
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0, 0.0
        
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return mape, rmse
    
    def _select_best_model(self, metric: str = 'accuracy') -> str:
        if not self.model_results:
            raise ValueError("No models were successfully trained")
        
        if metric == 'accuracy':
            return max(self.model_results.keys(), key=lambda k: self.model_results[k].accuracy)
        elif metric == 'mape':
            return min(self.model_results.keys(), key=lambda k: self.model_results[k].mape)
        elif metric == 'rmse':
            return min(self.model_results.keys(), key=lambda k: self.model_results[k].rmse)
        else:
            return max(self.model_results.keys(), key=lambda k: self.model_results[k].accuracy)
    
    def _generate_future_predictions(self, prepared: Dict[str, Any], steps: int) -> np.ndarray:
        best = self.model_results[self.best_model_name]
        
        if self.best_model_name == 'lstm' and best.model is not None:
            if len(prepared['X_test_lstm']) > 0:
                future_preds = predict_future(
                    best.model,
                    prepared['X_test_lstm'][-1],
                    prepared['target_scaler'],
                    steps=steps
                )
                return future_preds
            else:
                logger.warning("LSTM selected but no test sequences available, falling back to next best model")
                fallback_model = self._get_fallback_model()
                if fallback_model:
                    self.best_model_name = fallback_model
                    best = self.model_results[fallback_model]
                else:
                    return np.zeros(steps)
        
        if len(prepared['X_test']) > 0:
            if self.best_model_name == 'xgboost' and 'xgboost' in self.model_results:
                return predict_future_xgb(
                    self.model_results['xgboost'].model,
                    prepared['X_test'][-1],
                    prepared['target_scaler'],
                    steps=steps,
                    feature_cols=prepared['feature_cols']
                )
            elif self.best_model_name == 'random_forest' and 'random_forest' in self.model_results:
                return predict_future_rf(
                    self.model_results['random_forest'].model,
                    prepared['X_test'][-1],
                    prepared['target_scaler'],
                    steps=steps,
                    feature_cols=prepared['feature_cols']
                )
            elif 'xgboost' in self.model_results:
                self.best_model_name = 'xgboost'
                return predict_future_xgb(
                    self.model_results['xgboost'].model,
                    prepared['X_test'][-1],
                    prepared['target_scaler'],
                    steps=steps,
                    feature_cols=prepared['feature_cols']
                )
            elif 'random_forest' in self.model_results:
                self.best_model_name = 'random_forest'
                return predict_future_rf(
                    self.model_results['random_forest'].model,
                    prepared['X_test'][-1],
                    prepared['target_scaler'],
                    steps=steps,
                    feature_cols=prepared['feature_cols']
                )
        
        return np.zeros(steps)
    
    def _get_fallback_model(self) -> Optional[str]:
        non_lstm_models = [k for k in self.model_results.keys() if k != 'lstm']
        if not non_lstm_models:
            return None
        return max(non_lstm_models, key=lambda k: self.model_results[k].accuracy)
    
    def _convert_results_for_legacy(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                'accuracy': m.accuracy,
                'predictions': m.predictions,
                'model': m.model,
                'feature_importance': m.feature_importance
            }
            for name, m in self.model_results.items()
        }


def create_forecast_csv(forecast_result: Dict[str, Any], filepath: str = 'static/forecast_download.csv') -> str:
    future_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in forecast_result['future_dates']],
        'Predicted Water Usage': [round(v, 1) for v in forecast_result['future_predictions']]
    })
    
    future_df.to_csv(filepath, index=False)
    return filepath
