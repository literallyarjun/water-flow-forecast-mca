import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from modules.data_preprocessing import preprocess_data, generate_sample_data, extract_date_features, create_lag_features
from modules.feature_engineering import prepare_data_for_models
from models.lstm_model import train_lstm, predict_future
from models.random_forest_model import train_random_forest, get_feature_importance, predict_future_rf
from models.xgboost_model import train_xgboost, get_xgb_feature_importance, predict_future_xgb

def train_all_models(data):
    prepared = prepare_data_for_models(data)
    
    results = {
        'lstm': {'accuracy': 0, 'predictions': [], 'model': None},
        'random_forest': {'accuracy': 0, 'predictions': [], 'model': None},
        'xgboost': {'accuracy': 0, 'predictions': [], 'model': None}
    }
    
    rf_model, rf_accuracy, rf_preds = train_random_forest(
        prepared['X_train'], prepared['y_train'],
        prepared['X_test'], prepared['y_test']
    )
    results['random_forest'] = {
        'accuracy': rf_accuracy,
        'predictions': rf_preds,
        'model': rf_model,
        'feature_importance': get_feature_importance(rf_model, prepared['feature_cols'])
    }
    
    xgb_model, xgb_accuracy, xgb_preds = train_xgboost(
        prepared['X_train'], prepared['y_train'],
        prepared['X_test'], prepared['y_test']
    )
    results['xgboost'] = {
        'accuracy': xgb_accuracy,
        'predictions': xgb_preds,
        'model': xgb_model,
        'feature_importance': get_xgb_feature_importance(xgb_model, prepared['feature_cols'])
    }
    
    if len(prepared['X_train_lstm']) >= 10:
        lstm_model, lstm_accuracy, lstm_preds = train_lstm(
            prepared['X_train_lstm'], prepared['y_train_lstm'],
            prepared['X_test_lstm'], prepared['y_test_lstm']
        )
        if lstm_model is not None:
            results['lstm'] = {
                'accuracy': lstm_accuracy,
                'predictions': lstm_preds,
                'model': lstm_model
            }
    
    return results, prepared

def choose_best_model(results):
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    return best_model, results[best_model]

def generate_forecast(data, forecast_days=30):
    results, prepared = train_all_models(data)
    
    best_model_name, best_result = choose_best_model(results)
    
    last_date = pd.to_datetime(data['date'].iloc[-1])
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    feature_cols = prepared['feature_cols']
    
    if best_model_name == 'lstm' and results['lstm']['model'] is not None:
        future_preds = predict_future(
            results['lstm']['model'],
            prepared['X_test_lstm'][-1],
            prepared['target_scaler'],
            steps=forecast_days
        )
    elif best_model_name == 'xgboost':
        future_preds = predict_future_xgb(
            results['xgboost']['model'],
            prepared['X_test'][-1],
            prepared['target_scaler'],
            steps=forecast_days,
            feature_cols=feature_cols
        )
    else:
        future_preds = predict_future_rf(
            results['random_forest']['model'],
            prepared['X_test'][-1],
            prepared['target_scaler'],
            steps=forecast_days,
            feature_cols=feature_cols
        )
    
    actual_values = prepared['target_scaler'].inverse_transform(
        prepared['y_test'].reshape(-1, 1)
    ).flatten()
    
    if best_model_name == 'lstm':
        predicted_values = prepared['target_scaler'].inverse_transform(
            results['lstm']['predictions'].reshape(-1, 1)
        ).flatten() if len(results['lstm']['predictions']) > 0 else []
    else:
        predicted_values = prepared['target_scaler'].inverse_transform(
            results[best_model_name]['predictions'].reshape(-1, 1)
        ).flatten()
    
    feature_importance = results.get('random_forest', {}).get('feature_importance', {})
    if not feature_importance:
        feature_importance = results.get('xgboost', {}).get('feature_importance', {})
    
    return {
        'best_model': best_model_name,
        'model_results': results,
        'future_dates': future_dates,
        'future_predictions': future_preds,
        'actual_values': actual_values,
        'predicted_values': predicted_values,
        'feature_importance': feature_importance,
        'data': data
    }

def create_forecast_csv(forecast_result, filepath='static/forecast_download.csv'):
    future_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in forecast_result['future_dates']],
        'Predicted Water Usage': [round(v, 1) for v in forecast_result['future_predictions']]
    })
    
    future_df.to_csv(filepath, index=False)
    return filepath
