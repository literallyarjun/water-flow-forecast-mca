import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'actual': '#3b82f6',
    'predicted': '#22c55e',
    'future': '#8b5cf6',
    'temperature': '#ef4444',
    'humidity': '#3b82f6',
    'rainfall': '#06b6d4',
    'wind': '#84cc16',
    'evaporation': '#f59e0b'
}

def ensure_plots_dir():
    os.makedirs('static/plots', exist_ok=True)

def create_consumption_trend_chart(data, save_path='static/plots/consumption_trend.png'):
    ensure_plots_dir()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = data['date']
    consumption = data['consumption']
    
    ax.plot(dates, consumption, color=COLORS['actual'], linewidth=2, label='Water Usage')
    ax.fill_between(dates, consumption, alpha=0.2, color=COLORS['actual'])
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Water Usage Level', fontsize=12)
    ax.set_title('Daily Water Consumption Trend', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def create_climate_graphs(data, save_path='static/plots/climate_graphs.png'):
    ensure_plots_dir()
    
    climate_cols = ['temperature', 'humidity', 'rainfall', 'wind', 'evaporation']
    available_cols = [col for col in climate_cols if col in data.columns]
    
    if not available_cols:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'No climate data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    fig, axes = plt.subplots(1, len(available_cols), figsize=(4*len(available_cols), 4))
    
    if len(available_cols) == 1:
        axes = [axes]
    
    labels = {
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'rainfall': 'Rainfall',
        'wind': 'Wind Speed',
        'evaporation': 'Evaporation'
    }
    
    for idx, col in enumerate(available_cols):
        ax = axes[idx]
        color = COLORS.get(col, '#3b82f6')
        
        ax.plot(data['date'], data[col], color=color, linewidth=1.5)
        ax.fill_between(data['date'], data[col], alpha=0.2, color=color)
        ax.set_title(labels.get(col, col), fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def create_flow_level_trend(hours, values, save_path='static/plots/flow_level_trend.png'):
    ensure_plots_dir()
    
    fig, ax = plt.subplots(figsize=(6, 2))
    
    ax.plot(hours, values, color=COLORS['actual'], linewidth=2)
    ax.fill_between(hours, values, alpha=0.3, color=COLORS['actual'])
    
    ax.set_xlim(0, 23)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_xlabel('Hour', fontsize=10)
    ax.set_title('24-Hour Flow Trend', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def create_forecast_chart(actual, predicted, future_dates, future_preds, 
                          forecast_range='30D', save_path='static/plots/forecast_plot.png'):
    ensure_plots_dir()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    actual_x = list(range(len(actual)))
    ax.plot(actual_x, actual, color=COLORS['actual'], linewidth=2, label='Actual Usage', alpha=0.8)
    
    if len(predicted) > 0:
        pred_x = actual_x[-len(predicted):]
        ax.plot(pred_x, predicted, color=COLORS['predicted'], linewidth=2, 
                label='Model Prediction', linestyle='--')
    
    future_x = list(range(len(actual), len(actual) + len(future_preds)))
    ax.plot(future_x, future_preds, color=COLORS['future'], linewidth=2, 
            label='Future Forecast', linestyle='-')
    ax.fill_between(future_x, future_preds, alpha=0.2, color=COLORS['future'])
    
    ax.axvline(x=len(actual)-1, color='gray', linestyle=':', alpha=0.5, label='Today')
    
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Water Usage Level', fontsize=12)
    ax.set_title(f'Water Consumption Forecast ({forecast_range})', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def create_model_accuracy_chart(model_results, save_path='static/plots/model_accuracy.png'):
    ensure_plots_dir()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = list(model_results.keys())
    accuracies = [model_results[m]['accuracy'] for m in models]
    
    model_labels = {
        'lstm': 'LSTM Neural Network',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }
    
    colors = ['#3b82f6', '#22c55e', '#f59e0b']
    
    bars = ax.barh([model_labels.get(m, m) for m in models], accuracies, color=colors[:len(models)], height=0.5)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.0f}%', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 110)
    ax.set_xlabel('Accuracy Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def create_feature_importance_chart(feature_importance, save_path='static/plots/feature_importance.png'):
    ensure_plots_dir()
    
    if not feature_importance:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No feature importance data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(feature_importance.keys())[:10]
    importances = [feature_importance[f] for f in features]
    
    feature_labels = {
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'rainfall': 'Rainfall',
        'wind': 'Wind Speed',
        'evaporation': 'Evaporation',
        'consumption_lag_1': 'Previous Day',
        'consumption_lag_7': 'Last Week',
        'consumption_lag_14': 'Two Weeks Ago',
        'consumption_lag_30': 'Last Month',
        'month': 'Month',
        'weekday': 'Day of Week',
        'day': 'Day',
        'season': 'Season'
    }
    
    labels = [feature_labels.get(f, f.replace('_', ' ').title()) for f in features]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    
    bars = ax.barh(labels, importances, color=colors)
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('What Affects Water Usage Most', fontsize=16, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def generate_all_charts(forecast_result, hours, flow_values, forecast_range='30D'):
    charts = {}
    
    charts['consumption_trend'] = create_consumption_trend_chart(forecast_result['data'])
    
    charts['climate_graphs'] = create_climate_graphs(forecast_result['data'])
    
    charts['flow_level_trend'] = create_flow_level_trend(hours, flow_values)
    
    charts['forecast_plot'] = create_forecast_chart(
        forecast_result['actual_values'],
        forecast_result['predicted_values'],
        forecast_result['future_dates'],
        forecast_result['future_predictions'],
        forecast_range
    )
    
    charts['model_accuracy'] = create_model_accuracy_chart(forecast_result['model_results'])
    
    charts['feature_importance'] = create_feature_importance_chart(forecast_result['feature_importance'])
    
    return charts
