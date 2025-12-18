import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    accuracy = max(0, 100 - (mae * 100))
    
    return model, accuracy, predictions

def get_xgb_feature_importance(model, feature_names):
    importances = model.feature_importances_
    
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_importance

def predict_future_xgb(model, last_features, target_scaler, steps=30, feature_cols=None):
    predictions_scaled = []
    current_features = last_features.copy()
    
    lag_indices = {}
    if feature_cols is not None:
        for i, col in enumerate(feature_cols):
            if 'lag_1' in col:
                lag_indices['lag_1'] = i
            elif 'lag_7' in col:
                lag_indices['lag_7'] = i
    
    past_preds = []
    
    for step in range(steps):
        pred = model.predict(current_features.reshape(1, -1))[0]
        predictions_scaled.append(pred)
        past_preds.append(pred)
        
        new_features = current_features.copy()
        
        if 'lag_1' in lag_indices:
            new_features[lag_indices['lag_1']] = pred
        
        if 'lag_7' in lag_indices and len(past_preds) >= 7:
            new_features[lag_indices['lag_7']] = past_preds[-7]
        
        current_features = new_features
    
    predictions = np.array(predictions_scaled).reshape(-1, 1)
    predictions = target_scaler.inverse_transform(predictions).flatten()
    
    return predictions
