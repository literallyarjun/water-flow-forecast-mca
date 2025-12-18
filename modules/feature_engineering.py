import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def normalize_features(df, feature_cols, target_col='consumption'):
    scaler = MinMaxScaler()
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    X_scaled = scaler.fit_transform(X)
    
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler, target_scaler

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    return X_train, X_test, y_train, y_test

def create_lstm_sequences(X, y, window_size=7):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    
    return np.array(X_seq), np.array(y_seq)

def prepare_data_for_models(df, target_col='consumption'):
    feature_cols = [col for col in df.columns if col not in ['date', target_col] and df[col].dtype in ['int64', 'float64']]
    
    if not feature_cols:
        feature_cols = ['month', 'weekday', 'day', 'season']
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
    
    X_scaled, y_scaled, scaler, target_scaler = normalize_features(df, feature_cols, target_col)
    
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    
    X_train_lstm, y_train_lstm = create_lstm_sequences(X_train, y_train)
    X_test_lstm, y_test_lstm = create_lstm_sequences(X_test, y_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_lstm': X_train_lstm,
        'X_test_lstm': X_test_lstm,
        'y_train_lstm': y_train_lstm,
        'y_test_lstm': y_test_lstm,
        'scaler': scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols
    }
