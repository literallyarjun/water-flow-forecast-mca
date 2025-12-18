import numpy as np
import os

TENSORFLOW_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

def build_lstm_model(input_shape):
    if not TENSORFLOW_AVAILABLE:
        return None
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(30, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    if not TENSORFLOW_AVAILABLE:
        return None, 0, []
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    if len(X_train) < 10:
        return None, 0, []
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    if model is None:
        return None, 0, []
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    accuracy = max(0, 100 - (mae * 100))
    
    predictions = model.predict(X_test, verbose=0).flatten()
    
    return model, accuracy, predictions

def predict_future(model, last_sequence, target_scaler, steps=30):
    if model is None:
        return np.array([])
    
    predictions_scaled = []
    current_seq = last_sequence.copy()
    n_features = current_seq.shape[1]
    
    for step in range(steps):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)[0, 0]
        predictions_scaled.append(pred)
        
        current_seq = np.roll(current_seq, -1, axis=0)
        
        new_row = current_seq[-2].copy()
        if n_features > 0:
            new_row[0] = pred
        current_seq[-1] = new_row
    
    predictions = np.array(predictions_scaled).reshape(-1, 1)
    predictions = target_scaler.inverse_transform(predictions).flatten()
    
    return predictions
