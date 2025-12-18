# MetroFlow - Water Consumption Forecasting System

## Overview
MetroFlow is a Flask-based visual water consumption forecasting system that predicts city water consumption using historical data, climate indicators, and live water flow levels. The system focuses on simple visual charts, easy-to-read values, and teacher-friendly UI without showing complex measurements or engineering terms.

## Project Structure
```
├── app.py                     # Main Flask application (routes only)
├── main.py                    # Entry point
├── models.py                  # Database models
├── services/
│   ├── data_ingestion.py      # Data loading, validation, provenance tracking
│   ├── preprocessing_pipeline.py # Feature engineering, normalization, train/test split
│   └── forecast_service.py    # Model training, selection, prediction
├── modules/
│   ├── data_preprocessing.py  # Legacy CSV loading, missing values, lag features
│   ├── live_water_flow.py     # Simulated live flow levels
│   ├── feature_engineering.py # Normalization, train-test split, LSTM sliding window
│   ├── forecasting_engine.py  # Legacy: Train all models, generate forecasts
│   └── visualization.py       # Generate matplotlib charts
├── models/
│   ├── lstm_model.py          # LSTM neural network model
│   ├── random_forest_model.py # Random Forest model
│   └── xgboost_model.py       # XGBoost model
├── templates/
│   ├── index.html             # Home page with upload interface
│   └── results.html           # Results dashboard with charts
├── static/
│   ├── css/style.css          # Custom styling
│   └── plots/                 # Generated chart images
└── uploads/                   # Uploaded CSV files and default dataset
```

## Key Features
1. **CSV Upload**: Drag-and-drop interface for water consumption data
2. **Live Flow Widget**: Shows current flow level with 24-hour trend
3. **ML Models**: LSTM, Random Forest, and XGBoost for predictions
4. **Visual Charts**: Consumption trend, climate indicators, forecast plots
5. **Model Comparison**: Bar chart showing model accuracy
6. **Feature Importance**: Shows what factors affect water usage most
7. **Downloadable CSV**: Export forecast data in simple format

## Technical Stack
- **Backend**: Flask, Flask-SQLAlchemy
- **ML Libraries**: scikit-learn (Random Forest), XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Database**: PostgreSQL

Note: TensorFlow/LSTM is optional due to storage constraints. The system gracefully falls back to Random Forest and XGBoost models when TensorFlow is not available.

## Architecture (Improved December 2025)

### Services Layer
- **DataIngestionService**: Handles all data loading with schema validation, column normalization, date parsing, and provenance tracking
- **PreprocessingPipeline**: Centralizes feature engineering, missing value handling, lag features, and train/test splitting
- **ForecastService**: Trains models, calculates metrics (accuracy, MAPE, RMSE), selects best model, generates predictions

### Data Flow
1. User uploads CSV or uses default dataset
2. DataIngestionService validates schema, normalizes columns, filters to 2025
3. PreprocessingPipeline extracts features, creates lags, normalizes data
4. ForecastService trains models and generates predictions
5. Visualization module creates charts
6. Results displayed to user with provenance information

## Data Configuration
- Default dataset: `metroflow_2025_full_year.csv` (place in uploads/ folder)
- Supported date formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, and more
- Year filter: 2025 data only
- When CSV files are uploaded, they are validated and used instead of the default

## Running the Application
The application runs on port 5000:
```bash
python app.py
```

For production deployment:
```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port app:app
```

## Design Principles
- No scientific jargon or complex units
- Teacher-friendly, easy to understand labels
- Clear color coding: blue (actual), green (predicted), purple (future)
- Large fonts suitable for classroom projection
- Responsive design for all screen sizes

## Recent Changes
- December 2025: Added Weekly/Monthly Forecast section with 7-day and 30-day prediction buttons
- December 2025: Added City Selector dropdown with presets for Chennai, Mumbai, Delhi, Bangalore
- December 2025: Modern UI redesign with gradient header, cards, icons, and smooth transitions
- December 2025: Added Chart.js for interactive forecast line charts
- December 2025: Improved mobile responsiveness
- December 2025: Major refactor - created services layer (DataIngestionService, PreprocessingPipeline, ForecastService) for better code organization and reliability
- December 2025: Added robust date parsing with multiple format support and fallback logic
- December 2025: Added data provenance tracking for transparency on data source
- December 2025: Improved model selection with MAPE/RMSE metrics
- December 2025: Updated to use 2025 dataset (metroflow_2025_full_year.csv)
- December 2025: Imported to Replit, configured for Replit environment, made TensorFlow optional
- December 2024: Initial implementation with all core features
