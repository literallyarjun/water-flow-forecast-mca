import os
import logging
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['DEFAULT_DATASET'] = 'metroflow_2025_full_year.csv'
app.config['TARGET_YEAR'] = 2025

db.init_app(app)

os.makedirs('uploads', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from modules.live_water_flow import generate_live_flow_level, generate_24hour_trend, get_flow_anomaly_tag
from modules.visualization import generate_all_charts
from services.data_ingestion import DataIngestionService
from services.forecast_service import ForecastService, create_forecast_csv

data_ingestion = DataIngestionService(
    upload_folder=app.config['UPLOAD_FOLDER'],
    default_dataset=app.config['DEFAULT_DATASET'],
    target_year=app.config['TARGET_YEAR']
)


@app.route('/')
def index():
    flow_level = generate_live_flow_level()
    status, status_msg = get_flow_anomaly_tag(flow_level)
    
    return render_template('index.html', 
                         flow_level=flow_level,
                         flow_status=status,
                         flow_status_msg=status_msg,
                         openweather_api_key=os.environ.get('OPENWEATHER_API_KEY', ''))


@app.route('/forecast', methods=['POST'])
def forecast():
    forecast_range = request.form.get('forecast_range', '30D')
    range_map = {'24H': 1, '7D': 7, '30D': 30}
    forecast_days = range_map.get(forecast_range, 30)
    
    files = request.files.getlist('files')
    valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]
    
    logger.info(f"Received {len(files)} files, {len(valid_files)} valid CSV files")
    
    saved_filepaths = []
    for file in valid_files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        saved_filepaths.append(filepath)
        logger.info(f"Saved uploaded file: {filename}")
    
    if saved_filepaths:
        data, provenance = data_ingestion.process_uploaded_files(saved_filepaths)
        if data is None or data.empty:
            for error in provenance.validation_errors:
                flash(error)
            logger.warning("Uploaded files invalid, falling back to default dataset")
            data, provenance = data_ingestion.load_default_dataset()
    else:
        logger.info("No files uploaded, loading default dataset")
        data, provenance = data_ingestion.load_default_dataset()
    
    if not provenance.is_valid or data is None or data.empty:
        flash('Could not load valid data. Please check your CSV file format.')
        return redirect(url_for('index'))
    
    logger.info(f"Data loaded: {provenance.filtered_rows} rows from {provenance.source}")
    
    try:
        forecast_service = ForecastService()
        forecast_result = forecast_service.generate_forecast(data, forecast_days=forecast_days)
        
        hours, flow_values = generate_24hour_trend()
        flow_level = generate_live_flow_level()
        status, status_msg = get_flow_anomaly_tag(flow_level)
        
        charts = generate_all_charts(forecast_result, hours, flow_values, forecast_range)
        
        csv_path = create_forecast_csv(forecast_result)
        
        session['forecast_complete'] = True
        
        best_model_name = forecast_result['best_model']
        model_accuracy = forecast_result['model_results'][best_model_name]['accuracy']
        
        return render_template('results.html',
                             flow_level=flow_level,
                             flow_status=status,
                             flow_status_msg=status_msg,
                             best_model=best_model_name.replace('_', ' ').title(),
                             model_accuracy=model_accuracy,
                             forecast_range=forecast_range,
                             charts=charts,
                             data_source=provenance.source_description,
                             data_rows=provenance.filtered_rows,
                             data_year=provenance.year)
    
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}", exc_info=True)
        flash(f'Error generating forecast: {str(e)}')
        return redirect(url_for('index'))


@app.route('/download')
def download():
    csv_path = 'static/forecast_download.csv'
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name='water_forecast.csv')
    else:
        flash('No forecast data available for download')
        return redirect(url_for('index'))


@app.route('/manual_predict', methods=['POST'])
def manual_predict():
    try:
        temperature = float(request.form.get('temperature', 25))
        humidity = float(request.form.get('humidity', 60))
        rainfall = float(request.form.get('rainfall', 0))
        population = int(request.form.get('population', 50000))
        day_of_week = int(request.form.get('day_of_week', 0))
        month = int(request.form.get('month', 1))
        
        base_usage = population * 150
        
        temp_factor = 1 + (temperature - 20) * 0.02
        humidity_factor = 1 - (humidity - 50) * 0.005
        rain_factor = 1 - min(rainfall * 0.01, 0.15)
        
        if day_of_week >= 5:
            day_factor = 1.1
        else:
            day_factor = 1.0
        
        if month in [6, 7, 8]:
            season_factor = 1.15
        elif month in [12, 1, 2]:
            season_factor = 0.9
        else:
            season_factor = 1.0
        
        predicted_usage = base_usage * temp_factor * humidity_factor * rain_factor * day_factor * season_factor
        predicted_usage = round(predicted_usage, 0)
        
        flow_level = generate_live_flow_level()
        status, status_msg = get_flow_anomaly_tag(flow_level)
        
        return render_template('index.html',
                             flow_level=flow_level,
                             flow_status=status,
                             flow_status_msg=status_msg,
                             prediction_result=f"{predicted_usage:,.0f}")
    
    except Exception as e:
        logger.error(f"Manual prediction error: {str(e)}", exc_info=True)
        flash(f'Error calculating prediction: {str(e)}')
        return redirect(url_for('index'))


with app.app_context():
    import models
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
