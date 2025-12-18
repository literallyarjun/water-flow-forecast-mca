from app import db
from datetime import datetime

class ForecastLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    forecast_range = db.Column(db.String(10))
    best_model = db.Column(db.String(50))
    accuracy = db.Column(db.Float)
    data_source = db.Column(db.String(100))
