import numpy as np
from datetime import datetime, timedelta

def generate_live_flow_level():
    hour = datetime.now().hour
    base_flow = 15
    if 6 <= hour < 10:
        base_flow = 22
    elif 10 <= hour < 17:
        base_flow = 18
    elif 17 <= hour < 22:
        base_flow = 25
    else:
        base_flow = 12
    
    noise = np.random.uniform(-2, 2)
    flow_level = max(5, min(35, base_flow + noise))
    
    return round(flow_level)

def generate_24hour_trend():
    np.random.seed(int(datetime.now().timestamp()) // 3600)
    hours = list(range(24))
    
    base_pattern = [12, 10, 8, 7, 6, 8, 15, 22, 25, 20, 18, 17, 
                   16, 17, 18, 19, 20, 24, 26, 23, 20, 17, 14, 12]
    
    trend = [max(5, min(35, val + np.random.uniform(-2, 2))) for val in base_pattern]
    
    return hours, trend

def get_flow_anomaly_tag(flow_level):
    if flow_level < 8:
        return "low", "Low flow detected"
    elif flow_level > 28:
        return "high", "High flow detected"
    else:
        return "normal", "Normal flow"

def get_flow_status_color(status):
    colors = {
        "low": "#f59e0b",
        "high": "#ef4444",
        "normal": "#22c55e"
    }
    return colors.get(status, "#22c55e")
