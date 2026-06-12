"""config.py — Central configuration for AI Digital Oilfield Platform"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_PRODUCTION_CSV  = DATA_RAW_DIR / "well_production.csv"
RAW_MAINTENANCE_CSV = DATA_RAW_DIR / "predictive_maintenance.csv"
PROCESSED_PROD_CSV  = DATA_PROCESSED_DIR / "production_clean.csv"
PROCESSED_MAINT_CSV = DATA_PROCESSED_DIR / "maintenance_clean.csv"

DB_DIR  = ROOT_DIR / "database"
DB_PATH = DB_DIR / "oilfield.db"

MODELS_DIR             = ROOT_DIR / "models"
PRODUCTION_MODEL_PATH  = MODELS_DIR / "production_forecaster.joblib"
MAINTENANCE_MODEL_PATH = MODELS_DIR / "failure_predictor.joblib"
MODEL_METRICS_PATH     = MODELS_DIR / "model_metrics.json"

WELL_METADATA = {
    "15/9-F-1 C":  {"type": "producer", "field": "Volve", "depth_m": 2750, "start_year": 2008},
    "15/9-F-11":   {"type": "producer", "field": "Volve", "depth_m": 2700, "start_year": 2008},
    "15/9-F-12":   {"type": "producer", "field": "Volve", "depth_m": 2720, "start_year": 2008},
    "15/9-F-14":   {"type": "producer", "field": "Volve", "depth_m": 2695, "start_year": 2008},
    "15/9-F-15 D": {"type": "producer", "field": "Volve", "depth_m": 2710, "start_year": 2010},
    "15/9-F-4":    {"type": "injector", "field": "Volve", "depth_m": 2760, "start_year": 2008},
    "15/9-F-5":    {"type": "injector", "field": "Volve", "depth_m": 2740, "start_year": 2008},
}

DEVICE_TYPE_MAP = {
    "S1": "Submersible Pump",
    "W1": "Wellhead Controller",
    "Z1": "Zone Isolation Valve",
}

SM3_TO_BARREL = 6.2898
SM3_TO_MCF    = 0.03531
OIL_PRICE_USD = 70.0
GAS_PRICE_USD = 3.5

RANDOM_STATE    = 42
TEST_SIZE       = 0.2
FORECAST_MONTHS = 6
CV_FOLDS        = 5

RISK_THRESHOLDS = {
    "critical": 0.70,
    "high":     0.40,
    "medium":   0.20,
    "low":      0.00,
}

THEME = {
    "primary":        "#00D4FF",
    "secondary":      "#FF6B35",
    "success":        "#00E676",
    "warning":        "#FFD600",
    "danger":         "#FF1744",
    "info":           "#40C4FF",
    "bg_dark":        "#0A0E1A",
    "bg_card":        "#111827",
    "bg_sidebar":     "#0D1117",
    "surface":        "#1A2332",
    "text_primary":   "#E8EAED",
    "text_secondary": "#9AA0A6",
    "text_muted":     "#5F6368",
}

PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT    = 420
CHART_MARGIN    = dict(l=40, r=20, t=40, b=40)
