"""
src/ml/train_all.py
────────────────────
Master training script.
Run:  python -m src.ml.train_all

Trains both production forecasting and predictive maintenance models,
saves results to models/model_metrics.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.logger import get_logger
from src.config import PROCESSED_PROD_CSV, PROCESSED_MAINT_CSV, MODEL_METRICS_PATH
from src.ml.production_forecaster import train_production_models, forecast_next_months
from src.ml.failure_predictor import train_failure_models, score_equipment_health

log = get_logger(__name__)


def train_all() -> dict:
    start = time.time()
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║   ML Training Pipeline — All Models          ║")
    log.info("╚══════════════════════════════════════════════╝")

    # ── Load processed data ───────────────────────────────────────────────────
    log.info("Loading processed datasets...")
    df_prod  = pd.read_csv(PROCESSED_PROD_CSV,  parse_dates=["DATEPRD"])
    df_maint = pd.read_csv(PROCESSED_MAINT_CSV, parse_dates=["date"])
    log.info(f"  Production:  {len(df_prod):,} rows")
    log.info(f"  Maintenance: {len(df_maint):,} rows")

    all_metrics = {}

    # ── Production Forecasting ────────────────────────────────────────────────
    log.info("\n── Production Forecasting ──────────────────────────")
    prod_results = train_production_models(df_prod)
    all_metrics["production_forecasting"] = prod_results

    # ── Predictive Maintenance ────────────────────────────────────────────────
    log.info("\n── Predictive Maintenance ──────────────────────────")
    maint_results = train_failure_models(df_maint)
    all_metrics["predictive_maintenance"] = maint_results

    # ── Save metrics ──────────────────────────────────────────────────────────
    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Convert non-serialisable objects
    def default_serial(obj):
        if isinstance(obj, (float, int)):
            return obj
        return str(obj)

    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2, default=default_serial)
    log.info(f"\nMetrics saved → {MODEL_METRICS_PATH.name}")

    elapsed = time.time() - start
    log.info(f"\nTotal training time: {elapsed:.1f}s")
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║   Training Complete ✓                        ║")
    log.info("╚══════════════════════════════════════════════╝")
    return all_metrics


if __name__ == "__main__":
    train_all()
