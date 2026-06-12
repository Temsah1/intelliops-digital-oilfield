"""
src/ml/production_forecaster.py
────────────────────────────────
Production Forecasting — Multi-Model ML Pipeline.

Models trained:
  1. Linear Regression (baseline)
  2. Random Forest Regressor
  3. XGBoost Regressor (if available)

Best model selected by lowest RMSE on held-out test set.
6-month ahead forecast generated per well.

Target: BORE_OIL_VOL (daily oil production, Sm³/day)
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger
from src.config import (
    PRODUCTION_MODEL_PATH, MODEL_METRICS_PATH,
    RANDOM_STATE, TEST_SIZE, FORECAST_MONTHS, CV_FOLDS
)

log = get_logger(__name__)

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
    log.info("XGBoost detected ✓")
except ImportError:
    XGB_AVAILABLE = False
    log.warning("XGBoost not available — using GradientBoosting fallback")


# ── Feature set for production forecasting ────────────────────────────────────
PRODUCTION_FEATURES = [
    "DAYS_ON_PRODUCTION",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DP_TUBING",
    "AVG_WHP_P",
    "AVG_CHOKE_SIZE_P",
    "ON_STREAM_HRS",
    "ROLLING_7D_OIL",
    "ROLLING_30D_OIL",
    "ROLLING_7D_PRESSURE",
    "WATER_CUT",
    "GOR",
    "PRESSURE_DRAWDOWN",
    "YEAR",
    "MONTH",
    "DAY_OF_YEAR",
]

TARGET = "BORE_OIL_VOL"


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filter to production rows with valid oil volumes.
    Returns (X, y) ready for sklearn.
    """
    mask = (
        (df["FLOW_KIND"] == "production")
        & (df["BORE_OIL_VOL"].notna())
        & (df["BORE_OIL_VOL"] > 0)
        & (df["IS_SHUT_IN"] == 0)
    )
    sub = df[mask].copy()

    # Only keep features that actually exist in the dataframe
    feats = [f for f in PRODUCTION_FEATURES if f in sub.columns]

    X = sub[feats].fillna(0)
    y = sub[TARGET]
    return X, y, sub["DATEPRD"]


def train_production_models(df: pd.DataFrame) -> dict:
    """
    Train all production forecasting models.

    Returns
    -------
    dict  — metrics and best model name
    """
    log.info("═══ Training Production Forecasting Models ═══")

    X, y, dates = prepare_features(df)
    log.info(f"  Training set: {len(X):,} rows, {X.shape[1]} features")

    # ── Time-aware train/test split ───────────────────────────────────────────
    # Use last 20% of dates as test (preserves temporal order)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    log.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Model Definitions ─────────────────────────────────────────────────────
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    # ── Train & Evaluate ──────────────────────────────────────────────────────
    metrics = {}
    trained_models = {}

    for name, model in models.items():
        log.info(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Production cannot be negative

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        # Time-series cross-validation (5-fold)
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
        cv_rmse = -cross_val_score(
            model, X_train, y_train,
            cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )

        metrics[name] = {
            "mae":      round(float(mae), 2),
            "rmse":     round(float(rmse), 2),
            "r2":       round(float(r2), 4),
            "cv_rmse_mean": round(float(cv_rmse.mean()), 2),
            "cv_rmse_std":  round(float(cv_rmse.std()), 2),
        }
        trained_models[name] = model
        log.info(f"    MAE={mae:.1f} | RMSE={rmse:.1f} | R²={r2:.3f} | CV-RMSE={cv_rmse.mean():.1f}±{cv_rmse.std():.1f}")

    # ── Select Best Model ─────────────────────────────────────────────────────
    best_name = min(metrics, key=lambda k: metrics[k]["rmse"])
    best_model = trained_models[best_name]
    log.info(f"  Best model: {best_name} (RMSE={metrics[best_name]['rmse']:.1f})")

    # Feature importance (if available)
    feat_importance = {}
    raw_model = best_model["model"] if hasattr(best_model, "named_steps") else best_model
    if hasattr(raw_model, "feature_importances_"):
        feats = [f for f in PRODUCTION_FEATURES if f in X.columns]
        feat_importance = dict(zip(feats, raw_model.feature_importances_.tolist()))

    # ── Save ──────────────────────────────────────────────────────────────────
    PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, PRODUCTION_MODEL_PATH)
    log.info(f"  Saved best model → {PRODUCTION_MODEL_PATH.name}")

    result = {
        "best_model": best_name,
        "metrics": metrics,
        "feature_importance": feat_importance,
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "features": [f for f in PRODUCTION_FEATURES if f in X.columns],
        "y_test":  y_test.tolist(),
        "y_pred":  y_pred.tolist(),
        "test_dates": dates.iloc[split_idx:].astype(str).tolist(),
    }
    return result


def forecast_next_months(
    df: pd.DataFrame,
    model=None,
    months: int = FORECAST_MONTHS,
) -> pd.DataFrame:
    """
    Generate a per-well monthly production forecast for the next N months.

    Uses the last observed state of each well as the starting point,
    then projects forward using rolling average features.
    """
    if model is None:
        if not PRODUCTION_MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Run train_production_models first.")
        model = joblib.load(PRODUCTION_MODEL_PATH)

    feats = [f for f in PRODUCTION_FEATURES if f in df.columns]
    prod = df[(df["FLOW_KIND"] == "production") & (df["IS_SHUT_IN"] == 0)].copy()

    forecast_rows = []
    for well in prod["NPD_WELL_BORE_NAME"].unique():
        well_df = prod[prod["NPD_WELL_BORE_NAME"] == well].sort_values("DATEPRD")
        if len(well_df) < 30:
            continue

        last_row  = well_df.iloc[-1]
        last_date = pd.Timestamp(last_row["DATEPRD"])
        last_days = float(last_row.get("DAYS_ON_PRODUCTION", 0))

        rolling_7  = float(last_row.get("ROLLING_7D_OIL", last_row.get("BORE_OIL_VOL", 500)))
        rolling_30 = float(last_row.get("ROLLING_30D_OIL", last_row.get("BORE_OIL_VOL", 500)))

        for m in range(1, months + 1):
            fcst_date = last_date + pd.DateOffset(months=m)
            future_days = last_days + m * 30.5

            row_data = {f: float(last_row.get(f, 0)) for f in feats}
            row_data["DAYS_ON_PRODUCTION"] = future_days
            row_data["YEAR"]  = fcst_date.year
            row_data["MONTH"] = fcst_date.month
            row_data["DAY_OF_YEAR"] = fcst_date.dayofyear
            # Gradually apply decline to rolling averages
            decay = np.exp(-0.003 * m * 30.5)
            row_data["ROLLING_7D_OIL"]  = rolling_7  * decay
            row_data["ROLLING_30D_OIL"] = rolling_30 * decay

            X_pred = pd.DataFrame([row_data])[feats]
            q_pred = float(np.maximum(model.predict(X_pred)[0], 0))

            forecast_rows.append({
                "well_name":    well,
                "forecast_date": fcst_date,
                "months_ahead":  m,
                "predicted_oil_sm3":   round(q_pred, 1),
                "predicted_oil_boepd": round(q_pred * 6.2898, 1),
                "lower_bound_sm3": round(q_pred * 0.80, 1),
                "upper_bound_sm3": round(q_pred * 1.20, 1),
            })

    return pd.DataFrame(forecast_rows)
