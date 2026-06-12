"""
src/ml/failure_predictor.py
────────────────────────────
Predictive Maintenance — Equipment Failure Classification.

Handles the severe class imbalance (0.085% failure rate) using:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - class_weight='balanced' in tree models

Models trained:
  1. Logistic Regression (baseline)
  2. Random Forest Classifier (primary)
  3. XGBoost Classifier (if available)

Outputs:
  - Failure probability per device-day
  - Equipment Health Score (0–100)
  - Risk Level (Low / Medium / High / Critical)
  - Feature Importance for explainability
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger
from src.config import (
    MAINTENANCE_MODEL_PATH, RANDOM_STATE, TEST_SIZE, RISK_THRESHOLDS
)

log = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Feature set for failure prediction ───────────────────────────────────────
MAINTENANCE_FEATURES = [
    "metric1", "metric2", "metric3", "metric4",
    "metric5", "metric6", "metric7", "metric9",
    "ROLLING_3D_metric7", "ROLLING_7D_metric7",
    "ROLLING_3D_metric4", "ROLLING_7D_metric4",
    "ROLLING_3D_metric2", "ROLLING_7D_metric2",
    "METRIC7_SPIKE", "METRIC4_SPIKE", "METRIC2_SPIKE",
    "METRIC1_DELTA",
    "ANOMALY_SCORE",
    "DAY_OF_WEEK", "MONTH", "IS_WEEKEND",
]

TARGET = "failure"


def prepare_maintenance_features(df: pd.DataFrame) -> tuple:
    """
    Prepare X, y for maintenance model training.
    Drops rows with all-null features and encodes device type.
    """
    feats = [f for f in MAINTENANCE_FEATURES if f in df.columns]
    sub = df[feats + [TARGET]].dropna(subset=feats).copy()
    X = sub[feats].fillna(0)
    y = sub[TARGET].astype(int)
    return X, y, feats


def train_failure_models(df: pd.DataFrame) -> dict:
    """
    Train all predictive maintenance classification models.

    Returns dict with metrics, confusion matrices, feature importances.
    """
    log.info("═══ Training Predictive Maintenance Models ═══")

    X, y, feats = prepare_maintenance_features(df)
    log.info(f"  Dataset: {len(X):,} rows | {X.shape[1]} features")
    log.info(f"  Class balance: 0={int((y==0).sum()):,}, 1={int((y==1).sum()):,}")

    # Stratified split (preserves failure rate in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    log.info(f"  Test failures: {int(y_test.sum())} ({y_test.mean()*100:.3f}%)")

    # ── Model Definitions ─────────────────────────────────────────────────────
    # SMOTE applied inside each pipeline to prevent data leakage
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)

    models = {
        "Logistic Regression": ImbPipeline([
            ("smote", smote),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
                C=0.1,
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", smote),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    if XGB_AVAILABLE:
        ratio = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
        models["XGBoost"] = ImbPipeline([
            ("smote", smote),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=ratio,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                verbosity=0,
                random_state=RANDOM_STATE,
            )),
        ])

    # ── Train & Evaluate ──────────────────────────────────────────────────────
    metrics_all = {}
    trained_models = {}

    for name, pipeline in models.items():
        log.info(f"  Training {name}...")
        pipeline.fit(X_train, y_train)

        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.0
        cm   = confusion_matrix(y_test, y_pred).tolist()

        metrics_all[name] = {
            "accuracy":  round(float(acc),  4),
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1":        round(float(f1),   4),
            "roc_auc":   round(float(auc),  4),
            "confusion_matrix": cm,
        }
        trained_models[name] = pipeline
        log.info(f"    Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | AUC={auc:.3f}")

    # ── Select Best Model (by F1, most important for imbalanced data) ─────────
    best_name = max(metrics_all, key=lambda k: metrics_all[k]["f1"])
    best_model = trained_models[best_name]
    log.info(f"  Best model: {best_name} (F1={metrics_all[best_name]['f1']:.3f})")

    # Feature importance
    feat_importance = {}
    raw_model = best_model["model"] if hasattr(best_model, "named_steps") else best_model
    if hasattr(raw_model, "feature_importances_"):
        feat_importance = dict(zip(feats, raw_model.feature_importances_.tolist()))
        top5 = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        log.info(f"  Top 5 features: {top5}")

    # Save
    MAINTENANCE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MAINTENANCE_MODEL_PATH)
    log.info(f"  Saved model → {MAINTENANCE_MODEL_PATH.name}")

    return {
        "best_model": best_name,
        "metrics": metrics_all,
        "feature_importance": feat_importance,
        "features": feats,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def score_equipment_health(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Generate per-device health scores and risk rankings.

    For each device returns:
      - failure_probability: model-predicted probability
      - health_score: 0–100 (100 = perfectly healthy)
      - risk_level: Low / Medium / High / Critical
      - last_metric7 / last_anomaly: last observed sensor values
    """
    if model is None:
        if not MAINTENANCE_MODEL_PATH.exists():
            raise FileNotFoundError("No maintenance model. Run train_failure_models first.")
        model = joblib.load(MAINTENANCE_MODEL_PATH)

    feats = [f for f in MAINTENANCE_FEATURES if f in df.columns]

    # Get most recent record per device
    latest = (
        df.sort_values("date")
          .groupby("device")
          .last()
          .reset_index()
    )

    X = latest[feats].fillna(0)
    proba = model.predict_proba(X)[:, 1]

    result = latest[["device", "DEVICE_TYPE", "ANOMALY_SCORE",
                      "HEALTH_SCORE_RULE"]].copy()
    if "metric7" in latest.columns:
        result["last_metric7"] = latest["metric7"].values
    if "metric4" in latest.columns:
        result["last_metric4"] = latest["metric4"].values

    result["failure_probability"] = proba
    result["health_score"]         = (1 - proba) * 100
    result["risk_level"] = pd.cut(
        result["failure_probability"],
        bins=[-np.inf,
              RISK_THRESHOLDS["medium"],
              RISK_THRESHOLDS["high"],
              RISK_THRESHOLDS["critical"],
              np.inf],
        labels=["Low", "Medium", "High", "Critical"],
    )

    result = result.sort_values("failure_probability", ascending=False)
    log.info(f"Health scored {len(result)} devices")
    log.info(f"Risk distribution: {result['risk_level'].value_counts().to_dict()}")
    return result
