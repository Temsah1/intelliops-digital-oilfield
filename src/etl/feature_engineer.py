"""
src/etl/feature_engineer.py
────────────────────────────
Petroleum Engineering Feature Engineering.

Creates all derived features used in analytics and ML.
Every generated column is documented with its engineering formula.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_production_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add petroleum engineering derived features to production data.

    Derived features (documented):
    ┌─────────────────────────────┬──────────────────────────────────────────────┐
    │ Feature                     │ Formula / Source                             │
    ├─────────────────────────────┼──────────────────────────────────────────────┤
    │ WATER_CUT                   │ WAT / (OIL + WAT)  [fraction 0–1]           │
    │ GOR                         │ GAS / OIL  [Sm³/Sm³]                        │
    │ LIQUID_RATE                 │ OIL + WAT  [Sm³/day]                        │
    │ PRODUCTION_EFFICIENCY       │ ON_STREAM_HRS / 24  [fraction 0–1]          │
    │ PRESSURE_DRAWDOWN           │ DOWNHOLE_PRESS – WHP  [bar]                 │
    │ TUBING_HEAD_DIFF            │ DP_TUBING – WHP  [bar]                      │
    │ OIL_RATE_BOEPD              │ OIL_VOL × 6.2898  [barrels/day]             │
    │ GAS_RATE_MMSCFD             │ GAS_VOL × 0.03531 / 1000  [MMscf/day]       │
    │ CUMULATIVE_OIL              │ cumsum(OIL) per well  [Sm³]                 │
    │ CUMULATIVE_GAS              │ cumsum(GAS) per well  [Sm³]                 │
    │ CUMULATIVE_WATER            │ cumsum(WAT) per well  [Sm³]                 │
    │ DAYS_ON_PRODUCTION          │ Days since first non-zero oil per well       │
    │ ROLLING_7D_OIL              │ 7-day rolling mean OIL per well             │
    │ ROLLING_30D_OIL             │ 30-day rolling mean OIL per well            │
    │ ROLLING_7D_PRESSURE         │ 7-day rolling mean DOWNHOLE_PRESS per well  │
    │ OIL_DECLINE_RATE            │ (prev – current) / prev OIL (14d lag)       │
    │ YEAR / MONTH / DAY_OF_YEAR  │ Temporal features for ML                   │
    │ REVENUE_USD                 │ OIL_BOEPD × OIL_PRICE (reference: $70/bbl) │
    └─────────────────────────────┴──────────────────────────────────────────────┘
    """
    log.info("═══ Engineering Production Features ═══")
    df = df.copy().sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"])

    from src.config import SM3_TO_BARREL, SM3_TO_MCF, OIL_PRICE_USD, GAS_PRICE_USD

    # ── Petroleum Ratios ──────────────────────────────────────────────────────
    # Water Cut: fraction of produced liquid that is water
    df["WATER_CUT"] = np.where(
        (df["BORE_OIL_VOL"] + df["BORE_WAT_VOL"]) > 0,
        df["BORE_WAT_VOL"] / (df["BORE_OIL_VOL"] + df["BORE_WAT_VOL"]),
        np.nan,
    )

    # Gas-Oil Ratio: Sm³ gas per Sm³ oil
    df["GOR"] = np.where(
        df["BORE_OIL_VOL"] > 0,
        df["BORE_GAS_VOL"] / df["BORE_OIL_VOL"],
        np.nan,
    )

    # Liquid rate: combined oil + water
    df["LIQUID_RATE"] = df["BORE_OIL_VOL"] + df["BORE_WAT_VOL"]

    # Production efficiency: fraction of day the well was on-stream
    df["PRODUCTION_EFFICIENCY"] = (df["ON_STREAM_HRS"] / 24.0).clip(0.0, 1.0)

    # ── Pressure-Based Features ───────────────────────────────────────────────
    # Pressure drawdown drives reservoir inflow
    df["PRESSURE_DRAWDOWN"] = (df["AVG_DOWNHOLE_PRESSURE"] - df["AVG_WHP_P"]).clip(lower=0)
    df["TUBING_HEAD_DIFF"]  = (df["AVG_DP_TUBING"] - df["AVG_WHP_P"]).clip(lower=0)

    # ── Unit Conversions ──────────────────────────────────────────────────────
    df["OIL_RATE_BOEPD"]   = df["BORE_OIL_VOL"] * SM3_TO_BARREL
    df["GAS_RATE_MMSCFD"]  = df["BORE_GAS_VOL"] * SM3_TO_MCF / 1000.0

    # ── Revenue Estimate ──────────────────────────────────────────────────────
    df["REVENUE_USD"] = (
        df["OIL_RATE_BOEPD"] * OIL_PRICE_USD
        + df["GAS_RATE_MMSCFD"] * 1000 * GAS_PRICE_USD
    )

    # ── Cumulative Production (per well, production rows only) ────────────────
    prod_mask = df["FLOW_KIND"] == "production"
    for col, cum_col in [("BORE_OIL_VOL", "CUMULATIVE_OIL"),
                          ("BORE_GAS_VOL", "CUMULATIVE_GAS"),
                          ("BORE_WAT_VOL", "CUMULATIVE_WATER")]:
        df[cum_col] = 0.0
        df.loc[prod_mask, cum_col] = (
            df.loc[prod_mask].groupby("NPD_WELL_BORE_NAME")[col].cumsum()
        )

    # ── Time Features ─────────────────────────────────────────────────────────
    df["YEAR"]        = df["DATEPRD"].dt.year
    df["MONTH"]       = df["DATEPRD"].dt.month
    df["DAY_OF_YEAR"] = df["DATEPRD"].dt.dayofyear

    # Days on production per well (time index for DCA)
    df["DAYS_ON_PRODUCTION"] = 0.0
    for well, grp in df[prod_mask].groupby("NPD_WELL_BORE_NAME"):
        active = grp[grp["BORE_OIL_VOL"] > 0]["DATEPRD"]
        if len(active) == 0:
            continue
        first_prod = active.min()
        df.loc[grp.index, "DAYS_ON_PRODUCTION"] = (grp["DATEPRD"] - first_prod).dt.days

    # ── Rolling Features (per well, production only) ──────────────────────────
    roll_feats = {
        "ROLLING_7D_OIL":      ("BORE_OIL_VOL", 7),
        "ROLLING_30D_OIL":     ("BORE_OIL_VOL", 30),
        "ROLLING_7D_PRESSURE": ("AVG_DOWNHOLE_PRESSURE", 7),
        "ROLLING_7D_GOR":      ("GOR", 7),
    }

    for new_col, (src_col, window) in roll_feats.items():
        df[new_col] = (
            df.groupby("NPD_WELL_BORE_NAME")[src_col]
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # ── Decline Rate ──────────────────────────────────────────────────────────
    # 14-day lag comparison: positive = declining, negative = increasing
    df["OIL_14D_LAG"] = df.groupby("NPD_WELL_BORE_NAME")["BORE_OIL_VOL"].transform(
        lambda x: x.shift(14)
    )
    df["OIL_DECLINE_RATE"] = np.where(
        df["OIL_14D_LAG"] > 0,
        (df["OIL_14D_LAG"] - df["BORE_OIL_VOL"]) / df["OIL_14D_LAG"],
        np.nan,
    )
    df = df.drop(columns=["OIL_14D_LAG"])

    log.info(f"  Added {len(roll_feats) + 12} derived features")
    log.info("  Production feature engineering complete ✓")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_maintenance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML-ready features to predictive maintenance data.

    Key derived features:
    ┌──────────────────────────────┬───────────────────────────────────────────────┐
    │ Feature                      │ Description                                   │
    ├──────────────────────────────┼───────────────────────────────────────────────┤
    │ ROLLING_3D_metric7           │ 3-day rolling mean (top failure predictor)    │
    │ ROLLING_7D_metric7           │ 7-day rolling mean                            │
    │ ROLLING_3D_metric4           │ 3-day rolling mean (2nd strongest predictor)  │
    │ ROLLING_7D_metric2           │ 7-day rolling mean                            │
    │ METRIC7_SPIKE                │ Binary: metric7 > 95th percentile             │
    │ METRIC4_SPIKE                │ Binary: metric4 > 95th percentile             │
    │ METRIC2_SPIKE                │ Binary: metric2 > 95th percentile             │
    │ METRIC1_DELTA                │ Day-over-day change in metric1 (cumulative)   │
    │ ANOMALY_SCORE                │ Weighted composite of spike indicators        │
    │ DAYS_SINCE_LAST_FAILURE      │ Per-device days since previous failure        │
    │ DAY_OF_WEEK / MONTH          │ Temporal features                             │
    │ IS_WEEKEND                   │ Binary weekend flag                           │
    └──────────────────────────────┴───────────────────────────────────────────────┘
    """
    log.info("═══ Engineering Maintenance Features ═══")
    df = df.copy().sort_values(["device", "date"])

    metric_cols = [c for c in df.columns if c.startswith("metric")]

    # ── Rolling Statistics (per device) ──────────────────────────────────────
    rolling_specs = [
        ("metric7", 3), ("metric7", 7),
        ("metric4", 3), ("metric4", 7),
        ("metric2", 3), ("metric2", 7),
        ("metric9", 7),
    ]

    for col, window in rolling_specs:
        if col not in df.columns:
            continue
        feat_name = f"ROLLING_{window}D_{col}"
        df[feat_name] = df.groupby("device")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Also add rolling std for anomaly detection
        std_name = f"ROLLSTD_{window}D_{col}"
        df[std_name] = df.groupby("device")[col].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    # ── Spike Indicators (binary flags based on percentile thresholds) ────────
    spike_thresholds = {}
    for col in ["metric2", "metric4", "metric7", "metric9"]:
        if col not in df.columns:
            continue
        p95 = df[col].quantile(0.95)
        spike_thresholds[col] = p95
        df[f"{col.upper()}_SPIKE"] = (df[col] > p95).astype(int)

    log.info(f"  Spike thresholds (p95): {spike_thresholds}")

    # ── Metric1 Delta (cumulative counter → daily increment) ─────────────────
    df["METRIC1_DELTA"] = df.groupby("device")["metric1"].transform(
        lambda x: x.diff().fillna(0).clip(lower=0)
    )

    # ── Anomaly Score (weighted composite) ───────────────────────────────────
    # Weights based on failure-ratio analysis:
    # metric7: 115×, metric4: 32×, metric2: 26×
    total_weight = 115 + 32 + 26
    df["ANOMALY_SCORE"] = (
        df.get("METRIC7_SPIKE", 0) * 115 / total_weight
        + df.get("METRIC4_SPIKE", 0) * 32 / total_weight
        + df.get("METRIC2_SPIKE", 0) * 26 / total_weight
    )

    # ── Days Since Last Failure (per device) ──────────────────────────────────
    df["DAYS_SINCE_LAST_FAILURE"] = 999  # default: never failed
    for device, grp in df.groupby("device"):
        fail_dates = grp.loc[grp["failure"] == 1, "date"]
        if fail_dates.empty:
            continue
        for idx, row in grp.iterrows():
            past_fails = fail_dates[fail_dates < row["date"]]
            if past_fails.empty:
                continue
            days = (row["date"] - past_fails.max()).days
            df.at[idx, "DAYS_SINCE_LAST_FAILURE"] = days

    df["DAYS_SINCE_LAST_FAILURE"] = df["DAYS_SINCE_LAST_FAILURE"].clip(upper=999)

    # ── Temporal Features ─────────────────────────────────────────────────────
    df["DAY_OF_WEEK"] = df["date"].dt.dayofweek
    df["MONTH"]       = df["date"].dt.month
    df["IS_WEEKEND"]  = (df["DAY_OF_WEEK"] >= 5).astype(int)
    df["DAY_OF_YEAR"] = df["date"].dt.dayofyear

    # ── Health Score (inverse failure probability proxy, pre-ML) ─────────────
    # Simple rule-based score before ML model is run
    df["HEALTH_SCORE_RULE"] = (
        100
        - df["ANOMALY_SCORE"] * 50
        - df.get("METRIC7_SPIKE", 0) * 20
        - df.get("METRIC4_SPIKE", 0) * 15
    ).clip(0, 100)

    log.info(f"  Added rolling features, spike flags, anomaly score, health score")
    log.info("  Maintenance feature engineering complete ✓")
    return df
