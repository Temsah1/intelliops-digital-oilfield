"""
src/etl/cleaner.py
──────────────────
Data Quality & Cleaning Layer.

Handles:
  - Missing value imputation (per-well median strategy)
  - Outlier detection and capping (IQR method)
  - Physical constraint enforcement
  - Duplicate removal
  - Data quality report generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION DATA CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

def clean_production(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for well production data.

    Steps:
    1. Remove exact duplicates
    2. Cap ON_STREAM_HRS to physical maximum of 24h
    3. Clip negative production volumes to zero
    4. Impute missing pressure/temperature with per-well median
    5. Generate a DATA_QUALITY_FLAG column
    6. Log a quality report
    """
    log.info("═══ Cleaning Production Data ═══")
    original_len = len(df)
    report = {}

    # ── Step 1: Remove duplicates ─────────────────────────────────────────────
    df = df.drop_duplicates(subset=["DATEPRD", "NPD_WELL_BORE_NAME"]).reset_index(drop=True)
    n_dupes = original_len - len(df)
    report["duplicates_removed"] = n_dupes
    log.info(f"  Duplicates removed: {n_dupes}")

    # ── Step 2: Physical constraints ──────────────────────────────────────────
    # ON_STREAM_HRS cannot exceed 24h per day
    n_hrs_violated = (df["ON_STREAM_HRS"] > 24).sum()
    df["ON_STREAM_HRS"] = df["ON_STREAM_HRS"].clip(upper=24.0)
    report["on_stream_hrs_capped"] = int(n_hrs_violated)
    log.info(f"  ON_STREAM_HRS capped at 24h: {n_hrs_violated} records")

    # Volumes cannot be negative (except BORE_WI_VOL may have data issues)
    vol_cols = ["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]
    for col in vol_cols:
        n_neg = (df[col] < 0).sum()
        df[col] = df[col].clip(lower=0.0)
        if n_neg:
            log.warning(f"  {col}: {n_neg} negative values clipped to 0")

    # BORE_WI_VOL: one value is -458 Sm³ — physical impossibility
    df["BORE_WI_VOL"] = df["BORE_WI_VOL"].clip(lower=0.0)

    # ── Step 3: Missing value imputation ──────────────────────────────────────
    pressure_temp_cols = [
        "AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE",
        "AVG_DP_TUBING", "AVG_ANNULUS_PRESS",
        "AVG_CHOKE_SIZE_P", "AVG_WHP_P", "AVG_WHT_P", "DP_CHOKE_SIZE",
    ]

    imputation_counts = {}
    for col in pressure_temp_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            # Per-well median imputation — better than global because each
            # well operates at different pressure/temperature regimes
            df[col] = df.groupby("NPD_WELL_BORE_NAME")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Remaining nulls (wells with 100% null for that column) → 0
            df[col] = df[col].fillna(0.0)
            imputation_counts[col] = int(n_missing)

    report["imputed"] = imputation_counts
    log.info(f"  Imputed missing values in {len(imputation_counts)} columns")

    # ON_STREAM_HRS: forward-fill within well (a well open yesterday was
    # probably open today unless explicitly 0)
    df["ON_STREAM_HRS"] = df.groupby("NPD_WELL_BORE_NAME")["ON_STREAM_HRS"].transform(
        lambda x: x.ffill().fillna(0.0)
    )

    # ── Step 4: Outlier detection (IQR method on oil volume) ─────────────────
    for col in ["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]:
        prod_mask = df["FLOW_KIND"] == "production"
        sub = df.loc[prod_mask, col]
        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3.0 * iqr   # Use 3× IQR (less aggressive than 1.5× for petroleum)
        n_out = (sub > upper).sum()
        df.loc[prod_mask, col] = sub.clip(upper=upper)
        if n_out:
            log.info(f"  {col}: {n_out} outliers capped at {upper:.0f} Sm³")
        report[f"outliers_{col}"] = int(n_out)

    # ── Step 5: Quality flag ──────────────────────────────────────────────────
    # Flag rows where the well was shut-in (on_stream = 0, no production)
    df["IS_SHUT_IN"] = (
        (df["ON_STREAM_HRS"] == 0) |
        ((df["FLOW_KIND"] == "production") & (df["BORE_OIL_VOL"] == 0) &
         (df["BORE_GAS_VOL"] == 0))
    ).astype(int)

    # ── Final report ──────────────────────────────────────────────────────────
    report["final_rows"] = len(df)
    report["null_pct_remaining"] = df.isnull().mean().round(4).to_dict()
    _log_quality_report(report)

    log.info(f"  Cleaned production: {len(df):,} rows")
    return df


def _log_quality_report(report: dict) -> None:
    log.info("  ── Data Quality Report ──────────────────────────")
    log.info(f"     Duplicates removed  : {report.get('duplicates_removed', 0)}")
    log.info(f"     Hrs violations capped: {report.get('on_stream_hrs_capped', 0)}")
    log.info(f"     Imputed columns     : {list(report.get('imputed', {}).keys())}")
    remaining = {k: v for k, v in report.get("null_pct_remaining", {}).items() if v > 0}
    if remaining:
        log.info(f"     Remaining nulls (%)  : {remaining}")
    else:
        log.info("     Remaining nulls     : 0 ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE DATA CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

def clean_maintenance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for predictive maintenance data.

    Steps:
    1. Remove duplicates
    2. Drop metric8 (identical to metric7 — verified in EDA)
    3. Clip extreme outliers (IQR method per metric)
    4. Add device type from prefix
    5. Validate binary target
    """
    log.info("═══ Cleaning Maintenance Data ═══")
    original_len = len(df)

    # ── Step 1: Duplicates ────────────────────────────────────────────────────
    df = df.drop_duplicates(subset=["date", "device"]).reset_index(drop=True)
    log.info(f"  Duplicates removed: {original_len - len(df)}")

    # ── Step 2: Drop metric8 (duplicate of metric7) ───────────────────────────
    if "metric8" in df.columns:
        df = df.drop(columns=["metric8"])
        log.info("  Dropped metric8 (duplicate of metric7)")

    # ── Step 3: Clip extreme outliers (3 × IQR) ───────────────────────────────
    metric_cols = [c for c in df.columns if c.startswith("metric")]
    for col in metric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper = q3 + 5.0 * iqr   # Very generous — we don't want to mask failures
        n_clipped = (df[col] > upper).sum()
        df[col] = df[col].clip(upper=upper)
        if n_clipped:
            log.info(f"  {col}: {n_clipped} extreme values clipped")

    # ── Step 4: Device type ───────────────────────────────────────────────────
    from src.config import DEVICE_TYPE_MAP
    df["DEVICE_PREFIX"] = df["device"].str[:2]
    df["DEVICE_TYPE"] = df["DEVICE_PREFIX"].map(DEVICE_TYPE_MAP).fillna("Unknown")

    # ── Step 5: Validate binary target ───────────────────────────────────────
    invalid_labels = ~df["failure"].isin([0, 1])
    if invalid_labels.any():
        log.warning(f"  {invalid_labels.sum()} invalid failure labels — setting to 0")
        df.loc[invalid_labels, "failure"] = 0

    log.info(f"  Cleaned maintenance: {len(df):,} rows")
    log.info(f"  Failure rate: {df['failure'].mean()*100:.4f}%")
    return df
