"""
src/etl/loader.py
─────────────────
Raw data ingestion layer.
Reads the two source CSVs and returns typed DataFrames
without any transformation — that is the cleaner's job.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger
from src.config import RAW_PRODUCTION_CSV, RAW_MAINTENANCE_CSV

log = get_logger(__name__)


# ── Production Data ────────────────────────────────────────────────────────────
PRODUCTION_DTYPES = {
    "NPD_WELL_BORE_NAME":        "str",
    "ON_STREAM_HRS":             "float64",
    "AVG_DOWNHOLE_PRESSURE":     "float64",
    "AVG_DOWNHOLE_TEMPERATURE":  "float64",
    "AVG_DP_TUBING":             "float64",
    "AVG_ANNULUS_PRESS":         "float64",
    "AVG_CHOKE_SIZE_P":          "float64",
    "AVG_WHP_P":                 "float64",
    "AVG_WHT_P":                 "float64",
    "DP_CHOKE_SIZE":             "float64",
    "BORE_OIL_VOL":              "float64",
    "BORE_GAS_VOL":              "float64",
    "BORE_WAT_VOL":              "float64",
    "BORE_WI_VOL":               "float64",
    "FLOW_KIND":                 "str",
}

# ── Maintenance Data ───────────────────────────────────────────────────────────
MAINTENANCE_DTYPES = {
    "device":   "str",
    "failure":  "int8",
    "metric1":  "int64",
    "metric2":  "int64",
    "metric3":  "int64",
    "metric4":  "int64",
    "metric5":  "int64",
    "metric6":  "int64",
    "metric7":  "int64",
    "metric8":  "int64",
    "metric9":  "int64",
}


def load_production(path: Path = RAW_PRODUCTION_CSV) -> pd.DataFrame:
    """
    Load the Volve Well Production CSV.
    Returns a DataFrame with DATEPRD parsed as datetime.
    """
    log.info(f"Loading production data from {path.name}")
    df = pd.read_csv(
        path,
        dtype={k: v for k, v in PRODUCTION_DTYPES.items() if k != "DATEPRD"},
        parse_dates=False,
    )
    # Parse date with dayfirst because format is "dd-Mon-yy" / "dd-Mon-yyyy"
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"], dayfirst=True, errors="coerce")
    df = df.sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"]).reset_index(drop=True)
    log.info(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    log.info(f"  Date range: {df['DATEPRD'].min().date()} → {df['DATEPRD'].max().date()}")
    log.info(f"  Wells: {sorted(df['NPD_WELL_BORE_NAME'].unique().tolist())}")
    return df


def load_maintenance(path: Path = RAW_MAINTENANCE_CSV) -> pd.DataFrame:
    """
    Load the Predictive Maintenance CSV.
    Returns a DataFrame with date parsed as datetime.
    """
    log.info(f"Loading maintenance data from {path.name}")
    df = pd.read_csv(
        path,
        dtype={k: v for k, v in MAINTENANCE_DTYPES.items() if k != "date"},
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["device", "date"]).reset_index(drop=True)
    log.info(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    log.info(f"  Devices: {df['device'].nunique():,}")
    log.info(f"  Failure events: {df['failure'].sum()} ({df['failure'].mean()*100:.4f}%)")
    return df


def validate_schema(df: pd.DataFrame, expected_cols: list[str], name: str) -> bool:
    """Verify all expected columns are present and log missing ones."""
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        log.error(f"{name} — missing columns: {missing}")
        return False
    log.info(f"{name} schema OK ✓")
    return True
