"""
src/etl/pipeline.py
────────────────────
ETL Pipeline Orchestrator.

Runs the full Extract → Transform → Load sequence:
  1. Load raw CSVs
  2. Validate schemas
  3. Clean data
  4. Engineer features
  5. Save processed CSVs
  6. Load into SQLite

Run directly:  python -m src.etl.pipeline
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.logger import get_logger
from src.config import (
    DATA_PROCESSED_DIR, PROCESSED_PROD_CSV, PROCESSED_MAINT_CSV,
    MODELS_DIR,
)
from src.etl.loader import load_production, load_maintenance, validate_schema
from src.etl.cleaner import clean_production, clean_maintenance
from src.etl.feature_engineer import (
    engineer_production_features, engineer_maintenance_features
)
from src.database.db_manager import DatabaseManager

log = get_logger(__name__)

PRODUCTION_REQUIRED_COLS = [
    "DATEPRD", "NPD_WELL_BORE_NAME", "FLOW_KIND",
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
]
MAINTENANCE_REQUIRED_COLS = [
    "date", "device", "failure",
    "metric1", "metric2", "metric3", "metric4",
    "metric5", "metric6", "metric7", "metric9",
]


def run_etl(force_reload: bool = True) -> dict:
    """
    Run the complete ETL pipeline.

    Parameters
    ----------
    force_reload : bool
        If True, drop and recreate the SQLite database.
        Set to False if you just want to reload ML predictions.

    Returns
    -------
    dict
        Summary statistics from the pipeline run.
    """
    start = time.time()
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   AI Digital Oilfield — ETL Pipeline Starting        ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    # ── EXTRACT ───────────────────────────────────────────────────────────────
    log.info("PHASE 1: EXTRACT")
    df_prod_raw  = load_production()
    df_maint_raw = load_maintenance()

    validate_schema(df_prod_raw,  PRODUCTION_REQUIRED_COLS,  "Production")
    validate_schema(df_maint_raw, MAINTENANCE_REQUIRED_COLS, "Maintenance")

    summary["raw_production_rows"]   = len(df_prod_raw)
    summary["raw_maintenance_rows"]  = len(df_maint_raw)

    # ── TRANSFORM ─────────────────────────────────────────────────────────────
    log.info("PHASE 2: TRANSFORM")

    # Production
    df_prod_clean = clean_production(df_prod_raw)
    df_prod_feat  = engineer_production_features(df_prod_clean)

    # Maintenance
    df_maint_clean = clean_maintenance(df_maint_raw)
    df_maint_feat  = engineer_maintenance_features(df_maint_clean)

    summary["clean_production_rows"]  = len(df_prod_feat)
    summary["clean_maintenance_rows"] = len(df_maint_feat)

    # ── SAVE PROCESSED CSVs ───────────────────────────────────────────────────
    log.info("PHASE 3: SAVE PROCESSED DATA")
    df_prod_feat.to_csv(PROCESSED_PROD_CSV, index=False)
    df_maint_feat.to_csv(PROCESSED_MAINT_CSV, index=False)
    log.info(f"  Saved {PROCESSED_PROD_CSV.name} ✓")
    log.info(f"  Saved {PROCESSED_MAINT_CSV.name} ✓")

    # ── LOAD (SQLite) ─────────────────────────────────────────────────────────
    log.info("PHASE 4: LOAD INTO SQLITE")
    db = DatabaseManager()

    if force_reload:
        db.drop_and_recreate()

    db.seed_wells()
    db.seed_equipment(df_maint_feat)
    db.insert_production(df_prod_feat)
    db.insert_maintenance(df_maint_feat)

    counts = db.get_row_counts()
    summary["db_table_counts"] = counts
    log.info(f"  Database row counts: {counts}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    summary["elapsed_seconds"] = round(elapsed, 2)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info(f"║   ETL Pipeline Complete — {elapsed:.1f}s                    ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    # Persist summary
    summary_path = MODELS_DIR / "etl_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    run_etl()
