"""
src/database/db_manager.py
───────────────────────────
SQLite Database Layer.

Creates the oilfield.db schema and provides read/write helpers.
Schema design reflects oilfield industry data models.

Tables:
  wells        — static well metadata
  production   — daily production measurements (Volve field)
  equipment    — device registry from maintenance dataset
  maintenance  — daily sensor readings + ML features
  predictions  — stored ML predictions and forecasts
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger
from src.config import DB_PATH, WELL_METADATA, DEVICE_TYPE_MAP

log = get_logger(__name__)


# ── DDL Statements ────────────────────────────────────────────────────────────

DDL_WELLS = """
CREATE TABLE IF NOT EXISTS wells (
    well_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    well_name        TEXT    NOT NULL UNIQUE,
    field_name       TEXT    NOT NULL DEFAULT 'Volve',
    well_type        TEXT    NOT NULL CHECK(well_type IN ('producer','injector')),
    depth_m          REAL,
    start_year       INTEGER,
    country          TEXT    DEFAULT 'Norway',
    basin            TEXT    DEFAULT 'North Sea',
    block            TEXT    DEFAULT '15/9',
    created_at       TEXT    DEFAULT (datetime('now'))
);
"""

DDL_PRODUCTION = """
CREATE TABLE IF NOT EXISTS production (
    prod_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    well_name                TEXT    NOT NULL,
    date                     TEXT    NOT NULL,
    flow_kind                TEXT    NOT NULL CHECK(flow_kind IN ('production','injection')),
    on_stream_hrs            REAL,
    avg_downhole_pressure    REAL,
    avg_downhole_temp        REAL,
    avg_dp_tubing            REAL,
    avg_annulus_press        REAL,
    avg_choke_size_p         REAL,
    avg_whp_p                REAL,
    avg_wht_p                REAL,
    dp_choke_size            REAL,
    bore_oil_vol             REAL,
    bore_gas_vol             REAL,
    bore_wat_vol             REAL,
    bore_wi_vol              REAL,
    water_cut                REAL,
    gor                      REAL,
    liquid_rate              REAL,
    production_efficiency    REAL,
    pressure_drawdown        REAL,
    oil_rate_boepd           REAL,
    gas_rate_mmscfd          REAL,
    revenue_usd              REAL,
    cumulative_oil           REAL,
    cumulative_gas           REAL,
    cumulative_water         REAL,
    days_on_production       REAL,
    rolling_7d_oil           REAL,
    rolling_30d_oil          REAL,
    rolling_7d_pressure      REAL,
    oil_decline_rate         REAL,
    is_shut_in               INTEGER DEFAULT 0,
    year                     INTEGER,
    month                    INTEGER,
    day_of_year              INTEGER,
    UNIQUE(well_name, date, flow_kind)
);
"""

DDL_EQUIPMENT = """
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id       TEXT    NOT NULL UNIQUE,
    device_prefix   TEXT,
    device_type     TEXT,
    total_readings  INTEGER DEFAULT 0,
    total_failures  INTEGER DEFAULT 0,
    failure_rate    REAL    DEFAULT 0.0,
    first_seen      TEXT,
    last_seen       TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);
"""

DDL_MAINTENANCE = """
CREATE TABLE IF NOT EXISTS maintenance (
    maint_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id                TEXT    NOT NULL,
    date                     TEXT    NOT NULL,
    failure                  INTEGER NOT NULL DEFAULT 0,
    metric1                  INTEGER,
    metric2                  INTEGER,
    metric3                  INTEGER,
    metric4                  INTEGER,
    metric5                  INTEGER,
    metric6                  INTEGER,
    metric7                  INTEGER,
    metric9                  INTEGER,
    device_prefix            TEXT,
    device_type              TEXT,
    rolling_3d_metric7       REAL,
    rolling_7d_metric7       REAL,
    rolling_3d_metric4       REAL,
    rolling_7d_metric4       REAL,
    rolling_3d_metric2       REAL,
    rolling_7d_metric2       REAL,
    metric7_spike            INTEGER,
    metric4_spike            INTEGER,
    metric2_spike            INTEGER,
    metric1_delta            REAL,
    anomaly_score            REAL,
    days_since_last_failure  INTEGER,
    health_score_rule        REAL,
    day_of_week              INTEGER,
    month                    INTEGER,
    is_weekend               INTEGER,
    UNIQUE(device_id, date)
);
"""

DDL_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    pred_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pred_type        TEXT    NOT NULL,   -- 'production_forecast' | 'failure_risk'
    entity_id        TEXT    NOT NULL,   -- well_name or device_id
    pred_date        TEXT    NOT NULL,
    predicted_value  REAL,
    actual_value     REAL,
    confidence_low   REAL,
    confidence_high  REAL,
    model_name       TEXT,
    created_at       TEXT    DEFAULT (datetime('now'))
);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Database Manager Class
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    Thread-safe SQLite manager for the Digital Oilfield Platform.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"DatabaseManager initialised → {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")     # Better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ── Schema ────────────────────────────────────────────────────────────────
    def create_schema(self) -> None:
        """Create all tables if they do not exist."""
        log.info("Creating database schema...")
        with self._connect() as conn:
            for ddl in [DDL_WELLS, DDL_PRODUCTION, DDL_EQUIPMENT,
                        DDL_MAINTENANCE, DDL_PREDICTIONS]:
                conn.execute(ddl)
            conn.commit()
        log.info("Schema created ✓")

    def drop_and_recreate(self) -> None:
        """Wipe all tables and recreate — use during ETL re-runs."""
        log.warning("Dropping all tables and recreating schema...")
        tables = ["predictions", "maintenance", "equipment", "production", "wells"]
        with self._connect() as conn:
            for t in tables:
                conn.execute(f"DROP TABLE IF EXISTS {t}")
            conn.commit()
        self.create_schema()

    # ── Seed Static Data ──────────────────────────────────────────────────────
    def seed_wells(self) -> None:
        """Insert well metadata from config."""
        rows = [
            (name, meta["field"], meta["type"], meta["depth_m"], meta["start_year"])
            for name, meta in WELL_METADATA.items()
        ]
        sql = """
            INSERT OR IGNORE INTO wells
                (well_name, field_name, well_type, depth_m, start_year)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()
        log.info(f"Seeded {len(rows)} wells into wells table ✓")

    def seed_equipment(self, df_maintenance: pd.DataFrame) -> None:
        """Build equipment registry from maintenance device list."""
        eq = (
            df_maintenance.groupby("device")
            .agg(
                device_prefix=("DEVICE_PREFIX", "first"),
                device_type=("DEVICE_TYPE", "first"),
                total_readings=("failure", "count"),
                total_failures=("failure", "sum"),
                first_seen=("date", "min"),
                last_seen=("date", "max"),
            )
            .reset_index()
        )
        eq["failure_rate"] = eq["total_failures"] / eq["total_readings"]
        eq["first_seen"] = eq["first_seen"].astype(str)
        eq["last_seen"]  = eq["last_seen"].astype(str)

        sql = """
            INSERT OR IGNORE INTO equipment
                (device_id, device_prefix, device_type, total_readings,
                 total_failures, failure_rate, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = list(eq.itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()
        log.info(f"Seeded {len(rows)} devices into equipment table ✓")

    # ── Bulk Inserts ──────────────────────────────────────────────────────────
    def insert_production(self, df: pd.DataFrame, chunksize: int = 5000) -> int:
        """Bulk-insert cleaned production DataFrame into SQLite."""
        col_map = {
            "NPD_WELL_BORE_NAME": "well_name",
            "DATEPRD": "date",
            "FLOW_KIND": "flow_kind",
            "ON_STREAM_HRS": "on_stream_hrs",
            "AVG_DOWNHOLE_PRESSURE": "avg_downhole_pressure",
            "AVG_DOWNHOLE_TEMPERATURE": "avg_downhole_temp",
            "AVG_DP_TUBING": "avg_dp_tubing",
            "AVG_ANNULUS_PRESS": "avg_annulus_press",
            "AVG_CHOKE_SIZE_P": "avg_choke_size_p",
            "AVG_WHP_P": "avg_whp_p",
            "AVG_WHT_P": "avg_wht_p",
            "DP_CHOKE_SIZE": "dp_choke_size",
            "BORE_OIL_VOL": "bore_oil_vol",
            "BORE_GAS_VOL": "bore_gas_vol",
            "BORE_WAT_VOL": "bore_wat_vol",
            "BORE_WI_VOL": "bore_wi_vol",
            "WATER_CUT": "water_cut",
            "GOR": "gor",
            "LIQUID_RATE": "liquid_rate",
            "PRODUCTION_EFFICIENCY": "production_efficiency",
            "PRESSURE_DRAWDOWN": "pressure_drawdown",
            "OIL_RATE_BOEPD": "oil_rate_boepd",
            "GAS_RATE_MMSCFD": "gas_rate_mmscfd",
            "REVENUE_USD": "revenue_usd",
            "CUMULATIVE_OIL": "cumulative_oil",
            "CUMULATIVE_GAS": "cumulative_gas",
            "CUMULATIVE_WATER": "cumulative_water",
            "DAYS_ON_PRODUCTION": "days_on_production",
            "ROLLING_7D_OIL": "rolling_7d_oil",
            "ROLLING_30D_OIL": "rolling_30d_oil",
            "ROLLING_7D_PRESSURE": "rolling_7d_pressure",
            "OIL_DECLINE_RATE": "oil_decline_rate",
            "IS_SHUT_IN": "is_shut_in",
            "YEAR": "year",
            "MONTH": "month",
            "DAY_OF_YEAR": "day_of_year",
        }
        subset = {k: v for k, v in col_map.items() if k in df.columns}
        insert_df = df[list(subset.keys())].rename(columns=subset)
        insert_df["date"] = insert_df["date"].astype(str)

        with self._connect() as conn:
            insert_df.to_sql("production", conn, if_exists="append",
                             index=False, method="multi", chunksize=chunksize)
        log.info(f"Inserted {len(insert_df):,} rows into production table ✓")
        return len(insert_df)

    def insert_maintenance(self, df: pd.DataFrame, chunksize: int = 5000) -> int:
        """Bulk-insert cleaned maintenance DataFrame into SQLite."""
        col_map = {
            "device": "device_id",
            "date": "date",
            "failure": "failure",
            "metric1": "metric1", "metric2": "metric2", "metric3": "metric3",
            "metric4": "metric4", "metric5": "metric5", "metric6": "metric6",
            "metric7": "metric7", "metric9": "metric9",
            "DEVICE_PREFIX": "device_prefix",
            "DEVICE_TYPE": "device_type",
            "ROLLING_3D_metric7": "rolling_3d_metric7",
            "ROLLING_7D_metric7": "rolling_7d_metric7",
            "ROLLING_3D_metric4": "rolling_3d_metric4",
            "ROLLING_7D_metric4": "rolling_7d_metric4",
            "ROLLING_3D_metric2": "rolling_3d_metric2",
            "ROLLING_7D_metric2": "rolling_7d_metric2",
            "METRIC7_SPIKE": "metric7_spike",
            "METRIC4_SPIKE": "metric4_spike",
            "METRIC2_SPIKE": "metric2_spike",
            "METRIC1_DELTA": "metric1_delta",
            "ANOMALY_SCORE": "anomaly_score",
            "DAYS_SINCE_LAST_FAILURE": "days_since_last_failure",
            "HEALTH_SCORE_RULE": "health_score_rule",
            "DAY_OF_WEEK": "day_of_week",
            "MONTH": "month",
            "IS_WEEKEND": "is_weekend",
        }
        subset = {k: v for k, v in col_map.items() if k in df.columns}
        insert_df = df[list(subset.keys())].rename(columns=subset)
        insert_df["date"] = insert_df["date"].astype(str)

        with self._connect() as conn:
            insert_df.to_sql("maintenance", conn, if_exists="append",
                             index=False, method="multi", chunksize=chunksize)
        log.info(f"Inserted {len(insert_df):,} rows into maintenance table ✓")
        return len(insert_df)

    # ── Query Helpers ─────────────────────────────────────────────────────────
    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a SELECT and return a DataFrame."""
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_table_names(self) -> list[str]:
        sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        with self._connect() as conn:
            return [r[0] for r in conn.execute(sql).fetchall()]

    def get_table_info(self, table: str) -> pd.DataFrame:
        return self.query(f"PRAGMA table_info({table})")

    def get_row_counts(self) -> dict:
        counts = {}
        for t in self.get_table_names():
            df = self.query(f"SELECT COUNT(*) AS n FROM {t}")
            counts[t] = int(df["n"].iloc[0])
        return counts
