"""
dashboard/utils.py — Shared helpers for all dashboard pages.
BUG FIX: base_layout no longer includes xaxis/yaxis to avoid duplicate kwarg TypeError.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    PROCESSED_PROD_CSV, PROCESSED_MAINT_CSV, MODEL_METRICS_PATH,
    THEME, PLOTLY_TEMPLATE, CHART_HEIGHT, CHART_MARGIN, DB_PATH
)

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_production() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PROD_CSV, parse_dates=["DATEPRD"])
    return df

@st.cache_data(show_spinner=False)
def load_maintenance() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_MAINT_CSV, parse_dates=["date"])
    return df

@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH) as f:
            return json.load(f)
    return {}

# ── KPI card HTML ─────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str = "", delta_dir: str = "flat",
             icon: str = "📊", color: str = None) -> str:
    c = color or THEME["primary"]
    delta_class = {"up": "up", "down": "down"}.get(delta_dir, "flat")
    arrow = {"up": "▲", "down": "▼"}.get(delta_dir, "●")
    delta_html = f'<div class="kpi-delta {delta_class}">{arrow} {delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{c}">{value}</div>
      {delta_html}
    </div>"""

def risk_badge(level: str) -> str:
    cls = {"Critical":"badge-critical","High":"badge-high",
           "Medium":"badge-medium","Low":"badge-low"}.get(level,"badge-low")
    return f'<span class="badge {cls}">{level}</span>'

def section_header(title: str, badge: str = ""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
      <h2>{title}</h2>{badge_html}
    </div>""", unsafe_allow_html=True)

# ── FIXED base_layout — NO xaxis/yaxis to prevent duplicate kwarg TypeError ───
def base_layout(title: str = "", height: int = CHART_HEIGHT) -> dict:
    """
    Returns a base Plotly layout dict WITHOUT xaxis/yaxis so callers can
    safely pass their own xaxis= and yaxis= without a TypeError.
    """
    return dict(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=CHART_MARGIN,
        title=dict(text=title, font=dict(size=14, color=THEME["text_secondary"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text_primary"]),
        # xaxis and yaxis intentionally removed — callers set these themselves
    )

def apply_axis_style(gridcolor: str = "rgba(255,255,255,0.05)") -> dict:
    """Returns a dict of axis style kwargs to merge when needed."""
    return dict(gridcolor=gridcolor, showgrid=True,
                linecolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")

def fmt_num(n, decimals=0, suffix="") -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if abs(n) >= 1_000_000_000:
        return f"{n/1e9:.1f}B{suffix}"
    if abs(n) >= 1_000_000:
        return f"{n/1e6:.1f}M{suffix}"
    if abs(n) >= 1_000:
        return f"{n/1e3:.1f}K{suffix}"
    return f"{n:.{decimals}f}{suffix}"
