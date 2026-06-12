"""
app.py — IntelliOps AI Digital Oilfield Platform
Premium Streamlit Dashboard Entry Point
"""
import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="IntelliOps — AI Digital Oilfield",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load premium CSS
css_path = ROOT / "src" / "dashboard" / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Auto-bootstrap on first run ────────────────────────────────────────────────
from src.config import PROCESSED_PROD_CSV, PRODUCTION_MODEL_PATH

if not PROCESSED_PROD_CSV.exists():
    with st.spinner("⚙️ Running ETL pipeline on first launch..."):
        from src.etl.pipeline import run_etl
        run_etl()

if not PRODUCTION_MODEL_PATH.exists():
    with st.spinner("🤖 Training ML models (first launch only — ~2 min)..."):
        from src.ml.train_all import train_all
        train_all()

# ── Navigation ─────────────────────────────────────────────────────────────────
PAGES = {
    "🏠  Executive Home":          "pg_home",
    "📈  Production Analytics":    "pg_production",
    "📉  Decline Curve Analysis":  "pg_decline",
    "🤖  Production Forecasting":  "pg_forecast",
    "🔧  Predictive Maintenance":  "pg_maintenance",
    "📤  User Data Upload":        "pg_upload",
    "🗄️  Database Explorer":       "pg_database",
    "📋  Engineering Reports":     "pg_reports",
}

with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="padding:20px 8px 8px;text-align:center">
      <div style="font-size:36px;line-height:1">🛢️</div>
      <div class="brand-title" style="margin-top:8px">IntelliOps</div>
      <div class="brand-subtitle">AI DIGITAL OILFIELD</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    sel_page = st.radio("", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Field status indicator
    try:
        from src.dashboard.utils import load_production
        df_tmp = load_production()
        n_wells = df_tmp["NPD_WELL_BORE_NAME"].nunique()
        last_date = df_tmp["DATEPRD"].max().date()
        st.markdown(f"""
        <div style="background:rgba(0,230,118,0.06);border:1px solid rgba(0,230,118,0.2);
             border-radius:10px;padding:12px 14px;margin:8px 0">
          <div style="font-size:10px;color:#4A5568;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
            FIELD STATUS
          </div>
          <div style="color:#00E676;font-weight:700;font-size:13px">● ONLINE</div>
          <div style="color:#8A95A3;font-size:11px;margin-top:4px">
            {n_wells} Wells · Last: {last_date}
          </div>
        </div>""", unsafe_allow_html=True)
    except: pass

    st.markdown("""
    <div style="font-size:10px;color:#4A5568;text-align:center;padding:12px 4px">
      Equinor Volve Open Dataset<br>Norwegian North Sea · Block 15/9<br>
      <span style="color:#00D4FF">IntelliOps v2.0</span>
    </div>""", unsafe_allow_html=True)

# ── Render selected page ───────────────────────────────────────────────────────
import importlib
module_name = PAGES[sel_page]
try:
    page_mod = importlib.import_module(f"src.dashboard.pages.{module_name}")
    page_mod.render()
except Exception as e:
    st.error(f"Page error: {e}")
    st.exception(e)
    st.info("If this is a first run, try refreshing. The models may still be loading.")
