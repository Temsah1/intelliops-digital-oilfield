"""Page 3 — Decline Curve Analysis (Bug-Fixed)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import load_production, section_header, base_layout
from src.analytics.decline_curve import fit_decline_curves
from src.config import THEME, SM3_TO_BARREL

def render():
    df = load_production()
    prod = df[df["FLOW_KIND"] == "production"].copy()
    wells = sorted(prod[prod["BORE_OIL_VOL"]>0]["NPD_WELL_BORE_NAME"].unique().tolist())

    with st.sidebar:
        st.markdown("### 🎛️ DCA Settings")
        sel_well = st.selectbox("Select Well", wells)
        forecast_days = st.slider("Forecast Horizon (days)", 60, 365, 180, 30)
        show_raw = st.checkbox("Show Raw Data Points", value=False)

    section_header("Decline Curve Analysis", "ARPS · DCA")
    st.markdown("""
    <div class="info-box">
      <strong>Arps Decline Curve Analysis (1945)</strong> — Industry-standard forecasting method.<br>
      • <strong>Exponential:</strong> q(t) = qᵢ·exp(−D·t) — constant % decline, most conservative<br>
      • <strong>Hyperbolic:</strong> q(t) = qᵢ / (1 + b·D·t)^(1/b) — variable decline (0 &lt; b ≤ 2)<br>
      • <strong>Harmonic:</strong> q(t) = qᵢ / (1 + D·t) — optimistic, b=1 special case
    </div>""", unsafe_allow_html=True)

    with st.spinner(f"Fitting decline curves for {sel_well}..."):
        result = fit_decline_curves(prod, sel_well, forecast_days=forecast_days)

    if "error" in result:
        st.warning(f"⚠️ {result['error']}")
        return

    hist = result["historical"]
    t_hist = np.array(hist["t"])
    q_hist = np.array(hist["q"])
    q_smooth = np.array(hist["q_smooth"])
    dates_hist = hist["dates"]
    available = [m for m in ["exponential","hyperbolic","harmonic"]
                 if m in result and "error" not in result[m]]

    # Model stat cards
    cols = st.columns(len(available))
    colors_map = {"exponential":THEME["primary"],"hyperbolic":THEME["secondary"],"harmonic":THEME["success"]}
    for i, mn in enumerate(available):
        m = result[mn]
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{mn.upper()} DECLINE</div>
              <div class="kpi-value" style="color:{colors_map[mn]}">R² = {m.get('r2',0):.4f}</div>
              <div style="color:{THEME['text_secondary']};font-size:12px;margin-top:8px">
                qᵢ = {m.get('qi',0):.0f} Sm³/d &nbsp;|&nbsp; D = {m.get('D_annual_pct',0):.1f}%/yr
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Main DCA chart
    fig = go.Figure()
    if show_raw:
        fig.add_trace(go.Scatter(x=dates_hist, y=q_hist, name="Raw", mode="markers",
            marker=dict(color="rgba(255,255,255,0.15)", size=3)))
    fig.add_trace(go.Scatter(x=dates_hist, y=q_smooth, name="Smoothed (14d MA)",
        line=dict(color=THEME["text_secondary"], width=1.5, dash="dot")))

    dash_map = {"exponential":"solid","hyperbolic":"dash","harmonic":"dashdot"}
    for mn in available:
        m = result[mn]
        fig.add_trace(go.Scatter(x=dates_hist, y=m["fit_q"],
            name=f"{mn.capitalize()} Fit (R²={m['r2']:.3f})",
            line=dict(color=colors_map[mn], width=2, dash=dash_map[mn])))
        fig.add_trace(go.Scatter(x=m["fcst_dates"], y=m["fcst_q"],
            name=f"{mn.capitalize()} Forecast",
            line=dict(color=colors_map[mn], width=2, dash=dash_map[mn]), opacity=0.55))
    fig.add_vline(x=dates_hist[-1], line_dash="dash",
        line_color="rgba(255,255,255,0.25)", annotation_text="→ Forecast")

    fig.update_layout(**base_layout(f"DCA — {sel_well}", height=500),
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Oil Rate (Sm³/day)", gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    # Equations
    section_header("Fitted Equations", "PARAMETERS")
    eq_cols = st.columns(len(available))
    for i, mn in enumerate(available):
        m = result[mn]
        with eq_cols[i]:
            b_line = f"\nb   = {m.get('b',0):.3f}" if 'b' in m else ""
            eur_line = f"\nEUR = {m.get('eur_sm3',0)*SM3_TO_BARREL:,.0f} bbl" if m.get('eur_sm3') else ""
            st.markdown(f"""<div class="equation-box">{mn.upper()}
{m.get('equation','N/A')}

qᵢ  = {m.get('qi',0):.1f} Sm³/day
D   = {m.get('D_annual_pct',0):.2f} %/year
R²  = {m.get('r2',0):.4f}{b_line}{eur_line}</div>""", unsafe_allow_html=True)

    # Forecast table
    if available:
        section_header("Forecast Table", "MONTHLY RATES")
        m0 = result[available[0]]
        rows = []
        step = max(1, len(m0["fcst_dates"]) // 6)
        for idx in range(0, min(len(m0["fcst_dates"]), 180), step):
            d = m0["fcst_dates"][idx]
            row = {"Date": d}
            for mn in available:
                mv = result[mn]
                if idx < len(mv["fcst_q"]):
                    row[mn.capitalize()] = f"{mv['fcst_q'][idx]:.1f} Sm³/d"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Date"), use_container_width=True)
