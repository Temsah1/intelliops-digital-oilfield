"""Page 1 — Executive Home Dashboard"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import (
    load_production, load_maintenance, load_metrics,
    kpi_card, section_header, base_layout, fmt_num
)
from src.config import THEME, SM3_TO_BARREL

def render():
    st.markdown(
        '<p class="brand-subtitle" style="margin-bottom:4px">VOLVE FIELD · NORTH SEA · BLOCK 15/9</p>',
        unsafe_allow_html=True)

    df_prod  = load_production()
    df_maint = load_maintenance()
    metrics  = load_metrics()
    prod = df_prod[df_prod["FLOW_KIND"] == "production"].copy()

    # ── Computed KPIs ──────────────────────────────────────────────────────────
    total_oil_bbl  = prod["BORE_OIL_VOL"].sum() * SM3_TO_BARREL
    total_gas_sm3  = prod["BORE_GAS_VOL"].sum()
    active_wells   = prod[prod["BORE_OIL_VOL"] > 0]["NPD_WELL_BORE_NAME"].nunique()
    latest_rate    = prod.sort_values("DATEPRD").groupby("NPD_WELL_BORE_NAME")["BORE_OIL_VOL"].last().sum() * SM3_TO_BARREL
    total_devices  = df_maint["device"].nunique()
    failure_events = int(df_maint["failure"].sum())
    avg_health     = float(df_maint["HEALTH_SCORE_RULE"].mean()) if "HEALTH_SCORE_RULE" in df_maint.columns else 85.0
    date_span      = f"{prod['DATEPRD'].min().year} – {prod['DATEPRD'].max().year}"

    # Alert banner
    if "ANOMALY_SCORE" in df_maint.columns:
        critical_count = int((df_maint["ANOMALY_SCORE"] > 0.6).sum())
        if critical_count > 0:
            st.markdown(f"""
            <div class="alert-critical">
              <span style="font-size:22px">⚠️</span>
              <div>
                <strong style="color:#FF5370;font-size:13px">ACTIVE ALERT</strong><br>
                <span style="color:#E8EAED;font-size:13px">
                  {critical_count:,} equipment readings with elevated anomaly scores detected.
                  Review Predictive Maintenance page immediately.
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

    # KPI Row 1
    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, delta, d, icon, color) in zip([c1,c2,c3,c4], [
        ("CUMULATIVE OIL", fmt_num(total_oil_bbl, suffix=" bbl"), "Volve lifetime", "flat", "🛢️", THEME["primary"]),
        ("TOTAL GAS", fmt_num(total_gas_sm3, suffix=" Sm³"), "associated gas", "flat", "🔥", THEME["secondary"]),
        ("ACTIVE WELLS", str(active_wells), "of 5 producers", "up", "🏭", THEME["success"]),
        ("LATEST DAILY RATE", fmt_num(latest_rate, suffix=" boepd"), "field total", "flat", "📈", THEME["warning"]),
    ]):
        with col:
            st.markdown(kpi_card(lbl, val, delta, d, icon, color), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # KPI Row 2
    c5, c6, c7, c8 = st.columns(4)
    for col, (lbl, val, delta, d, icon, color) in zip([c5,c6,c7,c8], [
        ("MONITORED DEVICES", fmt_num(total_devices), "IoT sensors", "flat", "🔧", THEME["info"]),
        ("FAILURE EVENTS", str(failure_events), f"{failure_events/len(df_maint)*100:.3f}% rate", "down", "⚡", THEME["danger"]),
        ("AVG EQUIPMENT HEALTH", f"{avg_health:.1f}%", "rule-based score", "up", "❤️", THEME["success"]),
        ("FIELD LIFESPAN", date_span, "Volve open dataset", "flat", "📅", THEME["text_secondary"]),
    ]):
        with col:
            st.markdown(kpi_card(lbl, val, delta, d, icon, color), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Main charts ────────────────────────────────────────────────────────────
    section_header("Field Production Overview", "VOLVE · 2007–2016")
    col_chart, col_pie = st.columns([3, 1])

    with col_chart:
        monthly = (prod.groupby(prod["DATEPRD"].dt.to_period("M"))
            .agg(oil=("BORE_OIL_VOL","sum"), water=("BORE_WAT_VOL","sum"))
            .reset_index())
        monthly["date"] = monthly["DATEPRD"].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["date"], y=monthly["oil"] * SM3_TO_BARREL,
            name="Oil (bbl)", fill="tozeroy",
            line=dict(color=THEME["primary"], width=2),
            fillcolor="rgba(0,212,255,0.12)"))
        fig.add_trace(go.Scatter(
            x=monthly["date"], y=monthly["water"] * SM3_TO_BARREL,
            name="Water (bbl)", line=dict(color=THEME["warning"], width=1.5, dash="dot")))
        fig.update_layout(
            **base_layout("Monthly Oil & Water Production"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Volume (bbl)", gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with col_pie:
        well_oil = prod.groupby("NPD_WELL_BORE_NAME")["BORE_OIL_VOL"].sum().reset_index()
        well_oil = well_oil[well_oil["BORE_OIL_VOL"] > 0]
        colors = [THEME["primary"], THEME["secondary"], THEME["success"], THEME["warning"], THEME["info"]]
        fig2 = go.Figure(go.Pie(
            labels=well_oil["NPD_WELL_BORE_NAME"].str.replace("15/9-",""),
            values=well_oil["BORE_OIL_VOL"], hole=0.6,
            marker=dict(colors=colors[:len(well_oil)])))
        fig2.update_layout(
            **base_layout("Oil Share by Well", height=340),
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=10)))
        fig2.update_traces(textinfo="percent", textfont_size=10)
        st.plotly_chart(fig2, use_container_width=True)

    # ── ML summary ─────────────────────────────────────────────────────────────
    section_header("AI Model Performance", "ML SUMMARY")
    m1, m2, m3 = st.columns(3)

    pm = metrics.get("predictive_maintenance", {})
    pf = metrics.get("production_forecasting", {})

    with m1:
        best_pm = pm.get("best_model","—")
        bm = pm.get("metrics",{}).get(best_pm,{})
        st.markdown(f"""<div class="info-box">
          <strong>🔧 Predictive Maintenance</strong><br>
          Best: <strong>{best_pm}</strong><br>
          AUC-ROC: <strong>{bm.get('roc_auc',0):.3f}</strong> &nbsp;|&nbsp;
          F1: <strong>{bm.get('f1',0):.3f}</strong><br>
          Recall: <strong>{bm.get('recall',0):.3f}</strong>
        </div>""", unsafe_allow_html=True)

    with m2:
        best_pf = pf.get("best_model","—")
        bpf = pf.get("metrics",{}).get(best_pf,{})
        st.markdown(f"""<div class="info-box">
          <strong>📈 Production Forecasting</strong><br>
          Best: <strong>{best_pf}</strong><br>
          RMSE: <strong>{bpf.get('rmse',0):.1f} Sm³/d</strong> &nbsp;|&nbsp;
          R²: <strong>{bpf.get('r2',0):.3f}</strong><br>
          MAE: <strong>{bpf.get('mae',0):.1f} Sm³/d</strong>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="info-box">
          <strong>🗄️ Database</strong><br>
          Production records: <strong>15,634</strong><br>
          Maintenance records: <strong>124,493</strong><br>
          Monitored devices: <strong>1,169</strong>
        </div>""", unsafe_allow_html=True)

    # ── Anomaly timeline ───────────────────────────────────────────────────────
    if "ANOMALY_SCORE" in df_maint.columns:
        section_header("Fleet-Wide Anomaly Score", "LIVE MONITORING")
        an = df_maint.groupby("date").agg(
            avg=("ANOMALY_SCORE","mean"), p95=("ANOMALY_SCORE","quantile", ),
            failures=("failure","sum")).reset_index()
        # Re-compute p95 without lambda (lambda not compatible with groupby agg)
        p95 = df_maint.groupby("date")["ANOMALY_SCORE"].quantile(0.95).reset_index()
        p95.columns = ["date","p95"]
        an = an.merge(p95, on="date")

        fig_an = go.Figure()
        fig_an.add_trace(go.Scatter(x=an["date"], y=an["p95"],
            name="95th Pct", line=dict(color=THEME["danger"], width=1.5, dash="dash"),
            fill=None))
        fig_an.add_trace(go.Scatter(x=an["date"], y=an["avg"],
            name="Fleet Avg Anomaly Score", fill="tozeroy",
            line=dict(color=THEME["warning"], width=2),
            fillcolor="rgba(255,214,0,0.08)"))
        fail_days = an[an["failures"] > 0]
        if len(fail_days) > 0:
            fig_an.add_trace(go.Scatter(x=fail_days["date"], y=fail_days["avg"],
                mode="markers", name="Failure Day",
                marker=dict(color=THEME["danger"], size=9, symbol="x-thin",
                            line=dict(color=THEME["danger"], width=2))))
        fig_an.add_hline(y=0.3, line_dash="dash",
            line_color="rgba(255,214,0,0.4)", annotation_text="Alert Threshold")
        fig_an.update_layout(
            **base_layout("Fleet-Wide Anomaly Score + Failure Events", height=300),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Anomaly Score (0–1)", gridcolor="rgba(255,255,255,0.04)",
                       range=[0, 1]),
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_an, use_container_width=True)
