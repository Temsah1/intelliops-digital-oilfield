"""Page 4 — Production Forecasting ML (Bug-Fixed)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import load_production, load_metrics, section_header, base_layout, fmt_num
from src.ml.production_forecaster import forecast_next_months
from src.config import THEME, SM3_TO_BARREL, PRODUCTION_MODEL_PATH

def render():
    df = load_production()
    metrics = load_metrics()
    pf = metrics.get("production_forecasting", {})

    section_header("Production Forecasting", "ML · 6-MONTH")

    all_models = pf.get("metrics", {})
    best_model = pf.get("best_model", "—")

    if all_models:
        cols = st.columns(len(all_models))
        for i, (name, m) in enumerate(all_models.items()):
            is_best = (name == best_model)
            with cols[i]:
                star = " ⭐ BEST" if is_best else ""
                border_style = f"border-color:{THEME['primary']};box-shadow:0 0 20px rgba(0,212,255,0.2)" if is_best else ""
                st.markdown(f"""
                <div class="kpi-card" style="{border_style}">
                  <div class="kpi-label">{name}{star}</div>
                  <div class="kpi-value" style="font-size:18px;color:{''+THEME['primary'] if is_best else THEME['text_secondary']}">
                    RMSE: {m['rmse']:.1f} Sm³/d
                  </div>
                  <div style="color:{THEME['text_secondary']};font-size:12px;margin-top:6px">
                    MAE={m['mae']:.1f} &nbsp;|&nbsp; R²={m['r2']:.3f}
                  </div>
                  <div style="color:{THEME['text_muted']};font-size:11px">
                    CV-RMSE: {m['cv_rmse_mean']:.1f} ± {m['cv_rmse_std']:.1f}
                  </div>
                </div>""", unsafe_allow_html=True)

    section_header("Actual vs Predicted (Test Set)", "VALIDATION")
    y_test = pf.get("y_test", [])
    y_pred = pf.get("y_pred", [])
    test_dates = pf.get("test_dates", [])

    if y_test and y_pred:
        col1, col2 = st.columns([2, 1])
        with col1:
            x_axis = test_dates[:len(y_test)] if test_dates else list(range(len(y_test)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=y_test, name="Actual",
                line=dict(color=THEME["primary"], width=2)))
            fig.add_trace(go.Scatter(x=x_axis, y=y_pred, name="Predicted",
                line=dict(color=THEME["secondary"], width=2, dash="dash")))
            fig.update_layout(**base_layout("Actual vs Predicted (Sm³/day)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Oil Rate (Sm³/d)", gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # FIXED: No duplicate xaxis/yaxis kwargs
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                marker=dict(color=THEME["primary"], size=4, opacity=0.4)))
            mn, mx = min(y_test), max(y_test)
            fig_sc.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                name="Perfect", line=dict(color=THEME["danger"], dash="dash", width=1.5)))
            fig_sc.update_layout(
                **base_layout("Predicted vs Actual"),
                showlegend=False,
                xaxis=dict(title="Actual (Sm³/d)", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Predicted (Sm³/d)", gridcolor="rgba(255,255,255,0.04)"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    section_header("6-Month Forecast", "FORWARD PROJECTION")
    if not PRODUCTION_MODEL_PATH.exists():
        st.warning("Model not found. Run ETL + Training first.")
        return

    with st.spinner("Generating 6-month forecast..."):
        forecast_df = forecast_next_months(df)

    if forecast_df.empty:
        st.info("Insufficient data to generate forecast (need ≥30 production days per well).")
        return

    wells = forecast_df["well_name"].unique().tolist()
    sel_w = st.selectbox("Well", ["All Wells"] + wells)

    if sel_w == "All Wells":
        view = forecast_df.groupby("forecast_date").agg(
            predicted_oil_sm3=("predicted_oil_sm3","sum"),
            lower_bound_sm3=("lower_bound_sm3","sum"),
            upper_bound_sm3=("upper_bound_sm3","sum"),
        ).reset_index()
        chart_title = "Field-Wide 6-Month Forecast"
    else:
        view = forecast_df[forecast_df["well_name"]==sel_w]
        chart_title = f"6-Month Forecast — {sel_w}"

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        x=list(view["forecast_date"]) + list(view["forecast_date"])[::-1],
        y=list(view["upper_bound_sm3"]) + list(view["lower_bound_sm3"])[::-1],
        fill="toself", fillcolor="rgba(0,212,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="±20% Band"))
    fig_f.add_trace(go.Scatter(x=view["forecast_date"], y=view["predicted_oil_sm3"],
        name="Forecast Sm³/d", mode="lines+markers",
        line=dict(color=THEME["primary"], width=3),
        marker=dict(size=8, color=THEME["primary"])))
    fig_f.add_trace(go.Scatter(x=view["forecast_date"],
        y=view["predicted_oil_sm3"]*SM3_TO_BARREL,
        name="Forecast bbl/d", mode="lines",
        line=dict(color=THEME["secondary"], width=2, dash="dot"), yaxis="y2"))
    fig_f.update_layout(
        **base_layout(chart_title, height=460),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Oil Rate (Sm³/d)", gridcolor="rgba(255,255,255,0.04)"),
        yaxis2=dict(title="Oil Rate (bbl/d)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_f, use_container_width=True)

    # Table
    show_df = view[["forecast_date","predicted_oil_sm3","predicted_oil_boepd",
                    "lower_bound_sm3","upper_bound_sm3"]].copy()
    show_df.columns = ["Date","Pred (Sm³/d)","Pred (boepd)","Low (Sm³/d)","High (Sm³/d)"]
    st.dataframe(show_df.style.format({c:"{:.1f}" for c in show_df.columns[1:]}), use_container_width=True)

    feat_imp = pf.get("feature_importance", {})
    if feat_imp:
        section_header("Feature Importance", "EXPLAINABILITY")
        fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature","Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(10)
        fig_fi = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker=dict(color=fi_df["Importance"],
            colorscale=[[0, THEME["surface"]], [1, THEME["primary"]]])))
        fig_fi.update_layout(**base_layout("Top Feature Importance", height=340),
            xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_fi, use_container_width=True)
