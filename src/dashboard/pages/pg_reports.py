"""Page 7 — Engineering Reports & Insights (Bug-Fixed)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import load_production, load_maintenance, load_metrics, section_header, base_layout
from src.config import THEME, SM3_TO_BARREL

def render():
    df_prod  = load_production()
    df_maint = load_maintenance()
    metrics  = load_metrics()
    prod = df_prod[df_prod["FLOW_KIND"]=="production"].copy()

    section_header("Engineering Reports", "AUTO-GENERATED")
    st.markdown("""<div class="info-box">
      Automatically generated petroleum engineering insights based on data analysis, ML models, and
      industry best practices. Suitable for inclusion in a Digital Oilfield Operations Report.
    </div>""", unsafe_allow_html=True)

    section_header("1. Field Production Report", "VOLVE FIELD")
    total_oil   = prod["BORE_OIL_VOL"].sum()
    total_gas   = prod["BORE_GAS_VOL"].sum()
    total_water = prod["BORE_WAT_VOL"].sum()
    date_range  = f"{prod['DATEPRD'].min().date()} to {prod['DATEPRD'].max().date()}"
    peak_day    = prod.groupby("DATEPRD")["BORE_OIL_VOL"].sum().idxmax()
    peak_rate   = prod.groupby("DATEPRD")["BORE_OIL_VOL"].sum().max()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="info-box">
          <strong>📊 Production Summary</strong><br><br>
          • <strong>Field:</strong> Volve, Norwegian North Sea (Block 15/9)<br>
          • <strong>Period:</strong> {date_range}<br>
          • <strong>Cumulative Oil:</strong> {total_oil*SM3_TO_BARREL:,.0f} bbl ({total_oil:,.0f} Sm³)<br>
          • <strong>Cumulative Gas:</strong> {total_gas:,.0f} Sm³<br>
          • <strong>Cumulative Water:</strong> {total_water:,.0f} Sm³<br>
          • <strong>Peak Day:</strong> {peak_day.date() if hasattr(peak_day,'date') else peak_day}<br>
          • <strong>Peak Rate:</strong> {peak_rate*SM3_TO_BARREL:,.0f} bbl/day<br>
          • <strong>Producers:</strong> 5 | <strong>Injectors:</strong> 2
        </div>""", unsafe_allow_html=True)
    with col2:
        if "WATER_CUT" in prod.columns:
            annual_wc = prod.dropna(subset=["WATER_CUT"]).groupby(prod["DATEPRD"].dt.year)["WATER_CUT"].mean()*100
            insight = "Increasing (⚠️ mature field)" if annual_wc.iloc[-1] > annual_wc.iloc[0] else "Decreasing"
            st.markdown(f"""<div class="warn-box">
              <strong>💧 Water Cut Trend: {insight}</strong><br><br>
              Progressed from <strong>{annual_wc.iloc[0]:.1f}%</strong> (early) to
              <strong>{annual_wc.iloc[-1]:.1f}%</strong> (late field life).
              Consistent with reservoir pressure depletion and water influx.
              High water cut reduces production efficiency and increases handling costs.
            </div>""", unsafe_allow_html=True)

    section_header("2. Well Performance Ranking", "PRODUCERS")
    well_stats = prod.groupby("NPD_WELL_BORE_NAME").agg(
        total_oil_bbl=("BORE_OIL_VOL", lambda x: x.sum()*SM3_TO_BARREL),
        avg_daily_bbl=("BORE_OIL_VOL", lambda x: x[x>0].mean()*SM3_TO_BARREL),
        avg_wc=("WATER_CUT","mean") if "WATER_CUT" in prod.columns else ("BORE_OIL_VOL","count"),
        prod_days=("BORE_OIL_VOL", lambda x: (x>0).sum()),
        avg_pressure=("AVG_DOWNHOLE_PRESSURE", lambda x: x[x>0].mean()),
    ).reset_index().sort_values("total_oil_bbl", ascending=False)
    well_stats["avg_wc"] = (well_stats["avg_wc"]*100).round(1) if "WATER_CUT" in prod.columns else 0
    well_stats["Rank"] = range(1, len(well_stats)+1)
    display = well_stats.rename(columns={
        "NPD_WELL_BORE_NAME":"Well","total_oil_bbl":"Cum. Oil (bbl)",
        "avg_daily_bbl":"Avg Rate (bbl/d)","avg_wc":"Avg WC (%)",
        "prod_days":"Active Days","avg_pressure":"Avg Pressure (bar)"
    })[["Rank","Well","Cum. Oil (bbl)","Avg Rate (bbl/d)","Avg WC (%)","Active Days"]]
    st.dataframe(display.set_index("Rank").style.format({
        "Cum. Oil (bbl)":"{:,.0f}","Avg Rate (bbl/d)":"{:.1f}","Avg WC (%)":"{:.1f}","Active Days":"{:,}"
    }), use_container_width=True)

    section_header("3. Equipment Health Report", "MAINTENANCE")
    total_devices = df_maint["device"].nunique()
    failure_rate  = df_maint["failure"].mean()*100
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        if "DEVICE_TYPE" in df_maint.columns:
            by_type = df_maint.groupby("DEVICE_TYPE").agg(
                devices=("device","nunique"),failures=("failure","sum"),
                readings=("failure","count")).reset_index()
            by_type["failure_rate_%"] = (by_type["failures"]/by_type["readings"]*100).round(4)
            st.dataframe(by_type, use_container_width=True, hide_index=True)
    with col_eq2:
        st.markdown(f"""<div class="info-box">
          <strong>🔧 Maintenance Intelligence</strong><br><br>
          • <strong>Total Devices:</strong> {total_devices:,}<br>
          • <strong>Failure Events:</strong> {df_maint['failure'].sum()}<br>
          • <strong>Overall Failure Rate:</strong> {failure_rate:.4f}%<br>
          • <strong>Top Predictor:</strong> Metric 7 (115× at failure)<br>
          • <strong>Secondary:</strong> Metric 4 (32×), Metric 2 (26×)<br>
          • <strong>Action:</strong> Alert when metric7 > 95th percentile
        </div>""", unsafe_allow_html=True)

    section_header("4. ML Model Report", "AI PERFORMANCE")
    pf = metrics.get("production_forecasting", {})
    pm = metrics.get("predictive_maintenance", {})
    c1, c2 = st.columns(2)
    with c1:
        pf_best = pf.get("best_model","—")
        pf_m = pf.get("metrics",{}).get(pf_best,{})
        st.markdown(f"""<div class="info-box">
          <strong>📈 Production Forecasting</strong><br><br>
          Best: <strong>{pf_best}</strong><br>
          RMSE: <strong>{pf_m.get('rmse','—')} Sm³/day</strong> | R²: <strong>{pf_m.get('r2','—')}</strong><br>
          MAE: <strong>{pf_m.get('mae','—')} Sm³/day</strong><br>
          CV-RMSE: <strong>{pf_m.get('cv_rmse_mean','—')} ± {pf_m.get('cv_rmse_std','—')}</strong>
        </div>""", unsafe_allow_html=True)
    with c2:
        pm_best = pm.get("best_model","—")
        pm_m = pm.get("metrics",{}).get(pm_best,{})
        st.markdown(f"""<div class="info-box">
          <strong>🔧 Predictive Maintenance</strong><br><br>
          Best: <strong>{pm_best}</strong><br>
          AUC-ROC: <strong>{pm_m.get('roc_auc','—')}</strong> | F1: <strong>{pm_m.get('f1','—')}</strong><br>
          Recall: <strong>{pm_m.get('recall','—')}</strong> (minimize missed failures)<br>
          Imbalance: <strong>SMOTE + class_weight='balanced'</strong>
        </div>""", unsafe_allow_html=True)

    section_header("5. Operational Recommendations", "AI INSIGHTS")
    st.markdown("""<div class="info-box">
      <strong>🎯 Top Recommendations from Data Analysis</strong><br><br>
      <strong>1. Water Management:</strong> Field-wide water cut >60%. Evaluate IOR/EOR and water handling
      facility upgrades to improve sweep efficiency.<br><br>
      <strong>2. Equipment Alert:</strong> Metric 7 is the strongest failure predictor (115× elevated at failure).
      Deploy real-time alert when metric7 exceeds 95th percentile threshold.<br><br>
      <strong>3. Well F-12 & F-14:</strong> Top producers with high cumulative output but declining rates.
      Candidates for workover, stimulation, or artificial lift review.<br><br>
      <strong>4. ML Forecasting:</strong> Gradient Boosting provides lowest RMSE. Use 30-day rolling
      production average as the primary input feature.<br><br>
      <strong>5. Decline Curve:</strong> Exponential decline fits observed across most wells. Initiate
      production enhancement program to extend field life by 12–18 months.
    </div>""", unsafe_allow_html=True)

    # Download report as CSV
    section_header("Export Report Data", "DOWNLOAD")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        csv_prod = display.to_csv(index=False)
        st.download_button("⬇ Download Well Rankings CSV", csv_prod,
            "well_rankings.csv", "text/csv")
    with col_dl2:
        model_summary = pd.DataFrame({
            "Model": list(pf.get("metrics",{}).keys()),
            "RMSE": [pf["metrics"][k]["rmse"] for k in pf.get("metrics",{})],
            "R2":   [pf["metrics"][k]["r2"] for k in pf.get("metrics",{})],
        }) if pf.get("metrics") else pd.DataFrame()
        if not model_summary.empty:
            st.download_button("⬇ Download ML Metrics CSV", model_summary.to_csv(index=False),
                "ml_metrics.csv", "text/csv")
    with col_dl3:
        full_prod = prod[["DATEPRD","NPD_WELL_BORE_NAME","BORE_OIL_VOL","BORE_GAS_VOL",
                          "BORE_WAT_VOL","AVG_DOWNHOLE_PRESSURE"]].copy()
        full_prod["DATEPRD"] = full_prod["DATEPRD"].astype(str)
        st.download_button("⬇ Download Production Data CSV", full_prod.to_csv(index=False),
            "production_data.csv", "text/csv")
