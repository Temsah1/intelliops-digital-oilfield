"""Page 5 — Predictive Maintenance (Bug-Fixed)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import load_maintenance, load_metrics, section_header, base_layout, risk_badge
from src.config import THEME

def render():
    df_maint = load_maintenance()
    metrics  = load_metrics()
    pm = metrics.get("predictive_maintenance", {})
    best_model = pm.get("best_model", "—")
    best_m = pm.get("metrics", {}).get(best_model, {})

    section_header("Predictive Maintenance", "EQUIPMENT HEALTH")

    k1, k2, k3, k4 = st.columns(4)
    total_devices = df_maint["device"].nunique()
    failure_events = int(df_maint["failure"].sum())
    failure_rate = df_maint["failure"].mean() * 100
    avg_health = float(df_maint["HEALTH_SCORE_RULE"].mean()) if "HEALTH_SCORE_RULE" in df_maint.columns else 85.0
    with k1: st.metric("Total Devices", f"{total_devices:,}")
    with k2: st.metric("Failure Events", str(failure_events))
    with k3: st.metric("Failure Rate", f"{failure_rate:.4f}%")
    with k4: st.metric("Avg Health Score", f"{avg_health:.1f}/100")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Equipment Health", "📊 Sensor Analytics",
        "🤖 ML Results", "⚡ Failure Analysis"
    ])

    with tab1:
        if "ANOMALY_SCORE" in df_maint.columns:
            latest = df_maint.sort_values("date").groupby("device").last().reset_index()
            latest["risk_level"] = pd.cut(latest["ANOMALY_SCORE"],
                bins=[-np.inf,0.1,0.3,0.6,np.inf], labels=["Low","Medium","High","Critical"])

            col_dist, col_pie = st.columns([2, 1])
            with col_dist:
                # FIXED: No duplicate xaxis/yaxis kwargs
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(x=latest["HEALTH_SCORE_RULE"], nbinsx=40,
                    marker_color=THEME["primary"], opacity=0.8))
                fig_h.add_vline(x=latest["HEALTH_SCORE_RULE"].mean(),
                    line_dash="dash", line_color=THEME["warning"],
                    annotation_text=f"Mean: {latest['HEALTH_SCORE_RULE'].mean():.1f}")
                fig_h.update_layout(
                    **base_layout("Equipment Health Score Distribution"),
                    xaxis=dict(title="Health Score (0–100)", gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(title="Device Count", gridcolor="rgba(255,255,255,0.04)"),
                )
                st.plotly_chart(fig_h, use_container_width=True)

            with col_pie:
                risk_counts = latest["risk_level"].value_counts()
                risk_colors = {"Critical":THEME["danger"],"High":THEME["secondary"],
                               "Medium":THEME["warning"],"Low":THEME["success"]}
                labels = risk_counts.index.tolist()
                colors = [risk_colors.get(l, THEME["text_secondary"]) for l in labels]
                fig_r = go.Figure(go.Pie(labels=labels, values=risk_counts.values.tolist(),
                    hole=0.6, marker=dict(colors=colors), textinfo="label+percent"))
                fig_r.update_layout(**base_layout("Risk Distribution", height=340), showlegend=False)
                st.plotly_chart(fig_r, use_container_width=True)

            section_header("High-Risk Devices", "TOP 15")
            at_risk = latest.sort_values("ANOMALY_SCORE", ascending=False).head(15)
            at_risk = at_risk[["device","DEVICE_TYPE","HEALTH_SCORE_RULE","ANOMALY_SCORE","risk_level"]].copy()
            at_risk.columns = ["Device ID","Type","Health Score","Anomaly Score","Risk Level"]
            at_risk["Health Score"] = at_risk["Health Score"].round(1)
            at_risk["Anomaly Score"] = at_risk["Anomaly Score"].round(4)
            st.dataframe(at_risk, use_container_width=True, hide_index=True)

    with tab2:
        device_types = df_maint["DEVICE_TYPE"].unique().tolist() if "DEVICE_TYPE" in df_maint.columns else ["All"]
        sel_type = st.selectbox("Device Type", ["All Types"] + device_types)
        sel_metric = st.selectbox("Sensor Metric", [c for c in df_maint.columns if c.startswith("metric") and "_" not in c])

        view = df_maint.copy()
        if sel_type != "All Types":
            view = view[view["DEVICE_TYPE"] == sel_type]

        fail_vals = view[view["failure"]==1][sel_metric].dropna()
        ok_vals   = view[view["failure"]==0][sel_metric].dropna()

        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=ok_vals.sample(min(5000,len(ok_vals))),
                name="Normal", marker_color=THEME["success"], boxpoints=False))
            if len(fail_vals) > 0:
                fig_box.add_trace(go.Box(y=fail_vals, name="Failure",
                    marker_color=THEME["danger"], boxpoints="all", jitter=0.3, pointpos=-1.8))
            fig_box.update_layout(**base_layout(f"{sel_metric} — Normal vs Failure"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title=sel_metric, gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_box, use_container_width=True)

        with col_b:
            daily = view.groupby("date").agg(
                mean_val=(sel_metric,"mean"), failure_count=("failure","sum")).reset_index()
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=daily["date"], y=daily["mean_val"],
                name=f"Avg {sel_metric}", line=dict(color=THEME["primary"], width=2)))
            fail_days = daily[daily["failure_count"]>0]
            if len(fail_days) > 0:
                fig_ts.add_trace(go.Scatter(x=fail_days["date"], y=fail_days["mean_val"],
                    mode="markers", name="Failure Day",
                    marker=dict(color=THEME["danger"], size=10, symbol="x")))
            fig_ts.update_layout(**base_layout(f"Daily Avg {sel_metric}"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_ts, use_container_width=True)

        if "ANOMALY_SCORE" in df_maint.columns:
            an_daily = df_maint.groupby("date").agg(
                avg_anomaly=("ANOMALY_SCORE","mean"), failures=("failure","sum")).reset_index()
            fig_an = go.Figure()
            fig_an.add_trace(go.Scatter(x=an_daily["date"], y=an_daily["avg_anomaly"],
                fill="tozeroy", name="Avg Anomaly Score",
                line=dict(color=THEME["warning"], width=2),
                fillcolor="rgba(255,214,0,0.08)"))
            fig_an.update_layout(**base_layout("Daily Anomaly Score Trend"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Anomaly Score (0–1)", gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_an, use_container_width=True)

    with tab3:
        if pm:
            section_header(f"Best Model: {best_model}", "CLASSIFICATION")
            col_m, col_cm = st.columns([1, 1])
            with col_m:
                st.markdown(f"""
                <div class="info-box">
                  <strong>Model Performance</strong><br><br>
                  <table class="metric-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Accuracy</td><td><strong>{best_m.get('accuracy',0):.4f}</strong></td></tr>
                    <tr><td>Precision</td><td><strong>{best_m.get('precision',0):.4f}</strong></td></tr>
                    <tr><td>Recall</td><td><strong>{best_m.get('recall',0):.4f}</strong></td></tr>
                    <tr><td>F1 Score</td><td><strong>{best_m.get('f1',0):.4f}</strong></td></tr>
                    <tr><td>ROC-AUC</td><td><strong>{best_m.get('roc_auc',0):.4f}</strong></td></tr>
                  </table>
                </div>""", unsafe_allow_html=True)
                st.markdown("""<div class="warn-box"><strong>⚠️ Class Imbalance</strong><br>
                  Failure rate = 0.085%. SMOTE applied. High Recall is prioritized —
                  a missed failure is more costly than a false alarm.</div>""", unsafe_allow_html=True)

            with col_cm:
                cm = best_m.get("confusion_matrix", [[0,0],[0,0]])
                if cm:
                    labels = ["Normal (0)", "Failure (1)"]
                    fig_cm = go.Figure(go.Heatmap(z=cm, x=labels, y=labels,
                        colorscale=[[0,THEME["bg_card"]],[1,THEME["primary"]]],
                        text=[[str(v) for v in row] for row in cm],
                        texttemplate="%{text}", textfont=dict(size=18, color="white"),
                        showscale=False))
                    fig_cm.update_layout(**base_layout("Confusion Matrix", height=320),
                        xaxis=dict(title="Predicted"),
                        yaxis=dict(title="Actual"))
                    st.plotly_chart(fig_cm, use_container_width=True)

            all_m = pm.get("metrics", {})
            if all_m:
                section_header("All Models Comparison", "")
                comp_df = pd.DataFrame(all_m).T.drop(columns=["confusion_matrix"], errors="ignore")
                st.dataframe(comp_df.style.format("{:.4f}").highlight_max(
                    subset=[c for c in ["f1","roc_auc","recall"] if c in comp_df.columns],
                    color="#1A3A2A"), use_container_width=True)

            feat_imp = pm.get("feature_importance", {})
            if feat_imp:
                fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature","Importance"])
                fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)
                fig_fi = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"],
                    orientation="h", marker=dict(color=fi_df["Importance"],
                    colorscale=[[0, THEME["surface"]],[1, THEME["primary"]]])))
                fig_fi.update_layout(**base_layout("Feature Importance — Failure Predictor", height=380),
                    xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_fi, use_container_width=True)

    with tab4:
        failures = df_maint[df_maint["failure"]==1].copy()
        if len(failures) == 0:
            st.info("No failure events in dataset.")
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if "DEVICE_TYPE" in failures.columns:
                    by_type = failures.groupby("DEVICE_TYPE").size().reset_index(name="failures")
                    fig_t = go.Figure(go.Bar(x=by_type["DEVICE_TYPE"], y=by_type["failures"],
                        marker_color=THEME["danger"], opacity=0.85))
                    fig_t.update_layout(**base_layout("Failures by Device Type"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.04)"))
                    st.plotly_chart(fig_t, use_container_width=True)
            with col_f2:
                fail_daily = failures.groupby("date").size().reset_index(name="count")
                fig_fd = go.Figure()
                fig_fd.add_trace(go.Bar(x=fail_daily["date"], y=fail_daily["count"],
                    marker_color=THEME["danger"], opacity=0.8))
                fig_fd.update_layout(**base_layout("Failure Events Over Time"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(title="Failures/Day", gridcolor="rgba(255,255,255,0.04)"))
                st.plotly_chart(fig_fd, use_container_width=True)

            metric_cols = [c for c in df_maint.columns if c.startswith("metric") and "_" not in c]
            fail_means = df_maint[df_maint["failure"]==1][metric_cols].mean()
            norm_means = df_maint[df_maint["failure"]==0][metric_cols].mean()
            ratio = (fail_means / norm_means.replace(0, np.nan)).round(2)
            cmp_df = pd.DataFrame({"Normal Mean":norm_means.round(2),
                                   "Failure Mean":fail_means.round(2),
                                   "Failure/Normal Ratio":ratio})
            st.markdown("##### Sensor Readings at Failure vs Normal")
            st.dataframe(cmp_df.style.background_gradient(subset=["Failure/Normal Ratio"], cmap="RdYlGn_r"),
                use_container_width=True)
