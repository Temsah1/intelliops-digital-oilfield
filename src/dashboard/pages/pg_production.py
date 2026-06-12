"""Page 2 — Production Analytics (Bug-Fixed + Premium UI)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import load_production, section_header, base_layout, fmt_num, apply_axis_style
from src.config import THEME, SM3_TO_BARREL

def render():
    df = load_production()
    prod = df[df["FLOW_KIND"] == "production"].copy()

    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        wells = ["All Wells"] + sorted(prod["NPD_WELL_BORE_NAME"].unique().tolist())
        sel_well = st.selectbox("Select Well", wells)
        years = sorted(prod["DATEPRD"].dt.year.unique().tolist())
        sel_years = st.select_slider("Year Range", options=years, value=(years[0], years[-1]))

    view = prod.copy()
    if sel_well != "All Wells":
        view = view[view["NPD_WELL_BORE_NAME"] == sel_well]
    view = view[view["DATEPRD"].dt.year.between(sel_years[0], sel_years[1])]

    section_header("Production Analytics", f"{sel_well} · {sel_years[0]}–{sel_years[1]}")

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Oil", fmt_num(view["BORE_OIL_VOL"].sum() * SM3_TO_BARREL, suffix=" bbl"))
    with k2: st.metric("Total Gas", fmt_num(view["BORE_GAS_VOL"].sum(), suffix=" Sm³"))
    with k3: st.metric("Total Water", fmt_num(view["BORE_WAT_VOL"].sum(), suffix=" Sm³"))
    with k4:
        eff = view["PRODUCTION_EFFICIENCY"].mean() * 100 if "PRODUCTION_EFFICIENCY" in view.columns else 0
        st.metric("Avg Efficiency", f"{eff:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Production Trends", "💧 Water Cut & GOR", "🌡️ Pressure", "📊 Well Comparison"])

    with tab1:
        agg = view.groupby(view["DATEPRD"].dt.to_period("M")).agg(
            oil=("BORE_OIL_VOL","sum"), gas=("BORE_GAS_VOL","sum"), water=("BORE_WAT_VOL","sum")
        ).reset_index()
        agg["date"] = agg["DATEPRD"].dt.to_timestamp()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["date"], y=agg["oil"]*SM3_TO_BARREL, name="Oil (bbl)",
            marker_color=THEME["primary"], opacity=0.85))
        fig.add_trace(go.Bar(x=agg["date"], y=agg["water"]*SM3_TO_BARREL, name="Water (bbl)",
            marker_color=THEME["warning"], opacity=0.7))
        ma = agg["oil"].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=agg["date"], y=ma*SM3_TO_BARREL,
            name="3M Moving Avg", line=dict(color=THEME["danger"], width=2.5)))
        fig.update_layout(**base_layout("Monthly Oil & Water Production"),
            barmode="stack", legend=dict(orientation="h", y=1.05),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Volume (bbl)", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig, use_container_width=True)

        if "CUMULATIVE_OIL" in view.columns:
            fig2 = go.Figure()
            for well, grp in prod[prod["DATEPRD"].dt.year.between(sel_years[0], sel_years[1])].groupby("NPD_WELL_BORE_NAME"):
                grp = grp.sort_values("DATEPRD")
                fig2.add_trace(go.Scatter(x=grp["DATEPRD"], y=grp["CUMULATIVE_OIL"]*SM3_TO_BARREL,
                    name=well.replace("15/9-",""), mode="lines"))
            fig2.update_layout(**base_layout("Cumulative Oil Production per Well (bbl)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Cumulative Oil (bbl)", gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if "WATER_CUT" in view.columns:
            wc = view.dropna(subset=["WATER_CUT"]).copy()
            wc_m = wc.groupby(wc["DATEPRD"].dt.to_period("M"))["WATER_CUT"].mean().reset_index()
            wc_m["date"] = wc_m["DATEPRD"].dt.to_timestamp()

            fig_wc = go.Figure()
            fig_wc.add_trace(go.Scatter(x=wc_m["date"], y=wc_m["WATER_CUT"]*100,
                fill="tozeroy", name="Water Cut (%)",
                line=dict(color=THEME["warning"], width=2),
                fillcolor="rgba(255,214,0,0.10)"))
            fig_wc.update_layout(**base_layout("Monthly Average Water Cut (%)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Water Cut (%)", range=[0,100],
                           gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_wc, use_container_width=True)

            if "GOR" in view.columns:
                gor = view[view["GOR"].between(0, 2000)].dropna(subset=["GOR"])
                gor_m = gor.groupby(gor["DATEPRD"].dt.to_period("M"))["GOR"].mean().reset_index()
                gor_m["date"] = gor_m["DATEPRD"].dt.to_timestamp()
                fig_gor = go.Figure()
                fig_gor.add_trace(go.Scatter(x=gor_m["date"], y=gor_m["GOR"],
                    name="GOR", line=dict(color=THEME["info"], width=2)))
                fig_gor.update_layout(**base_layout("Gas-Oil Ratio (GOR)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(title="GOR (Sm³/Sm³)", gridcolor="rgba(255,255,255,0.04)"))
                st.plotly_chart(fig_gor, use_container_width=True)
        else:
            st.info("Water Cut data not available for this selection.")

    with tab3:
        p_cols = ["AVG_DOWNHOLE_PRESSURE","AVG_WHP_P","AVG_DP_TUBING"]
        p_labels = {"AVG_DOWNHOLE_PRESSURE":"Downhole (bar)","AVG_WHP_P":"Wellhead (bar)","AVG_DP_TUBING":"Tubing ΔP (bar)"}
        fig_p = go.Figure()
        for col, color in zip(p_cols, [THEME["primary"], THEME["success"], THEME["warning"]]):
            if col in view.columns:
                pd_ = view[view[col]>0].groupby(view["DATEPRD"].dt.to_period("M"))[col].mean().reset_index()
                pd_["date"] = pd_["DATEPRD"].dt.to_timestamp()
                fig_p.add_trace(go.Scatter(x=pd_["date"], y=pd_[col],
                    name=p_labels[col], line=dict(color=color, width=2)))
        fig_p.update_layout(**base_layout("Pressure Trends"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Pressure (bar)", gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_p, use_container_width=True)

        if "PRESSURE_DRAWDOWN" in view.columns:
            dd = view[view["PRESSURE_DRAWDOWN"]>0].groupby(
                view["DATEPRD"].dt.to_period("M"))["PRESSURE_DRAWDOWN"].mean().reset_index()
            dd["date"] = dd["DATEPRD"].dt.to_timestamp()
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd["date"], y=dd["PRESSURE_DRAWDOWN"],
                fill="tozeroy", name="Drawdown (bar)",
                line=dict(color=THEME["secondary"], width=2),
                fillcolor="rgba(255,107,53,0.10)"))
            fig_dd.update_layout(**base_layout("Pressure Drawdown (Downhole − Wellhead)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Drawdown (bar)", gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_dd, use_container_width=True)

    with tab4:
        well_stats = prod.groupby("NPD_WELL_BORE_NAME").agg(
            total_oil=("BORE_OIL_VOL","sum"),
            total_gas=("BORE_GAS_VOL","sum"),
            total_water=("BORE_WAT_VOL","sum"),
        ).reset_index()
        well_stats["oil_bbl"] = well_stats["total_oil"] * SM3_TO_BARREL
        well_stats["well_short"] = well_stats["NPD_WELL_BORE_NAME"].str.replace("15/9-","")

        fig_bar = px.bar(well_stats.sort_values("oil_bbl", ascending=True),
            x="oil_bbl", y="well_short", orientation="h",
            color="oil_bbl", color_continuous_scale=["#131D2E", THEME["primary"]],
            labels={"oil_bbl":"Total Oil (bbl)","well_short":"Well"})
        fig_bar.update_layout(**base_layout("Total Oil Production by Well (bbl)"),
            coloraxis_showscale=False,
            xaxis=dict(title="Total Oil (bbl)", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_bar, use_container_width=True)

        display = well_stats[["NPD_WELL_BORE_NAME","oil_bbl","total_gas","total_water"]].copy()
        display.columns = ["Well","Total Oil (bbl)","Total Gas (Sm³)","Total Water (Sm³)"]
        st.dataframe(display.set_index("Well").style.format("{:,.0f}"), use_container_width=True)
