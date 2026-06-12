"""
Page — User Data Upload & Analysis
Supports Excel/CSV files up to multi-GB via chunked streaming.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, os, time, gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import section_header, base_layout, fmt_num, kpi_card
from src.config import THEME, SM3_TO_BARREL, DATA_PROCESSED_DIR

# Where we store user-uploaded processed files
USER_DATA_DIR = DATA_PROCESSED_DIR / "user_uploads"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50_000   # rows per chunk for streaming

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_excel_chunked(uploaded_file, sheet=0, max_rows: int = None) -> pd.DataFrame:
    """Stream-read large Excel files in chunks using openpyxl."""
    import openpyxl
    st.info("📂 Reading large Excel file in chunks — this may take a moment for files >100MB...")
    file_bytes = uploaded_file.read()
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c) if c is not None else f"Col_{i}" for i, c in enumerate(next(rows_iter))]
    chunks = []
    chunk_rows = []
    n_total = 0
    progress = st.progress(0, text="Loading data...")
    for i, row in enumerate(rows_iter):
        chunk_rows.append(row)
        if len(chunk_rows) >= CHUNK_SIZE:
            chunks.append(pd.DataFrame(chunk_rows, columns=header))
            chunk_rows = []
            n_total += CHUNK_SIZE
            progress.progress(min(n_total / (ws.max_row or 1), 0.99),
                text=f"Loaded {fmt_num(n_total)} rows...")
            if max_rows and n_total >= max_rows:
                break
        gc.collect()
    if chunk_rows:
        chunks.append(pd.DataFrame(chunk_rows, columns=header))
    progress.progress(1.0, text="✅ Done!")
    wb.close()
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    return df

def read_csv_chunked(uploaded_file, max_rows: int = None) -> pd.DataFrame:
    """Stream-read large CSV files."""
    st.info("📂 Reading large CSV in chunks...")
    progress = st.progress(0, text="Loading...")
    chunks = []
    file_bytes = uploaded_file.read()
    total = len(file_bytes)
    buf = io.BytesIO(file_bytes)
    n_read = 0
    for chunk in pd.read_csv(buf, chunksize=CHUNK_SIZE, low_memory=False):
        chunks.append(chunk)
        n_read += len(chunk)
        pct = min(n_read / max(total / 100, 1), 0.99)  # rough estimate
        progress.progress(pct, text=f"Loaded {fmt_num(n_read)} rows...")
        if max_rows and n_read >= max_rows:
            break
        gc.collect()
    progress.progress(1.0, text="✅ Done!")
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    return df

def auto_detect_columns(df: pd.DataFrame) -> dict:
    """Try to auto-map common petroleum/maintenance column names."""
    mapping = {}
    col_lower = {c.lower().replace(" ","_"): c for c in df.columns}
    candidates = {
        "date":     ["date","dateprd","timestamp","time","period","datetime"],
        "oil":      ["oil","bore_oil_vol","oil_vol","oil_rate","production","bore_oil","oil_sm3"],
        "gas":      ["gas","bore_gas_vol","gas_vol","gas_rate","bore_gas","gas_sm3"],
        "water":    ["water","bore_wat_vol","water_vol","bore_water","water_sm3"],
        "pressure": ["pressure","avg_downhole_pressure","downhole_pressure","reservoir_pressure"],
        "well":     ["well","npd_well_bore_name","well_name","wellname","well_id"],
        "device":   ["device","device_id","equipment","asset","tag"],
        "failure":  ["failure","fault","alarm","breakdown","failed"],
    }
    for key, cands in candidates.items():
        for c in cands:
            if c in col_lower:
                mapping[key] = col_lower[c]
                break
    return mapping

def smart_analyze(df: pd.DataFrame, col_map: dict) -> dict:
    """Run smart analysis based on detected column types."""
    results = {}
    # Parse date
    if "date" in col_map:
        try:
            df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
            results["date_range"] = (df[col_map["date"]].min(), df[col_map["date"]].max())
        except: pass

    # Numeric summaries
    for key in ["oil","gas","water","pressure"]:
        if key in col_map:
            col = col_map[key]
            df[col] = pd.to_numeric(df[col], errors="coerce")
            results[key] = {
                "sum": df[col].sum(),
                "mean": df[col].mean(),
                "max": df[col].max(),
                "min": df[col][df[col]>0].min() if (df[col]>0).any() else 0,
                "std": df[col].std(),
                "missing_pct": df[col].isna().mean()*100,
            }
    # Failure analysis
    if "failure" in col_map:
        col = col_map["failure"]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        results["failure"] = {
            "total": int(df[col].sum()),
            "rate": df[col].mean()*100,
        }
    return results


def render():
    section_header("User Data Upload & Analysis", "BRING YOUR OWN DATA")

    st.markdown("""
    <div class="info-box">
      <strong>📤 Upload your own production or maintenance data</strong> — Excel (.xlsx, .xls) or CSV.
      Files up to <strong>1GB+</strong> are supported via chunked streaming.<br>
      The platform auto-detects petroleum columns and runs instant analytics.
    </div>""", unsafe_allow_html=True)

    # ── Upload ─────────────────────────────────────────────────────────────────
    tab_upload, tab_schema, tab_analyze, tab_ml = st.tabs([
        "📁 Upload File", "🗂️ Schema & Preview",
        "📊 Auto Analytics", "🤖 ML on Your Data"
    ])

    with tab_upload:
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded = st.file_uploader(
                "Drop Excel or CSV file here",
                type=["xlsx","xls","csv"],
                help="Supports files >1GB via chunked streaming. Excel .xlsx recommended.",
                label_visibility="collapsed"
            )
            if uploaded is None:
                st.markdown("""
                <div class="upload-zone">
                  <div style="font-size:48px;margin-bottom:16px">📊</div>
                  <div style="color:{p};font-size:16px;font-weight:700;margin-bottom:8px">
                    Drag & Drop Your Data File
                  </div>
                  <div style="color:{s};font-size:13px">
                    Excel (.xlsx, .xls) or CSV · Up to 1GB+ supported
                  </div>
                </div>""".format(p=THEME["primary"], s=THEME["text_secondary"]),
                unsafe_allow_html=True)

        with col_u2:
            st.markdown(f"""
            <div class="info-box">
              <strong>Supported Formats</strong><br><br>
              📗 Excel 2007+ (.xlsx)<br>
              📗 Excel 97 (.xls)<br>
              📄 CSV (UTF-8)<br><br>
              <strong>Auto-detected columns:</strong><br>
              • Date / Timestamp<br>
              • Oil / Gas / Water volumes<br>
              • Pressure readings<br>
              • Well / Device IDs<br>
              • Failure / Alarm flags
            </div>""", unsafe_allow_html=True)

        if uploaded is not None:
            file_size_mb = uploaded.size / 1024 / 1024
            st.success(f"✅ File received: **{uploaded.name}** ({file_size_mb:.1f} MB)")

            # Sheet selection for Excel
            sheet_name = 0
            if uploaded.name.endswith((".xlsx",".xls")):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(io.BytesIO(uploaded.getvalue()), read_only=True)
                    sheets = wb.sheetnames
                    wb.close()
                    if len(sheets) > 1:
                        sheet_name = st.selectbox("Select Sheet", sheets)
                except: pass

            max_rows_opt = st.selectbox("Load rows", ["All rows","1,000,000","500,000","100,000"],
                help="For very large files, limit rows to speed up preview")
            max_rows = None
            if max_rows_opt != "All rows":
                max_rows = int(max_rows_opt.replace(",",""))

            if st.button("▶ Load & Analyze File"):
                t0 = time.time()
                try:
                    if uploaded.name.endswith(".csv"):
                        df_user = read_csv_chunked(uploaded, max_rows=max_rows)
                    else:
                        df_user = read_excel_chunked(uploaded, sheet=sheet_name, max_rows=max_rows)

                    # Cache in session state
                    st.session_state["user_df"]   = df_user
                    st.session_state["user_fname"] = uploaded.name
                    st.session_state["col_map"]   = auto_detect_columns(df_user)

                    elapsed = time.time() - t0
                    st.success(f"✅ Loaded **{len(df_user):,}** rows × **{df_user.shape[1]}** columns in {elapsed:.1f}s")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")

    # ── Schema ─────────────────────────────────────────────────────────────────
    with tab_schema:
        if "user_df" not in st.session_state:
            st.info("Upload a file first.")
        else:
            df_user = st.session_state["user_df"]
            col_map = st.session_state["col_map"]

            st.markdown(f"**File:** `{st.session_state['user_fname']}` — **{len(df_user):,} rows × {df_user.shape[1]} columns**")

            # Auto-detected mappings
            if col_map:
                st.markdown("##### 🔍 Auto-Detected Column Mappings")
                map_df = pd.DataFrame([(k.upper(), v, "✅ Auto-detected") for k,v in col_map.items()],
                    columns=["Data Type","Column Name","Status"])
                st.dataframe(map_df, use_container_width=True, hide_index=True)

            # Manual override
            with st.expander("⚙️ Override column mappings (optional)"):
                all_cols = ["— None —"] + list(df_user.columns)
                new_map = {}
                col_keys = ["date","well","device","oil","gas","water","pressure","failure"]
                cols_form = st.columns(2)
                for i, key in enumerate(col_keys):
                    with cols_form[i % 2]:
                        default = col_map.get(key, "— None —")
                        default_idx = all_cols.index(default) if default in all_cols else 0
                        chosen = st.selectbox(f"{key.capitalize()} column", all_cols,
                            index=default_idx, key=f"map_{key}")
                        if chosen != "— None —":
                            new_map[key] = chosen
                if st.button("Apply Mapping"):
                    st.session_state["col_map"] = new_map
                    col_map = new_map
                    st.success("Mapping updated!")

            # Schema table
            schema_info = pd.DataFrame({
                "Column": df_user.columns,
                "Type": df_user.dtypes.astype(str),
                "Non-Null": df_user.count().values,
                "Null %": (df_user.isna().mean()*100).round(2).values,
                "Sample Value": [str(df_user[c].dropna().iloc[0]) if df_user[c].dropna().any() else "NaN"
                                 for c in df_user.columns],
            })
            st.dataframe(schema_info, use_container_width=True, hide_index=True)

            # Preview
            st.markdown("##### Preview (first 100 rows)")
            st.dataframe(df_user.head(100), use_container_width=True)

    # ── Analytics ──────────────────────────────────────────────────────────────
    with tab_analyze:
        if "user_df" not in st.session_state:
            st.info("Upload a file first.")
        else:
            df_user = st.session_state["user_df"]
            col_map = st.session_state["col_map"]
            results = smart_analyze(df_user.copy(), col_map)

            # KPI row
            kpi_cols = st.columns(4)
            kpi_data = [
                ("TOTAL ROWS", fmt_num(len(df_user)), "", "flat", "📋", THEME["primary"]),
                ("COLUMNS", str(df_user.shape[1]), "features", "flat", "📐", THEME["info"]),
                ("NULL CELLS %", f"{df_user.isna().mean().mean()*100:.1f}%", "data quality", "flat", "🔍", THEME["warning"]),
                ("NUMERIC COLS", str(df_user.select_dtypes(include=np.number).shape[1]),
                 "auto-detected", "flat", "🔢", THEME["success"]),
            ]
            for col, (lbl, val, delta, d, icon, color) in zip(kpi_cols, kpi_data):
                with col:
                    st.markdown(kpi_card(lbl, val, delta, d, icon, color), unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            # Date range
            if "date_range" in results:
                dr = results["date_range"]
                st.markdown(f"""<div class="info-box">📅 <strong>Date range:</strong>
                {dr[0].date() if hasattr(dr[0],'date') else dr[0]} →
                {dr[1].date() if hasattr(dr[1],'date') else dr[1]}</div>""", unsafe_allow_html=True)

            # Production summary
            prod_keys = [k for k in ["oil","gas","water","pressure"] if k in results]
            if prod_keys:
                section_header("Production / Sensor Summary", "AUTO-DETECTED")
                p_cols = st.columns(len(prod_keys))
                for pc, key in zip(p_cols, prod_keys):
                    r = results[key]
                    with pc:
                        st.metric(f"Total {key.capitalize()}", fmt_num(r["sum"]))
                        st.caption(f"Avg: {fmt_num(r['mean'])} | Max: {fmt_num(r['max'])} | Missing: {r['missing_pct']:.1f}%")

            # Time series charts
            if "date" in col_map:
                date_col = col_map["date"]
                section_header("Time Series Charts", "AUTO-GENERATED")
                numeric_cols = df_user.select_dtypes(include=np.number).columns.tolist()
                sel_chart_col = st.selectbox("Select column to chart",
                    [c for c in numeric_cols if c != col_map.get("failure")])

                if sel_chart_col:
                    try:
                        ts = df_user[[date_col, sel_chart_col]].dropna().copy()
                        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
                        ts = ts.dropna().sort_values(date_col)
                        if "well" in col_map:
                            group_col = col_map["well"]
                        elif "device" in col_map:
                            group_col = col_map["device"]
                        else:
                            group_col = None

                        if group_col and df_user[group_col].nunique() <= 20:
                            fig_ts = go.Figure()
                            for grp, grp_df in df_user.groupby(group_col):
                                grp_ts = grp_df[[date_col, sel_chart_col]].dropna()
                                grp_ts[date_col] = pd.to_datetime(grp_ts[date_col], errors="coerce")
                                grp_ts = grp_ts.dropna().sort_values(date_col)
                                if len(grp_ts) > 0:
                                    fig_ts.add_trace(go.Scatter(x=grp_ts[date_col], y=grp_ts[sel_chart_col],
                                        name=str(grp), mode="lines"))
                            fig_ts.update_layout(**base_layout(f"{sel_chart_col} by {group_col}"),
                                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                                yaxis=dict(title=sel_chart_col, gridcolor="rgba(255,255,255,0.04)"),
                                legend=dict(orientation="h", y=1.05))
                        else:
                            # Resample to monthly mean if many rows
                            if len(ts) > 10000:
                                ts = ts.set_index(date_col).resample("M").mean().reset_index()
                            fig_ts = go.Figure()
                            fig_ts.add_trace(go.Scatter(x=ts[date_col], y=ts[sel_chart_col],
                                name=sel_chart_col, fill="tozeroy",
                                line=dict(color=THEME["primary"], width=2),
                                fillcolor="rgba(0,212,255,0.08)"))
                            fig_ts.update_layout(**base_layout(f"{sel_chart_col} Over Time"),
                                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                                yaxis=dict(title=sel_chart_col, gridcolor="rgba(255,255,255,0.04)"))
                        st.plotly_chart(fig_ts, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not plot: {e}")

            # Correlation heatmap
            section_header("Correlation Heatmap", "NUMERIC FEATURES")
            num_df = df_user.select_dtypes(include=np.number).dropna(axis=1, how="all")
            if len(num_df.columns) >= 2:
                corr = num_df.corr()
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                    colorscale=[[0, THEME["danger"]], [0.5, THEME["surface"]], [1, THEME["primary"]]],
                    zmin=-1, zmax=1, text=corr.round(2).values,
                    texttemplate="%{text}", textfont=dict(size=10)))
                fig_corr.update_layout(**base_layout("Feature Correlation Matrix", height=max(350, len(corr)*28)),
                    xaxis=dict(tickangle=-45, gridcolor="rgba(0,0,0,0)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_corr, use_container_width=True)

            # Statistical summary
            section_header("Statistical Summary", "DESCRIPTIVE STATS")
            st.dataframe(df_user.describe().T.style.format("{:.3f}"), use_container_width=True)

    # ── ML on User Data ────────────────────────────────────────────────────────
    with tab_ml:
        if "user_df" not in st.session_state:
            st.info("Upload a file first.")
        else:
            df_user = st.session_state["user_df"]
            col_map = st.session_state["col_map"]
            st.markdown("""<div class="info-box">
              <strong>🤖 Train ML models directly on your uploaded data.</strong><br>
              Select a target column and features, and the platform will train a
              Random Forest model with cross-validation.
            </div>""", unsafe_allow_html=True)

            num_cols = df_user.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns to train an ML model.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    target_col = st.selectbox("🎯 Target Column (what to predict)", num_cols,
                        index=num_cols.index(col_map["failure"]) if "failure" in col_map and col_map["failure"] in num_cols else 0)
                    task_type = st.radio("Task Type", ["Classification (binary target)", "Regression"],
                        help="Classification for failure/alarm columns; Regression for rates/volumes")

                with col_b:
                    feat_cols = st.multiselect("📐 Feature Columns (inputs)",
                        [c for c in num_cols if c != target_col],
                        default=[c for c in num_cols if c != target_col][:min(8, len(num_cols)-1)],
                        help="Select up to 15 features for best performance")

                if st.button("🚀 Train Model") and feat_cols:
                    with st.spinner("Training model on your data..."):
                        try:
                            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                            from sklearn.model_selection import cross_val_score, train_test_split
                            from sklearn.metrics import classification_report, r2_score, mean_squared_error
                            from sklearn.preprocessing import StandardScaler

                            sub = df_user[feat_cols + [target_col]].dropna().copy()
                            X = sub[feat_cols].values
                            y = sub[target_col].values

                            if len(sub) < 50:
                                st.error("Need at least 50 rows with no missing values to train.")
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                is_classification = "Classification" in task_type
                                if is_classification:
                                    model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                        random_state=42, n_jobs=-1)
                                else:
                                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                                # Metrics
                                col_r1, col_r2 = st.columns(2)
                                with col_r1:
                                    if is_classification:
                                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                                        acc = accuracy_score(y_test, y_pred)
                                        f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                                        st.markdown(f"""<div class="success-box">
                                          <strong>✅ Classification Results</strong><br><br>
                                          Accuracy: <strong>{acc:.4f}</strong><br>
                                          F1 Score: <strong>{f1:.4f}</strong><br>
                                          Training samples: <strong>{len(X_train):,}</strong><br>
                                          Test samples: <strong>{len(X_test):,}</strong>
                                        </div>""", unsafe_allow_html=True)
                                    else:
                                        r2   = r2_score(y_test, y_pred)
                                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                        st.markdown(f"""<div class="success-box">
                                          <strong>✅ Regression Results</strong><br><br>
                                          R²: <strong>{r2:.4f}</strong><br>
                                          RMSE: <strong>{rmse:.3f}</strong><br>
                                          Training samples: <strong>{len(X_train):,}</strong><br>
                                          Test samples: <strong>{len(X_test):,}</strong>
                                        </div>""", unsafe_allow_html=True)

                                with col_r2:
                                    # Feature importance chart
                                    fi = pd.DataFrame({"Feature": feat_cols,
                                        "Importance": model.feature_importances_})
                                    fi = fi.sort_values("Importance", ascending=True).tail(10)
                                    fig_fi = go.Figure(go.Bar(x=fi["Importance"], y=fi["Feature"],
                                        orientation="h", marker=dict(color=fi["Importance"],
                                        colorscale=[[0, THEME["surface"]], [1, THEME["primary"]]])))
                                    fig_fi.update_layout(**base_layout("Feature Importance", height=300),
                                        xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.04)"),
                                        yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                                    st.plotly_chart(fig_fi, use_container_width=True)

                                # Actual vs Predicted
                                if not is_classification:
                                    fig_av = go.Figure()
                                    fig_av.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test,
                                        name="Actual", line=dict(color=THEME["primary"], width=2)))
                                    fig_av.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred,
                                        name="Predicted", line=dict(color=THEME["secondary"],width=2,dash="dash")))
                                    fig_av.update_layout(**base_layout("Actual vs Predicted"),
                                        xaxis=dict(title="Sample Index", gridcolor="rgba(255,255,255,0.04)"),
                                        yaxis=dict(title=target_col, gridcolor="rgba(255,255,255,0.04)"),
                                        legend=dict(orientation="h", y=1.05))
                                    st.plotly_chart(fig_av, use_container_width=True)

                        except Exception as e:
                            st.error(f"Training error: {e}")
                            st.exception(e)
