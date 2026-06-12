"""Page 6 — Database Explorer (Bug-Fixed)"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.utils import section_header
from src.config import DB_PATH
from src.database.db_manager import DatabaseManager

def render():
    section_header("Database Explorer", "SQLite · LIVE")

    if not DB_PATH.exists():
        st.error("Database not found. Run the ETL pipeline first.")
        return

    db = DatabaseManager()
    tables = [t for t in db.get_table_names() if t != "sqlite_sequence"]

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        sel_table = st.selectbox("Select Table", tables)
        row_count = db.query(f"SELECT COUNT(*) AS n FROM {sel_table}")["n"].iloc[0]
        st.markdown(f"""<div class="info-box">
          <strong>Table:</strong> {sel_table}<br>
          <strong>Rows:</strong> {row_count:,}<br>
          <strong>DB:</strong> oilfield.db
        </div>""", unsafe_allow_html=True)

    with col_info:
        schema = db.get_table_info(sel_table)
        st.markdown("**Schema**")
        st.dataframe(schema[["name","type","notnull","pk"]].rename(
            columns={"name":"Column","type":"Type","notnull":"Not Null","pk":"PK"}),
            use_container_width=True, hide_index=True, height=180)

    st.markdown("---")
    section_header("SQL Query Interface", "LIVE QUERY")
    sql_query = st.text_area("SQL Query", value=f"SELECT * FROM {sel_table} LIMIT 100", height=90)

    if st.button("▶ Run Query"):
        try:
            result_df = db.query(sql_query)
            st.success(f"✅ {len(result_df):,} rows returned")
            st.dataframe(result_df, use_container_width=True)
            st.download_button("⬇ Download CSV", result_df.to_csv(index=False),
                f"{sel_table}_result.csv", "text/csv")
        except Exception as e:
            st.error(f"Query error: {e}")

    st.markdown("---")
    section_header("All Tables", "OVERVIEW")
    rows = []
    for t in tables:
        cnt = db.query(f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0]
        cols = db.get_table_info(t)
        rows.append({"Table": t, "Rows": cnt, "Columns": len(cols)})
    st.dataframe(pd.DataFrame(rows).set_index("Table").style.format({"Rows":"{:,}"}),
        use_container_width=True)
