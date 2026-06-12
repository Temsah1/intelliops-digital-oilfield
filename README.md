# 🛢️ AI-Powered Digital Oilfield Platform

> **Portfolio project** demonstrating end-to-end Petroleum Data Science, Machine Learning, and Dashboard Engineering — built on the **Equinor Volve Open Field Dataset**.

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)](https://xgboost.ai)
[![SQLite](https://img.shields.io/badge/Database-SQLite-yellow)](https://sqlite.org)

---

## 📌 Project Overview

A production-grade **Digital Oilfield Intelligence Platform** that integrates:

| Domain | Technologies |
|---|---|
| **Data Engineering** | Pandas ETL pipeline, SQLite, schema validation |
| **Petroleum Analytics** | DCA (Arps), Water Cut, GOR, Pressure Drawdown |
| **Machine Learning** | Gradient Boosting, Random Forest, XGBoost, SMOTE |
| **Dashboard** | Streamlit multi-page dark-mode app, Plotly charts |
| **Deployment** | Streamlit Community Cloud ready |

---

## 🌊 Datasets

### 1. Volve Field — Well Production Data
- **Source:** Equinor Volve Open Dataset (Norwegian North Sea, Block 15/9)
- **Period:** September 2007 – December 2016 (9 years)
- **Wells:** 5 producers + 2 injectors
- **Rows:** 15,634 daily production records
- **Key columns:** `BORE_OIL_VOL`, `BORE_GAS_VOL`, `AVG_DOWNHOLE_PRESSURE`, `FLOW_KIND`

### 2. Predictive Maintenance Dataset
- **Source:** IoT sensor readings from 1,169 industrial devices
- **Period:** January – November 2015
- **Rows:** 124,494 device-day records
- **Target:** `failure` (binary, 0.085% rate — severe class imbalance)
- **Strongest predictors:** metric7 (115× at failure), metric4 (32×), metric2 (26×)

---

## 🏗️ Architecture

```
AI_Digital_Oilfield/
├── app.py                        # Streamlit entry point
├── data/
│   ├── raw/                      # Original CSVs
│   └── processed/                # Cleaned + feature-engineered CSVs
├── database/
│   └── oilfield.db               # SQLite (5 tables)
├── models/
│   ├── production_forecaster.joblib
│   ├── failure_predictor.joblib
│   └── model_metrics.json
├── src/
│   ├── config.py                 # All paths, constants, theme
│   ├── logger.py                 # Coloured logging
│   ├── etl/
│   │   ├── loader.py             # Raw data ingestion
│   │   ├── cleaner.py            # Data quality & cleaning
│   │   ├── feature_engineer.py   # Petroleum feature creation
│   │   └── pipeline.py           # ETL orchestrator
│   ├── analytics/
│   │   └── decline_curve.py      # Arps DCA (exponential/hyperbolic/harmonic)
│   ├── ml/
│   │   ├── production_forecaster.py  # ML production forecasting
│   │   ├── failure_predictor.py      # Equipment failure classification
│   │   └── train_all.py              # Master training script
│   ├── database/
│   │   └── db_manager.py         # SQLite CRUD layer
│   └── dashboard/
│       ├── utils.py              # Shared chart helpers
│       ├── assets/style.css      # Premium dark theme
│       └── pages/
│           ├── pg_home.py        # Executive dashboard
│           ├── pg_production.py  # Production analytics
│           ├── pg_decline.py     # Decline curve analysis
│           ├── pg_forecast.py    # ML forecasting
│           ├── pg_maintenance.py # Predictive maintenance
│           ├── pg_database.py    # DB explorer
│           └── pg_reports.py     # Engineering reports
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AI_Digital_Oilfield.git
cd AI_Digital_Oilfield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ETL pipeline (creates database + processed data)
python -m src.etl.pipeline

# 4. Train ML models
python -m src.ml.train_all

# 5. Launch dashboard
streamlit run app.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Executive Home | KPI cards, field overview, model summary |
| 📈 Production Analytics | Oil/gas/water trends, well comparison, pressure |
| 📉 Decline Curve Analysis | Arps DCA fitting with 6-month forecast |
| 🤖 Production Forecasting | ML 6-month forecast with uncertainty bands |
| 🔧 Predictive Maintenance | Health scores, failure risk, confusion matrix |
| 🗄️ Database Explorer | Live SQLite query interface |
| 📋 Reports & Insights | Auto-generated engineering recommendations |

---

## 🤖 Machine Learning

### Production Forecasting
- Models: Ridge Regression, Random Forest, Gradient Boosting, XGBoost
- Target: `BORE_OIL_VOL` (Sm³/day)
- Features: 15 engineered features (rolling averages, pressure, time)
- Selection: Lowest test-set RMSE

### Predictive Maintenance
- Models: Logistic Regression, Random Forest, XGBoost
- Target: `failure` (binary)
- Class Imbalance: SMOTE + `class_weight='balanced'`
- Selection: Highest F1 score (critical for imbalanced data)

---

## ☁️ Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set **Main file:** `app.py`
5. Add `requirements.txt` to repo root
6. Click **Deploy**

> **Note:** Add raw CSV files to `data/raw/` before deploying, or use `st.secrets` for cloud data paths.

---

## 🔮 Future Improvements

- Real-time SCADA data integration via OPC-UA
- SHAP values for model explainability
- Reservoir simulation integration (Eclipse/tNavigator)
- Automated report PDF export
- Multi-field comparison dashboard
- Carbon footprint tracking module

---

*Built with ❤️ for the oil & gas industry | Equinor Volve Open Dataset*
