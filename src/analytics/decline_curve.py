"""
src/analytics/decline_curve.py
────────────────────────────────
Decline Curve Analysis (DCA) — Arps Decline Models.

Implements:
  - Exponential Decline:  q(t) = qi * exp(-D * t)
  - Hyperbolic Decline:   q(t) = qi / (1 + b*D*t)^(1/b)
  - Harmonic Decline:     q(t) = qi / (1 + D*t)   [special case b=1]

All three are members of the Arps family (1945), the industry standard
for production forecasting in oil & gas.

Parameters:
  qi = initial production rate [Sm³/day]
  D  = nominal decline rate [1/day]
  b  = hyperbolic exponent [dimensionless, 0 < b ≤ 2]
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Arps Decline Functions
# ═══════════════════════════════════════════════════════════════════════════════

def exponential_decline(t: np.ndarray, qi: float, D: float) -> np.ndarray:
    """
    Exponential (constant-percentage) decline.
    q(t) = qi · exp(−D · t)
    Most conservative; common in dry gas wells.
    """
    return qi * np.exp(-D * t)


def hyperbolic_decline(t: np.ndarray, qi: float, D: float, b: float) -> np.ndarray:
    """
    Hyperbolic decline.
    q(t) = qi / (1 + b·D·t)^(1/b)
    General case; most flexible. b→0 → exponential; b=1 → harmonic.
    """
    b = np.clip(b, 1e-6, 2.0)
    return qi / np.power(1.0 + b * D * t, 1.0 / b)


def harmonic_decline(t: np.ndarray, qi: float, D: float) -> np.ndarray:
    """
    Harmonic decline (b = 1 special case).
    q(t) = qi / (1 + D·t)
    """
    return qi / (1.0 + D * t)


# ═══════════════════════════════════════════════════════════════════════════════
# Fitting Engine
# ═══════════════════════════════════════════════════════════════════════════════

def fit_decline_curves(
    df_prod: pd.DataFrame,
    well: str,
    forecast_days: int = 180,
    min_points: int = 30,
) -> dict:
    """
    Fit all three Arps decline models to a single well's production history.

    Parameters
    ----------
    df_prod : DataFrame  — cleaned production DataFrame with feature columns
    well    : str        — well bore name
    forecast_days : int  — days to forecast beyond last data point
    min_points : int     — minimum active production days required

    Returns
    -------
    dict with keys: well, historical, exponential, hyperbolic, harmonic
    """
    log.info(f"Fitting decline curves for {well}")

    # ── Filter: production wells, non-zero oil days ───────────────────────────
    mask = (
        (df_prod["NPD_WELL_BORE_NAME"] == well)
        & (df_prod["FLOW_KIND"] == "production")
        & (df_prod["BORE_OIL_VOL"] > 0)
    )
    sub = df_prod[mask].sort_values("DATEPRD").reset_index(drop=True)

    if len(sub) < min_points:
        log.warning(f"  {well}: only {len(sub)} active days — skipping DCA")
        return {"well": well, "error": f"Insufficient data ({len(sub)} points)"}

    # ── Build time axis (days since first production) ─────────────────────────
    t = sub["DAYS_ON_PRODUCTION"].values.astype(float)
    q = sub["BORE_OIL_VOL"].values.astype(float)

    # Smooth with 14-day rolling average for fitting (reduces noise)
    q_smooth = pd.Series(q).rolling(14, min_periods=1).mean().values

    qi_guess = float(q_smooth[:5].mean())
    D_guess  = 0.001

    results = {
        "well": well,
        "historical": {
            "t":    t.tolist(),
            "q":    q.tolist(),
            "q_smooth": q_smooth.tolist(),
            "dates": sub["DATEPRD"].astype(str).tolist(),
        },
    }

    # ── Forecast time axis ────────────────────────────────────────────────────
    t_last = float(t[-1])
    t_fcst = np.linspace(t_last, t_last + forecast_days, forecast_days)
    last_date = sub["DATEPRD"].iloc[-1]
    fcst_dates = pd.date_range(last_date, periods=forecast_days, freq="D")

    # ── Fit Exponential ───────────────────────────────────────────────────────
    try:
        popt_exp, _ = curve_fit(
            exponential_decline, t, q_smooth,
            p0=[qi_guess, D_guess],
            bounds=([0, 0], [np.inf, 1.0]),
            maxfev=10000,
        )
        q_fit_exp  = exponential_decline(t, *popt_exp)
        q_fcst_exp = exponential_decline(t_fcst, *popt_exp)
        r2_exp     = _r2_score(q_smooth, q_fit_exp)
        Di_annual  = float(popt_exp[1]) * 365

        results["exponential"] = {
            "qi": float(popt_exp[0]),
            "D_daily": float(popt_exp[1]),
            "D_annual_pct": Di_annual * 100,
            "r2": r2_exp,
            "fit_q":    q_fit_exp.tolist(),
            "fcst_t":   t_fcst.tolist(),
            "fcst_q":   q_fcst_exp.tolist(),
            "fcst_dates": fcst_dates.strftime("%Y-%m-%d").tolist(),
            "equation": f"q(t) = {popt_exp[0]:.1f} · exp(−{popt_exp[1]:.5f}·t)",
            "eur_sm3":  float(popt_exp[0] / popt_exp[1]) if popt_exp[1] > 0 else None,
        }
        log.info(f"  Exponential: qi={popt_exp[0]:.0f}, D={Di_annual*100:.1f}%/yr, R²={r2_exp:.3f}")
    except Exception as e:
        log.warning(f"  Exponential fit failed: {e}")
        results["exponential"] = {"error": str(e)}

    # ── Fit Hyperbolic ────────────────────────────────────────────────────────
    try:
        popt_hyp, _ = curve_fit(
            hyperbolic_decline, t, q_smooth,
            p0=[qi_guess, D_guess, 0.5],
            bounds=([0, 0, 0.01], [np.inf, 1.0, 2.0]),
            maxfev=10000,
        )
        q_fit_hyp  = hyperbolic_decline(t, *popt_hyp)
        q_fcst_hyp = hyperbolic_decline(t_fcst, *popt_hyp)
        r2_hyp     = _r2_score(q_smooth, q_fit_hyp)
        b          = float(popt_hyp[2])

        results["hyperbolic"] = {
            "qi": float(popt_hyp[0]),
            "D_daily": float(popt_hyp[1]),
            "D_annual_pct": float(popt_hyp[1]) * 365 * 100,
            "b": b,
            "r2": r2_hyp,
            "fit_q":    q_fit_hyp.tolist(),
            "fcst_t":   t_fcst.tolist(),
            "fcst_q":   q_fcst_hyp.tolist(),
            "fcst_dates": fcst_dates.strftime("%Y-%m-%d").tolist(),
            "equation": f"q(t) = {popt_hyp[0]:.1f} / (1 + {b:.2f}·{popt_hyp[1]:.5f}·t)^(1/{b:.2f})",
        }
        log.info(f"  Hyperbolic: qi={popt_hyp[0]:.0f}, b={b:.2f}, R²={r2_hyp:.3f}")
    except Exception as e:
        log.warning(f"  Hyperbolic fit failed: {e}")
        results["hyperbolic"] = {"error": str(e)}

    # ── Fit Harmonic ──────────────────────────────────────────────────────────
    try:
        popt_har, _ = curve_fit(
            harmonic_decline, t, q_smooth,
            p0=[qi_guess, D_guess],
            bounds=([0, 0], [np.inf, 1.0]),
            maxfev=10000,
        )
        q_fit_har  = harmonic_decline(t, *popt_har)
        q_fcst_har = harmonic_decline(t_fcst, *popt_har)
        r2_har     = _r2_score(q_smooth, q_fit_har)

        results["harmonic"] = {
            "qi": float(popt_har[0]),
            "D_daily": float(popt_har[1]),
            "D_annual_pct": float(popt_har[1]) * 365 * 100,
            "r2": r2_har,
            "fit_q":    q_fit_har.tolist(),
            "fcst_t":   t_fcst.tolist(),
            "fcst_q":   q_fcst_har.tolist(),
            "fcst_dates": fcst_dates.strftime("%Y-%m-%d").tolist(),
            "equation": f"q(t) = {popt_har[0]:.1f} / (1 + {popt_har[1]:.5f}·t)",
        }
        log.info(f"  Harmonic: qi={popt_har[0]:.0f}, D={float(popt_har[1])*365*100:.1f}%/yr, R²={r2_har:.3f}")
    except Exception as e:
        log.warning(f"  Harmonic fit failed: {e}")
        results["harmonic"] = {"error": str(e)}

    return results


def fit_all_wells(df_prod: pd.DataFrame, forecast_days: int = 180) -> dict:
    """Run DCA for all production wells."""
    producers = (
        df_prod[df_prod["FLOW_KIND"] == "production"]["NPD_WELL_BORE_NAME"]
        .unique().tolist()
    )
    all_results = {}
    for well in producers:
        all_results[well] = fit_decline_curves(df_prod, well, forecast_days)
    return all_results


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
