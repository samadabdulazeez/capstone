# dashboard.py
"""
Manitoba Farmer Dashboard (ensemble integration, conservative calibration)

- Loads trained artifacts via utils.model_loader
- Uses utils.dashboard_predict.predict_df for ensemble inference
- Converts model units to farmer units (kg/ha <-> lb/acre) with sidebar toggle
- Applies shrinkage toward a robust baseline when model confidence is low
- Caps percent-change messaging and clamps plotted forecast points to avoid misleading optimism
- Falls back to in-app linear regression if artifacts are missing or fail
"""
import os
import datetime
import math
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import r2_score

from utils.data_loader import (
    list_statscan_commodities_mb,
    load_statscan_yields_mb,
    load_mb_weather,
    load_openmeteo_forecast,
)
from utils.ndvi import compute_ndvi_map_mb

# Try to import model helpers
try:
    from utils.model_loader import load_artifacts
    from utils.dashboard_predict import predict_df
    ARTIFACTS_AVAILABLE = True
except Exception:
    ARTIFACTS_AVAILABLE = False

# Page config and logger
st.set_page_config(page_title="Manitoba Farmer Dashboard", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# Sidebar controls
st.sidebar.title("Manitoba Farmer Dashboard")
if st.sidebar.button("🔄 Refresh data"):
    try:
        st.cache_data.clear()
    except Exception:
        pass

commodities = list_statscan_commodities_mb()
commodity = st.sidebar.selectbox("Commodity yield series (StatsCan)", commodities)
weather_source = st.sidebar.radio("Weather source", ["obs", "reanalysis", "union"], index=2)

st.sidebar.markdown("### Units and display")
unit_choice = st.sidebar.radio("Display yield units", ["lb/acre", "kg/ha"], index=0)
# conversion constants
KG_PER_HA_TO_LB_PER_ACRE = 0.892179  # 1 kg/ha = 0.892179 lb/acre
LB_PER_ACRE_TO_KG_PER_HA = 1.0 / KG_PER_HA_TO_LB_PER_ACRE

st.sidebar.markdown("### Sentinel‑2 NDVI (optional)")
today = datetime.date.today()
date_start = st.sidebar.date_input("Start date", datetime.date(today.year, 5, 1), max_value=today)
date_end = st.sidebar.date_input("End date", datetime.date(today.year, 9, 30), max_value=today)
cloud_max = st.sidebar.slider("Max cloud cover (%)", 0, 100, 30)

st.sidebar.markdown("### Advisory thresholds")
yield_drop_pct_threshold = st.sidebar.slider("Yield change threshold (%)", 5, 30, 10)
precip_extreme_mm = st.sidebar.slider("7‑day heavy rain threshold (mm)", 10, 30, 20)
heat_days_threshold = st.sidebar.slider("7‑day hot days threshold (>30°C)", 0, 3, 1)

# Helper: farmer advisory (always shown)
def render_farmer_advisory(annual_df: pd.DataFrame, forecast_df: pd.DataFrame):
    st.subheader("Farmer Advisory")
    st.markdown("Plain-language guidance based on forecasted precipitation and temperature compared to historical averages:")

    hist_precip_mean = float(annual_df["Annual_Precip_mm"].mean()) if (annual_df is not None and not annual_df.empty) else math.nan
    hist_temp_mean = float(annual_df["Annual_Tmean_C"].mean()) if (annual_df is not None and not annual_df.empty) else math.nan

    seven_day_precip = float(np.nansum(forecast_df["Precipitation"])) if (forecast_df is not None and not forecast_df.empty) else math.nan
    seven_day_temp_mean = float(((forecast_df["Temp_Max"] + forecast_df["Temp_Min"]) / 2.0).mean()) if (forecast_df is not None and not forecast_df.empty) else math.nan

    precip_diff = None if math.isnan(hist_precip_mean) or math.isnan(seven_day_precip) else seven_day_precip - hist_precip_mean
    temp_diff = None if math.isnan(hist_temp_mean) or math.isnan(seven_day_temp_mean) else seven_day_temp_mean - hist_temp_mean

    bullets = []
    if precip_diff is None:
        bullets.append("• **Precipitation:** Historical or forecast data missing; check data sources.")
    else:
        if precip_diff < -20:
            bullets.append("• ⚠️ **Drier than average** forecast — consider irrigation or drought‑tolerant varieties.")
        elif precip_diff > 20:
            bullets.append("• 🌧️ **Wetter than average** forecast — watch for waterlogging and consider drainage.")
        else:
            bullets.append("• ✅ **Precipitation near average** — no major precipitation-related adjustments expected.")

    if temp_diff is None:
        bullets.append("• **Temperature:** Historical or forecast data missing; check data sources.")
    else:
        if temp_diff > 2:
            bullets.append("• 🔥 **Warmer than average** forecast — monitor for heat stress and adjust timing.")
        elif temp_diff < -2:
            bullets.append("• ❄️ **Cooler than average** forecast — growth may be slower; consider later planting if timing allows.")
        else:
            bullets.append("• ✅ **Temperature near average** — standard temperature-related practices apply.")

    if not math.isnan(seven_day_precip):
        if seven_day_precip >= 20:
            bullets.append(f"• Short-term: **{seven_day_precip:.1f} mm** expected over 7 days — delay spraying and heavy fieldwork.")
        elif seven_day_precip < 2:
            bullets.append(f"• Short-term: **{seven_day_precip:.1f} mm** expected over 7 days — low rainfall; check soil moisture before seeding or top-dress.")
    else:
        bullets.append("• Short-term forecast unavailable.")

    for b in bullets:
        st.markdown(b)

    st.markdown(
        "_Note: This advisory is a practical guide based on short-term forecasts and historical annual averages. "
        "It does not replace local scouting or agronomic advice._"
    )

# Header
st.header("Manitoba Agriculture Insights")

# Load data
try:
    yields_df = load_statscan_yields_mb(commodity)
    annual_df = load_mb_weather(weather_source)
except Exception as e:
    logger.exception("Error loading data")
    st.error(f"Error loading data: {e}")
    st.stop()

# Ensure Year int
def _ensure_year_int(df):
    df = df.copy()
    if "Year" not in df.columns:
        return df
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    return df

yields_df = _ensure_year_int(yields_df)
annual_df = _ensure_year_int(annual_df)

# Forecast
try:
    forecast_df = load_openmeteo_forecast(lat=49.9, lon=-97.2)
except Exception as e:
    logger.exception("Forecast fetch failed")
    forecast_df = pd.DataFrame(columns=["Date", "Temp_Max", "Temp_Min", "Precipitation"])

# Always show farmer advisory
render_farmer_advisory(annual_df, forecast_df)

# Merge on Year
merged = yields_df.merge(annual_df, on="Year", how="inner")

if merged.empty:
    st.warning("No overlapping years between yield and climate data. Try a different commodity or weather source.")
    if not yields_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(yields_df["Year"], yields_df["Yield"], marker="o", color="green")
        ax.set_title(f"{commodity} yields in Manitoba")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"Yield ({yields_df['UOM'].iloc[0] if 'UOM' in yields_df.columns and not yields_df.empty else ''})")
        st.pyplot(fig)
    if not annual_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(annual_df["Year"], annual_df["Annual_Tmean_C"], color="blue", label="Mean Temp (°C)")
        ax2 = ax.twinx()
        ax2.plot(annual_df["Year"], annual_df["Annual_Precip_mm"], color="orange", label="Precip (mm)")
        ax.set_title("Annual climate")
        st.pyplot(fig)
    st.stop()

# Aggregate forecast to seasonal-like proxies
avg_precip_7d = float(np.nansum(forecast_df["Precipitation"])) if not forecast_df.empty else 0.0
avg_temp_7d = float(((forecast_df["Temp_Max"] + forecast_df["Temp_Min"]) / 2.0).mean()) if not forecast_df.empty else merged["Annual_Tmean_C"].mean()

# Build seasonal proxy: scale 7-day anomaly to seasonal window (simple heuristic)
hist_precip_mean = merged["Annual_Precip_mm"].mean()
hist_temp_mean = merged["Annual_Tmean_C"].mean()
seven_day_clim_precip = forecast_df["Precipitation"].mean() if not forecast_df.empty else 0.0
precip_anom = avg_precip_7d - seven_day_clim_precip
seasonal_precip_proxy = hist_precip_mean + np.clip(precip_anom * 8.0, -0.5 * hist_precip_mean, 0.5 * hist_precip_mean)
temp_anom = avg_temp_7d - (forecast_df[["Temp_Max", "Temp_Min"]].mean(axis=1).mean() if not forecast_df.empty else avg_temp_7d)
seasonal_temp_proxy = hist_temp_mean + np.clip(temp_anom, -3.0, 3.0)

# Historical baseline stats
hist_median_yield = float(np.nanmedian(merged["Yield"]))
hist_mean_yield = float(np.nanmean(merged["Yield"]))
hist_std_yield = float(np.nanstd(merged["Yield"]))

# Default placeholders
predicted_model_y = np.nan
pred_ridge_fore = np.nan
pred_rf_fore = np.nan
pi_low = np.nan
pi_high = np.nan
r2 = np.nan
confidence = "Low"
conf_color = "red"
pi_width_pct = np.nan
oof_mae = None
display_predicted = np.nan

# Run ensemble if available
if ARTIFACTS_AVAILABLE:
    try:
        ART = load_artifacts()
        models_dir = ART.get("models_dir", os.path.join(os.path.dirname(__file__), "..", "models"))
        # Build forecast row for model (use proxies and last known yield for lag)
        forecast_row = pd.DataFrame({
            "Year": [int(datetime.date.today().year)],
            "Annual_Precip_mm": [seasonal_precip_proxy],
            "Annual_Tmean_C": [seasonal_temp_proxy],
            "obs_Mean_Temp_mean": [seasonal_temp_proxy],
            "obs_Precip_mm_sum": [seasonal_precip_proxy],
            "Yield_kg_ha": [merged["Yield"].iloc[-1] if not merged.empty else np.nan],
        })
        preds_forecast = predict_df(forecast_row)
        predicted_model_y = float(preds_forecast["pred_Final"].iloc[0])
        pred_ridge_fore = float(preds_forecast["pred_Ridge"].iloc[0])
        pred_rf_fore = float(preds_forecast["pred_RF"].iloc[0])

        # Try to compute OOF MAE from artifacts
        oof_path = os.path.join(models_dir, "oof_with_meta_final.csv")
        if os.path.exists(oof_path):
            try:
                oof_meta = pd.read_csv(oof_path)
                if "pred_meta_final" in oof_meta.columns and "Yield_kg_ha" in oof_meta.columns:
                    mask_oof = oof_meta[["pred_meta_final", "Yield_kg_ha"]].notnull().all(axis=1)
                    if mask_oof.sum() >= 3:
                        from sklearn.metrics import mean_absolute_error
                        oof_mae = mean_absolute_error(oof_meta.loc[mask_oof, "Yield_kg_ha"], oof_meta.loc[mask_oof, "pred_meta_final"])
            except Exception:
                oof_mae = None

        # Try to read manifest for an OOF R2 hint
        try:
            manifest_path = os.path.join(models_dir, "manifest.txt")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as mf:
                    manifest_txt = mf.read()
                import re
                m = re.search(r"R2[:=]?\s*([0-9]*\.?[0-9]+)", manifest_txt)
                if m:
                    r2 = float(m.group(1))
        except Exception:
            r2 = r2 if not np.isnan(r2) else 0.0

    except Exception as e:
        logger.exception("Model artifacts present but prediction failed; falling back to regression. Error: %s", e)
        ARTIFACTS_AVAILABLE = False

# Fallback: in-app linear regression if artifacts not available
if not ARTIFACTS_AVAILABLE:
    try:
        X = merged[["Annual_Precip_mm", "Annual_Tmean_C"]].values.astype(float)
        y = merged["Yield"].values.astype(float)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        beta, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        intercept, coef_precip, coef_temp = beta
        y_pred = X_with_intercept @ beta
        r2 = r2_score(y, y_pred)

        resid = y - y_pred
        n, p = X_with_intercept.shape[0], X_with_intercept.shape[1]
        sigma2 = np.sum(resid**2) / max(n - p, 1)
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)

        x0 = np.array([1.0, seasonal_precip_proxy, seasonal_temp_proxy])
        predicted_model_y = float(x0 @ beta)

    except Exception as e:
        logger.exception("Fallback regression failed: %s", e)
        predicted_model_y = np.nan
        r2 = np.nan

# --- Conservative calibration and optimism limits ---
# Parameters (tunable)
MAX_PCT_DISPLAY = 30.0            # maximum percent change shown to users (absolute)
MIN_ALPHA = 0.2                   # minimum weight on model when confidence is very low
MAX_ALPHA = 0.9                   # maximum weight on model when confidence is high
OOF_MAE_TRUST_SCALE = 1.0         # scale factor for OOF MAE when computing alpha

# 1) Determine a confidence score in [0,1]
conf_score = 0.0
if np.isfinite(r2) and not np.isnan(r2):
    conf_score = float(np.clip(r2, 0.0, 1.0))
else:
    if np.isfinite(pi_width_pct) and pi_width_pct > 0:
        conf_score = float(np.clip(1.0 - (pi_width_pct / 200.0), 0.0, 1.0))

# 2) Compute alpha weight for shrinkage toward baseline (median)
alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * conf_score
if oof_mae is not None and hist_median_yield > 0:
    rel_mae = oof_mae / max(hist_median_yield, 1e-6)
    alpha = alpha * float(np.clip(1.0 - rel_mae * OOF_MAE_TRUST_SCALE, 0.0, 1.0))

# 3) Convert model prediction to display units and compute baseline_display
pred_model_train_units = predicted_model_y  # model output (kg/ha)
if unit_choice == "lb/acre":
    pred_model_display = pred_model_train_units * KG_PER_HA_TO_LB_PER_ACRE
else:
    pred_model_display = pred_model_train_units

# Heuristic baseline conversion: if historical yields are large (>1000) assume kg/ha
baseline = hist_median_yield
baseline_display = baseline
if unit_choice == "lb/acre":
    if baseline > 1000:
        baseline_display = baseline * KG_PER_HA_TO_LB_PER_ACRE

# 4) Shrink prediction toward baseline_display
final_display = alpha * pred_model_display + (1.0 - alpha) * baseline_display

# 5) Cap percent change shown to user
raw_pct_change = (final_display - baseline_display) / max(baseline_display, 1e-6) * 100.0
pct_change_capped = float(np.clip(raw_pct_change, -MAX_PCT_DISPLAY, MAX_PCT_DISPLAY))

# 6) Recompute PI conservatively
if oof_mae is not None:
    oof_mae_display = oof_mae * (KG_PER_HA_TO_LB_PER_ACRE if unit_choice == "lb/acre" else 1.0)
    pi_half = 1.96 * oof_mae_display
else:
    # ensemble spread proxy (use available preds if present)
    ensemble_vals = np.array([v for v in [pred_ridge_fore, pred_rf_fore, predicted_model_y] if not np.isnan(v)])
    if ensemble_vals.size > 0:
        ensemble_vals_display = ensemble_vals * (KG_PER_HA_TO_LB_PER_ACRE if unit_choice == "lb/acre" else 1.0)
        ensemble_std = float(np.nanstd(ensemble_vals_display))
    else:
        ensemble_std = 0.05 * max(abs(final_display), 1.0)
    pi_half = 1.96 * max(ensemble_std, 0.05 * max(abs(final_display), 1.0))

# Inflate PI when confidence is low
pi_inflation = 1.0 + (1.0 - conf_score) * 1.5  # up to 2.5x when conf_score=0
pi_low = final_display - pi_half * pi_inflation
pi_high = final_display + pi_half * pi_inflation

# 7) Clamp plotted forecast point to a reasonable envelope
hist_vals = merged["Yield"].values
hist_y_display = hist_vals
if unit_choice == "lb/acre" and np.nanmedian(hist_vals) > 1000:
    hist_y_display = hist_vals * KG_PER_HA_TO_LB_PER_ACRE
min_env = np.nanmin(hist_y_display) * 0.5
max_env = np.nanmax(hist_y_display) * 1.5
final_display_clamped = float(np.clip(final_display, min_env, max_env))

# Use final_display_clamped for plotting and final messages
display_predicted = final_display_clamped
pct_change_capped = float(np.clip(pct_change_capped, -MAX_PCT_DISPLAY, MAX_PCT_DISPLAY))

# Confidence label
if conf_score >= 0.5 and (pi_high - pi_low) / max(abs(display_predicted), 1e-6) * 100 < 30:
    confidence = "High"
    conf_color = "green"
elif conf_score >= 0.3 or (pi_high - pi_low) / max(abs(display_predicted), 1e-6) * 100 < 60:
    confidence = "Medium"
    conf_color = "orange"
else:
    confidence = "Low"
    conf_color = "red"

# Advisory wording (bounded and farmer-friendly)
advice_lines = []
if pct_change_capped <= -yield_drop_pct_threshold:
    advice_lines.append("Model indicates modestly lower yield potential than recent seasons. Prioritize scouting and consider irrigation where feasible.")
elif pct_change_capped >= yield_drop_pct_threshold:
    advice_lines.append("Model indicates modestly higher yield potential than recent seasons. Consider targeted top‑dress fertilizer and timely operations.")
else:
    advice_lines.append("Yield expected near typical levels. Continue standard management and monitor short‑term forecasts for operations.")

seven_day_precip = float(np.nansum(forecast_df["Precipitation"])) if not forecast_df.empty else 0.0
hot_days = int(np.sum((forecast_df["Temp_Max"] >= 30).astype(int))) if not forecast_df.empty else 0
if seven_day_precip >= precip_extreme_mm:
    advice_lines.append(f"Short-term: heavy rain expected ({seven_day_precip:.1f} mm over 7 days). Delay spraying and heavy fieldwork.")
if hot_days >= heat_days_threshold:
    advice_lines.append(f"Short-term: {hot_days} hot day(s) forecast (>30°C). Monitor for heat stress and consider irrigation if available.")
if seven_day_precip < 1 and hot_days == 0 and abs(pct_change_capped) < yield_drop_pct_threshold:
    advice_lines.append("Short-term: no major weather hazards in the next 7 days. Proceed with planned field operations.")

# Decision rules
decision_rules = []
if pct_change_capped <= -yield_drop_pct_threshold:
    decision_rules.append("If yield risk: prioritize high-value fields for irrigation; delay expensive inputs.")
if pct_change_capped >= yield_drop_pct_threshold:
    decision_rules.append("If yield potential: consider timely top-up fertilizer on responsive fields.")
if seven_day_precip >= precip_extreme_mm:
    decision_rules.append("If heavy rain forecast: postpone herbicide/pesticide application for 48–72 hours after rain.")
if hot_days >= heat_days_threshold:
    decision_rules.append("If heat stress likely: ensure irrigation scheduling targets critical growth stages.")

# Visualization: scatter + forecast point (seasonal proxy)
st.header(f"{commodity} — Yield vs Climate (Farmer‑ready)")

mean_precip = merged["Annual_Precip_mm"].mean()
mean_temp = merged["Annual_Tmean_C"].mean()

xt = np.linspace(merged["Annual_Tmean_C"].min(), merged["Annual_Tmean_C"].max(), 200)
xp = np.linspace(merged["Annual_Precip_mm"].min(), merged["Annual_Precip_mm"].max(), 200)

# Attempt to compute fallback regression lines for context
y_temp_pred = np.full_like(xt, np.nan)
y_prec_pred = np.full_like(xp, np.nan)
try:
    X = merged[["Annual_Precip_mm", "Annual_Tmean_C"]].values.astype(float)
    y = merged["Yield"].values.astype(float)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    y_temp_pred = beta[0] + beta[2] * xt + beta[1] * mean_precip
    y_prec_pred = beta[0] + beta[1] * xp + beta[2] * mean_temp
except Exception:
    pass

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Temperature panel
ax = axes[0]
# Prepare historical Y values in display units
hist_y_vals = merged["Yield"].values
hist_y_display = hist_y_vals
if unit_choice == "lb/acre" and np.nanmedian(hist_y_vals) > 1000:
    hist_y_display = hist_y_vals * KG_PER_HA_TO_LB_PER_ACRE

ax.scatter(merged["Annual_Tmean_C"], hist_y_display, color="tomato", alpha=0.8, label="Observed (historical)")
if not np.isnan(y_temp_pred).all():
    y_temp_pred_display = y_temp_pred
    if unit_choice == "lb/acre" and np.nanmedian(hist_y_vals) > 1000:
        y_temp_pred_display = y_temp_pred * KG_PER_HA_TO_LB_PER_ACRE
    ax.plot(xt, y_temp_pred_display, color="black", linestyle="--", label="Historical regression (precip at mean)")

ax.scatter([seasonal_temp_proxy], [display_predicted], color="purple", s=120, marker="X", label="Forecast (seasonal proxy)")
ax.axvline(seasonal_temp_proxy, color="purple", linestyle=":", alpha=0.7)
ax.set_xlabel("Annual Mean Temp (°C)")
ax.set_ylabel(f"Yield ({unit_choice})")
ax.set_title("Yield vs Temperature (precip at historical mean)")
ax.legend(loc="upper left")
ymin = min(np.nanmin(hist_y_display) * 0.9, display_predicted * 0.9)
ymax = max(np.nanmax(hist_y_display) * 1.1, display_predicted * 1.1)
ax.set_ylim(ymin, ymax)
ax.set_xlim(min(merged["Annual_Tmean_C"].min(), seasonal_temp_proxy) - 1, max(merged["Annual_Tmean_C"].max(), seasonal_temp_proxy) + 1)
ax.annotate("Forecast (seasonal proxy)", xy=(seasonal_temp_proxy, display_predicted),
            xytext=(seasonal_temp_proxy + 0.5, display_predicted * 1.02),
            arrowprops=dict(arrowstyle="->", color="purple"), fontsize=9)

# Precipitation panel
ax = axes[1]
ax.scatter(merged["Annual_Precip_mm"], hist_y_display, color="steelblue", alpha=0.8, label="Observed (historical)")
if not np.isnan(y_prec_pred).all():
    y_prec_pred_display = y_prec_pred
    if unit_choice == "lb/acre" and np.nanmedian(hist_y_vals) > 1000:
        y_prec_pred_display = y_prec_pred * KG_PER_HA_TO_LB_PER_ACRE
    ax.plot(xp, y_prec_pred_display, color="black", linestyle="--", label="Historical regression (temp at mean)")

ax.scatter([seasonal_precip_proxy], [display_predicted], color="purple", s=120, marker="X", label="Forecast (seasonal proxy)")
ax.axvline(seasonal_precip_proxy, color="purple", linestyle=":", alpha=0.7)
ax.set_xlabel("Annual Precipitation (mm)")
ax.set_ylabel(f"Yield ({unit_choice})")
ax.set_title("Yield vs Precipitation (temp at historical mean)")
ax.legend(loc="upper left")
ymin = min(np.nanmin(hist_y_display) * 0.9, display_predicted * 0.9)
ymax = max(np.nanmax(hist_y_display) * 1.1, display_predicted * 1.1)
ax.set_ylim(ymin, ymax)
ax.set_xlim(min(merged["Annual_Precip_mm"].min(), seasonal_precip_proxy) - 10, max(merged["Annual_Precip_mm"].max(), seasonal_precip_proxy) + 10)
ax.annotate("Forecast (seasonal proxy)", xy=(seasonal_precip_proxy, display_predicted),
            xytext=(seasonal_precip_proxy + 10, display_predicted * 1.02),
            arrowprops=dict(arrowstyle="->", color="purple"), fontsize=9)

st.pyplot(fig)

# Quick advisory card
st.markdown("### Quick advisory")
col1, col2 = st.columns([2, 1])

with col1:
    if np.isfinite(display_predicted):
        st.markdown(f"**Predicted yield:** **{display_predicted:.0f} {unit_choice}**")
        st.markdown(f"**95% prediction interval:** {pi_low:.0f} — {pi_high:.0f} {unit_choice}")
    else:
        st.markdown("**Predicted yield:** unavailable")
    st.markdown(f"**Confidence:** **{confidence}**")
    if confidence == "Low":
        st.markdown("_Model confidence is low for this commodity; treat predictions as a seasonal signal, not a guarantee._")
    st.markdown("**Actionable summary:**")
    for line in advice_lines:
        st.write(f"- {line}")

with col2:
    st.markdown("**Decision rules**")
    if decision_rules:
        for rule in decision_rules:
            st.write(f"- {rule}")
    else:
        st.write("- No specific decision rules triggered.")

# 7-day forecast visualization and CSV download
st.header("7‑day weather forecast (Winnipeg)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(forecast_df["Date"], forecast_df["Temp_Max"], label="Max Temp", color="red", marker="o")
ax.plot(forecast_df["Date"], forecast_df["Temp_Min"], label="Min Temp", color="blue", marker="o")
ax.bar(forecast_df["Date"], forecast_df["Precipitation"], label="Precipitation (mm)", alpha=0.3, color="green")
ax.legend()
ax.set_title("7‑day forecast (Winnipeg)")
st.pyplot(fig)

st.download_button(
    label="Download forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv",
)

# NDVI (optional)
st.header("Satellite NDVI (optional)")
st.caption("NDVI shows crop health and flood extent. Use as a visual complement to the advisory.")
with st.spinner("Querying Sentinel‑2 and computing NDVI..."):
    try:
        ndvi, stats = compute_ndvi_map_mb(lat=49.9, lon=-97.2,
                                          date_start=str(date_start),
                                          date_end=str(date_end),
                                          cloud_max=cloud_max)
        st.write(f"NDVI stats — min: {stats['min']:.3f}, max: {stats['max']:.3f}, mean: {stats['mean']:.3f}")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(ndvi, cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label="NDVI")
        ax.set_title("Sentinel‑2 NDVI (Crop Health / Flood Extent)")
        st.pyplot(fig)
    except Exception as e:
        logger.exception("NDVI query failed")
        st.write("NDVI unavailable:", e)

# Interpretation guide
st.markdown("""
**NDVI interpretation guide**
- **0.6 – 1.0 (dark green):** Healthy, dense vegetation  
- **0.2 – 0.6 (yellow‑green):** Moderate vegetation, possible stress  
- **0.0 – 0.2 (orange):** Bare soil or sparse vegetation  
- **< 0.0 (blue/black):** Water, flooding, or non‑vegetated surfaces  

**How to use NDVI with the advisory**
- Use NDVI to confirm or localize the advisory: low NDVI in areas flagged by the forecast suggests prioritizing irrigation there.  
- High NDVI with a negative yield outlook may indicate nutrient or pest issues rather than climate; investigate those fields first.  
- Combine NDVI with the 7‑day forecast before scheduling field operations.
""")
# End of dashboard
