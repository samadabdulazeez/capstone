# dashboard.py
"""
Farmer-ready Manitoba Crop Dashboard (full file)

- Regression-based yield vs climate plots with prediction intervals
- Annotates forecasted climate point and predicted yield (with 95% PI)
- Provides a concise, actionable advisory and confidence indicator (always visible)
- Optional Sentinel-2 NDVI visualization
"""
import streamlit as st
import matplotlib.pyplot as plt
import datetime
import numpy as np
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math

from utils.data_loader import (
    list_statscan_commodities_mb,
    load_statscan_yields_mb,
    load_mb_weather,
    load_openmeteo_forecast,
)
from utils.ndvi import compute_ndvi_map_mb

# --- Page config and logger ---
st.set_page_config(page_title="Manitoba Farmer Dashboard", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# --- Sidebar controls ---
st.sidebar.title("Manitoba Farmer Dashboard")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

commodities = list_statscan_commodities_mb()
commodity = st.sidebar.selectbox("Commodity yield series (StatsCan)", commodities)
weather_source = st.sidebar.radio("Weather source", ["obs", "reanalysis", "union"], index=2)

st.sidebar.markdown("### Sentinel‑2 NDVI (optional)")
today = datetime.date.today()
date_start = st.sidebar.date_input("Start date", datetime.date(today.year, 5, 1), max_value=today)
date_end = st.sidebar.date_input("End date", datetime.date(today.year, 9, 30), max_value=today)
cloud_max = st.sidebar.slider("Max cloud cover (%)", 0, 100, 30)

# Quick user-tunable thresholds for advisory
st.sidebar.markdown("### Advisory thresholds")
yield_drop_pct_threshold = st.sidebar.slider("Yield drop threshold (%)", 5, 30, 10)
precip_extreme_mm = st.sidebar.slider("7‑day heavy rain threshold (mm)", 10, 30, 20)
heat_days_threshold = st.sidebar.slider("7‑day hot days threshold (>30°C)", 0, 3, 1)

# --- Helper: render farmer advisory (always shown) ---
def render_farmer_advisory(annual_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """
    Render a plain-language farmer advisory based on forecast vs historical annual means.
    Always shown on the dashboard.
    """
    st.subheader("Farmer Advisory")
    st.markdown("The dashboard always shows plain-language guidance based on forecasted precipitation and temperature compared to historical averages:")

    # Compute simple historical means (fallbacks if data missing)
    hist_precip_mean = float(annual_df["Annual_Precip_mm"].mean()) if (annual_df is not None and not annual_df.empty) else math.nan
    hist_temp_mean = float(annual_df["Annual_Tmean_C"].mean()) if (annual_df is not None and not annual_df.empty) else math.nan

    # Aggregate 7-day forecast
    seven_day_precip = float(np.nansum(forecast_df["Precipitation"])) if (forecast_df is not None and not forecast_df.empty) else math.nan
    seven_day_temp_mean = float(((forecast_df["Temp_Max"] + forecast_df["Temp_Min"]) / 2.0).mean()) if (forecast_df is not None and not forecast_df.empty) else math.nan

    # Compare with simple thresholds
    precip_diff = None if math.isnan(hist_precip_mean) or math.isnan(seven_day_precip) else seven_day_precip - hist_precip_mean
    temp_diff = None if math.isnan(hist_temp_mean) or math.isnan(seven_day_temp_mean) else seven_day_temp_mean - hist_temp_mean

    # Build advisory bullets
    bullets = []
    # Precipitation guidance
    if precip_diff is None:
        bullets.append("• **Precipitation:** Historical or forecast data missing; check data sources.")
    else:
        if precip_diff < -20:
            bullets.append("• ⚠️ **Drier than average** forecast — consider irrigation or drought‑tolerant varieties.")
        elif precip_diff > 20:
            bullets.append("• 🌧️ **Wetter than average** forecast — watch for waterlogging and consider drainage.")
        else:
            bullets.append("• ✅ **Precipitation near average** — no major precipitation-related adjustments expected.")

    # Temperature guidance
    if temp_diff is None:
        bullets.append("• **Temperature:** Historical or forecast data missing; check data sources.")
    else:
        if temp_diff > 2:
            bullets.append("• 🔥 **Warmer than average** forecast — monitor for heat stress and adjust planting or irrigation timing.")
        elif temp_diff < -2:
            bullets.append("• ❄️ **Cooler than average** forecast — growth may be slower; consider later planting if timing allows.")
        else:
            bullets.append("• ✅ **Temperature near average** — standard temperature-related practices apply.")

    # Short-term operational tips from forecast totals
    if not math.isnan(seven_day_precip):
        if seven_day_precip >= 20:
            bullets.append(f"• Short-term: **{seven_day_precip:.1f} mm** expected over 7 days — delay spraying and heavy fieldwork.")
        elif seven_day_precip < 2:
            bullets.append(f"• Short-term: **{seven_day_precip:.1f} mm** expected over 7 days — low rainfall; check soil moisture before seeding or top-dress.")
    else:
        bullets.append("• Short-term forecast unavailable.")

    # Render bullets
    for b in bullets:
        st.markdown(b)

    # Small note about limitations
    st.markdown(
        "_Note: This advisory is a simple, practical guide based on short-term forecasts and historical annual averages. "
        "It does not replace local scouting or agronomic advice. For field-level decisions, combine this guidance with soil, crop stage, and local observations._"
    )

# --- Header ---
st.header("Manitoba Agriculture Insights")

# --- Load data ---
try:
    yields_df = load_statscan_yields_mb(commodity)
    annual_df = load_mb_weather(weather_source)
except Exception as e:
    logger.exception("Error loading data")
    st.error(f"Error loading data: {e}")
    st.stop()

# Ensure Year is integer and drop bad rows
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

# --- Forecast (Open-Meteo) ---
try:
    forecast_df = load_openmeteo_forecast(lat=49.9, lon=-97.2)
except Exception as e:
    logger.exception("Forecast fetch failed")
    forecast_df = pd.DataFrame(columns=["Date", "Temp_Max", "Temp_Min", "Precipitation"])

# Always render the plain-language advisory
render_farmer_advisory(annual_df, forecast_df)

# Merge on Year
merged = yields_df.merge(annual_df, on="Year", how="inner")

if merged.empty:
    st.warning("No overlapping years between yield and climate data. Try a different commodity or weather source.")
    # Show simple plots for inspection
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

# --- Fit linear model (Yield ~ Precip + Tmean) ---
X = merged[["Annual_Precip_mm", "Annual_Tmean_C"]].values.astype(float)
y = merged["Yield"].values.astype(float)
X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])  # shape (n, 3)

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Residual standard error (sigma)
resid = y - y_pred
n, p = X_with_intercept.shape[0], X_with_intercept.shape[1]
sigma2 = np.sum(resid**2) / max(n - p, 1)
sigma = np.sqrt(sigma2)

# Covariance matrix of coefficients
XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
cov_beta = sigma2 * XtX_inv

# Aggregate forecast to seasonal-like totals/means for prediction
avg_precip = float(np.nansum(forecast_df["Precipitation"])) if not forecast_df.empty else 0.0
avg_temp = float(((forecast_df["Temp_Max"] + forecast_df["Temp_Min"]) / 2.0).mean()) if not forecast_df.empty else merged["Annual_Tmean_C"].mean()

# Build x0 for prediction (intercept, precip, tmean)
x0 = np.array([1.0, avg_precip, avg_temp])
predicted_y = float(model.predict([[avg_precip, avg_temp]])[0])

# Prediction interval for the new point
pred_var = sigma2 * (1.0 + x0.T @ XtX_inv @ x0)
pred_se = np.sqrt(pred_var)
z = 1.96
pi_low = predicted_y - z * pred_se
pi_high = predicted_y + z * pred_se

# Historical mean yield for context
hist_mean_yield = float(np.nanmean(merged["Yield"]))
hist_std_yield = float(np.nanstd(merged["Yield"]))

# Confidence indicator heuristic
pi_width_pct = (pi_high - pi_low) / max(abs(predicted_y), 1e-6) * 100
if r2 >= 0.5 and pi_width_pct < 30:
    confidence = "High"
    conf_color = "green"
elif r2 >= 0.3 or pi_width_pct < 60:
    confidence = "Medium"
    conf_color = "orange"
else:
    confidence = "Low"
    conf_color = "red"

# Actionable advisory generation
advice_lines = []
pct_change = (predicted_y - hist_mean_yield) / max(hist_mean_yield, 1e-6) * 100
if pct_change <= -yield_drop_pct_threshold:
    advice_lines.append(f"Expected yield **{pct_change:.0f}% below** historical mean. Consider prioritizing irrigation and reducing non-essential input spend.")
elif pct_change >= yield_drop_pct_threshold:
    advice_lines.append(f"Expected yield **{pct_change:.0f}% above** historical mean. Consider optimizing input use to capture higher yield potential.")
else:
    advice_lines.append("Expected yield close to historical average. Maintain standard management practices.")

seven_day_precip = float(np.nansum(forecast_df["Precipitation"])) if not forecast_df.empty else 0.0
hot_days = int(np.sum((forecast_df["Temp_Max"] >= 30).astype(int))) if not forecast_df.empty else 0
if seven_day_precip >= precip_extreme_mm:
    advice_lines.append(f"Short-term: heavy rain expected ({seven_day_precip:.1f} mm over 7 days). Delay field operations and spraying.")
if hot_days >= heat_days_threshold:
    advice_lines.append(f"Short-term: {hot_days} hot day(s) forecast (>30°C). Monitor for heat stress during flowering and consider irrigation if available.")
if seven_day_precip < 1 and hot_days == 0 and abs(pct_change) < yield_drop_pct_threshold:
    advice_lines.append("Short-term: no major weather hazards in the next 7 days. Proceed with planned field operations.")

# Decision rules
decision_rules = []
if pct_change <= -yield_drop_pct_threshold:
    decision_rules.append("If yield drop > threshold: prioritize high-value fields for irrigation; delay expensive inputs.")
if pct_change >= yield_drop_pct_threshold:
    decision_rules.append("If yield increase > threshold: consider timely top-up fertilizer on responsive fields.")
if seven_day_precip >= precip_extreme_mm:
    decision_rules.append("If heavy rain forecast: postpone herbicide/pesticide application for 48–72 hours after rain.")
if hot_days >= heat_days_threshold:
    decision_rules.append("If heat stress likely: ensure irrigation scheduling targets critical growth stages.")

# --- Visualization: two-panel with regression, PI band, and forecasted point ---
st.header(f"{commodity} — Yield vs Climate (Farmer‑ready)")

mean_precip = merged["Annual_Precip_mm"].mean()
mean_temp = merged["Annual_Tmean_C"].mean()

# Temperature plot arrays
xt = np.linspace(merged["Annual_Tmean_C"].min(), merged["Annual_Tmean_C"].max(), 200)
X_temp_plot = np.column_stack([np.ones_like(xt), np.full_like(xt, mean_precip), xt])
y_temp_pred = X_temp_plot @ np.concatenate([[model.intercept_], model.coef_])
se_temp = np.sqrt(sigma2 * (1.0 + np.sum(X_temp_plot @ XtX_inv * X_temp_plot, axis=1)))
temp_low = y_temp_pred - z * se_temp
temp_high = y_temp_pred + z * se_temp

# Precipitation plot arrays
xp = np.linspace(merged["Annual_Precip_mm"].min(), merged["Annual_Precip_mm"].max(), 200)
X_prec_plot = np.column_stack([np.ones_like(xp), xp, np.full_like(xp, mean_temp)])
y_prec_pred = X_prec_plot @ np.concatenate([[model.intercept_], model.coef_])
se_prec = np.sqrt(sigma2 * (1.0 + np.sum(X_prec_plot @ XtX_inv * X_prec_plot, axis=1)))
prec_low = y_prec_pred - z * se_prec
prec_high = y_prec_pred + z * se_prec

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Temperature panel
ax = axes[0]
ax.scatter(merged["Annual_Tmean_C"], merged["Yield"], color="tomato", alpha=0.8, label="Observed")
ax.plot(xt, y_temp_pred, color="black", linestyle="--", label="Regression (precip at mean)")
ax.fill_between(xt, temp_low, temp_high, color="black", alpha=0.12, label="95% PI")
ax.scatter([avg_temp], [predicted_y], color="purple", s=120, marker="X", label="Forecasted climate → predicted yield")
ax.axvline(avg_temp, color="purple", linestyle=":", alpha=0.7)
ax.set_xlabel("Annual Mean Temp (°C)")
ax.set_ylabel(f"Yield ({yields_df['UOM'].iloc[0] if 'UOM' in yields_df.columns and not yields_df.empty else ''})")
ax.set_title("Yield vs Temperature (precip at historical mean)")
ax.legend(loc="upper left")
ax.text(0.02, 0.95, f"R² = {r2:.2f}\nPredicted yield = {predicted_y:.1f}\n95% PI: [{pi_low:.1f}, {pi_high:.1f}]\nConfidence: {confidence}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(facecolor=conf_color, alpha=0.08, edgecolor='none'))

# Precipitation panel
ax = axes[1]
ax.scatter(merged["Annual_Precip_mm"], merged["Yield"], color="steelblue", alpha=0.8, label="Observed")
ax.plot(xp, y_prec_pred, color="black", linestyle="--", label="Regression (temp at mean)")
ax.fill_between(xp, prec_low, prec_high, color="black", alpha=0.12, label="95% PI")
ax.scatter([avg_precip], [predicted_y], color="purple", s=120, marker="X", label="Forecasted climate → predicted yield")
ax.axvline(avg_precip, color="purple", linestyle=":", alpha=0.7)
ax.set_xlabel("Annual Precipitation (mm)")
ax.set_ylabel(f"Yield ({yields_df['UOM'].iloc[0] if 'UOM' in yields_df.columns and not yields_df.empty else ''})")
ax.set_title("Yield vs Precipitation (temp at historical mean)")
ax.legend(loc="upper left")

st.pyplot(fig)

# --- Farmer-facing summary card ---
st.markdown("### Quick advisory")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"**Predicted yield:** **{predicted_y:.1f} {yields_df['UOM'].iloc[0] if 'UOM' in yields_df.columns and not yields_df.empty else ''}**")
    st.markdown(f"**95% prediction interval:** {pi_low:.1f} — {pi_high:.1f}")
    st.markdown(f"**Confidence:** {confidence}  (R² = {r2:.2f}, PI width = {pi_width_pct:.0f}%)")
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

# --- 7-day forecast visualization and CSV download ---
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

# --- NDVI (optional) ---
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

# Interpretation guide (final notes for the farmer)
st.markdown("""
**NDVI interpretation guide**
- **0.6 – 1.0 (dark green):** Healthy, dense vegetation  
- **0.2 – 0.6 (yellow‑green):** Moderate vegetation, possible stress  
- **0.0 – 0.2 (orange):** Bare soil or sparse vegetation  
- **< 0.0 (blue/black):** Water, flooding, or non‑vegetated surfaces  

**How to use NDVI with the advisory**
- Use NDVI to confirm or localize the advisory: low NDVI in areas flagged by the forecast (drought/heat) suggests prioritizing irrigation there.  
- High NDVI with a negative yield outlook may indicate nutrient or pest issues rather than climate; investigate those fields first.  
- Combine NDVI with the 7‑day forecast before scheduling field operations (spraying, seeding, harvest).
""")

# End of dashboard
