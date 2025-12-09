import streamlit as st
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.data_loader import (
    list_statscan_commodities_mb,
    load_statscan_yields_mb,
    list_eccc_stations_mb,
    load_eccc_daily_annual_mb,
    load_openmeteo_forecast,
)
from utils.ndvi import compute_ndvi_map_mb

st.set_page_config(page_title="Manitoba Crop Dashboard", layout="wide")

st.sidebar.title("Manitoba Crop Dashboard")
st.sidebar.caption("Live data: StatsCan CSV + ECCC + Open‑Meteo + Planetary Computer")

# Refresh button
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

# Commodity + station lists (no loop)
commodities = list_statscan_commodities_mb()
stations = list_eccc_stations_mb()

commodity = st.sidebar.selectbox("Commodity yield series (StatsCan)", commodities, index=0)
station_name = st.sidebar.selectbox("ECCC station", [s["name"] for s in stations], index=0)
station = next(s for s in stations if s["name"] == station_name)
station_id, lat, lon = station["id"], station["lat"], station["lon"]

# Sentinel‑2 search window
st.sidebar.subheader("Sentinel‑2 imagery search")
today = datetime.date.today()
default_start = datetime.date(today.year, 5, 1)
default_end = datetime.date(today.year, 9, 30)
date_start = st.sidebar.date_input("Start date", default_start, max_value=today)
date_end = st.sidebar.date_input("End date", default_end, max_value=today)
cloud_max = st.sidebar.slider("Max cloud cover (%)", 0, 100, 30)

# Yield + climate data
st.header("Manitoba crop yields (StatsCan) vs annual climate (ECCC)")
yields_df = load_statscan_yields_mb(commodity)
annual_df = load_eccc_daily_annual_mb(station_id)

st.subheader(f"Yields — {commodity}")
st.dataframe(yields_df)
st.subheader(f"Annual climate — {station_name}")
st.dataframe(annual_df)

merged = yields_df.merge(annual_df, on="Year", how="inner")

if merged.empty:
    st.warning("No overlapping years between yield and climate data. Showing separate plots instead.")
    if not yields_df.empty:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(yields_df["Year"], yields_df["Yield"], marker="o", color="green")
        ax1.set_title(f"{commodity} yields in Manitoba")
        ax1.set_xlabel("Year")
        ax1.set_ylabel(f"Yield ({yields_df['UOM'].iloc[0]})")
        st.pyplot(fig1)
    if not annual_df.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(annual_df["Year"], annual_df["Annual_Tmean_C"], label="Mean Temp (°C)", color="blue")
        ax2.set_ylabel("Mean Temp (°C)", color="blue")
        ax2b = ax2.twinx()
        ax2b.plot(annual_df["Year"], annual_df["Annual_Precip_mm"], label="Precipitation (mm)", color="orange")
        ax2b.set_ylabel("Precipitation (mm)", color="orange")
        ax2.set_title(f"Annual climate — {station_name}")
        st.pyplot(fig2)
else:
    st.success("Yield and climate overlap found.")
    X, y = merged[["Annual_Precip_mm", "Annual_Tmean_C"]], merged["Yield"]
    model = LinearRegression().fit(X, y)
    forecast_df = load_openmeteo_forecast(lat, lon)
    avg_precip = forecast_df["Precipitation"].sum()
    avg_temp = (forecast_df["Temp_Max"] + forecast_df["Temp_Min"]).mean() / 2
    predicted_yield = model.predict([[avg_precip, avg_temp]])[0]
    st.subheader("Predicted Yield (based on forecasted climate)")
    st.write(f"Estimated yield for {commodity}: **{predicted_yield:.2f} {yields_df['UOM'].iloc[0]}**")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(merged["Annual_Precip_mm"], merged["Yield"], c="blue")
    ax[0].set_xlabel("Annual Precipitation (mm)")
    ax[0].set_ylabel("Yield")
    ax[0].set_title("Yield vs Precipitation")
    x_vals = np.linspace(merged["Annual_Precip_mm"].min(), merged["Annual_Precip_mm"].max(), 100)
    y_vals = model.predict(np.column_stack([x_vals, np.full_like(x_vals, merged["Annual_Tmean_C"].mean())]))
    ax[0].plot(x_vals, y_vals, color="black", linestyle="--")
    ax[1].scatter(merged["Annual_Tmean_C"], merged["Yield"], c="red")
    ax[1].set_xlabel("Annual Mean Temp (°C)")
    ax[1].set_ylabel("Yield")
    ax[1].set_title("Yield vs Temperature")
    x_vals = np.linspace(merged["Annual_Tmean_C"].min(), merged["Annual_Tmean_C"].max(), 100)
    y_vals = model.predict(np.column_stack([np.full_like(x_vals, merged["Annual_Precip_mm"].mean()), x_vals]))
    ax[1].plot(x_vals, y_vals, color="black", linestyle="--")
    st.pyplot(fig)

# Forecast
st.header("7‑day forecast (Open‑Meteo)")
forecast_df = load_openmeteo_forecast(lat, lon)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(forecast_df["Date"], forecast_df["Temp_Max"], label="Max Temp", color="red", marker="o")
ax.plot(forecast_df["Date"], forecast_df["Temp_Min"], label="Min Temp", color="blue", marker="o")
ax.bar(forecast_df["Date"], forecast_df["Precipitation"], label="Precipitation", alpha=0.3, color="green")
ax.legend()
ax.set_title("Daily forecast")
st.pyplot(fig)

st.download_button(
    label="Download forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv",
)

# Farmer advisory (always shown)
st.subheader("Farmer Advisory (based on forecasted climate)")
avg_precip = forecast_df["Precipitation"].sum()
avg_temp = (forecast_df["Temp_Max"] + forecast_df["Temp_Min"]).mean() / 2
if not annual_df.empty:
    hist_precip_mean = annual_df["Annual_Precip_mm"].mean()
    hist_temp_mean = annual_df["Annual_Tmean_C"].mean()
else:
    hist_precip_mean, hist_temp_mean = avg_precip, avg_temp
precip_diff, temp_diff = avg_precip - hist_precip_mean, avg_temp - hist_temp_mean
advice = []
if precip_diff < -20:
    advice.append("⚠️ Drier than average forecast. Consider irrigation or drought‑tolerant varieties.")
elif precip_diff > 20:
    advice.append("🌧️ Wetter than average forecast. Watch for waterlogging and consider drainage.")
if temp_diff > 2:
    advice.append("🔥 Warmer than average forecast. Monitor for heat stress and adjust planting dates.")
elif temp_diff < -2:
    advice.append("❄️ Cooler than average forecast. Growth may be slower; consider later planting.")
if not advice:
    advice.append("✅ Forecast conditions are close to average. Standard planting practices should be suitable.")
for tip in advice:
    st.write(tip)

# NDVI
st.header("Satellite NDVI (Sentinel‑2 L2A via Planetary Computer)")
st.caption("NDVI shows crop health and flood extent. High values = healthy vegetation, low/negative = water or stressed crops.")
with st.spinner("Querying Sentinel‑2 and computing NDVI..."):
    ndvi, stats = compute_ndvi_map_mb(lat=lat, lon=lon,
                                      date_start=str(date_start),
                                      date_end=str(date_end),
                                      cloud_max=cloud_max)
st.write(f"NDVI stats — min: {stats['min']:.3f}, max: {stats['max']:.3f}, mean: {stats['mean']:.3f}")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(im, ax=ax, label="NDVI")
ax.set_title("Sentinel‑2 NDVI (Crop Health / Flood Extent)")
st.markdown("""
**NDVI interpretation guide:**
- **0.6 – 1.0 (dark green):** Healthy, dense vegetation  
- **0.2 – 0.6 (yellow‑green):** Moderate vegetation, possible stress  
- **0.0 – 0.2 (orange):** Bare soil or sparse vegetation  
- **< 0.0 (blue/black):** Water, flooding, or non‑vegetated surfaces  
""")

st.pyplot(fig)