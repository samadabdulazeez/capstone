```markdown
# Manitoba Crop Dashboard 🌾

A Streamlit-based dashboard for monitoring crop yields, climate conditions, and satellite-derived vegetation health in Manitoba.  
This project integrates **StatsCan yield data**, **ECCC climate data**, **Open‑Meteo forecasts**, and **Sentinel‑2 NDVI imagery** from Microsoft’s Planetary Computer.

---

## Features

- **StatsCan yields**: Annual crop yield series for Manitoba commodities.  
- **ECCC climate**: Daily weather data aggregated to annual summaries.  
- **Regression analysis**: Correlation and regression between yields and climate (annual mean temperature and annual precipitation).  
- **Forecast-based prediction**: Uses a 7‑day Open‑Meteo forecast to produce a quick, transparent yield estimate.  
- **Farmer advisory**: Plain‑language guidance based on forecast vs historical averages (always shown).  
- **Satellite NDVI**: Sentinel‑2 imagery processed to NDVI maps for crop health and flood extent.  
- **Interactive dashboard**: Built with Streamlit, easy to run locally or deploy.

---

## Data sources

The dashboard uses the following raw CSV files hosted on GitHub (Manitoba weather data):

```python
files = {
    "Weather Obs.csv": "https://raw.githubusercontent.com/vlyubchich/Manitoba/master/data/Weather%20Obs.csv",
    "Weather Reanalysis.csv": "https://raw.githubusercontent.com/vlyubchich/Manitoba/master/data/Weather%20Reanalysis.csv"
}
```

Other data sources:
- **StatsCan yields**: downloaded from the official StatsCan table (zipped CSV).  
- **Open‑Meteo**: 7‑day forecast API used for short-term predictions.  
- **Sentinel‑2 (Planetary Computer)**: NDVI imagery via Microsoft Planetary Computer.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/.../manitoba-crop-dashboard.git
   cd manitoba-crop-dashboard
   ```

2. Create and activate a Python environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Dashboard

Run Streamlit from the project root:

```bash
streamlit run dashboard.py
```

Open the provided local URL in your browser (usually `http://localhost:8501`).

---

## Quick walkthrough

1. **Pick a crop** from the sidebar dropdown (only crops with Manitoba yield data are shown).  
2. **Choose the weather source**: use `union` to combine observations and reanalysis for the longest climate record.  
3. **Optional NDVI window**: set start/end dates and cloud threshold for Sentinel‑2 imagery.  
4. **Read the Quick Advisory**: a concise, plain‑language summary (predicted yield, prediction interval, confidence, and short operational tips).  
5. **Inspect the two-panel plot**: left = yield vs temperature, right = yield vs precipitation. The purple X marks the forecasted climate and the model’s predicted yield.  
6. **Check the 7‑day forecast** for immediate operational decisions (spraying, irrigation, fieldwork).  
7. **Use NDVI** to localize stressed or flooded areas and confirm advisory signals.

---

## Quick advisory and plot explanation

**What the Quick Advisory shows**
- **Predicted yield**: a single, transparent estimate derived from a simple linear model that uses annual temperature and precipitation plus the next 7 days of forecasted weather.  
- **95% prediction interval (PI)**: a range that expresses uncertainty around the predicted yield; a wide PI means low confidence.  
- **Confidence indicator**: a simple High / Medium / Low label based on model fit (R²) and PI width.  
- **Actionable tips**: short, specific recommendations (for example: delay spraying after heavy rain; prioritize irrigation during heat spells).  
- **Decision rules**: clear if/then rules tied to thresholds (e.g., if predicted yield drops > X% then prioritize irrigation).

**How to read the two-panel regression plot**
- **Left panel (Yield vs Temperature)**: each point is a historical year; the dashed line is the regression holding precipitation at its historical mean. The shaded band is the 95% PI. The purple X is the forecasted seasonal temperature (from the 7‑day forecast aggregated) and the model’s predicted yield for that climate.  
- **Right panel (Yield vs Precipitation)**: same idea but varying precipitation and holding temperature at its historical mean. The purple X shows forecasted precipitation and predicted yield.  
- **R² and PI**: R² indicates how much of yield variability the model explains; PI width shows prediction uncertainty. Use both to judge how much weight to give the prediction.

**Practical interpretation**
- **7‑day forecast** is actionable for operations (spraying, fieldwork, short irrigation decisions).  
- **Predicted yield** is a probabilistic seasonal signal — useful for planning and prioritizing but not a guarantee.  
- **Combine signals**: use NDVI + forecast + advisory to localize actions (e.g., irrigate fields with low NDVI and drought signal).

---

## Farmer Advisory (always shown)

The dashboard always shows **plain-language guidance** based on forecasted precipitation and temperature compared to historical averages:

- ⚠️ **Drier than average** → consider irrigation or drought‑tolerant varieties.  
- 🌧️ **Wetter than average** → watch for waterlogging; consider drainage.  
- 🔥 **Warmer than average** → monitor for heat stress; adjust planting dates or irrigation timing.  
- ❄️ **Cooler than average** → growth may be slower; consider later planting if appropriate.  
- ✅ **Near average** → standard practices are likely suitable.

Short-term operational tips are also provided from the 7‑day forecast (for example: delay spraying after heavy rain; prioritize irrigation during heat spells).  
**Limitations:** this advisory is a practical guide based on short-term forecasts and historical annual averages; it does not replace local scouting or agronomic advice.

---

## NDVI Interpretation Guide

- **0.6 – 1.0 (dark green):** Healthy, dense vegetation  
- **0.2 – 0.6 (yellow‑green):** Moderate vegetation, possible stress  
- **0.0 – 0.2 (orange):** Bare soil or sparse vegetation  
- **< 0.0 (blue/black):** Water, flooding, or non‑vegetated surfaces

**How to use NDVI with the advisory**
- Use NDVI to confirm or localize the advisory: low NDVI in areas flagged by the forecast suggests prioritizing irrigation there.  
- High NDVI with a negative yield outlook may indicate nutrient or pest issues rather than climate; investigate those fields first.  
- Combine NDVI with the 7‑day forecast before scheduling field operations.

---

## Future Improvements

- **Better climate data**: integrate ERA5 or NOAA GHCN for continuous overlap with StatsCan yields.  
- **Crop‑specific advisories**: tailor guidance and thresholds to each commodity.  
- **Seasonal alignment**: match climate data to crop growing stages rather than calendar years.  
- **Field-level predictions**: allow farmers to select specific fields or soil classes.  
- **Deployment and mobile UI**: host on Streamlit Cloud or Azure and optimize for phones/tablets.  
- **Model improvements**: move from simple annual regressions to stage-aware models and include management variables.

---

## Notes

- Current ECCC data often has gaps, so regression predictions may not always be available. Use the `union` weather source to maximize overlap.  
- Farmer advisory guidance is always shown, even if regression cannot run.  
- Sentinel‑2 imagery requires internet access to Microsoft Planetary Computer.

---

## Authors

Developed as part of the **Manitoba Agri Capstone Project**.  
Powered by **Streamlit**, **Planetary Computer**, and open data sources.
```