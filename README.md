Here’s a clear and complete **README.md** you can use for your project. It explains what the Manitoba Crop Dashboard does, how to set it up and run it, and outlines future improvements.

---

```markdown
# Manitoba Crop Dashboard 🌾

A Streamlit-based dashboard for monitoring crop yields, climate conditions, and satellite-derived vegetation health in Manitoba.  
This project integrates **StatsCan yield data**, **ECCC climate data**, **Open-Meteo forecasts**, and **Sentinel‑2 NDVI imagery** from Microsoft’s Planetary Computer.

---

## Features

- **StatsCan yields**: Annual crop yield series for Manitoba commodities.
- **ECCC climate**: Daily weather data aggregated to annual summaries.
- **Regression analysis**: Correlation and regression between yields and climate.
- **Forecast-based prediction**: Uses 7‑day weather forecasts to estimate yield.
- **Farmer advisory**: Plain‑language guidance based on forecast vs historical averages (always shown).
- **Satellite NDVI**: Sentinel‑2 imagery processed to NDVI maps for crop health and flood extent.
- **Interactive dashboard**: Built with Streamlit, easy to run locally or deploy.

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

## Project Structure

```
.
├── dashboard.py             # Main Streamlit app
├── utils/
│   ├── data_loader.py       # Functions for StatsCan, ECCC, Open-Meteo
│   └── ndvi.py              # Sentinel-2 NDVI processing
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Farmer Advisory

The dashboard always shows **plain-language guidance** based on forecasted precipitation and temperature compared to historical averages:

- ⚠️ Drier than average → consider irrigation or drought‑tolerant varieties.
- 🌧️ Wetter than average → watch for waterlogging, consider drainage.
- 🔥 Warmer than average → monitor for heat stress, adjust planting dates.
- ❄️ Cooler than average → slower growth, consider later planting.
- ✅ Near average → standard practices suitable.

---

## NDVI Interpretation Guide

- **0.6 – 1.0 (dark green):** Healthy, dense vegetation  
- **0.2 – 0.6 (yellow‑green):** Moderate vegetation, possible stress  
- **0.0 – 0.2 (orange):** Bare soil or sparse vegetation  
- **< 0.0 (blue/black):** Water, flooding, or non‑vegetated surfaces  

---

## Future Improvements

- **Better climate data**: ECCC stations are sparse; integrate ERA5 reanalysis or NOAA GHCN for continuous overlap with StatsCan yields.
- **Crop‑specific advisories**: Tailor guidance to each commodity (e.g., barley vs canola).
- **Seasonal alignment**: Match climate data to crop growing seasons instead of calendar years.
- **Deployment**: Host on Streamlit Cloud or Azure for easy farmer access.
- **Mobile optimization**: Improve layout for smartphones and tablets.
- **Data caching**: Optimize queries to reduce load times when fetching large datasets.

---

## Notes

- Current ECCC data often has gaps, so regression predictions may not always be available.  
- Farmer advisory guidance is always shown, even if regression cannot run.  
- Sentinel‑2 imagery requires internet access to Microsoft Planetary Computer.

---

## Authors

Developed as part of the **Manitoba Agri Capstone Project**.  
Powered by **Streamlit**, **Planetary Computer**, and open data sources.
```