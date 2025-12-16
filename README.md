# Manitoba Crop Dashboard

A Streamlit dashboard for Manitoba crop yield insights that combines **StatsCan yields**, **climate data**, **short‑term forecasts**, and **Sentinel‑2 NDVI** to produce farmer‑facing advisories and ensemble model predictions.

---

## Features

- **Annual yield series** from StatsCan for Manitoba commodities.  
- **Climate aggregation**: daily ECCC/observational and reanalysis data summarized to annual metrics.  
- **7‑day forecast integration** via Open‑Meteo used as a seasonal proxy for quick operational guidance.  
- **Satellite NDVI** from Microsoft Planetary Computer for field‑level visual checks.  
- **Trained ensemble models** (Ridge, Random Forest, stacked ensemble) for yield prediction with OOF‑based diagnostics.  
- **Conservative advisory logic**: predictions are shrunk toward historical baselines when model confidence is low, percent changes are capped, and prediction intervals are inflated when uncertainty is high.  
- **Farmer‑friendly UI**: clear units toggle (lb/acre or kg/ha), plain‑language advisories, decision rules, and downloadable forecast CSV.

---

## Data sources

- **StatsCan yields**: official table (zipped CSV).  
- **Manitoba weather**: observational and reanalysis CSVs hosted on GitHub.  
- **Open‑Meteo**: 7‑day forecast API for short‑term operational guidance.  
- **Sentinel‑2 L2A**: NDVI via Microsoft Planetary Computer STAC.

---

## Installation

1. **Clone the repo**
```bash
git clone https://github.com/.../manitoba-crop-dashboard.git
cd manitoba-crop-dashboard
```

2. **Create and activate a Python environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Running the dashboard

From the project root run:
```bash
streamlit run dashboard.py
```
Open the local URL shown by Streamlit (usually `http://localhost:8501`).

**Quick walkthrough**
- Select a **commodity** from the sidebar.  
- Choose **weather source** (`obs`, `reanalysis`, or `union`).  
- Toggle **units** between lb/acre and kg/ha.  
- Optionally set NDVI date window and cloud threshold.  
- Read the **Farmer Advisory** and **Quick advisory** card for predicted yield, prediction interval, confidence, and actionable tips.  
- Use NDVI and the 7‑day forecast to localize operational decisions.

---

## Model artifacts and verification

**Required artifacts location**
Place trained artifacts under:
```
models/manitoba_artifacts_for_dashboard/
```
Required files:
- `ridge_tuned.joblib`  
- `rf_tuned.joblib`  
- `stacker_ridge_final.joblib`  
- `feature_list.json`

Optional but recommended:
- `scaler.joblib`  
- `pca.joblib`  
- `oof_with_meta_final.csv`  
- `manifest.txt`

**Quick verification**
Run the included verifier to confirm artifacts load and a sample prediction runs:
```bash
python verify_models.py
```
If verification succeeds, start the dashboard.

**How predictions are used**
- `utils/model_loader.py` loads artifacts once at startup.  
- `utils/dashboard_predict.py` applies the same preprocessing used in training (lags, rolling features, PCA) and returns ensemble predictions.  
- The dashboard applies conservative calibration: predictions are blended with a robust historical baseline when model confidence is low, percent changes are capped, and prediction intervals are computed from OOF MAE or ensemble spread and inflated when uncertainty is high.

---

## Interpretation guidance and limitations

**What the Quick advisory means**
- **Predicted yield** is a seasonal signal, not a guarantee.  
- **Confidence** reflects model fit and OOF diagnostics; low confidence means rely more on scouting and local knowledge.  
- **Prediction interval** expresses uncertainty; a wide interval indicates low reliability.

**Practical limitations**
- Models use **annual** climate summaries; they do not model crop stage‑specific impacts.  
- Short‑term forecasts are used as a **proxy** for seasonal conditions and are scaled conservatively.  
- Low explained variance for some commodities is expected; use the dashboard as a decision support tool, not a replacement for field scouting.

---

## Next steps and improvements

- Replace annual aggregation with **growing‑stage windows** aligned to crop phenology.  
- Incorporate **soil and management** variables for field‑level predictions.  
- Calibrate prediction intervals using proper meta‑learner variance estimation.  
- Deploy artifacts to cloud storage and fetch at startup for reproducible releases.

---

## Contact and credits

Developed as part of the **Manitoba Agri Capstone Project**. Built with **Streamlit**, **Planetary Computer**, and open data sources.