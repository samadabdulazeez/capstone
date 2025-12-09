# utils/data_loader.py
import pandas as pd
import requests
from io import StringIO
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- StatsCan crop yields ---
STATSCAN_CSV_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/32100359-eng.zip"
MB_GEO = "Manitoba"

# Known yield-related units of measure
YIELD_UOMS = [
    "Bushels per acre",
    "Tons per acre",
    "Hundredweight per acre",
    "Pounds per acre",
    "Kilograms per hectare",
]

@st.cache_data
def _download_statscan_csv() -> pd.DataFrame:
    logging.info("Downloading StatsCan CSV from %s", STATSCAN_CSV_URL)
    r = requests.get(STATSCAN_CSV_URL, timeout=60)
    r.raise_for_status()
    from zipfile import ZipFile
    import io
    z = ZipFile(io.BytesIO(r.content))
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    df = pd.read_csv(z.open(csv_name), low_memory=False)
    logging.info("CSV columns: %s", list(df.columns))
    return df

def _commodity_column(df: pd.DataFrame) -> str:
    for candidate in ["Field crop", "Type of crop", "Commodity", "Crop"]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"No commodity column found. Columns: {list(df.columns)}")

def list_statscan_commodities_mb() -> list:
    df = _download_statscan_csv()
    col = _commodity_column(df)
    df_mb = df[df["GEO"] == MB_GEO]
    df_yield = df_mb[df_mb["UOM"].isin(YIELD_UOMS)]
    commodities = sorted(df_yield[col].unique().tolist())
    logging.info("Commodities found: %s", commodities)
    return commodities

def load_statscan_yields_mb(commodity: str) -> pd.DataFrame:
    df = _download_statscan_csv()
    col = _commodity_column(df)
    df_mb = df[(df["GEO"] == MB_GEO) & (df[col] == commodity)]
    df_yield = df_mb[df_mb["UOM"].isin(YIELD_UOMS)]
    if df_yield.empty:
        logging.warning("No yield data found for %s", commodity)
        return pd.DataFrame()
    df_yield["Year"] = pd.to_datetime(df_yield["REF_DATE"], errors="coerce").dt.year
    return df_yield[["Year", "VALUE", "UOM"]].rename(columns={"VALUE": "Yield"}).dropna().sort_values("Year")

# --- ECCC stations (Manitoba) via GeoMet API, with fallback ---
GEOMET_STATIONS_URL = "https://api.weather.gc.ca/collections/climate-stations/items"
GEOMET_PARAMS_MB = {"prov": "MB", "f": "json", "limit": 10000}

MB_STATIONS_FALLBACK = [
    {"id": "27174", "name": "Winnipeg A CS", "lat": 49.91, "lon": -97.23},
    {"id": "27616", "name": "Brandon A", "lat": 49.91, "lon": -99.95},
    {"id": "28038", "name": "Portage Southport", "lat": 49.90, "lon": -98.27},
    {"id": "50821", "name": "Gimli Harbour", "lat": 50.62, "lon": -96.99},
    {"id": "26798", "name": "The Pas A", "lat": 53.97, "lon": -101.10},
]

def list_eccc_stations_mb():
    try:
        r = requests.get(GEOMET_STATIONS_URL, params=GEOMET_PARAMS_MB, timeout=30)
        r.raise_for_status()
        js = r.json()
        features = js.get("features", [])
        stations = []
        for f in features:
            props = f.get("properties", {})
            geom_obj = f.get("geometry", {})
            coords = geom_obj.get("coordinates")
            if not props or not coords:
                continue
            sid = str(props.get("station_id") or props.get("stationID") or "").strip()
            name = (props.get("name") or props.get("station_name") or "Unknown").strip()
            lon, lat = coords[:2]
            if not (-102.0 <= lon <= -95.0 and 48.99 <= lat <= 60.0):
                continue
            if sid:
                stations.append({"id": sid, "name": name, "lat": float(lat), "lon": float(lon)})
        if stations:
            unique = {}
            for s in stations:
                unique[s["id"]] = s
            logging.info("Stations found dynamically: %s", list(unique.values()))
            return list(unique.values())
    except Exception as e:
        logging.warning("Falling back to static station list due to error: %s", e)
    return MB_STATIONS_FALLBACK

def load_eccc_daily_annual_mb(station_id: str) -> pd.DataFrame:
    url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    params = {"format": "csv", "stationID": station_id, "timeframe": "2", "submit": "Download+Data"}
    logging.info("Downloading ECCC daily data for station %s", station_id)
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    date_col = next((c for c in df.columns if "Date/Time" in c), None)
    tmean_col = next((c for c in df.columns if "Mean Temp" in c), None)
    precip_col = next((c for c in df.columns if "Total Precip" in c), None)
    if not date_col or not tmean_col or not precip_col:
        logging.warning("Missing expected columns in ECCC data")
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Tmean"] = pd.to_numeric(df[tmean_col], errors="coerce")
    df["Precip"] = pd.to_numeric(df[precip_col], errors="coerce")
    annual = df.groupby("Year", as_index=False).agg(
        Annual_Tmean_C=("Tmean", "mean"),
        Annual_Precip_mm=("Precip", "sum"),
    )
    logging.info("Annual climate summary:\n%s", annual.head())
    return annual

# --- Open-Meteo forecast ---
def load_openmeteo_forecast(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto",
    }
    logging.info("Fetching 7-day forecast for lat=%s, lon=%s", lat, lon)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["daily"]
    forecast_df = pd.DataFrame({
        "Date": pd.to_datetime(data["time"]),
        "Temp_Max": data["temperature_2m_max"],
        "Temp_Min": data["temperature_2m_min"],
        "Precipitation": data["precipitation_sum"],
    })
    logging.info("Forecast preview:\n%s", forecast_df.head())
    return forecast_df

# --- Sentinel-2 search helper ---
def sentinel2_search(lat: float, lon: float, start: str, end: str, max_cloud: int = 30):
    if pd.to_datetime(end) > pd.Timestamp.today():
        raise ValueError("End date cannot be in the future.")
    return {
        "lat": lat,
        "lon": lon,
        "start": start,
        "end": end,
        "max_cloud": max_cloud,
    }
