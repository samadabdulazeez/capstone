# utils/data_loader.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
import logging
from zipfile import ZipFile
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_loader")

MB_WEATHER_FILES = {
    "obs": "https://raw.githubusercontent.com/vlyubchich/Manitoba/master/data/Weather%20Obs.csv",
    "reanalysis": "https://raw.githubusercontent.com/vlyubchich/Manitoba/master/data/Weather%20Reanalysis.csv",
}

STATSCAN_CSV_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/32100359-eng.zip"
MB_GEO = "Manitoba"

YIELD_UOMS = [
    "Bushels per acre",
    "Tons per acre",
    "Hundredweight per acre",
    "Pounds per acre",
    "Kilograms per hectare",
]

@st.cache_data(show_spinner=False)
def _download_statscan_csv() -> pd.DataFrame:
    """
    Download and return the StatsCan CSV (unzipped) as a DataFrame.
    """
    logger.info("Downloading StatsCan CSV from %s", STATSCAN_CSV_URL)
    r = requests.get(STATSCAN_CSV_URL, timeout=60)
    r.raise_for_status()
    z = ZipFile(io.BytesIO(r.content))
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    df = pd.read_csv(z.open(csv_name), low_memory=False)
    logger.info("StatsCan CSV loaded with %d rows", len(df))
    return df

def _commodity_column(df: pd.DataFrame) -> str:
    for candidate in ["Type of crop", "Field crop", "Commodity", "Crop"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("No commodity column found in StatsCan CSV")

def _disposition_column(df: pd.DataFrame) -> str:
    for candidate in ["Harvest disposition", "Statistic", "Measure"]:
        if candidate in df.columns:
            return candidate
    return None

@st.cache_data(show_spinner=False)
def list_statscan_commodities_mb() -> list:
    """
    Return commodities that have Manitoba yield rows (strictly filtered).
    """
    df = _download_statscan_csv()
    col = _commodity_column(df)
    disp_col = _disposition_column(df)

    df_mb = df[df["GEO"] == MB_GEO].copy()
    if disp_col:
        df_mb = df_mb[df_mb[disp_col].str.contains("Yield", case=False, na=False)]
    df_mb = df_mb[df_mb["UOM"].isin(YIELD_UOMS)].copy()

    # Parse REF_DATE to year where possible for filtering
    df_mb["REF_DATE_sample"] = df_mb["REF_DATE"].astype(str).str[:10]
    # Try to parse to year; fallback to numeric cast
    df_mb["Year_parsed"] = pd.to_datetime(df_mb["REF_DATE_sample"], errors="coerce").dt.year
    df_mb["Year_parsed"] = df_mb["Year_parsed"].fillna(pd.to_numeric(df_mb["REF_DATE"].astype(str).str.extract(r'(\d{4})')[0], errors="coerce"))

    df_mb = df_mb.dropna(subset=["Year_parsed"])
    commodities = sorted(df_mb[col].dropna().unique().tolist())
    logger.info("list_statscan_commodities_mb: found %d commodities with parsed years", len(commodities))
    return commodities


@st.cache_data(show_spinner=False)
def load_statscan_yields_mb(commodity: str) -> pd.DataFrame:
    """
    Load StatsCan yields for Manitoba and a selected commodity.
    Returns DataFrame with columns: Year (int), Yield (float), UOM (str)
    Includes diagnostic logging of REF_DATE samples and unique years/UOMs.
    """
    df = _download_statscan_csv()
    col = _commodity_column(df)
    disp_col = _disposition_column(df)

    df_mb = df[(df["GEO"] == MB_GEO) & (df[col] == commodity)].copy()
    if df_mb.empty:
        logger.warning("load_statscan_yields_mb: no rows for commodity '%s' in Manitoba", commodity)
        return pd.DataFrame(columns=["Year", "Yield", "UOM"])

    if disp_col:
        df_mb = df_mb[df_mb[disp_col].str.contains("Yield", case=False, na=False)]
    df_mb = df_mb[df_mb["UOM"].isin(YIELD_UOMS)].copy()

    # Diagnostics: show a few REF_DATE values and UOMs
    sample_dates = df_mb["REF_DATE"].astype(str).unique()[:10].tolist()
    sample_uoms = df_mb["UOM"].astype(str).unique().tolist()
    logger.info("load_statscan_yields_mb: commodity=%s sample REF_DATEs=%s UOMs=%s rows=%d",
                commodity, sample_dates, sample_uoms, len(df_mb))

    # Robust Year parsing:
    # 1) Try pandas to_datetime (handles YYYY, YYYY-MM, YYYY-MM-DD)
    # 2) If that fails, extract first 4-digit year with regex
    df_mb["REF_DATE_str"] = df_mb["REF_DATE"].astype(str)
    df_mb["Year"] = pd.to_datetime(df_mb["REF_DATE_str"], errors="coerce").dt.year
    if df_mb["Year"].isna().all():
        # fallback: extract 4-digit year substring
        df_mb["Year"] = pd.to_numeric(df_mb["REF_DATE_str"].str.extract(r'(\d{4})')[0], errors="coerce")

    # Convert Yield to numeric
    df_mb["Yield"] = pd.to_numeric(df_mb["VALUE"], errors="coerce")

    # Final diagnostics: unique years after parsing
    unique_years = sorted(df_mb["Year"].dropna().unique().tolist())
    logger.info("load_statscan_yields_mb: after parsing Year unique_years (sample up to 20)=%s", unique_years[:20])

    out = (
        df_mb[["Year", "Yield", "UOM"]]
        .dropna(subset=["Year", "Yield"])
        .sort_values("Year")
        .reset_index(drop=True)
    )

    # Convert Year to int
    out["Year"] = out["Year"].astype(int)
    logger.info("load_statscan_yields_mb: returning %d rows for '%s' (years %s - %s)",
                len(out), commodity, out["Year"].min() if not out.empty else "N/A", out["Year"].max() if not out.empty else "N/A")
    return out


@st.cache_data(show_spinner=False)
def _load_obs_daily() -> pd.DataFrame:
    df = pd.read_csv(MB_WEATHER_FILES["obs"])
    # Coerce types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    for col in ["Mean_Temp", "Total Rain (mm)", "Total Snow"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning("Obs CSV missing expected column: %s", col)
    return df

@st.cache_data(show_spinner=False)
def _load_rean_daily() -> pd.DataFrame:
    df = pd.read_csv(MB_WEATHER_FILES["reanalysis"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Mean_Temp" in df.columns:
        df["Mean_Temp"] = pd.to_numeric(df["Mean_Temp"], errors="coerce")
    else:
        logger.warning("Reanalysis CSV missing 'Mean_Temp' column")
    return df

@st.cache_data(show_spinner=False)
def load_mb_weather(source: str = "union") -> pd.DataFrame:
    """
    Aggregate daily weather into annual summaries.
    source: 'obs', 'reanalysis', or 'union' (obs + reanalysis-only years).
    Returns DataFrame with columns: Year (int), Annual_Tmean_C (float), Annual_Precip_mm (float or NaN)
    """
    source = source.lower()
    if source not in {"obs", "reanalysis", "union"}:
        raise ValueError("source must be 'obs', 'reanalysis', or 'union'")

    obs = _load_obs_daily()
    rean = _load_rean_daily()

    # Build annual obs
    if not obs.empty:
        annual_obs = (
            obs.groupby("Year", as_index=False)
            .agg({
                "Mean_Temp": "mean",
                "Total Rain (mm)": "sum",
                "Total Snow": "sum",
            })
        )
        annual_obs["Annual_Tmean_C"] = annual_obs["Mean_Temp"]
        annual_obs["Annual_Precip_mm"] = annual_obs["Total Rain (mm)"].fillna(0) + annual_obs["Total Snow"].fillna(0)
        annual_obs = annual_obs[["Year", "Annual_Tmean_C", "Annual_Precip_mm"]]
        logger.info("Aggregated annual_obs years: %s - %s", annual_obs["Year"].min(), annual_obs["Year"].max())
    else:
        annual_obs = pd.DataFrame(columns=["Year", "Annual_Tmean_C", "Annual_Precip_mm"])
        logger.warning("Obs daily DataFrame is empty")

    # Build annual reanalysis
    if not rean.empty:
        annual_rean = (
            rean.groupby("Year", as_index=False)
            .agg({"Mean_Temp": "mean"})
        )
        annual_rean["Annual_Tmean_C"] = annual_rean["Mean_Temp"]
        annual_rean["Annual_Precip_mm"] = np.nan
        annual_rean = annual_rean[["Year", "Annual_Tmean_C", "Annual_Precip_mm"]]
        logger.info("Aggregated annual_rean years: %s - %s", annual_rean["Year"].min(), annual_rean["Year"].max())
    else:
        annual_rean = pd.DataFrame(columns=["Year", "Annual_Tmean_C", "Annual_Precip_mm"])
        logger.warning("Reanalysis daily DataFrame is empty")

    if source == "obs":
        out = annual_obs
    elif source == "reanalysis":
        out = annual_rean
    else:  # union
        # Combine obs and reanalysis-only years
        missing_years = set(annual_rean["Year"]) - set(annual_obs["Year"])
        fill = annual_rean[annual_rean["Year"].isin(missing_years)].copy()
        out = pd.concat([annual_obs, fill], ignore_index=True)
        logger.info("Unioned annual weather years: %s - %s", out["Year"].min() if not out.empty else "N/A",
                    out["Year"].max() if not out.empty else "N/A")

    out = out.dropna(subset=["Year", "Annual_Tmean_C"]).sort_values("Year").reset_index(drop=True)
    logger.info("Final annual weather rows: %d (years %s - %s)",
                len(out), out["Year"].min() if not out.empty else "N/A", out["Year"].max() if not out.empty else "N/A")
    return out

@st.cache_data(show_spinner=False)
def load_openmeteo_forecast(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("daily", {})
    if not data:
        logger.warning("Open-Meteo returned no daily data")
        return pd.DataFrame(columns=["Date", "Temp_Max", "Temp_Min", "Precipitation"])
    return pd.DataFrame({
        "Date": pd.to_datetime(data["time"]),
        "Temp_Max": data["temperature_2m_max"],
        "Temp_Min": data["temperature_2m_min"],
        "Precipitation": data["precipitation_sum"],
    })
