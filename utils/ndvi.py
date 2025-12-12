# utils/ndvi.py
import planetary_computer as pc
import pystac_client
import stackstac
import numpy as np
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)

@st.cache_data(show_spinner=False)
def compute_ndvi_map_mb(lat: float, lon: float, date_start: str, date_end: str, cloud_max: int = 30):
    """
    Query Sentinel-2 L2A imagery from Planetary Computer for a bounding box around (lat, lon),
    filter by date range and cloud cover, and compute NDVI.

    Parameters
    ----------
    lat, lon : float
        Center coordinates in Manitoba.
    date_start, date_end : str
        Date range in YYYY-MM-DD format.
    cloud_max : int
        Maximum cloud cover percentage.

    Returns
    -------
    ndvi : 2D numpy array
        NDVI raster.
    stats : dict
        Summary statistics (min, max, mean).
    """

    logging.info("Querying Sentinel-2 imagery for lat=%s, lon=%s, %s to %s (cloud <= %s%%)",
                 lat, lon, date_start, date_end, cloud_max)

    # Define bounding box ~0.1° around point
    bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]

    # Open Planetary Computer STAC with signed requests
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": cloud_max}},
        limit=1,
    )

    items = list(search.get_items())
    if not items:
        logging.warning("No Sentinel-2 scenes found for this query.")
        return np.zeros((100, 100), dtype="float32"), {"min": 0.0, "max": 0.0, "mean": 0.0}

    item = items[0]
    logging.info("Using scene: %s", item.id)

    stack = stackstac.stack(
        [item],
        assets=["B04", "B08"],  # Red and NIR
        resolution=20,
        bounds_latlon=bbox,
        epsg=32614,             # UTM zone 14N (southern Manitoba)
    ).isel(time=0)

    red = stack.sel(band="B04").astype("float32")
    nir = stack.sel(band="B08").astype("float32")

    # NDVI calculation with small epsilon to avoid division by zero
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndvi = ndvi.compute().values

    stats = {
        "min": float(np.nanmin(ndvi)),
        "max": float(np.nanmax(ndvi)),
        "mean": float(np.nanmean(ndvi)),
    }

    logging.info("NDVI stats: %s", stats)
    return ndvi, stats
