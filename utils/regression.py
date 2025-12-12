# utils/regression.py
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def run_regression(df, crop_name: str):
    """
    Run a simple linear regression of Yield vs Annual_Tmean_C and Annual_Precip_mm.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ["Yield", "Annual_Tmean_C", "Annual_Precip_mm"]
    crop_name : str
        Name of the crop (for logging)

    Returns
    -------
    dict with keys:
        intercept, coef_temp, coef_precip, r2
    """
    if df.empty or len(df) < 5:
        logging.warning("Not enough data for regression on %s", crop_name)
        return {"intercept": np.nan, "coef_temp": np.nan,
                "coef_precip": np.nan, "r2": np.nan}

    X = df[["Annual_Tmean_C", "Annual_Precip_mm"]].values
    y = df["Yield"].values

    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    intercept, coef_temp, coef_precip = beta

    # R²
    y_pred = X_with_intercept @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot

    logging.info("%s regression: intercept=%.2f, temp=%.3f, precip=%.6f, R²=%.2f",
                 crop_name, intercept, coef_temp, coef_precip, r2)

    return {
        "intercept": intercept,
        "coef_temp": coef_temp,
        "coef_precip": coef_precip,
        "r2": r2,
    }

def predict_yield(regression_result: dict, temp: float, precip: float) -> float:
    """
    Predict yield given regression coefficients and climate inputs.
    """
    if any(np.isnan([regression_result["intercept"],
                     regression_result["coef_temp"],
                     regression_result["coef_precip"]])):
        return np.nan
    return (regression_result["intercept"] +
            regression_result["coef_temp"] * temp +
            regression_result["coef_precip"] * precip)
