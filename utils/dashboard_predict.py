# utils/dashboard_predict.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.model_loader import load_artifacts

ART = load_artifacts()
RIDGE = ART["ridge"]
RF = ART["rf"]
STACKER = ART["stacker"]
FEATURES = ART["features"]
SCALER = ART["scaler"]
PCA_OBJ = ART["pca"]

def _compute_lags_rolls(df):
    df = df.copy()
    bases = ["obs_Mean_Temp_mean", "obs_GDD_day_sum", "obs_heat_day_sum", "obs_Precip_mm_sum"]
    for b in bases:
        if b in df.columns:
            # use ffill/bfill instead of deprecated fillna(method=...)
            df[b + "_lag1"] = df[b].shift(1).ffill().fillna(0.0)
            df[b + "_roll2_mean"] = df[b].shift(1).rolling(2, min_periods=1).mean().ffill().fillna(0.0)
    if "Year" in df.columns:
        df["Year_trend"] = df["Year"] - df["Year"].min()
    if "Yield_kg_ha" in df.columns:
        df["Yield_lag1"] = df["Yield_kg_ha"].shift(1).ffill().fillna(0.0)
    return df

def _compute_temp_pcs(df):
    df = df.copy()
    temp_cols = [c for c in df.columns if any(k in c.lower() for k in ("temp", "gdd", "heat"))]
    if not temp_cols:
        df["temp_PC1"] = 0.0
        df["temp_PC2"] = 0.0
        return df

    block = df[temp_cols].astype(float).ffill().bfill().fillna(0.0)

    # If saved scaler/pca exist, try to use them
    if SCALER is not None and PCA_OBJ is not None:
        try:
            scaled = SCALER.transform(block)
            pcs = PCA_OBJ.transform(scaled)
            df["temp_PC1"] = pcs[:, 0]
            df["temp_PC2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0
            return df
        except Exception:
            # fall through to local PCA fallback
            pass

    # Fallback: compute PCA locally but guard n_components
    n_samples, n_features = block.shape
    max_components = min(2, n_samples, n_features)
    if max_components <= 0:
        df["temp_PC1"] = 0.0
        df["temp_PC2"] = 0.0
        return df
    if max_components == 1:
        # Use the first standardized column's z-score as a single PC proxy
        sc = StandardScaler()
        scaled = sc.fit_transform(block)
        # collapse to a single component by projecting onto the first feature's standardized values
        df["temp_PC1"] = scaled[:, 0]
        df["temp_PC2"] = 0.0
        return df

    # Normal case: at least 2 components possible
    sc = StandardScaler()
    scaled = sc.fit_transform(block)
    p = PCA(n_components=2).fit_transform(scaled)
    df["temp_PC1"] = p[:, 0]
    df["temp_PC2"] = p[:, 1] if p.shape[1] > 1 else 0.0
    return df

def _ensure_features(df):
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    return df

def predict_df(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with at least the raw columns used to compute features (Year, weather bases, etc.)
    Output: DataFrame with added columns: pred_Ridge, pred_RF, pred_Final
    """
    df = df_input.copy().reset_index(drop=True)
    df = _compute_lags_rolls(df)
    df = _compute_temp_pcs(df)
    df = _ensure_features(df)
    X = df[FEATURES].astype(float).fillna(0.0)
    pred_ridge = RIDGE.predict(X)
    pred_rf = RF.predict(X)
    stack_X = np.column_stack([pred_ridge, pred_rf])
    pred_final = STACKER.predict(stack_X)
    out = df.copy()
    out["pred_Ridge"] = pred_ridge
    out["pred_RF"] = pred_rf
    out["pred_Final"] = pred_final
    return out
