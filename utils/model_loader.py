# utils/model_loader.py
import os
import joblib
import pandas as pd
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root /manitoba_agri_capstone_final
MODELS_DIR = os.path.join(BASE_DIR, "models", "manitoba_artifacts_for_dashboard")

# Expected artifact filenames (adjust if you renamed files)
ARTIFACTS = {
    "ridge": "ridge_tuned.joblib",
    "rf": "rf_tuned.joblib",
    "stacker": "stacker_ridge_final.joblib",
    "feature_list": "feature_list.json",
    "scaler": "scaler.joblib",
    "pca": "pca.joblib",
}

def _path(name):
    return os.path.join(MODELS_DIR, ARTIFACTS[name])

def list_missing():
    missing = []
    for k, fname in ARTIFACTS.items():
        p = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(p):
            missing.append(fname)
    return missing

@lru_cache(maxsize=1)
def load_artifacts():
    """
    Load and return a dict with keys:
      ridge, rf, stacker, features (list), scaler (or None), pca (or None)
    Raises FileNotFoundError if required artifacts are missing.
    """
    missing = list_missing()
    if missing:
        raise FileNotFoundError(f"Missing model artifacts in {MODELS_DIR}: {missing}")

    ridge = joblib.load(_path("ridge"))
    rf = joblib.load(_path("rf"))
    stacker = joblib.load(_path("stacker"))
    features = pd.read_json(_path("feature_list"), typ="series").tolist()
    scaler = joblib.load(_path("scaler")) if os.path.exists(_path("scaler")) else None
    pca = joblib.load(_path("pca")) if os.path.exists(_path("pca")) else None

    return {
        "ridge": ridge,
        "rf": rf,
        "stacker": stacker,
        "features": features,
        "scaler": scaler,
        "pca": pca,
        "models_dir": MODELS_DIR,
    }
