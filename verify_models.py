# verify_models.py
import pandas as pd
from utils.model_loader import load_artifacts
from utils.dashboard_predict import predict_df

ART = load_artifacts()
print("Loaded artifacts from:", ART["models_dir"])
print("Feature count:", len(ART["features"]))

# Build a tiny sample row (use realistic values)
sample = pd.DataFrame([{
    "Year": 2025,
    "Annual_Precip_mm": 300.0,
    "Annual_Tmean_C": 6.5,
    "obs_Mean_Temp_mean": 6.5,
    "obs_Precip_mm_sum": 300.0,
    "Yield_kg_ha": 3500.0
}])
out = predict_df(sample)
print("Predictions:", out[["pred_Ridge", "pred_RF", "pred_Final"]].to_dict(orient="records"))
