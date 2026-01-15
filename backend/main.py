# Backend API
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# ------------------------------
# Load model artifacts
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are expected in a 'models' directory at the project root (sibling to backend folder)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "xgb_model.pkl"))
FEATS_PATH = os.getenv("FEATS_PATH", os.path.join(MODELS_DIR, "selected_features.pkl"))
THRESH_PATH = os.getenv("THRESH_PATH", os.path.join(MODELS_DIR, "best_threshold.pkl"))

# Fallback selected features
FALLBACK_FEATURES = [
    "sugar_intake","bmi","cholesterol","sleep_hours",
    "physical_activity","work_hours","blood_pressure",
    "calorie_intake","water_intake",
]

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[Backend] Failed to load model from {MODEL_PATH}: {e}")

# Load features list
try:
    sel_feats = joblib.load(FEATS_PATH)
    if not isinstance(sel_feats, (list, tuple)):
        sel_feats = FALLBACK_FEATURES
except Exception:
    sel_feats = FALLBACK_FEATURES

# Load tuned threshold
try:
    THRESH = float(joblib.load(THRESH_PATH))
except Exception:
    THRESH = 0.5

MODEL_VERSION = os.getenv("MODEL_VERSION", "xgb_red_v1")

# ------------------------------
# App + CORS
# ------------------------------
app = FastAPI(title="Lifestyle Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Defaults for missing fields
# ------------------------------
DEFAULTS: Dict[str, float] = {
    "sugar_intake": 50.0,
    "bmi": 22.0,
    "cholesterol": 180.0,
    "sleep_hours": 7.0,
    "physical_activity": 5.0,
    "work_hours": 8.0,
    "blood_pressure": 120.0,
    "calorie_intake": 2000.0,
    "water_intake": 2.0,
}

# ------------------------------
# Routes
# ------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "features": list(sel_feats),
        "threshold": float(THRESH),
        "model_version": MODEL_VERSION,
        "model_loaded": bool(model is not None),
    }

@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded. Check server logs. Expected at: {MODEL_PATH}")

    data = dict(payload or {})

    # Ensure all required features exist
    for f in sel_feats:
        if data.get(f) is None:
            data[f] = DEFAULTS.get(f, 0.0)

    # Build dataframe in correct order
    df = pd.DataFrame([data]).reindex(columns=sel_feats, fill_value=0)

    proba = model.predict_proba(df)[:, 1]
    p = float(proba[0])
    pred = "At Risk" if p > THRESH else "Healthy"

    return {
        "prediction": pred,
        "probability": p,
        "threshold": float(THRESH),
        "features_used": list(sel_feats),
        "model_version": MODEL_VERSION,
    }
