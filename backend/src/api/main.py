from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

from src.model.train import load_models
from src.model.explainability import F1Explainer

app = FastAPI(title="F1 Race Outcome Predictor API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models from disk if they exist
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    MODEL_DIR = os.path.join(BASE_DIR, "models") + os.sep
    reg_model, clf_model = load_models(path_prefix=MODEL_DIR)
    explainer = F1Explainer(reg_model)
    print(f"✅ XGBoost Models Loaded Successfully from {MODEL_DIR}")
except Exception as e:
    print(f"⚠️  MODELS NOT FOUND. Run: python src/scripts/run_training.py  →  Error: {e}")
    reg_model, clf_model, explainer = None, None, None


class RaceStateInput(BaseModel):
    driver_id: str
    grid_position: int
    avg_lap_time: float
    tire_degradation_rate: float
    driver_form: float
    team_strength: float
    historical_safety_car_prob: float
    driver_morale: float   # NEW: 1–10 driver confidence level


FEATURE_COLS = [
    'grid_position', 'avg_lap_time', 'tire_degradation_rate',
    'driver_form', 'team_strength', 'historical_safety_car_prob', 'driver_morale'
]


@app.get("/")
def health_check():
    return {"status": "active", "models_loaded": reg_model is not None, "version": "4.0"}


@app.post("/predict")
def predict_outcome(state: RaceStateInput):
    """
    Predicts finishing position (regression) and podium probability (classification)
    using XGBoost trained on 8,000 historical-hybrid race state vectors.
    Features: grid_position, avg_lap_time, tire_degradation_rate, driver_form,
              team_strength, historical_safety_car_prob, driver_morale
    """
    if reg_model is None:
        return {"error": "Models not trained. Please run src/scripts/run_training.py"}

    input_dict = state.model_dump()
    driver_id = input_dict.pop("driver_id")

    df = pd.DataFrame([input_dict])[FEATURE_COLS]

    pos_pred = float(reg_model.predict(df)[0])
    prob_pred = float(clf_model.predict_proba(df)[0][1])

    # Clamp prediction: dropping max 6 from grid, gaining up to 15 (miracle run possible)
    grid = state.grid_position
    pos_clamped = int(np.clip(round(pos_pred), max(1, grid - 15), min(20, grid + 6)))

    # SHAP explanation
    try:
        raw_exp = explainer.explain_prediction(df)
        formatted = explainer.format_for_frontend(raw_exp, df)
        explanation = formatted['feature_contributions']
    except Exception:
        explanation = []

    return {
        "driver_id": driver_id,
        "predicted_position": pos_clamped,
        "podium_probability": round(prob_pred, 4),
        "explanation": explanation
    }


@app.get("/metrics")
def get_metrics():
    return {
        "algorithm": "Gradient Boosted Trees (XGBoost)",
        "features_mapped": 7,
        "training_vectors": 8000,
        "feature_list": FEATURE_COLS,
        "regression_parameters": "max_depth=4, lr=0.05, n_estimators=300",
        "drop_cap": "Max ±6 positions from car weakness (historical realism)",
        "morale_miracle": "7% miracle-run chance when morale ≥ 8",
        "historical_coverage": "2020–2026 Hybrid Era statistical profiles",
        "ready": reg_model is not None
    }
