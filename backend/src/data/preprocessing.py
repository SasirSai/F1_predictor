"""
backend/src/data/preprocessing.py
──────────────────────────────────
Prepares the generated DataFrame for XGBoost training.
Splits into feature matrix X, regression target y_reg, and classification target y_clf.
"""

import pandas as pd


FEATURE_COLS = [
    "grid_position",
    "avg_lap_time",
    "tire_degradation_rate",
    "driver_form",
    "team_strength",
    "historical_safety_car_prob",
    "driver_morale",
]


def prepare_for_model(df: pd.DataFrame):
    """
    Splits a race-state DataFrame into XGBoost-ready tensors.

    Targets
    -------
    y_reg : continuous finish position (1–20)
    y_clf : binary podium flag (1 = top-3, 0 = outside podium)

    Returns
    -------
    X, y_reg, y_clf  –  all as pandas Series / DataFrame so column names are preserved.
    """
    if df is None or df.empty:
        return None, None, None

    y_reg = df["position"].astype(float)
    y_clf = (df["position"] <= 3).astype(int)
    X = df[FEATURE_COLS]

    return X, y_reg, y_clf
