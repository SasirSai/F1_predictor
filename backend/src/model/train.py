"""
backend/src/model/train.py
───────────────────────────
XGBoost dual-model training and persistence helpers.

Models
------
Regressor  : predicts continuous finish position (1–20).
Classifier : predicts podium probability (binary: finish position ≤ 3).
"""

import os
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score


# ── Shared hyperparameter base ────────────────────────────────────────────────
_BASE_PARAMS = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


def train_models(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_clf: pd.Series,
) -> tuple:
    """
    Trains an XGBoost Regressor and Classifier on the provided feature matrix.

    Parameters
    ----------
    X     : feature DataFrame (must NOT contain target columns)
    y_reg : continuous position target  (float)
    y_clf : binary podium target        (0 or 1)

    Returns
    -------
    (reg_model, clf_model)
    """
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # ── Regressor ─────────────────────────────────────────────────────────────
    reg_model = xgb.XGBRegressor(objective="reg:squarederror", **_BASE_PARAMS)
    reg_model.fit(X_train, yr_train, eval_set=[(X_test, yr_test)], verbose=False)

    reg_preds = reg_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(yr_test, reg_preds)))
    print(f"   Regressor  RMSE : {rmse:.4f}")

    # ── Classifier ────────────────────────────────────────────────────────────
    clf_model = xgb.XGBClassifier(objective="binary:logistic", **_BASE_PARAMS)
    clf_model.fit(X_train, yc_train, eval_set=[(X_test, yc_test)], verbose=False)

    clf_probs = clf_model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(yc_test, clf_probs))
    print(f"   Classifier AUC  : {auc:.4f}")

    return reg_model, clf_model


def save_models(reg_model, clf_model, path_prefix: str = "models/") -> None:
    """Persists both models to disk using joblib."""
    os.makedirs(path_prefix, exist_ok=True)
    joblib.dump(reg_model, os.path.join(path_prefix, "xgb_regressor.pkl"))
    joblib.dump(clf_model, os.path.join(path_prefix, "xgb_classifier.pkl"))
    print(f"   Models saved → {os.path.abspath(path_prefix)}")


def load_models(path_prefix: str = "models/") -> tuple:
    """Loads persisted XGBoost models from disk."""
    reg_model = joblib.load(os.path.join(path_prefix, "xgb_regressor.pkl"))
    clf_model = joblib.load(os.path.join(path_prefix, "xgb_classifier.pkl"))
    return reg_model, clf_model
