"""
backend/src/model/explainability.py
─────────────────────────────────────
SHAP-based explainability wrapper for the trained XGBoost models.
Uses TreeExplainer for exact, fast Shapley values without sampling.
"""

import shap
import numpy as np
import pandas as pd


class F1Explainer:
    """Wraps a trained XGBoost model with SHAP TreeExplainer for per-feature attribution."""

    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)
        base = self.explainer.expected_value
        # Normalise: binary classifiers expose a list; regressors expose a scalar.
        self.base_value = float(base[-1] if isinstance(base, (list, np.ndarray)) else base)

    def explain_prediction(self, X: pd.DataFrame) -> dict:
        """
        Compute SHAP values for a single-row input DataFrame.

        Returns
        -------
        dict with keys: base_value, shap_values (list), features (list of str)
        """
        shap_vals = self.explainer.shap_values(X)

        # For binary classifiers, shap_values is a list [neg_class, pos_class]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[-1]

        return {
            "base_value": self.base_value,
            "shap_values": shap_vals.tolist() if isinstance(shap_vals, np.ndarray) else shap_vals,
            "features": X.columns.tolist(),
        }

    def format_for_frontend(self, explanation: dict, X_row: pd.DataFrame) -> dict:
        """
        Converts a raw SHAP dict into a sorted list of feature contributions
        ready for the React frontend chart.

        Returns
        -------
        dict with keys: base_value, prediction, feature_contributions (list)
        """
        shap_vals = explanation["shap_values"][0]   # single prediction row
        features  = explanation["features"]

        contributions = [
            {
                "feature":      features[i],
                "value":        float(X_row.iloc[0][features[i]]),
                "contribution": float(shap_vals[i]),
            }
            for i in range(len(features))
        ]
        # Most impactful features first
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "base_value":          explanation["base_value"],
            "prediction":          explanation["base_value"] + sum(shap_vals),
            "feature_contributions": contributions,
        }
