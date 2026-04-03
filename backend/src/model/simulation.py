"""
backend/src/model/simulation.py
────────────────────────────────
Monte Carlo simulation layer for F1 race outcome uncertainty estimation.

Adds controlled stochastic noise to the point-estimate XGBoost prediction,
producing a distribution of likely finish positions rather than a single value.
"""

import numpy as np
import pandas as pd


def run_monte_carlo(
    base_features: pd.DataFrame,
    model,
    num_simulations: int = 1000,
) -> dict:
    """
    Perturbs the base feature vector `num_simulations` times and runs
    XGBoost inference on each perturbation to model uncertainty.

    Perturbations applied
    ---------------------
    - tire_degradation_rate : Gaussian noise σ = 0.05
    - avg_lap_time          : Gaussian noise σ = 0.8 s
    - Race incident (SC/crash) : 15% chance → +5 s lap time penalty for one driver

    Parameters
    ----------
    base_features   : pd.DataFrame – single-row feature vector
    model           : trained XGBoost regressor
    num_simulations : int

    Returns
    -------
    dict with:
        mean_position    – float
        std_position     – float
        podium_probability – float (fraction of sims where position ≤ 3)
        p10_position     – 10th percentile finish (optimistic scenario)
        p90_position     – 90th percentile finish (pessimistic scenario)
    """
    positions = []

    for _ in range(num_simulations):
        state = base_features.copy()

        if "tire_degradation_rate" in state.columns:
            state["tire_degradation_rate"] += np.random.normal(0, 0.05, len(state))

        if "avg_lap_time" in state.columns:
            state["avg_lap_time"] += np.random.normal(0, 0.8, len(state))

        # Race incident: 15% chance of a significant pace loss
        if np.random.rand() < 0.15 and "avg_lap_time" in state.columns:
            idx = np.random.randint(0, len(state))
            state.at[idx, "avg_lap_time"] += 5.0

        pred = float(np.clip(model.predict(state)[0], 1, 20))
        positions.append(pred)

    arr = np.array(positions)
    return {
        "mean_position":      round(float(arr.mean()), 2),
        "std_position":       round(float(arr.std()), 2),
        "podium_probability": round(float((arr <= 3).mean()), 4),
        "p10_position":       round(float(np.percentile(arr, 10)), 1),
        "p90_position":       round(float(np.percentile(arr, 90)), 1),
    }


def safety_car_uncertainty_multiplier(sc_probability: float) -> float:
    """
    Returns a scalar multiplier for prediction uncertainty based on
    circuit safety-car likelihood.

    Higher SC probability → more variance in final position.

    Parameters
    ----------
    sc_probability : float in [0, 1]

    Returns
    -------
    float – multiplier ≥ 1.0
    """
    return round(1.0 + sc_probability * 0.5, 3)
