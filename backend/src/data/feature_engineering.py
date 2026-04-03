"""
backend/src/data/feature_engineering.py
────────────────────────────────────────
Feature engineering helpers for building race-state vectors from raw telemetry.

These functions are called by run_training.py (for synthetic generation) and can
also be used against real FastF1 lap data once a live data pipeline is wired in.
"""

import pandas as pd
import numpy as np


def compute_driver_features(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives per-driver metrics from a FastF1 laps DataFrame.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Must contain 'Driver' and 'LapTime' (timedelta) columns.

    Returns
    -------
    pd.DataFrame with columns: Driver, avg_lap_time (s), lap_time_variance
    """
    if laps_df is None or laps_df.empty or "LapTime" not in laps_df.columns:
        return pd.DataFrame()

    laps = laps_df.dropna(subset=["LapTime"]).copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    rows = []
    for driver, group in laps.groupby("Driver"):
        s = group["LapTime_s"]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        clean = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
        rows.append(
            {
                "Driver": driver,
                "avg_lap_time": clean.mean() if not clean.empty else np.nan,
                "lap_time_variance": clean.var() if not clean.empty else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_driver_form(historical_results_df: pd.DataFrame, driver_id: str) -> float:
    """
    Rolling average finish position over the last 5 races for a given driver.

    Parameters
    ----------
    historical_results_df : pd.DataFrame
        Columns: driverId, raceId, position  (sorted chronologically).
    driver_id : str

    Returns
    -------
    float – mean position (lower is better); NaN if no history.
    """
    history = historical_results_df[historical_results_df["driverId"] == driver_id]
    recent = history.tail(5)
    return recent["position"].mean() if not recent.empty else np.nan


def compute_tire_degradation(stint_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates tire degradation rate per stint.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Columns: Driver, Stint, Compound, Initial_LapTime, Final_LapTime, Stint_Length

    Returns
    -------
    pd.DataFrame with tire_degradation_rate column added.
    """
    out = stint_df.copy()
    out["tire_degradation_rate"] = (
        (out["Final_LapTime"] - out["Initial_LapTime"])
        / out["Stint_Length"].replace(0, 1)
    )
    return out


def encode_race_context(race_data: dict) -> dict:
    """
    Maps a race context dictionary to numeric features.

    Parameters
    ----------
    race_data : dict
        Expected keys: circuitId, grid

    Returns
    -------
    dict with grid_position, circuit_type, overtaking_difficulty, historical_safety_car_prob
    """
    # Extend this mapping as more circuits are added.
    CIRCUIT_MAP = {
        "monaco":      {"circuit_type": 0, "overtaking_diff": 0.90, "sc_prob": 0.85},
        "singapore":   {"circuit_type": 0, "overtaking_diff": 0.85, "sc_prob": 0.85},
        "baku":        {"circuit_type": 0, "overtaking_diff": 0.50, "sc_prob": 0.75},
        "jeddah":      {"circuit_type": 0, "overtaking_diff": 0.45, "sc_prob": 0.75},
        "silverstone": {"circuit_type": 1, "overtaking_diff": 0.35, "sc_prob": 0.30},
        "monza":       {"circuit_type": 1, "overtaking_diff": 0.30, "sc_prob": 0.30},
        "spa":         {"circuit_type": 1, "overtaking_diff": 0.40, "sc_prob": 0.65},
        "bahrain":     {"circuit_type": 2, "overtaking_diff": 0.40, "sc_prob": 0.40},
        "abu_dhabi":   {"circuit_type": 2, "overtaking_diff": 0.45, "sc_prob": 0.30},
    }
    props = CIRCUIT_MAP.get(race_data.get("circuitId", ""), {"circuit_type": 1, "overtaking_diff": 0.50, "sc_prob": 0.30})

    return {
        "grid_position":              race_data.get("grid", 0),
        "circuit_type":               props["circuit_type"],
        "overtaking_difficulty":      props["overtaking_diff"],
        "historical_safety_car_prob": props["sc_prob"],
    }


def compute_team_strength(constructors_df: pd.DataFrame, constructor_id: str) -> float:
    """
    Average constructor championship points over the last 5 rounds.

    Parameters
    ----------
    constructors_df : pd.DataFrame
        Columns: constructorId, points  (sorted chronologically).
    constructor_id : str

    Returns
    -------
    float – mean points; NaN if no data.
    """
    history = constructors_df[constructors_df["constructorId"] == constructor_id]
    recent = history.tail(5)
    return recent["points"].mean() if not recent.empty else np.nan
