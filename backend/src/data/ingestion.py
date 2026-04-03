"""
backend/src/data/ingestion.py
──────────────────────────────
Data ingestion helpers for FastF1 telemetry and the Ergast REST API.

Usage
-----
    from src.data.ingestion import fetch_telemetry_data, fetch_historical_race_results

Note: FastF1 requires an internet connection on first load; subsequent calls are served
from the local cache at backend/cache/.
"""

import os
import fastf1
import pandas as pd
import requests

# FastF1 disk cache — keeps downloaded sessions so they are not re-downloaded
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

ERGAST_BASE = "http://ergast.com/api/f1"
REQUEST_TIMEOUT = 10  # seconds


def fetch_telemetry_data(year: int, race_name: str, session_type: str = "R") -> pd.DataFrame | None:
    """
    Downloads lap and telemetry data for a single race session via FastF1.

    Parameters
    ----------
    year        : int   – e.g. 2024
    race_name   : str   – e.g. 'Bahrain'
    session_type: str   – 'R' (Race), 'Q' (Qualifying), 'FP1', etc.

    Returns
    -------
    pd.DataFrame of laps, or None on failure.
    """
    try:
        session = fastf1.get_session(year, race_name, session_type)
        session.load(telemetry=True, weather=True)
        return session.laps
    except Exception as exc:
        print(f"[ingestion] FastF1 error for {year} {race_name} {session_type}: {exc}")
        return None


def fetch_historical_race_results(year: int, round_num: int) -> list[dict]:
    """
    Fetches race results from the Ergast API for a given season round.

    Parameters
    ----------
    year      : int – season year
    round_num : int – round number within the season

    Returns
    -------
    list of result dicts from the Ergast JSON payload, or [] on failure.
    """
    url = f"{ERGAST_BASE}/{year}/{round_num}/results.json"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        races = response.json()["MRData"]["RaceTable"]["Races"]
        return races[0]["Results"] if races else []
    except Exception as exc:
        print(f"[ingestion] Ergast error for {year} round {round_num}: {exc}")
        return []


def get_race_schedule(year: int) -> list[dict]:
    """
    Returns the full race calendar for a given season from the Ergast API.

    Parameters
    ----------
    year : int

    Returns
    -------
    list of race dicts, or [] on failure.
    """
    url = f"{ERGAST_BASE}/{year}.json"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["MRData"]["RaceTable"]["Races"]
    except Exception as exc:
        print(f"[ingestion] Ergast schedule error for {year}: {exc}")
        return []
