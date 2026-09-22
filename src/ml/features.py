"""Feature engineering for coarse historical risk forecasting."""

from __future__ import annotations

import pandas as pd


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["started_at"] = pd.to_datetime(result["started_at"], utc=True)
    result["hour"] = result["started_at"].dt.hour
    result["day_of_week"] = result["started_at"].dt.dayofweek
    result["month"] = result["started_at"].dt.month
    return result


def add_recent_counts(df: pd.DataFrame, windows=(6, 24, 168)) -> pd.DataFrame:
    """Add historical event counts by oblast and time window."""
    result = df.sort_values(["oblast", "started_at"]).copy()
    result["started_at"] = pd.to_datetime(result["started_at"], utc=True)
    result = result.set_index("started_at")

    for hours in windows:
        col = f"attacks_last_{hours}h"
        counts = (
            result.groupby("oblast")["attack_id"]
            .rolling(f"{hours}h", closed="left")
            .count()
            .reset_index(level=0, drop=True)
        )
        result[col] = counts

    return result.reset_index()
