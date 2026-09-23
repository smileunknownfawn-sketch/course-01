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
    """Add prior-event counts by oblast and time window.

    Only events strictly earlier than the current timestamp are counted.
    This prevents future leakage (підглядання в майбутні дані).
    """
    result = df.copy()
    result["started_at"] = pd.to_datetime(result["started_at"], utc=True)
    result["_original_order"] = range(len(result))
    result = result.sort_values(["oblast", "started_at", "_original_order"])

    for hours in windows:
        col = f"attacks_last_{hours}h"
        result[col] = 0

        for _, group in result.groupby("oblast", dropna=False, sort=False):
            ordered = group.sort_values(["started_at", "_original_order"])
            history = pd.Series(1, index=ordered["started_at"])
            counts = (
                history.rolling(f"{hours}h", closed="left")
                .count()
                .fillna(0)
                .astype("int64")
                .to_numpy()
            )
            result.loc[ordered.index, col] = counts

    return (
        result.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )
