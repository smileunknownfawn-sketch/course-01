"""Build leakage-safe daily oblast datasets for coarse 24-hour risk modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "oblast",
    "day_of_week",
    "month",
    "attacks_prev_1d",
    "attacks_prev_7d",
    "attacks_prev_30d",
    "days_since_last_attack",
]


def build_daily_oblast_dataset(
    attacks: pd.DataFrame,
    attack_regions: pd.DataFrame,
    min_history_days: int = 30,
) -> pd.DataFrame:
    """Create one row per oblast/day with a next-day attack label.

    The target is intentionally coarse: whether at least one historical attack
    affected an oblast during the following UTC calendar day.
    Features use only days strictly before the prediction day.
    """
    required_attacks = {"attack_id", "started_at"}
    required_regions = {"attack_id", "oblast"}
    missing_attacks = required_attacks.difference(attacks.columns)
    missing_regions = required_regions.difference(attack_regions.columns)
    if missing_attacks:
        raise ValueError(f"Missing attack columns: {sorted(missing_attacks)}")
    if missing_regions:
        raise ValueError(f"Missing attack-region columns: {sorted(missing_regions)}")

    event_rows = attack_regions[["attack_id", "oblast"]].merge(
        attacks[["attack_id", "started_at"]],
        on="attack_id",
        how="inner",
        validate="many_to_one",
    )
    event_rows["started_at"] = pd.to_datetime(
        event_rows["started_at"], utc=True, errors="coerce"
    )
    event_rows = event_rows.dropna(subset=["started_at", "oblast"])
    if event_rows.empty:
        raise ValueError("No dated oblast-level attack records are available")

    event_rows["event_day"] = event_rows["started_at"].dt.floor("D")
    daily_events = (
        event_rows.groupby(["oblast", "event_day"], as_index=False)
        .size()
        .rename(columns={"size": "attack_count"})
    )

    start_day = daily_events["event_day"].min()
    end_day = daily_events["event_day"].max()
    oblasts = sorted(daily_events["oblast"].dropna().unique())

    all_days = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    grid = pd.MultiIndex.from_product(
        [oblasts, all_days], names=["oblast", "day"]
    ).to_frame(index=False)

    result = grid.merge(
        daily_events.rename(columns={"event_day": "day"}),
        on=["oblast", "day"],
        how="left",
    )
    result["attack_count"] = result["attack_count"].fillna(0).astype("int64")
    result = result.sort_values(["oblast", "day"]).reset_index(drop=True)

    # Prediction target: attack occurrence on the following day.
    result["target_next_24h"] = (
        result.groupby("oblast")["attack_count"].shift(-1).fillna(0).gt(0).astype("int8")
    )

    # Historical features; shift(1) prevents using the prediction day's events.
    previous_counts = result.groupby("oblast")["attack_count"].shift(1).fillna(0)
    result["attacks_prev_1d"] = previous_counts.astype("int64")

    for days in (7, 30):
        result[f"attacks_prev_{days}d"] = (
            previous_counts.groupby(result["oblast"])
            .rolling(days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype("int64")
        )

    result["day_of_week"] = result["day"].dt.dayofweek.astype("int8")
    result["month"] = result["day"].dt.month.astype("int8")

    days_since: list[float] = []
    for _, group in result.groupby("oblast", sort=False):
        last_attack_day: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            if last_attack_day is None:
                days_since.append(np.nan)
            else:
                days_since.append(float((row.day - last_attack_day).days))
            if row.attack_count > 0:
                last_attack_day = row.day
    result["days_since_last_attack"] = days_since

    result["history_days"] = result.groupby("oblast").cumcount()
    result = result[result["history_days"] >= min_history_days].copy()
    result = result.drop(columns=["history_days"])

    # The final day has no complete following-day label in the source range.
    max_day = result["day"].max()
    result = result[result["day"] < max_day].reset_index(drop=True)

    return result


def chronological_split(
    dataset: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, never randomly, to preserve realistic evaluation."""
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    if dataset.empty:
        raise ValueError("Dataset is empty")

    unique_days = sorted(dataset["day"].dropna().unique())
    if len(unique_days) < 10:
        raise ValueError("At least 10 distinct days are required")

    split_index = max(1, int(len(unique_days) * (1 - test_fraction)))
    split_day = unique_days[split_index]
    train = dataset[dataset["day"] < split_day].copy()
    test = dataset[dataset["day"] >= split_day].copy()

    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    return train, test
