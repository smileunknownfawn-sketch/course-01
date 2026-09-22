from __future__ import annotations

import pandas as pd

OBLAST_ALIASES = {
    "київ": "Київ",
    "м. київ": "Київ",
    "київська": "Київська область",
    "київська область": "Київська область",
    "одеська": "Одеська область",
    "одеська область": "Одеська область",
    "дніпропетровська": "Дніпропетровська область",
    "дніпропетровська область": "Дніпропетровська область",
    "львівська": "Львівська область",
    "львівська область": "Львівська область",
    "харківська": "Харківська область",
    "харківська область": "Харківська область",
}


def normalize_oblast(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return OBLAST_ALIASES.get(text, str(value).strip())


def add_event_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["started_at"] = pd.to_datetime(result["started_at"], utc=True, errors="coerce")
    result["oblast_normalized"] = result["oblast"].map(normalize_oblast)
    result["event_key"] = (
        result["started_at"].dt.floor("h").astype("string")
        + "|" + result["oblast_normalized"].astype("string")
        + "|" + result["attack_type"].astype("string").str.lower()
    )
    return result


def find_possible_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    result = add_event_key(df)
    counts = result.groupby("event_key", dropna=False)["attack_id"].transform("size")
    return result[counts > 1].sort_values("event_key")
