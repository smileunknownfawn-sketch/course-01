"""Validation helpers for normalized historical event data."""

from __future__ import annotations

import pandas as pd

REQUIRED_ATTACK_COLUMNS = {
    "attack_id",
    "started_at",
    "oblast",
    "attack_type",
    "confidence",
}

VALID_CONFIDENCE = {"confirmed", "probable", "reported", "unverified"}


def validate_attacks(df: pd.DataFrame) -> list[str]:
    """Return validation errors instead of silently changing source data."""
    errors: list[str] = []

    missing = REQUIRED_ATTACK_COLUMNS.difference(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        return errors

    if df["attack_id"].duplicated().any():
        errors.append("attack_id contains duplicates")

    if df["started_at"].isna().any():
        errors.append("started_at contains missing values")

    invalid_confidence = set(df["confidence"].dropna()) - VALID_CONFIDENCE
    if invalid_confidence:
        errors.append(f"Invalid confidence values: {sorted(invalid_confidence)}")

    return errors


def normalize_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types without inventing missing facts."""
    result = df.copy()
    result["started_at"] = pd.to_datetime(result["started_at"], utc=True, errors="coerce")
    if "ended_at" in result.columns:
        result["ended_at"] = pd.to_datetime(result["ended_at"], utc=True, errors="coerce")
    result["oblast"] = result["oblast"].astype("string").str.strip()
    result["attack_type"] = result["attack_type"].astype("string").str.strip().str.lower()
    result["confidence"] = result["confidence"].astype("string").str.strip().str.lower()
    return result
