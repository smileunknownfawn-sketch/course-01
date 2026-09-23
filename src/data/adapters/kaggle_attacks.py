"""Adapter for the public Kaggle missile/UAV historical dataset."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pandas as pd

from src.data.cleaning import OBLAST_ALIASES, normalize_oblast

REQUIRED_COLUMNS = {"time_start", "model", "launched", "destroyed", "source"}
CANONICAL_REGIONS = set(OBLAST_ALIASES.values())


def normalize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known column-name variants without changing source values."""
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]

    if "affected region" in result.columns and "affected_region" not in result.columns:
        result = result.rename(columns={"affected region": "affected_region"})

    return result


def validate_source_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing Kaggle columns: {sorted(missing)}")


def classify_weapon(model: object) -> str:
    text = "" if pd.isna(model) else str(model).lower()
    if any(token in text for token in ("uav", "shahed", "geran", "gerbera", "drone")):
        return "uav"
    if any(token in text for token in ("kab", "guided bomb")):
        return "guided_bomb"
    return "missile"


def _candidate_regions(value: object) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text or text in {"[]", "{}"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in re.split(r"\s+and\s+|,\s*", text) if item.strip()]


def parse_affected_regions(value: object) -> list[str]:
    """Return only recognized Ukrainian administrative regions."""
    regions: list[str] = []
    for candidate in _candidate_regions(value):
        normalized = normalize_oblast(candidate)
        if normalized in CANONICAL_REGIONS and normalized not in regions:
            regions.append(normalized)
    return regions


def _stable_attack_id(row: pd.Series) -> int:
    payload = "|".join(
        [
            str(row.get("time_start", "")),
            str(row.get("time_end", "")),
            str(row.get("model", "")),
            str(row.get("source", "")),
            str(row.get("launched", "")),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


def transform_kaggle_attacks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Transform Kaggle rows into normalized project tables."""
    source = normalize_source_columns(df)
    validate_source_columns(source)

    source["started_at"] = pd.to_datetime(source["time_start"], utc=True, errors="coerce")
    if source["started_at"].isna().any():
        bad_rows = source.index[source["started_at"].isna()].tolist()
        raise ValueError(f"Invalid time_start values at rows: {bad_rows[:10]}")

    if "time_end" in source.columns:
        source["ended_at"] = pd.to_datetime(source["time_end"], utc=True, errors="coerce")
    else:
        source["ended_at"] = pd.NaT

    source["attack_id"] = source.apply(_stable_attack_id, axis=1)
    if source["attack_id"].duplicated().any():
        raise ValueError("Stable attack_id collision or duplicate source rows detected")

    source["weapon_category"] = source["model"].map(classify_weapon)
    source["quantity"] = pd.to_numeric(source["launched"], errors="coerce").astype("Int64")
    source["intercepted_quantity"] = pd.to_numeric(source["destroyed"], errors="coerce").astype("Int64")

    affected_column = (
        source["affected_region"]
        if "affected_region" in source.columns
        else pd.Series([pd.NA] * len(source), index=source.index)
    )
    source["affected_regions"] = affected_column.map(parse_affected_regions)

    source["oblast"] = source["affected_regions"].map(
        lambda regions: regions[0] if len(regions) == 1 else pd.NA
    )

    attacks = source[
        ["attack_id", "started_at", "ended_at", "oblast", "weapon_category", "model"]
    ].rename(columns={"weapon_category": "attack_type", "model": "description"})
    attacks["confidence"] = "reported"
    attacks = attacks[
        [
            "attack_id",
            "started_at",
            "ended_at",
            "oblast",
            "attack_type",
            "confidence",
            "description",
        ]
    ]

    region_records: list[dict[str, object]] = []
    for attack_id, regions in zip(source["attack_id"], source["affected_regions"]):
        for region in regions:
            region_records.append({"attack_id": attack_id, "oblast": region})
    attack_regions = pd.DataFrame(region_records, columns=["attack_id", "oblast"])

    weapons = source[
        ["attack_id", "weapon_category", "model", "quantity", "intercepted_quantity"]
    ].rename(columns={"weapon_category": "category", "model": "type"})

    provenance = source[["attack_id", "source"]].rename(
        columns={"source": "source_reference"}
    )
    provenance["source_event_id"] = provenance["attack_id"].astype("string")

    return {
        "attacks": attacks,
        "attack_regions": attack_regions,
        "weapons": weapons,
        "provenance": provenance,
    }


def load_and_transform(path: str | Path) -> dict[str, pd.DataFrame]:
    source = pd.read_csv(path)
    return transform_kaggle_attacks(source)
