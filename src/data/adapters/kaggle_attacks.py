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
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if pd.isna(value):
        return []

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


def find_unmapped_regions(value: object) -> list[str]:
    """Return source region tokens that are not in the canonical dictionary."""
    unmapped: list[str] = []
    for candidate in _candidate_regions(value):
        normalized = normalize_oblast(candidate)
        if normalized not in CANONICAL_REGIONS and candidate not in unmapped:
            unmapped.append(candidate)
    return unmapped


def _source_record_hash(row: pd.Series, raw_columns: list[str]) -> str:
    """Hash the complete normalized source row for stable provenance."""
    parts: list[str] = []
    for column in sorted(raw_columns):
        value = row.get(column)
        text = "" if pd.isna(value) else str(value).strip()
        parts.append(f"{column}={text}")
    payload = "\\x1f".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _attack_id_from_hash(record_hash: str) -> int:
    # 15 hex chars fit safely inside a signed BIGINT and are deterministic.
    return int(record_hash[:15], 16)


def transform_kaggle_attacks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Transform Kaggle rows into normalized project tables."""
    source = normalize_source_columns(df)
    validate_source_columns(source)
    raw_columns = list(source.columns)
    source["source_record_hash"] = source.apply(
        lambda row: _source_record_hash(row, raw_columns), axis=1
    )

    exact_duplicate_mask = source["source_record_hash"].duplicated(keep=False)
    source_duplicates = source.loc[
        exact_duplicate_mask,
        ["source_record_hash", "time_start", "model", "source"],
    ].copy()
    source_duplicates["duplicate_count"] = source_duplicates.groupby(
        "source_record_hash"
    )["source_record_hash"].transform("size")
    source_duplicates = source_duplicates.drop_duplicates("source_record_hash")

    # Byte-equivalent source records are collapsed once and separately reported.
    source = source.drop_duplicates("source_record_hash", keep="first").copy()

    source["started_at"] = pd.to_datetime(source["time_start"], utc=True, errors="coerce", format="mixed")
    if source["started_at"].isna().any():
        bad_rows = source.index[source["started_at"].isna()].tolist()
        raise ValueError(f"Invalid time_start values at rows: {bad_rows[:10]}")

    if "time_end" in source.columns:
        source["ended_at"] = pd.to_datetime(source["time_end"], utc=True, errors="coerce", format="mixed")
    else:
        source["ended_at"] = pd.NaT

    source["attack_id"] = source["source_record_hash"].map(_attack_id_from_hash)
    if source["attack_id"].duplicated().any():
        raise ValueError("Stable attack_id hash collision detected")

    source["weapon_category"] = source["model"].map(classify_weapon)
    source["quantity"] = pd.to_numeric(source["launched"], errors="coerce").astype("Int64")
    source["intercepted_quantity"] = pd.to_numeric(source["destroyed"], errors="coerce").astype("Int64")

    affected_column = (
        source["affected_region"]
        if "affected_region" in source.columns
        else pd.Series([pd.NA] * len(source), index=source.index)
    )
    source["affected_regions"] = affected_column.map(parse_affected_regions)
    source["unmapped_regions"] = affected_column.map(find_unmapped_regions)

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

    unmapped_records: list[dict[str, object]] = []
    for attack_id, regions in zip(source["attack_id"], source["unmapped_regions"]):
        for raw_region in regions:
            unmapped_records.append(
                {"attack_id": attack_id, "raw_region": raw_region}
            )
    unmapped_regions = pd.DataFrame(
        unmapped_records, columns=["attack_id", "raw_region"]
    )

    weapons = source[
        ["attack_id", "weapon_category", "model", "quantity", "intercepted_quantity"]
    ].rename(columns={"weapon_category": "category", "model": "type"})

    provenance = source[["attack_id", "source", "source_record_hash"]].rename(
        columns={"source": "source_reference"}
    )
    provenance["source_event_id"] = provenance["attack_id"].astype("string")

    return {
        "attacks": attacks,
        "attack_regions": attack_regions,
        "weapons": weapons,
        "provenance": provenance,
        "unmapped_regions": unmapped_regions,
        "source_duplicates": source_duplicates,
    }


def load_and_transform(path: str | Path) -> dict[str, pd.DataFrame]:
    source = pd.read_csv(path)
    return transform_kaggle_attacks(source)
