"""Build normalized multi-source historical context.

This script intentionally keeps different source semantics separate:
- Kaggle: normalized attack records (built elsewhere)
- VIINA: geocoded media-derived aerial incident evidence
- eTryvoga volunteer data: air-raid alert intervals, not attacks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaning import normalize_oblast
from src.data.config import PROCESSED_DIR

RAW = ROOT / "data" / "raw"
VIINA_DIR = RAW / "viina"
SIRENS_PATH = RAW / "volunteer_air_raids_uk.csv"
GEO_PATH = RAW / "ukraine_admin1.geojson"


def build_viina(boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in sorted(VIINA_DIR.glob("viina_incidents_*.csv")):
        frame = pd.read_csv(
            path,
            usecols=["datetime", "lon", "lat", "place_name", "event_type", "headline"],
            low_memory=False,
        )
        event_type = frame["event_type"].fillna("").astype(str)
        aerial = event_type.str.contains(
            r"(?:^|\|)(?:uav|airstrike)(?:\||$)",
            regex=True,
        )
        frame = frame[aerial].copy()
        if frame.empty:
            continue

        frame["datetime"] = pd.to_datetime(
            frame["datetime"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame = frame.dropna(subset=["datetime", "lon", "lat"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "day",
                "oblast",
                "viina_events",
                "viina_uav_events",
                "viina_airstrike_events",
            ]
        )

    incidents = pd.concat(frames, ignore_index=True)
    incidents = incidents.drop_duplicates(
        subset=["datetime", "lon", "lat", "event_type", "headline"]
    )

    points = gpd.GeoDataFrame(
        incidents,
        geometry=gpd.points_from_xy(incidents["lon"], incidents["lat"]),
        crs="EPSG:4326",
    )

    matched = gpd.sjoin(
        points,
        boundaries[["name", "geometry"]],
        how="inner",
        predicate="within",
    ).rename(columns={"name": "oblast"})

    matched["day"] = matched["datetime"].dt.floor("D")
    matched["viina_uav_event"] = (
        matched["event_type"]
        .fillna("")
        .astype(str)
        .str.contains(r"(?:^|\|)uav(?:\||$)", regex=True)
        .astype("int64")
    )
    matched["viina_airstrike_event"] = (
        matched["event_type"]
        .fillna("")
        .astype(str)
        .str.contains(r"(?:^|\|)airstrike(?:\||$)", regex=True)
        .astype("int64")
    )

    return (
        matched.groupby(["day", "oblast"], as_index=False)
        .agg(
            viina_events=("event_type", "size"),
            viina_uav_events=("viina_uav_event", "sum"),
            viina_airstrike_events=("viina_airstrike_event", "sum"),
        )
        .sort_values(["day", "oblast"])
    )


def build_sirens() -> pd.DataFrame:
    if not SIRENS_PATH.exists():
        return pd.DataFrame(
            columns=["day", "oblast", "alert_count", "alert_minutes"]
        )

    alerts = pd.read_csv(SIRENS_PATH, low_memory=False)
    required = {"region", "started_at", "finished_at"}
    missing = required.difference(alerts.columns)
    if missing:
        raise ValueError(f"Missing siren columns: {sorted(missing)}")

    alerts["started_at"] = pd.to_datetime(
        alerts["started_at"], utc=True, errors="coerce", format="mixed"
    )
    alerts["finished_at"] = pd.to_datetime(
        alerts["finished_at"], utc=True, errors="coerce", format="mixed"
    )
    alerts["oblast"] = alerts["region"].map(normalize_oblast)
    alerts = alerts.dropna(subset=["started_at", "oblast"])
    alerts["day"] = alerts["started_at"].dt.floor("D")
    alerts["alert_minutes"] = (
        (alerts["finished_at"] - alerts["started_at"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0, upper=24 * 60)
    )

    return (
        alerts.groupby(["day", "oblast"], as_index=False)
        .agg(
            alert_count=("started_at", "size"),
            alert_minutes=("alert_minutes", "sum"),
        )
        .sort_values(["day", "oblast"])
    )


def main() -> None:
    if not GEO_PATH.exists():
        raise FileNotFoundError(
            "Ukraine GeoJSON is missing. Run scripts/download_public_context.py first."
        )

    boundaries = gpd.read_file(GEO_PATH)
    if "admin_level" in boundaries.columns:
        boundaries = boundaries[
            boundaries["admin_level"].astype(str).eq("4")
        ].copy()
    boundaries = boundaries.dropna(subset=["name", "geometry"])
    if boundaries.crs is None:
        boundaries = boundaries.set_crs("EPSG:4326")
    else:
        boundaries = boundaries.to_crs("EPSG:4326")

    viina = build_viina(boundaries)
    sirens = build_sirens()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    viina.to_csv(PROCESSED_DIR / "viina_oblast_daily.csv", index=False)
    sirens.to_csv(PROCESSED_DIR / "siren_oblast_daily.csv", index=False)

    status = {
        "sources": {
            "kaggle_missile_attacks": {
                "role": "attack_records",
                "active": True,
            },
            "viina": {
                "role": "independent_aerial_incident_evidence",
                "active": not viina.empty,
                "rows": int(len(viina)),
                "start": str(viina["day"].min()) if not viina.empty else None,
                "end": str(viina["day"].max()) if not viina.empty else None,
            },
            "etryvoga_volunteer": {
                "role": "air_raid_context_not_attack_labels",
                "active": not sirens.empty,
                "rows": int(len(sirens)),
                "start": str(sirens["day"].min()) if not sirens.empty else None,
                "end": str(sirens["day"].max()) if not sirens.empty else None,
            },
            "acled": {
                "role": "conflict_events_optional",
                "active": False,
                "reason": "API authentication required",
            },
            "ohchr": {
                "role": "civilian_casualty_verification",
                "active": False,
                "reason": "Public reports are kept as a separate verification layer",
            },
        }
    }
    (PROCESSED_DIR / "multisource_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "viina_daily_rows": len(viina),
                "siren_daily_rows": len(sirens),
                "status": str(PROCESSED_DIR / "multisource_status.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
