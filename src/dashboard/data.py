"""Build small, safe dashboard snapshots from normalized historical data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analysis.improvement import (
    build_improvement_recommendations,
    project_readiness_score,
)

DASHBOARD_DIR = Path("data/dashboard")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dashboard_tables(
    attacks: pd.DataFrame,
    attack_regions: pd.DataFrame,
    weapons: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    attacks = attacks.copy()
    attack_regions = attack_regions.copy()
    weapons = weapons.copy()

    attacks["started_at"] = pd.to_datetime(
        attacks["started_at"], utc=True, errors="coerce", format="mixed"
    )
    attacks = attacks.dropna(subset=["started_at", "attack_id"])
    attacks["day"] = attacks["started_at"].dt.floor("D")

    # National series counts each source attack record once.
    weapon_rollup = (
        weapons.groupby("attack_id", as_index=False)
        .agg(
            launched_reported=("quantity", "sum"),
            intercepted_reported=("intercepted_quantity", "sum"),
        )
    )
    national = attacks.merge(
        weapon_rollup,
        on="attack_id",
        how="left",
        validate="one_to_one",
    )
    national["launched_reported"] = pd.to_numeric(
        national["launched_reported"], errors="coerce"
    )
    national["intercepted_reported"] = pd.to_numeric(
        national["intercepted_reported"], errors="coerce"
    )

    for category in ("uav", "missile", "guided_bomb"):
        national[f"{category}_event"] = (
            national["attack_type"].eq(category).astype("int64")
        )

    national_daily = (
        national.groupby("day", as_index=False)
        .agg(
            attack_records=("attack_id", "nunique"),
            launched_reported=("launched_reported", "sum"),
            intercepted_reported=("intercepted_reported", "sum"),
            uav_events=("uav_event", "sum"),
            missile_events=("missile_event", "sum"),
            guided_bomb_events=("guided_bomb_event", "sum"),
        )
        .sort_values("day")
    )

    # Regional views count event-region links, never duplicate weapon quantities.
    regional = attack_regions[["attack_id", "oblast"]].merge(
        attacks[["attack_id", "day", "attack_type"]],
        on="attack_id",
        how="inner",
        validate="many_to_one",
    )
    regional = regional.dropna(subset=["oblast", "day"]).drop_duplicates(
        ["attack_id", "oblast"]
    )

    for category in ("uav", "missile", "guided_bomb"):
        regional[f"{category}_event"] = (
            regional["attack_type"].eq(category).astype("int64")
        )

    oblast_daily = (
        regional.groupby(["day", "oblast"], as_index=False)
        .agg(
            attack_events=("attack_id", "nunique"),
            uav_events=("uav_event", "sum"),
            missile_events=("missile_event", "sum"),
            guided_bomb_events=("guided_bomb_event", "sum"),
        )
        .sort_values(["day", "oblast"])
    )

    oblast_summary = (
        regional.groupby("oblast", as_index=False)
        .agg(
            attack_events=("attack_id", "nunique"),
            active_days=("day", "nunique"),
            first_seen=("day", "min"),
            last_seen=("day", "max"),
            uav_events=("uav_event", "sum"),
            missile_events=("missile_event", "sum"),
            guided_bomb_events=("guided_bomb_event", "sum"),
        )
        .sort_values(["attack_events", "oblast"], ascending=[False, True])
    )

    weapon_summary = (
        attacks.groupby("attack_type", as_index=False)
        .agg(attack_records=("attack_id", "nunique"))
        .sort_values("attack_records", ascending=False)
    )

    return {
        "national_daily": national_daily,
        "oblast_daily": oblast_daily,
        "oblast_summary": oblast_summary,
        "weapon_summary": weapon_summary,
    }


def write_dashboard_snapshot(
    attacks: pd.DataFrame,
    attack_regions: pd.DataFrame,
    weapons: pd.DataFrame,
    quality_report: dict[str, object] | None = None,
    learning_report: dict[str, object] | None = None,
    output_dir: Path = DASHBOARD_DIR,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_dashboard_tables(attacks, attack_regions, weapons)

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    quality_report = quality_report or {}
    learning_report = learning_report or {}

    latest_source = pd.to_datetime(
        attacks["started_at"], utc=True, errors="coerce", format="mixed"
    ).max()

    recommendations = build_improvement_recommendations(
        quality_report,
        learning_report,
        latest_source_event_at=(
            latest_source.isoformat() if pd.notna(latest_source) else None
        ),
    )
    readiness_score = project_readiness_score(
        quality_report,
        learning_report,
        latest_source_event_at=(
            latest_source.isoformat() if pd.notna(latest_source) else None
        ),
    )

    metadata: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_source_event_at": (
            latest_source.isoformat() if pd.notna(latest_source) else None
        ),
        "source_name": "Massive Missile Attacks on Ukraine (Kaggle / piterfm)",
        "source_scope": "Historical public aggregate data",
        "attack_rows": int(len(attacks)),
        "region_links": int(len(attack_regions)),
        "weapon_rows": int(len(weapons)),
        "quality": quality_report,
        "learning": learning_report,
        "technical_readiness_score": readiness_score,
        "improvement_recommendations": recommendations,
        "safety_scope": (
            "Historical aggregated oblast-level analysis only; no live routes, "
            "exact targets, launch coordinates, or operational timing."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history_path = output_dir / "health_history.csv"
    candidate_metrics = learning_report.get("candidate_metrics") or {}
    history_row = pd.DataFrame(
        [
            {
                "generated_at": metadata["generated_at"],
                "latest_source_event_at": metadata["latest_source_event_at"],
                "technical_readiness_score": readiness_score,
                "region_coverage_rate": quality_report.get("region_coverage_rate"),
                "recent_region_coverage_90d": quality_report.get(
                    "recent_region_coverage_90d"
                ),
                "candidate_average_precision": candidate_metrics.get(
                    "average_precision"
                ),
                "candidate_brier_score": candidate_metrics.get("brier_score"),
                "model_ready_for_serving": learning_report.get(
                    "model_ready_for_serving", False
                ),
                "warnings": quality_report.get("warnings"),
            }
        ]
    )
    if history_path.exists():
        previous_history = pd.read_csv(history_path)
        history = pd.concat(
            [previous_history, history_row],
            ignore_index=True,
        )
    else:
        history = history_row

    history = history.drop_duplicates(
        subset=["generated_at"],
        keep="last",
    ).tail(104)
    history.to_csv(history_path, index=False)

    return metadata
