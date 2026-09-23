import json

import pandas as pd

from src.dashboard.data import build_dashboard_tables, write_dashboard_snapshot


def test_dashboard_tables_do_not_duplicate_national_weapon_quantities():
    attacks = pd.DataFrame([
        {
            "attack_id": 1,
            "started_at": "2026-01-01T10:00:00Z",
            "attack_type": "uav",
        },
        {
            "attack_id": 2,
            "started_at": "2026-01-02T10:00:00Z",
            "attack_type": "missile",
        },
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
        {"attack_id": 1, "oblast": "Київська область"},
        {"attack_id": 2, "oblast": "Одеська область"},
    ])
    weapons = pd.DataFrame([
        {
            "attack_id": 1,
            "quantity": 10,
            "intercepted_quantity": 8,
        },
        {
            "attack_id": 2,
            "quantity": 3,
            "intercepted_quantity": 1,
        },
    ])

    tables = build_dashboard_tables(attacks, regions, weapons)

    assert tables["national_daily"]["launched_reported"].sum() == 13
    assert tables["oblast_daily"]["attack_events"].sum() == 3
    assert (
        tables["oblast_summary"]
        .set_index("oblast")
        .loc["Одеська область", "attack_events"]
        == 2
    )


def test_dashboard_snapshot_writes_metadata_and_tables(tmp_path):
    attacks = pd.DataFrame([
        {
            "attack_id": 1,
            "started_at": "2026-01-01T10:00:00Z",
            "attack_type": "uav",
        },
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
    ])
    weapons = pd.DataFrame([
        {
            "attack_id": 1,
            "quantity": 2,
            "intercepted_quantity": 1,
        },
    ])

    metadata = write_dashboard_snapshot(
        attacks,
        regions,
        weapons,
        quality_report={"region_coverage_rate": 1.0},
        learning_report={"promoted": False},
        output_dir=tmp_path,
    )

    assert metadata["attack_rows"] == 1
    assert (tmp_path / "national_daily.csv").exists()
    assert (tmp_path / "oblast_daily.csv").exists()
    assert (tmp_path / "oblast_summary.csv").exists()
    assert (tmp_path / "weapon_summary.csv").exists()

    stored = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert stored["quality"]["region_coverage_rate"] == 1.0
