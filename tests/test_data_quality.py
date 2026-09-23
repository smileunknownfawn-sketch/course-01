import pandas as pd
import pytest

from src.data.quality import (
    assess_model_readiness,
    assert_quality_gate,
    build_quality_report,
)


def test_quality_report_blocks_duplicate_attack_ids():
    attacks = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-01-01T00:00:00Z", "attack_type": "uav"},
        {"attack_id": 1, "started_at": "2026-01-02T00:00:00Z", "attack_type": "missile"},
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
    ])
    weapons = pd.DataFrame([
        {"attack_id": 1, "quantity": 2, "intercepted_quantity": 1},
    ])

    report = build_quality_report(attacks, regions, weapons)
    assert report.duplicate_attack_ids == 1

    with pytest.raises(ValueError, match="quality gate failed"):
        assert_quality_gate(report)


def test_quality_report_keeps_unmapped_regions_as_warning():
    attacks = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-01-01T00:00:00Z", "attack_type": "uav"},
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
    ])
    weapons = pd.DataFrame([
        {"attack_id": 1, "quantity": 2, "intercepted_quantity": 1},
    ])
    unmapped = pd.DataFrame([
        {"attack_id": 1, "raw_region": "south"},
    ])

    report = build_quality_report(attacks, regions, weapons, unmapped)
    assert report.blocking_errors == 0
    assert report.unmapped_region_rows == 1
    assert report.warnings == 1

    assert_quality_gate(report)


def test_model_readiness_detects_stale_region_labels():
    attacks = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-01-01T00:00:00Z", "attack_type": "uav"},
        {"attack_id": 2, "started_at": "2026-04-10T00:00:00Z", "attack_type": "missile"},
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
    ])
    weapons = pd.DataFrame([
        {"attack_id": 1, "quantity": 2, "intercepted_quantity": 1},
        {"attack_id": 2, "quantity": 1, "intercepted_quantity": 0},
    ])

    report = build_quality_report(attacks, regions, weapons)
    ready, reasons = assess_model_readiness(report)

    assert ready is False
    assert report.region_label_lag_days > 14
    assert reasons


def test_model_readiness_accepts_fresh_complete_region_labels():
    attacks = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-04-01T00:00:00Z", "attack_type": "uav"},
        {"attack_id": 2, "started_at": "2026-04-10T00:00:00Z", "attack_type": "missile"},
        {"attack_id": 3, "started_at": "2026-04-20T00:00:00Z", "attack_type": "uav"},
        {"attack_id": 4, "started_at": "2026-04-30T00:00:00Z", "attack_type": "missile"},
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
        {"attack_id": 2, "oblast": "Одеська область"},
        {"attack_id": 3, "oblast": "Одеська область"},
        {"attack_id": 4, "oblast": "Одеська область"},
    ])
    weapons = pd.DataFrame([
        {"attack_id": i, "quantity": 1, "intercepted_quantity": 0}
        for i in range(1, 5)
    ])

    report = build_quality_report(attacks, regions, weapons)
    ready, reasons = assess_model_readiness(report)

    assert ready is True
    assert reasons == []
