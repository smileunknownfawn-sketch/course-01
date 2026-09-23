import pandas as pd
import pytest

from src.data.quality import assert_quality_gate, build_quality_report


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
