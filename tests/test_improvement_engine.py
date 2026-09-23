from datetime import datetime, timezone

from src.analysis.improvement import (
    build_improvement_recommendations,
    project_readiness_score,
)


def test_recommendations_prioritize_low_region_coverage():
    quality = {
        "blocking_errors": 0,
        "region_coverage_rate": 0.066,
        "recent_region_coverage_90d": 0.081,
        "unmapped_region_rows": 5,
        "warnings": 6,
    }
    learning = {
        "beats_baseline": True,
        "model_ready_for_serving": False,
        "candidate_metrics": {"average_precision": 0.19},
        "baseline_metrics": {"average_precision": 0.02},
    }

    items = build_improvement_recommendations(
        quality,
        learning,
        latest_source_event_at="2026-09-18T18:00:00+00:00",
        now=datetime(2026, 9, 23, tzinfo=timezone.utc),
    )

    titles = [item["title"] for item in items]
    assert "Збільшити регіональне покриття" in titles
    assert "Не показувати модель як готовий прогноз" in titles
    assert items == sorted(items, key=lambda item: item["priority"], reverse=True)


def test_readiness_score_improves_with_better_data():
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)

    weak = project_readiness_score(
        {
            "blocking_errors": 0,
            "region_coverage_rate": 0.06,
            "recent_region_coverage_90d": 0.08,
        },
        {"beats_baseline": True},
        "2026-09-18T00:00:00+00:00",
        now=now,
    )

    strong = project_readiness_score(
        {
            "blocking_errors": 0,
            "region_coverage_rate": 0.45,
            "recent_region_coverage_90d": 0.50,
        },
        {"beats_baseline": True},
        "2026-09-22T00:00:00+00:00",
        now=now,
    )

    assert strong > weak
    assert 0 <= weak <= 100
    assert 0 <= strong <= 100
