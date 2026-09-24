import pandas as pd
import pytest

from src.ml.dataset import build_daily_oblast_dataset, chronological_split
from src.ml.training import (
    ModelMetrics, beats_baseline, decide_promotion, should_promote,
    validate_training_dataset, champion_holdout_is_clean,
)


def test_daily_dataset_uses_only_prior_days():
    attacks = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-01-01T10:00:00Z"},
        {"attack_id": 2, "started_at": "2026-01-03T10:00:00Z"},
        {"attack_id": 3, "started_at": "2026-01-10T10:00:00Z"},
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
        {"attack_id": 2, "oblast": "Одеська область"},
        {"attack_id": 3, "oblast": "Одеська область"},
    ])

    dataset = build_daily_oblast_dataset(attacks, regions, min_history_days=0)

    jan2 = dataset.loc[dataset["day"] == pd.Timestamp("2026-01-02", tz="UTC")].iloc[0]
    jan3 = dataset.loc[dataset["day"] == pd.Timestamp("2026-01-03", tz="UTC")].iloc[0]

    assert pd.isna(jan2["target_next_24h"])
    assert jan2["attacks_prev_1d"] == 1

    assert jan3["target_next_24h"] == 1
    assert jan3["attacks_prev_1d"] == 0
    assert jan3["attacks_prev_7d"] == 1


def test_certified_complete_day_can_be_negative_but_other_days_remain_unknown():
    attacks = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-01-01T10:00:00Z"},
        {"attack_id": 2, "started_at": "2026-01-04T10:00:00Z"},
    ])
    regions = pd.DataFrame([
        {"attack_id": 1, "oblast": "Одеська область"},
        {"attack_id": 2, "oblast": "Одеська область"},
    ])
    observations = pd.DataFrame([
        {"oblast": "Одеська область", "day": "2026-01-02", "observed_complete": True,
         "source_reference": "audited-source-2026-01-02"},
    ])
    result = build_daily_oblast_dataset(attacks, regions, min_history_days=0,
                                        observation_days=observations)
    assert result.loc[result["day"].eq(pd.Timestamp("2026-01-02", tz="UTC")),
                      "target_next_24h"].iloc[0] == 0
    assert pd.isna(result.loc[result["day"].eq(pd.Timestamp("2026-01-03", tz="UTC")),
                            "target_next_24h"].iloc[0])
    assert validate_training_dataset(result)[-1] == "Target contains unverified oblast/day outcomes"


def test_unverified_or_unreferenced_observations_cannot_create_negative_labels():
    attacks = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-01-01T10:00:00Z"},
        {"attack_id": 2, "started_at": "2026-01-03T10:00:00Z"},
    ])
    regions = pd.DataFrame([
        {"attack_id": i, "oblast": "Одеська область"} for i in (1, 2)
    ])
    for complete, reference in ((False, "source"), (True, "")):
        observations = pd.DataFrame([{
            "oblast": "Одеська область", "day": "2026-01-02",
            "observed_complete": complete, "source_reference": reference,
        }])
        with pytest.raises(ValueError, match="complete|source reference"):
            build_daily_oblast_dataset(attacks, regions, min_history_days=0,
                                       observation_days=observations)


def test_chronological_split_never_mixes_future_into_train():
    attacks = pd.DataFrame([
        {"attack_id": i, "started_at": f"2026-01-{i:02d}T10:00:00Z"}
        for i in range(1, 21)
    ])
    regions = pd.DataFrame([
        {"attack_id": i, "oblast": "Одеська область"} for i in range(1, 21)
    ])
    dataset = build_daily_oblast_dataset(attacks, regions, min_history_days=0)
    train, test = chronological_split(dataset, test_fraction=0.2)

    assert train["day"].max() < test["day"].min()


def test_champion_cannot_be_evaluated_on_days_seen_during_training():
    holdout = pd.Timestamp("2026-05-01", tz="UTC")
    assert champion_holdout_is_clean({"data_end": "2026-04-30"}, holdout)
    assert not champion_holdout_is_clean({"data_end": "2026-05-01"}, holdout)
    assert not champion_holdout_is_clean({"data_end": "2026-05-02"}, holdout)
    assert not champion_holdout_is_clean({}, holdout)


def test_candidate_promotion_requires_meaningful_improvement():
    champion = ModelMetrics(
        rows=100,
        positives=20,
        positive_rate=0.2,
        brier_score=0.20,
        average_precision=0.40,
        roc_auc=0.70,
        balanced_accuracy=0.65,
    )
    better = ModelMetrics(
        rows=100,
        positives=20,
        positive_rate=0.2,
        brier_score=0.19,
        average_precision=0.40,
        roc_auc=0.71,
        balanced_accuracy=0.66,
    )
    worse = ModelMetrics(
        rows=100,
        positives=20,
        positive_rate=0.2,
        brier_score=0.22,
        average_precision=0.38,
        roc_auc=0.68,
        balanced_accuracy=0.63,
    )

    assert should_promote(better, champion)[0] is True
    assert should_promote(worse, champion)[0] is False


def test_candidate_must_beat_prevalence_baseline():
    baseline = ModelMetrics(
        rows=100,
        positives=10,
        positive_rate=0.1,
        brier_score=0.09,
        average_precision=0.10,
        roc_auc=0.5,
        balanced_accuracy=0.5,
    )
    useful = ModelMetrics(
        rows=100,
        positives=10,
        positive_rate=0.1,
        brier_score=0.08,
        average_precision=0.15,
        roc_auc=0.7,
        balanced_accuracy=0.6,
    )
    miscalibrated = ModelMetrics(
        rows=100,
        positives=10,
        positive_rate=0.1,
        brier_score=0.20,
        average_precision=0.20,
        roc_auc=0.8,
        balanced_accuracy=0.6,
    )

    assert beats_baseline(useful, baseline)[0] is True
    assert beats_baseline(miscalibrated, baseline)[0] is False

    baseline_pass, promoted, reason = decide_promotion(
        useful, baseline, None, quality_ready=False
    )
    assert baseline_pass is True
    assert promoted is False
    assert reason == "Data-quality gate blocks promotion"

    assert decide_promotion(
        useful, baseline, None, quality_ready=True
    )[1] is True
