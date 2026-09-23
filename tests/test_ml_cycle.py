import pandas as pd

from src.ml.dataset import build_daily_oblast_dataset, chronological_split
from src.ml.training import ModelMetrics, should_promote


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

    assert jan2["target_next_24h"] == 0
    assert jan2["attacks_prev_1d"] == 1

    assert jan3["target_next_24h"] == 1
    assert jan3["attacks_prev_1d"] == 0
    assert jan3["attacks_prev_7d"] == 1


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
