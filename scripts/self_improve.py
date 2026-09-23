"""Run one controlled self-improvement cycle for the historical risk model."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.config import PROCESSED_DIR
from src.ml.dataset import build_daily_oblast_dataset, chronological_split
from src.ml.registry import (
    MODEL_DIR,
    load_champion_model,
    promote_candidate,
    save_candidate,
)
from src.ml.training import evaluate_model, should_promote, train_model


def main() -> None:
    attacks_path = PROCESSED_DIR / "attacks.csv"
    regions_path = PROCESSED_DIR / "attack_regions.csv"

    if not attacks_path.exists() or not regions_path.exists():
        raise FileNotFoundError(
            "Processed data is missing. Run scripts/build_processed_kaggle.py first."
        )

    attacks = pd.read_csv(attacks_path)
    attack_regions = pd.read_csv(regions_path)

    dataset = build_daily_oblast_dataset(attacks, attack_regions)
    dataset_path = PROCESSED_DIR / "ml_daily_oblast.csv"
    dataset.to_csv(dataset_path, index=False)

    train, test = chronological_split(dataset, test_fraction=0.2)

    candidate = train_model(train)
    candidate_metrics = evaluate_model(candidate, test)

    champion = load_champion_model()
    champion_metrics = evaluate_model(champion, test) if champion is not None else None

    data_start = str(dataset["day"].min())
    data_end = str(dataset["day"].max())
    save_candidate(candidate, candidate_metrics, data_start, data_end)

    promote, reason = should_promote(candidate_metrics, champion_metrics)
    if promote:
        promote_candidate(reason)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "data_start": data_start,
        "data_end": data_end,
        "train_rows": len(train),
        "test_rows": len(test),
        "candidate_metrics": asdict(candidate_metrics),
        "champion_metrics_on_same_holdout": (
            asdict(champion_metrics) if champion_metrics is not None else None
        ),
        "promoted": promote,
        "decision_reason": reason,
    }
    report_path = MODEL_DIR / "learning_cycle_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
