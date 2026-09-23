"""Create a machine-readable data-quality report and enforce blocking checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.config import PROCESSED_DIR
from src.data.quality import (
    assess_model_readiness,
    assert_quality_gate,
    build_quality_report,
    report_to_dict,
)


def _read(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path)


def main() -> None:
    attacks = _read("attacks.csv")
    attack_regions = _read("attack_regions.csv")
    weapons = _read("weapons.csv")

    unmapped_path = PROCESSED_DIR / "unmapped_regions.csv"
    unmapped = pd.read_csv(unmapped_path) if unmapped_path.exists() else None

    duplicates_path = PROCESSED_DIR / "source_duplicates.csv"
    source_duplicates = (
        pd.read_csv(duplicates_path) if duplicates_path.exists() else None
    )

    report = build_quality_report(
        attacks,
        attack_regions,
        weapons,
        unmapped,
        source_duplicates,
    )
    payload = report_to_dict(report)
    model_ready, readiness_reasons = assess_model_readiness(report)
    payload["model_ready_for_serving"] = model_ready
    payload["model_readiness_reasons"] = readiness_reasons

    output = PROCESSED_DIR / "data_quality_report.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    assert_quality_gate(report)


if __name__ == "__main__":
    main()
