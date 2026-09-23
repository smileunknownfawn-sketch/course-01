"""Build compact dashboard files from the normalized project data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dashboard.data import write_dashboard_snapshot
from src.data.config import PROCESSED_DIR

MODEL_DIR = ROOT_DIR / "models"
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    attacks_path = PROCESSED_DIR / "attacks.csv"
    regions_path = PROCESSED_DIR / "attack_regions.csv"
    weapons_path = PROCESSED_DIR / "weapons.csv"

    missing = [
        str(path)
        for path in (attacks_path, regions_path, weapons_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing processed data: " + ", ".join(missing)
        )

    attacks = pd.read_csv(attacks_path)
    attack_regions = pd.read_csv(regions_path)
    weapons = pd.read_csv(weapons_path)

    quality = _read_json(PROCESSED_DIR / "data_quality_report.json")
    learning = _read_json(MODEL_DIR / "learning_cycle_report.json")

    metadata = write_dashboard_snapshot(
        attacks,
        attack_regions,
        weapons,
        quality_report=quality,
        learning_report=learning,
        output_dir=DASHBOARD_DIR,
    )

    print(
        json.dumps(
            {
                "dashboard_dir": str(DASHBOARD_DIR),
                "attack_rows": metadata["attack_rows"],
                "region_links": metadata["region_links"],
                "generated_at": metadata["generated_at"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
