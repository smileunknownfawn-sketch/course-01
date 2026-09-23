"""Build normalized CSV files from the Kaggle historical dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.adapters.kaggle_attacks import load_and_transform
from src.data.config import PROCESSED_DIR, RAW_DIR


def main() -> None:
    source_path = RAW_DIR / "missile_attacks_daily.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}. Put missile_attacks_daily.csv in data/raw first."
        )

    tables = load_and_transform(source_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        output = PROCESSED_DIR / f"{name}.csv"
        table.to_csv(output, index=False)
        print(f"{name}: {len(table)} rows -> {output}")


if __name__ == "__main__":
    main()
