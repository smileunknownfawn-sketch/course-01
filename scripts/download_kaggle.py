"""Download the current public Kaggle source files into data/raw."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import kagglehub

from src.data.config import RAW_DIR

DATASET = "piterfm/massive-missile-attacks-on-ukraine"
FILES = ("missile_attacks_daily.csv", "missiles_and_uavs.csv")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        path = kagglehub.dataset_download(
            DATASET,
            path=filename,
            output_dir=str(RAW_DIR),
            force_download=True,
        )
        print(f"{filename} -> {path}")


if __name__ == "__main__":
    main()
