"""Download the current public Kaggle source files into data/raw."""

from __future__ import annotations

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
        )
        print(f"{filename} -> {path}")


if __name__ == "__main__":
    main()
