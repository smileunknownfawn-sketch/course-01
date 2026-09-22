"""Data ingestion (завантаження даних) helpers.

Secrets are read from environment variables and are never stored in Git.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import RAW_DIR


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to your local .env file; never commit the secret."
        )
    return value.strip()


def save_raw_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save an untouched source extract to data/raw/."""
    if not filename.endswith(".csv"):
        raise ValueError("Raw data filename must end with .csv")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    return path
