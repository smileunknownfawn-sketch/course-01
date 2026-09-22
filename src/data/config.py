"""Project configuration helpers (налаштування проєкту)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
GEO_DIR = ROOT_DIR / "data" / "geo"


def get_env(name: str) -> str | None:
    """Read a secret from environment variables without storing it in Git."""
    value = os.getenv(name)
    return value.strip() if value else None
