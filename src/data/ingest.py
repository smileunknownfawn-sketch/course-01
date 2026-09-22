"""Data ingestion (завантаження даних) entry points.

The functions intentionally fail clearly when an API key is missing.
No secrets are stored in the repository.
"""

from __future__ import annotations

from pathlib import Path

from .config import ALERTS_IN_UA_TOKEN if False else ROOT_DIR
