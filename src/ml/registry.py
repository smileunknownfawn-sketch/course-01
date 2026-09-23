"""Local model registry for champion/challenger training."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.ml.training import ModelMetrics

MODEL_DIR = Path("models")
CHAMPION_MODEL = MODEL_DIR / "champion.joblib"
CHAMPION_METADATA = MODEL_DIR / "champion.json"
CANDIDATE_MODEL = MODEL_DIR / "candidate.joblib"
CANDIDATE_METADATA = MODEL_DIR / "candidate.json"


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_candidate(
    model: object,
    metrics: ModelMetrics,
    data_start: str,
    data_end: str,
) -> None:
    ensure_model_dir()
    joblib.dump(model, CANDIDATE_MODEL)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_start": data_start,
        "data_end": data_end,
        "metrics": asdict(metrics),
    }
    CANDIDATE_METADATA.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_champion_model() -> object | None:
    if not CHAMPION_MODEL.exists():
        return None
    return joblib.load(CHAMPION_MODEL)


def load_champion_metadata() -> dict[str, object] | None:
    if not CHAMPION_METADATA.exists():
        return None
    return json.loads(CHAMPION_METADATA.read_text(encoding="utf-8"))


def promote_candidate(reason: str) -> None:
    ensure_model_dir()
    if not CANDIDATE_MODEL.exists() or not CANDIDATE_METADATA.exists():
        raise FileNotFoundError("Candidate model or metadata is missing")

    metadata = json.loads(CANDIDATE_METADATA.read_text(encoding="utf-8"))
    metadata["promotion_reason"] = reason
    metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()

    shutil.copy2(CANDIDATE_MODEL, CHAMPION_MODEL)
    CHAMPION_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
