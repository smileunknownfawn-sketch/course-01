"""Training and evaluation for the coarse oblast/day risk model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.dataset import FEATURE_COLUMNS

TARGET_COLUMN = "target_next_24h"
CATEGORICAL_FEATURES = ["oblast"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]


@dataclass(frozen=True)
class ModelMetrics:
    rows: int
    positives: int
    positive_rate: float
    brier_score: float
    average_precision: float | None
    roc_auc: float | None
    balanced_accuracy: float


def validate_training_dataset(dataset: pd.DataFrame) -> list[str]:
    """Return quality-gate errors that should block training."""
    errors: list[str] = []
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, "day"])
    missing = required.difference(dataset.columns)
    if missing:
        return [f"Missing training columns: {sorted(missing)}"]

    if len(dataset) < 500:
        errors.append("Training dataset has fewer than 500 rows")

    if dataset["day"].nunique() < 60:
        errors.append("Training dataset has fewer than 60 distinct days")

    if dataset["oblast"].nunique() < 3:
        errors.append("Training dataset has fewer than 3 oblasts")

    if dataset.duplicated(["oblast", "day"]).any():
        errors.append("Training dataset contains duplicate oblast/day rows")

    target = dataset[TARGET_COLUMN]
    if target.isna().any():
        errors.append("Target contains missing values")
    else:
        positive_rate = float(target.mean())
        if not 0.005 <= positive_rate <= 0.8:
            errors.append(
                f"Target positive rate {positive_rate:.4f} is outside the expected range"
            )

    return errors


def build_model() -> Pipeline:
    """Create an interpretable baseline probability model."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "oblast",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(train: pd.DataFrame) -> Pipeline:
    errors = validate_training_dataset(train)
    if errors:
        raise ValueError("Training quality gate failed: " + "; ".join(errors))

    model = build_model()
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


def evaluate_model(model: Pipeline, test: pd.DataFrame) -> ModelMetrics:
    if test.empty:
        raise ValueError("Test dataset is empty")

    y_true = test[TARGET_COLUMN].astype("int8")
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")

    average_precision: float | None
    roc_auc: float | None
    if y_true.nunique() > 1:
        average_precision = float(average_precision_score(y_true, probabilities))
        roc_auc = float(roc_auc_score(y_true, probabilities))
    else:
        average_precision = None
        roc_auc = None

    return ModelMetrics(
        rows=len(test),
        positives=int(y_true.sum()),
        positive_rate=float(y_true.mean()),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        average_precision=average_precision,
        roc_auc=roc_auc,
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
    )


def evaluate_prevalence_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> ModelMetrics:
    """Evaluate a constant-probability baseline using training prevalence."""
    if train.empty or test.empty:
        raise ValueError("Train and test datasets must be non-empty")

    y_true = test[TARGET_COLUMN].astype("int8")
    probability = float(train[TARGET_COLUMN].mean())
    probabilities = np.full(len(test), probability, dtype=float)
    predictions = (probabilities >= 0.5).astype("int8")

    average_precision: float | None
    roc_auc: float | None
    if y_true.nunique() > 1:
        average_precision = float(average_precision_score(y_true, probabilities))
        roc_auc = 0.5
    else:
        average_precision = None
        roc_auc = None

    return ModelMetrics(
        rows=len(test),
        positives=int(y_true.sum()),
        positive_rate=float(y_true.mean()),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        average_precision=average_precision,
        roc_auc=roc_auc,
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
    )


def metrics_to_dict(metrics: ModelMetrics) -> dict[str, object]:
    return asdict(metrics)


def beats_baseline(
    candidate: ModelMetrics,
    baseline: ModelMetrics,
) -> tuple[bool, str]:
    """Require a candidate to outperform a simple prevalence baseline."""
    candidate_ap = candidate.average_precision
    baseline_ap = baseline.average_precision

    if candidate.brier_score >= baseline.brier_score:
        return False, "Candidate probability calibration is worse than baseline"

    if (
        candidate_ap is not None
        and baseline_ap is not None
        and candidate_ap <= baseline_ap
    ):
        return False, "Candidate ranking quality does not exceed baseline"

    return True, "Candidate exceeds prevalence baseline"


def should_promote(
    candidate: ModelMetrics,
    champion: ModelMetrics | None,
) -> tuple[bool, str]:
    """Decide whether a candidate is meaningfully better on the same holdout."""
    if champion is None:
        return True, "No champion exists yet"

    candidate_ap = candidate.average_precision
    champion_ap = champion.average_precision

    brier_improvement = champion.brier_score - candidate.brier_score
    ap_change = (
        0.0
        if candidate_ap is None or champion_ap is None
        else candidate_ap - champion_ap
    )

    if brier_improvement >= 0.002 and ap_change >= -0.02:
        return True, "Brier score improved without material AP degradation"

    if ap_change >= 0.02 and candidate.brier_score <= champion.brier_score + 0.002:
        return True, "Average precision improved without material calibration degradation"

    return False, "Candidate did not exceed promotion thresholds"


def decide_promotion(
    candidate: ModelMetrics,
    baseline: ModelMetrics,
    champion: ModelMetrics | None,
    *,
    quality_ready: bool,
) -> tuple[bool, bool, str]:
    """Evaluate a candidate while keeping incomplete regional data out of production."""
    baseline_pass, baseline_reason = beats_baseline(candidate, baseline)
    if not quality_ready:
        return baseline_pass, False, "Data-quality gate blocks promotion"
    if not baseline_pass:
        return False, False, baseline_reason
    promote, reason = should_promote(candidate, champion)
    return True, promote, reason
