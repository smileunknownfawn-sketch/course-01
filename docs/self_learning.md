# Self-learning cycle

The project uses controlled retraining rather than uncontrolled self-modification.

## What “self-learning” means here

1. Download the newest public historical source snapshot.
2. Normalize source records into stable project tables.
3. Run data-quality checks.
4. Build a leakage-safe daily oblast dataset.
5. Train a new challenger model.
6. Evaluate the challenger on a chronological holdout (майбутній відрізок даних, який не використовувався для навчання).
7. Evaluate the current champion on the same holdout.
8. Promote the challenger only when it exceeds explicit thresholds.
9. Preserve a machine-readable report for audit.

The model does not edit raw source facts, invent missing values, or silently “correct” source records.

## Forecast scope

The ML task is intentionally coarse:

- geography: oblast;
- horizon: one UTC calendar day / approximately 24 hours;
- target: whether at least one historical attack affected the oblast;
- features: calendar information and prior historical event counts.

The project does not forecast exact targets, live routes, launch coordinates, or precise operational timing.

## Quality gate

Retraining is blocked when the normalized data contains:

- duplicate attack IDs;
- invalid event timestamps;
- missing attack types;
- region links pointing to nonexistent attacks;
- negative weapon quantities.

Warnings that do not automatically rewrite the source include:

- unmapped source region labels;
- intercepted/destroyed counts greater than launched counts.

Warnings remain visible for human review because source semantics may differ.

## Model promotion

The current model is the **champion**.
A newly trained model is the **challenger**.

The challenger is evaluated against the champion on the same time-based holdout.
Promotion requires a meaningful improvement in probability calibration (Brier score) or average precision without material degradation in the other metric.

If no champion exists, the first model becomes the champion.

## Persistence

Generated models and reports live under `models/` and are not committed to Git.
GitHub Actions preserves the champion between learning-cycle runs with an Actions cache and uploads model/report artifacts for inspection.

## Automation

`.github/workflows/learning-cycle.yml` runs:

- every Monday at 03:00 UTC;
- manually through GitHub Actions;
- when ML/data pipeline code changes on `main`.

Pipeline:

```text
Kaggle snapshot
  -> normalized tables
  -> quality gate
  -> daily oblast ML dataset
  -> chronological train/test split
  -> challenger training
  -> champion vs challenger evaluation
  -> conditional promotion
  -> saved report/artifact
```

## Local run

```bash
python scripts/download_kaggle.py
python scripts/build_processed_kaggle.py
python scripts/data_quality_report.py
python scripts/self_improve.py
```

Then run:

```bash
pytest
```
