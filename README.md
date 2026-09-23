# Course 1

Data Science project: analysis and visualization of historical attacks in Ukraine using aggregated open data.

## Project
- Historical analysis by oblast
- Casualty statistics
- Weapon categories
- Historical movement/direction visualization at a generalized regional level
- Coarse-grained civilian safety risk forecasting

## Data pipeline

```text
Source data
  -> raw snapshots
  -> cleaning and validation
  -> normalized processed tables
  -> EDA / statistics
  -> ML features
  -> dashboard
```

## First data source

The first adapter supports the public Kaggle dataset
`piterfm/massive-missile-attacks-on-ukraine`.

Supported input files:
- `data/raw/missile_attacks_daily.csv`
- `data/raw/missiles_and_uavs.csv`

The adapter accepts both `affected region` and `affected_region` column names.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

## Download current Kaggle data

```bash
python scripts/download_kaggle.py
```

For public resources Kaggle may work without login. If Kaggle requires consent/authentication, configure a Kaggle API token locally. Never commit tokens to Git.

## Build processed tables

```bash
python scripts/build_processed_kaggle.py
```

Generated files:
- `data/processed/attacks.csv`
- `data/processed/attack_regions.csv`
- `data/processed/weapons.csv`
- `data/processed/provenance.csv`

Raw and processed datasets are ignored by Git and remain local.

## Tests

```bash
pytest
```

GitHub Actions also runs the test suite on pushes and pull requests to `main`.

## Data-quality rules

- Unknown values stay unknown.
- Air alerts are not treated as confirmed attacks.
- Possible duplicates are flagged instead of silently deleted.
- One historical attack can be linked to several affected regions without duplicating weapon counts.
- Source provenance (походження даних) is preserved.
- ML features use only past observations to prevent future leakage (підглядання в майбутні дані).
- Forecasts are aggregated by broad geography/time windows; the project does not model exact future targets, live routes, or launch coordinates.
