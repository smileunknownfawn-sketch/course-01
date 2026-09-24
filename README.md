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


## Controlled self-learning

The project now supports a champion/challenger learning cycle (чинна модель / нова модель-кандидат):

```text
fresh historical data
  -> quality checks
  -> oblast/day histories with unknown outcomes kept unknown
  -> verify complete observations for negative labels
  -> train challenger only when outcomes are verified
  -> compare on a holdout unseen by the champion
  -> promote only if quality and metrics pass
```

Run locally:

```bash
python scripts/download_kaggle.py
python scripts/build_processed_kaggle.py
python scripts/data_quality_report.py
python scripts/self_improve.py
```

The scheduled GitHub Actions workflow repeats the checks weekly. When verified
negative outcomes are unavailable, it reports why training is blocked and
continues to update the historical dashboard. Raw facts are never rewritten
automatically; suspicious or unmapped values are reported for review.

## Dashboard navigation

Run `streamlit run streamlit_app.py`. Choose a period and oblast at the top of
the page. The six clearly named sections show the historical overview,
launches and reported interceptions, oblast map, oblast comparison, source
reliability, and model status. Source dates and definitions are under
“Джерела та дати останніх записів”. A missing source period is shown as “—”,
not as zero incidents.

The interface also includes a lightweight decorative hero image, an exact
source-coverage timeline, a visual equation for launches/interceptions, a
regional-coverage threshold chart, and a step-by-step learning-cycle diagram.
Decorative imagery is explicitly separated from factual maps and charts; the
oblast map continues to use the repository's GeoJSON boundaries.

The dashboard also provides a 90-day air-alert calendar, equal-period
comparisons, selectable map layers, side-by-side oblast comparison, and a
transparent observed-activity index that is explicitly not a forecast. Two
additional optimized illustrations introduce the calendar and source-quality
sections without presenting decorative artwork as evidence.

When an oblast is selected, the map section also shows curated official event
cards from the last 30 days. Each card includes the date, reported weapon
categories, damage and casualty summary, up to three non-graphic official
photos, attribution, and a direct source link. Missing cards are explicitly
described as missing verified publications, not as evidence that no attack
occurred. See `docs/verified_events.md` for curation and safety rules.

See `docs/self_learning.md` for the full lifecycle.

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
