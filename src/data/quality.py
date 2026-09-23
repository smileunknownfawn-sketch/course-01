"""Data-quality diagnostics for normalized historical attack data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    attack_rows: int
    region_rows: int
    weapon_rows: int
    attacks_with_region: int
    region_coverage_rate: float
    recent_attack_rows_90d: int
    recent_attacks_with_region_90d: int
    recent_region_coverage_90d: float
    latest_attack_date: str | None
    latest_region_labeled_date: str | None
    region_label_lag_days: int | None
    duplicate_attack_ids: int
    invalid_attack_dates: int
    missing_attack_types: int
    orphan_region_links: int
    negative_weapon_quantities: int
    intercepted_above_launched: int
    unmapped_region_rows: int
    source_duplicate_groups: int
    blocking_errors: int
    warnings: int


def build_quality_report(
    attacks: pd.DataFrame,
    attack_regions: pd.DataFrame,
    weapons: pd.DataFrame,
    unmapped_regions: pd.DataFrame | None = None,
    source_duplicates: pd.DataFrame | None = None,
) -> DataQualityReport:
    attacks = attacks.copy()
    attack_regions = attack_regions.copy()
    weapons = weapons.copy()

    parsed_dates = pd.to_datetime(
        attacks.get("started_at"), utc=True, errors="coerce", format="mixed"
    )
    attack_ids = attacks.get("attack_id", pd.Series(dtype=object))
    duplicate_attack_ids = int(attack_ids.duplicated().sum())
    invalid_attack_dates = int(parsed_dates.isna().sum())
    missing_attack_types = int(
        attacks.get("attack_type", pd.Series(dtype=object)).isna().sum()
    )

    known_ids = set(attack_ids.dropna())
    region_ids = attack_regions.get("attack_id", pd.Series(dtype=object))
    orphan_region_links = int((~region_ids.isin(known_ids)).sum())

    region_attack_ids = set(region_ids.dropna()).intersection(known_ids)
    attacks_with_region = len(region_attack_ids)
    region_coverage_rate = (
        attacks_with_region / len(attacks) if len(attacks) else 0.0
    )

    dated = attacks[["attack_id"]].copy()
    dated["started_at"] = parsed_dates
    valid_dated = dated.dropna(subset=["started_at"])
    latest_attack = (
        valid_dated["started_at"].max() if not valid_dated.empty else None
    )

    region_dated = attack_regions[["attack_id"]].drop_duplicates().merge(
        valid_dated,
        on="attack_id",
        how="inner",
    )
    latest_region = (
        region_dated["started_at"].max() if not region_dated.empty else None
    )

    region_label_lag_days: int | None
    if latest_attack is not None and latest_region is not None:
        region_label_lag_days = max(0, int((latest_attack - latest_region).days))
    else:
        region_label_lag_days = None

    recent_attack_rows_90d = 0
    recent_attacks_with_region_90d = 0
    recent_region_coverage_90d = 0.0
    if latest_attack is not None:
        recent_start = latest_attack - pd.Timedelta(days=90)
        recent = valid_dated[valid_dated["started_at"] >= recent_start]
        recent_attack_rows_90d = len(recent)
        if recent_attack_rows_90d:
            recent_ids = set(recent["attack_id"])
            recent_attacks_with_region_90d = len(
                recent_ids.intersection(region_attack_ids)
            )
            recent_region_coverage_90d = (
                recent_attacks_with_region_90d / recent_attack_rows_90d
            )

    quantity = pd.to_numeric(
        weapons.get("quantity", pd.Series(dtype=float)), errors="coerce"
    )
    intercepted = pd.to_numeric(
        weapons.get("intercepted_quantity", pd.Series(dtype=float)), errors="coerce"
    )
    negative_weapon_quantities = int(
        ((quantity < 0).fillna(False) | (intercepted < 0).fillna(False)).sum()
    )
    intercepted_above_launched = int(
        ((intercepted > quantity) & quantity.notna() & intercepted.notna()).sum()
    )

    unmapped_region_rows = 0 if unmapped_regions is None else len(unmapped_regions)
    source_duplicate_groups = (
        0 if source_duplicates is None else len(source_duplicates)
    )

    blocking_errors = (
        duplicate_attack_ids
        + invalid_attack_dates
        + missing_attack_types
        + orphan_region_links
        + negative_weapon_quantities
    )
    warnings = (
        intercepted_above_launched
        + unmapped_region_rows
        + source_duplicate_groups
        + int(region_label_lag_days is not None and region_label_lag_days > 14)
        + int(recent_region_coverage_90d < 0.25)
    )

    return DataQualityReport(
        attack_rows=len(attacks),
        region_rows=len(attack_regions),
        weapon_rows=len(weapons),
        attacks_with_region=attacks_with_region,
        region_coverage_rate=float(region_coverage_rate),
        recent_attack_rows_90d=recent_attack_rows_90d,
        recent_attacks_with_region_90d=recent_attacks_with_region_90d,
        recent_region_coverage_90d=float(recent_region_coverage_90d),
        latest_attack_date=latest_attack.isoformat() if latest_attack is not None else None,
        latest_region_labeled_date=(
            latest_region.isoformat() if latest_region is not None else None
        ),
        region_label_lag_days=region_label_lag_days,
        duplicate_attack_ids=duplicate_attack_ids,
        invalid_attack_dates=invalid_attack_dates,
        missing_attack_types=missing_attack_types,
        orphan_region_links=orphan_region_links,
        negative_weapon_quantities=negative_weapon_quantities,
        intercepted_above_launched=intercepted_above_launched,
        unmapped_region_rows=unmapped_region_rows,
        source_duplicate_groups=source_duplicate_groups,
        blocking_errors=blocking_errors,
        warnings=warnings,
    )


def report_to_dict(report: DataQualityReport) -> dict[str, object]:
    return asdict(report)


def assert_quality_gate(report: DataQualityReport) -> None:
    """Block model training when factual integrity checks fail."""
    if report.blocking_errors:
        raise ValueError(
            f"Data quality gate failed with {report.blocking_errors} blocking errors"
        )


def assess_model_readiness(report: DataQualityReport) -> tuple[bool, list[str]]:
    """Assess whether regional labels are fresh/complete enough for serving."""
    reasons: list[str] = []

    if report.region_coverage_rate < 0.25:
        reasons.append(
            f"Overall region-label coverage is only {report.region_coverage_rate:.1%}"
        )

    if report.recent_region_coverage_90d < 0.25:
        reasons.append(
            "Recent 90-day region-label coverage is below 25%"
        )

    if report.region_label_lag_days is None:
        reasons.append("Cannot determine regional label freshness")
    elif report.region_label_lag_days > 14:
        reasons.append(
            f"Regional labels lag source attacks by {report.region_label_lag_days} days"
        )

    return len(reasons) == 0, reasons
