"""Data-quality diagnostics for normalized historical attack data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    attack_rows: int
    region_rows: int
    weapon_rows: int
    duplicate_attack_ids: int
    invalid_attack_dates: int
    missing_attack_types: int
    orphan_region_links: int
    negative_weapon_quantities: int
    intercepted_above_launched: int
    unmapped_region_rows: int
    blocking_errors: int
    warnings: int


def build_quality_report(
    attacks: pd.DataFrame,
    attack_regions: pd.DataFrame,
    weapons: pd.DataFrame,
    unmapped_regions: pd.DataFrame | None = None,
) -> DataQualityReport:
    attacks = attacks.copy()
    attack_regions = attack_regions.copy()
    weapons = weapons.copy()

    parsed_dates = pd.to_datetime(attacks.get("started_at"), utc=True, errors="coerce")
    duplicate_attack_ids = int(attacks.get("attack_id", pd.Series(dtype=object)).duplicated().sum())
    invalid_attack_dates = int(parsed_dates.isna().sum())
    missing_attack_types = int(
        attacks.get("attack_type", pd.Series(dtype=object)).isna().sum()
    )

    known_ids = set(attacks.get("attack_id", pd.Series(dtype=object)).dropna())
    region_ids = attack_regions.get("attack_id", pd.Series(dtype=object))
    orphan_region_links = int((~region_ids.isin(known_ids)).sum())

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

    blocking_errors = (
        duplicate_attack_ids
        + invalid_attack_dates
        + missing_attack_types
        + orphan_region_links
        + negative_weapon_quantities
    )
    warnings = intercepted_above_launched + unmapped_region_rows

    return DataQualityReport(
        attack_rows=len(attacks),
        region_rows=len(attack_regions),
        weapon_rows=len(weapons),
        duplicate_attack_ids=duplicate_attack_ids,
        invalid_attack_dates=invalid_attack_dates,
        missing_attack_types=missing_attack_types,
        orphan_region_links=orphan_region_links,
        negative_weapon_quantities=negative_weapon_quantities,
        intercepted_above_launched=intercepted_above_launched,
        unmapped_region_rows=unmapped_region_rows,
        blocking_errors=blocking_errors,
        warnings=warnings,
    )


def report_to_dict(report: DataQualityReport) -> dict[str, int]:
    return asdict(report)


def assert_quality_gate(report: DataQualityReport) -> None:
    """Block model training when factual integrity checks fail."""
    if report.blocking_errors:
        raise ValueError(
            f"Data quality gate failed with {report.blocking_errors} blocking errors"
        )
