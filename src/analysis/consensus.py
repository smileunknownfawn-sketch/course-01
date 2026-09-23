"""Multi-source aggregation for coarse historical oblast analytics."""

from __future__ import annotations

import pandas as pd


def build_oblast_consensus(
    all_oblasts: list[str],
    kaggle_daily: pd.DataFrame,
    viina_daily: pd.DataFrame | None = None,
    siren_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a source-balanced historical oblast summary.

    Attack-like sources (Kaggle and VIINA aerial incidents) are normalized
    within each source and then averaged. Air-raid alerts remain context only
    and are never treated as attack labels.
    """
    base = pd.DataFrame({"oblast": sorted(set(all_oblasts))})

    kaggle = kaggle_daily.copy()
    if kaggle.empty:
        kaggle_summary = pd.DataFrame(columns=["oblast", "kaggle_events"])
    else:
        kaggle_summary = (
            kaggle.groupby("oblast", as_index=False)
            .agg(kaggle_events=("attack_events", "sum"))
        )
    base = base.merge(kaggle_summary, on="oblast", how="left")
    base["kaggle_events"] = pd.to_numeric(
        base["kaggle_events"], errors="coerce"
    ).fillna(0).astype("int64")

    viina_daily = viina_daily if viina_daily is not None else pd.DataFrame()
    if viina_daily.empty:
        viina_summary = pd.DataFrame(columns=["oblast", "viina_events"])
    else:
        viina_summary = (
            viina_daily.groupby("oblast", as_index=False)
            .agg(viina_events=("viina_events", "sum"))
        )
    base = base.merge(viina_summary, on="oblast", how="left")
    base["viina_events"] = pd.to_numeric(
        base["viina_events"], errors="coerce"
    ).fillna(0).astype("int64")

    siren_daily = siren_daily if siren_daily is not None else pd.DataFrame()
    if siren_daily.empty:
        siren_summary = pd.DataFrame(
            columns=["oblast", "alert_count", "alert_minutes"]
        )
    else:
        siren_summary = (
            siren_daily.groupby("oblast", as_index=False)
            .agg(
                alert_count=("alert_count", "sum"),
                alert_minutes=("alert_minutes", "sum"),
            )
        )
    base = base.merge(siren_summary, on="oblast", how="left")
    base["alert_count"] = pd.to_numeric(
        base["alert_count"], errors="coerce"
    ).fillna(0).astype("int64")
    base["alert_minutes"] = pd.to_numeric(
        base["alert_minutes"], errors="coerce"
    ).fillna(0.0)

    shares: list[str] = []

    kaggle_total = float(base["kaggle_events"].sum())
    if kaggle_total > 0:
        base["kaggle_share_pct"] = base["kaggle_events"] / kaggle_total * 100
        shares.append("kaggle_share_pct")
    else:
        base["kaggle_share_pct"] = 0.0

    viina_total = float(base["viina_events"].sum())
    if viina_total > 0:
        base["viina_share_pct"] = base["viina_events"] / viina_total * 100
        shares.append("viina_share_pct")
    else:
        base["viina_share_pct"] = 0.0

    if shares:
        base["consensus_share_pct"] = base[shares].mean(axis=1)
    else:
        base["consensus_share_pct"] = 0.0

    base["evidence_sources"] = (
        base["kaggle_events"].gt(0).astype("int8")
        + base["viina_events"].gt(0).astype("int8")
    )
    base["active_attack_sources"] = len(shares)

    return base.sort_values(
        ["consensus_share_pct", "oblast"],
        ascending=[False, True],
    ).reset_index(drop=True)
