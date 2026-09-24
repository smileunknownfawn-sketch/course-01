"""Check that the visible summary follows the controls a reader actually uses."""

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _metric(app: AppTest, label: str) -> str:
    return next(item.value for item in app.metric if item.label == label)


def _format_count(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def test_period_and_oblast_controls_update_the_summary():
    national = pd.read_csv(ROOT / "data/dashboard/national_daily.csv", parse_dates=["day"])
    regional = pd.read_csv(ROOT / "data/dashboard/oblast_daily.csv", parse_dates=["day"])
    latest = national["day"].max()
    since = latest - pd.Timedelta(days=29)

    app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 6
    chart_by_key = {chart.key: chart for chart in app.get("plotly_chart")}
    assert {
        "source_coverage",
        "interception_composition",
        "region_coverage_quality",
    }.issubset(chart_by_key)
    assert _metric(app, "Kaggle · записів з областю") != "—"
    assert _metric(app, "VIINA · повітряних інцидентів") != "—"
    assert _metric(app, "Частка області-лідера").endswith("%")
    coverage_spec = json.loads(chart_by_key["source_coverage"].proto.spec)
    assert coverage_spec["layout"]["font"]["size"] >= 18
    assert any("2025-08-27" in str(trace["x"]) for trace in coverage_spec["data"])

    app.selectbox[0].set_value("Останні 30 днів").run()
    assert not app.exception
    expected_national = int(national.loc[national["day"] >= since, "attack_records"].sum())
    assert _metric(app, "Записів атак · Kaggle") == _format_count(expected_national)
    assert _metric(app, "Записів атак") == _format_count(expected_national)
    assert _metric(app, "Повітряних інцидентів · VIINA") == "—"

    app.selectbox[1].set_value("Одеська область").run()
    assert not app.exception
    expected_regional = int(regional.loc[
        regional["day"].ge(since) & regional["oblast"].eq("Одеська область"),
        "attack_events",
    ].sum())
    assert _metric(app, "Записів атак · Kaggle") == _format_count(expected_regional)


def test_hero_asset_is_wide_and_optimized():
    hero = ROOT / "assets/dashboard-hero-v3.webp"
    assert hero.exists()
    assert hero.stat().st_size < 150_000
    with Image.open(hero) as image:
        assert image.width >= 1800
        assert image.width / image.height >= 2.5
