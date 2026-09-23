"""Streamlit dashboard for historical aggregated attack analytics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/darmat1/ukraine-geo-data/"
    "main/geodata/Ukraine.geojson"
)

st.set_page_config(
    page_title="Ukraine Historical Attack Analytics",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_csv(name: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_metadata() -> dict[str, object]:
    path = DASHBOARD_DIR / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=86400)
def load_geojson() -> dict[str, object] | None:
    try:
        response = requests.get(GEOJSON_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def fmt_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column], utc=True, errors="coerce", format="mixed"
            )
    return result


metadata = load_metadata()
national_daily = parse_dates(load_csv("national_daily.csv"), ["day"])
oblast_daily = parse_dates(load_csv("oblast_daily.csv"), ["day"])
oblast_summary = parse_dates(
    load_csv("oblast_summary.csv"), ["first_seen", "last_seen"]
)
weapon_summary = load_csv("weapon_summary.csv")

st.title("Історична аналітика атак по Україні")
st.caption(
    "Агрегований навчальний проєкт: історичні відкриті дані по областях. "
    "Не показує live-маршрути, точні цілі, координати запусків або оперативний час."
)

if not metadata or national_daily.empty:
    st.error(
        "Dashboard snapshot ще не створений. Запусти "
        "python scripts/build_dashboard_data.py після підготовки даних."
    )
    st.stop()

generated_at = pd.to_datetime(metadata.get("generated_at"), utc=True, errors="coerce")
latest_source = pd.to_datetime(
    metadata.get("latest_source_event_at"), utc=True, errors="coerce"
)

quality = metadata.get("quality") or {}
learning = metadata.get("learning") or {}

status_cols = st.columns([1.3, 1.3, 1.3, 2.2])
status_cols[0].metric("Записів атак", fmt_int(metadata.get("attack_rows")))
status_cols[1].metric("Зв'язків з областями", fmt_int(metadata.get("region_links")))
status_cols[2].metric(
    "Покриття областями",
    fmt_pct(quality.get("region_coverage_rate")),
)
status_cols[3].metric(
    "Остання дата у джерелі",
    latest_source.strftime("%d.%m.%Y") if pd.notna(latest_source) else "—",
)

if pd.notna(generated_at):
    st.caption(
        "Знімок dashboard оновлено: "
        + generated_at.strftime("%d.%m.%Y %H:%M UTC")
    )

min_day = national_daily["day"].min()
max_day = national_daily["day"].max()

with st.sidebar:
    st.header("Фільтри")
    date_range = st.date_input(
        "Період",
        value=(min_day.date(), max_day.date()),
        min_value=min_day.date(),
        max_value=max_day.date(),
    )

    oblast_options = ["Усі області"]
    if not oblast_summary.empty:
        oblast_options += sorted(oblast_summary["oblast"].dropna().unique().tolist())
    selected_oblast = st.selectbox("Область", oblast_options)

    st.divider()
    st.caption("Джерело")
    st.write(metadata.get("source_name", "—"))
    st.caption(
        "Дані є історичними та можуть мати неповну регіональну розмітку."
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_day = pd.Timestamp(date_range[0], tz="UTC")
    end_day = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
else:
    start_day = min_day
    end_day = max_day + pd.Timedelta(days=1)

filtered_national = national_daily[
    (national_daily["day"] >= start_day)
    & (national_daily["day"] < end_day)
].copy()

filtered_oblast_daily = oblast_daily[
    (oblast_daily["day"] >= start_day)
    & (oblast_daily["day"] < end_day)
].copy()

if selected_oblast != "Усі області":
    filtered_oblast_daily = filtered_oblast_daily[
        filtered_oblast_daily["oblast"] == selected_oblast
    ]

overview_tab, regions_tab, quality_tab, ml_tab = st.tabs(
    ["Огляд", "Області", "Якість даних", "ML"]
)

with overview_tab:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Записів атак за період",
        fmt_int(filtered_national["attack_records"].sum()),
    )
    k2.metric(
        "Повідомлено запущено",
        fmt_int(filtered_national["launched_reported"].sum()),
    )
    k3.metric(
        "Повідомлено перехоплено",
        fmt_int(filtered_national["intercepted_reported"].sum()),
    )

    launched = filtered_national["launched_reported"].sum()
    intercepted = filtered_national["intercepted_reported"].sum()
    interception_rate = intercepted / launched if launched else None
    k4.metric("Співвідношення перехоплень", fmt_pct(interception_rate))

    st.subheader("Динаміка історичних записів")
    timeline = filtered_national.melt(
        id_vars=["day"],
        value_vars=["attack_records", "uav_events", "missile_events", "guided_bomb_events"],
        var_name="series",
        value_name="count",
    )
    labels = {
        "attack_records": "Усі записи",
        "uav_events": "БпЛА",
        "missile_events": "Ракети",
        "guided_bomb_events": "Керовані авіабомби",
    }
    timeline["series"] = timeline["series"].map(labels)
    fig = px.line(
        timeline,
        x="day",
        y="count",
        color="series",
        labels={"day": "Дата", "count": "Кількість", "series": "Категорія"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Структура записів за типом")
    if not weapon_summary.empty:
        weapon_chart = px.bar(
            weapon_summary,
            x="attack_type",
            y="attack_records",
            labels={
                "attack_type": "Категорія",
                "attack_records": "Кількість записів",
            },
        )
        st.plotly_chart(weapon_chart, use_container_width=True)

with regions_tab:
    period_summary = (
        filtered_oblast_daily.groupby("oblast", as_index=False)
        .agg(
            attack_events=("attack_events", "sum"),
            active_days=("day", "nunique"),
            uav_events=("uav_events", "sum"),
            missile_events=("missile_events", "sum"),
            guided_bomb_events=("guided_bomb_events", "sum"),
        )
        .sort_values("attack_events", ascending=False)
    )

    left, right = st.columns([1.45, 1.0])

    with left:
        st.subheader("Карта історичних подій")
        geojson = load_geojson()
        if geojson and not period_summary.empty:
            map_fig = px.choropleth(
                period_summary,
                geojson=geojson,
                locations="oblast",
                featureidkey="properties.name",
                color="attack_events",
                hover_name="oblast",
                hover_data={
                    "attack_events": True,
                    "active_days": True,
                    "uav_events": True,
                    "missile_events": True,
                    "guided_bomb_events": True,
                },
                labels={"attack_events": "Подій"},
            )
            map_fig.update_geos(
                fitbounds="locations",
                visible=False,
            )
            map_fig.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                coloraxis_colorbar_title="Подій",
            )
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.info("GeoJSON недоступний — показую рейтинг областей.")
            if not period_summary.empty:
                st.bar_chart(period_summary.set_index("oblast")["attack_events"])

    with right:
        st.subheader("Області")
        st.dataframe(
            period_summary.head(15),
            use_container_width=True,
            hide_index=True,
        )

    if selected_oblast != "Усі області" and not filtered_oblast_daily.empty:
        st.subheader(selected_oblast)
        region_fig = px.bar(
            filtered_oblast_daily,
            x="day",
            y=["uav_events", "missile_events", "guided_bomb_events"],
            labels={"value": "Кількість подій", "day": "Дата", "variable": "Тип"},
        )
        st.plotly_chart(region_fig, use_container_width=True)

with quality_tab:
    st.subheader("Контроль якості даних")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Blocking errors", fmt_int(quality.get("blocking_errors")))
    q2.metric("Попередження", fmt_int(quality.get("warnings")))
    q3.metric(
        "Покриття областями (все)",
        fmt_pct(quality.get("region_coverage_rate")),
    )
    q4.metric(
        "Покриття областями (90 днів)",
        fmt_pct(quality.get("recent_region_coverage_90d")),
    )

    ready = bool(quality.get("model_ready_for_serving", False))
    if ready:
        st.success("Регіональні дані відповідають поточним порогам готовності.")
    else:
        st.warning(
            "Регіональні дані поки недостатньо повні для використання "
            "ML як актуального сервісного прогнозу."
        )

    reasons = quality.get("model_readiness_reasons") or []
    if reasons:
        st.write("Причини:")
        for reason in reasons:
            st.write(f"- {reason}")

    quality_table = pd.DataFrame(
        [
            {"Показник": key, "Значення": value}
            for key, value in quality.items()
            if key not in {"model_readiness_reasons"}
        ]
    )
    st.dataframe(quality_table, use_container_width=True, hide_index=True)

with ml_tab:
    st.subheader("Стан навчання моделі")
    st.caption(
        "ML працює лише як агрегований історичний експеримент по області/добі. "
        "Точні цілі, маршрути й оперативні прогнози не моделюються."
    )

    candidate = learning.get("candidate_metrics") or {}
    baseline = learning.get("baseline_metrics") or {}
    champion = learning.get("champion_metrics_on_same_holdout") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Candidate ROC AUC",
        f"{candidate.get('roc_auc'):.3f}" if candidate.get("roc_auc") is not None else "—",
    )
    m2.metric(
        "Candidate AP",
        f"{candidate.get('average_precision'):.3f}"
        if candidate.get("average_precision") is not None
        else "—",
    )
    m3.metric(
        "Candidate Brier",
        f"{candidate.get('brier_score'):.3f}"
        if candidate.get("brier_score") is not None
        else "—",
    )
    m4.metric(
        "Baseline Brier",
        f"{baseline.get('brier_score'):.3f}"
        if baseline.get("brier_score") is not None
        else "—",
    )

    promoted = learning.get("promoted")
    if promoted is True:
        st.success("Нова модель пройшла promotion (підвищення до champion).")
    elif promoted is False:
        st.info("Нова модель не замінила champion — поріг покращення не пройдений.")

    st.write("Рішення:", learning.get("decision_reason", "—"))
    st.write(
        "Готовність до показу як сервісного прогнозу:",
        "так" if learning.get("model_ready_for_serving") else "ні",
    )

    comparison_rows = []
    for name, metrics in (
        ("Candidate", candidate),
        ("Baseline", baseline),
        ("Champion", champion),
    ):
        if metrics:
            comparison_rows.append(
                {
                    "Модель": name,
                    "ROC AUC": metrics.get("roc_auc"),
                    "Average Precision": metrics.get("average_precision"),
                    "Brier": metrics.get("brier_score"),
                    "Balanced Accuracy": metrics.get("balanced_accuracy"),
                }
            )
    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "Навчальний Data Science проєкт. Показники залежать від повноти відкритих джерел; "
    "відсутність запису не означає відсутність події."
)
