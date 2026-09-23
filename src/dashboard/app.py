"""Streamlit dashboard for historical aggregated attack analytics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.analysis.improvement import (
    build_improvement_recommendations,
    project_readiness_score,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/darmat1/ukraine-geo-data/"
    "main/geodata/Ukraine.geojson"
)

ATTACK_TYPE_UA = {
    "uav": "БпЛА",
    "missile": "Ракети",
    "guided_bomb": "Керовані авіабомби",
    "combined": "Комбіновані",
    "unknown": "Невідомо",
}

QUALITY_LABELS_UA = {
    "attack_rows": "Записів атак",
    "region_rows": "Зв'язків з областями",
    "weapon_rows": "Записів про засоби ураження",
    "attacks_with_region": "Атак із визначеною областю",
    "region_coverage_rate": "Покриття областями",
    "recent_attack_rows_90d": "Записів атак за останні 90 днів",
    "recent_attacks_with_region_90d": "Атак з областю за останні 90 днів",
    "recent_region_coverage_90d": "Покриття областями за 90 днів",
    "latest_attack_date": "Остання дата атаки у джерелі",
    "latest_region_labeled_date": "Остання дата з регіональною міткою",
    "region_label_lag_days": "Відставання регіональної розмітки, днів",
    "duplicate_attack_ids": "Дублікати ID атак",
    "invalid_attack_dates": "Некоректні дати",
    "missing_attack_types": "Записи без типу атаки",
    "orphan_region_links": "Некоректні зв'язки з областями",
    "negative_weapon_quantities": "Від'ємні значення кількості",
    "intercepted_above_launched": "Перехоплено більше, ніж запущено",
    "unmapped_region_rows": "Нерозпізнані регіональні значення",
    "source_duplicate_groups": "Групи дублікатів у джерелі",
    "blocking_errors": "Критичні помилки",
    "warnings": "Попередження",
    "model_ready_for_serving": "Модель готова до використання",
}

READINESS_REASON_UA = {
    "Overall region-label coverage is only 6.6%":
        "Загальне покриття подій регіональними мітками становить лише 6,6%.",
    "Recent 90-day region-label coverage is below 25%":
        "Покриття регіональними мітками за останні 90 днів нижче 25%.",
}

DECISION_UA = {
    "Candidate did not exceed promotion thresholds":
        "Нова модель не перевищила пороги, необхідні для заміни чинної.",
    "Candidate exceeds prevalence baseline":
        "Нова модель перевершує просту базову модель.",
    "Candidate probability calibration is worse than baseline":
        "Калібрування ймовірностей нової моделі гірше за базову.",
    "Candidate ranking quality does not exceed baseline":
        "Якість ранжування нової моделі не перевищує базову.",
    "No champion exists yet":
        "Чинної моделі ще немає.",
    "Brier score improved without material AP degradation":
        "Показник Брієра покращився без суттєвого погіршення точності ранжування.",
    "Average precision improved without material calibration degradation":
        "Середня точність покращилася без суттєвого погіршення калібрування.",
}

st.set_page_config(
    page_title="Історична аналітика атак по Україні",
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
        return f"{float(value):.1%}".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def fmt_pct_points(value: object) -> str:
    try:
        return f"{float(value):.1f}%".replace(".", ",")
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


def translate_reason(reason: object) -> str:
    text = str(reason)
    if text in READINESS_REASON_UA:
        return READINESS_REASON_UA[text]
    if "Overall region-label coverage is only" in text:
        value = text.split("only", 1)[-1].strip()
        return f"Загальне покриття подій регіональними мітками становить лише {value}."
    if "Recent 90-day region-label coverage is below" in text:
        value = text.split("below", 1)[-1].strip()
        return (
            "Покриття регіональними мітками за останні 90 днів "
            f"нижче {value}."
        )
    return text


def selected_map_oblast(event: object) -> str | None:
    """Extract clicked choropleth location from Streamlit Plotly selection."""
    try:
        points = event.selection.points
    except (AttributeError, TypeError):
        try:
            points = event.get("selection", {}).get("points", [])
        except AttributeError:
            points = []

    if not points:
        return None

    point = points[0]
    if hasattr(point, "get"):
        location = point.get("location")
        if location:
            return str(location)

        customdata = point.get("customdata")
        if isinstance(customdata, (list, tuple)) and customdata:
            return str(customdata[0])

    return None


metadata = load_metadata()
national_daily = parse_dates(load_csv("national_daily.csv"), ["day"])
oblast_daily = parse_dates(load_csv("oblast_daily.csv"), ["day"])
oblast_summary = parse_dates(
    load_csv("oblast_summary.csv"), ["first_seen", "last_seen"]
)
weapon_summary = load_csv("weapon_summary.csv")

if "attack_type" in weapon_summary.columns:
    weapon_summary["Категорія"] = (
        weapon_summary["attack_type"].map(ATTACK_TYPE_UA).fillna(
            weapon_summary["attack_type"]
        )
    )

st.title("Історична аналітика атак по Україні")
st.caption(
    "Агрегований навчальний проєкт на основі відкритих історичних даних. "
    "Не показує оперативні маршрути, точні цілі, координати запусків "
    "або точний час майбутніх ударів."
)

if not metadata or national_daily.empty:
    st.error(
        "Знімок даних для панелі ще не створений. Потрібно виконати "
        "python scripts/build_dashboard_data.py після підготовки даних."
    )
    st.stop()

generated_at = pd.to_datetime(metadata.get("generated_at"), utc=True, errors="coerce")
latest_source = pd.to_datetime(
    metadata.get("latest_source_event_at"), utc=True, errors="coerce"
)

quality = metadata.get("quality") or {}
learning = metadata.get("learning") or {}
improvement_recommendations = metadata.get("improvement_recommendations") or (
    build_improvement_recommendations(
        quality,
        learning,
        latest_source_event_at=metadata.get("latest_source_event_at"),
    )
)
technical_readiness_score = metadata.get("technical_readiness_score")
if technical_readiness_score is None:
    technical_readiness_score = project_readiness_score(
        quality,
        learning,
        latest_source_event_at=metadata.get("latest_source_event_at"),
    )

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
        "Дані панелі оновлено: "
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
    st.caption("Джерело даних")
    st.write("Kaggle — відкритий історичний набір piterfm")
    st.caption(
        "Регіональна розмітка джерела неповна, тому відсутність запису "
        "не означає відсутність події."
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

period_oblast_daily = oblast_daily[
    (oblast_daily["day"] >= start_day)
    & (oblast_daily["day"] < end_day)
].copy()

overview_tab, regions_tab, risk_tab, quality_tab, ml_tab, improve_tab = st.tabs(
    [
        "Огляд",
        "Області",
        "Відсотки та оцінка",
        "Якість даних",
        "Модель",
        "Що покращити",
    ]
)

with overview_tab:
    last_30_start = max_day - pd.Timedelta(days=29)
    previous_30_start = last_30_start - pd.Timedelta(days=30)

    current_30 = national_daily[
        (national_daily["day"] >= last_30_start)
        & (national_daily["day"] <= max_day)
    ]
    previous_30 = national_daily[
        (national_daily["day"] >= previous_30_start)
        & (national_daily["day"] < last_30_start)
    ]

    current_count = float(current_30["attack_records"].sum())
    previous_count = float(previous_30["attack_records"].sum())
    change_pct = (
        (current_count - previous_count) / previous_count * 100
        if previous_count
        else None
    )

    recent_regions = oblast_daily[
        (oblast_daily["day"] >= last_30_start)
        & (oblast_daily["day"] <= max_day)
    ]
    recent_region_summary = (
        recent_regions.groupby("oblast", as_index=False)["attack_events"]
        .sum()
        .sort_values("attack_events", ascending=False)
    )
    leading_region = (
        recent_region_summary.iloc[0]["oblast"]
        if not recent_region_summary.empty
        else "—"
    )

    category_totals = {
        "БпЛА": float(current_30["uav_events"].sum()),
        "Ракети": float(current_30["missile_events"].sum()),
        "Керовані авіабомби": float(current_30["guided_bomb_events"].sum()),
    }
    leading_category = max(category_totals, key=category_totals.get)

    st.subheader("Що змінило за останні 30 днів")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Записів за 30 днів", fmt_int(current_count))
    s2.metric(
        "Зміна до попередніх 30 днів",
        "—" if change_pct is None else fmt_pct_points(change_pct),
    )
    s3.metric("Найчастіше розмічена область", str(leading_region))
    s4.metric("Переважна категорія", leading_category)

    st.caption(
        "Цей блок описує лише історичні зміни у відкритому джерелі "
        "та не є прогнозом майбутніх подій."
    )

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
    k4.metric("Частка перехоплень", fmt_pct(interception_rate))

    st.subheader("Динаміка історичних записів")
    timeline = filtered_national.melt(
        id_vars=["day"],
        value_vars=[
            "attack_records",
            "uav_events",
            "missile_events",
            "guided_bomb_events",
        ],
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
            x="Категорія",
            y="attack_records",
            labels={
                "Категорія": "Категорія",
                "attack_records": "Кількість записів",
            },
        )
        st.plotly_chart(weapon_chart, use_container_width=True)

with regions_tab:
    period_summary = (
        period_oblast_daily.groupby("oblast", as_index=False)
        .agg(
            attack_events=("attack_events", "sum"),
            active_days=("day", "nunique"),
            uav_events=("uav_events", "sum"),
            missile_events=("missile_events", "sum"),
            guided_bomb_events=("guided_bomb_events", "sum"),
        )
        .sort_values("attack_events", ascending=False)
    )

    total_regional_events = float(period_summary["attack_events"].sum())
    period_summary["historical_share_pct"] = (
        period_summary["attack_events"] / total_regional_events * 100
        if total_regional_events
        else 0.0
    )

    left, right = st.columns([1.45, 1.0])
    clicked_oblast: str | None = None

    with left:
        st.subheader("Інтерактивна карта областей")
        st.caption(
            "Натисни на область — нижче відкриється її статистика за вибраний період."
        )
        geojson = load_geojson()
        if geojson and not period_summary.empty:
            map_fig = px.choropleth(
                period_summary,
                geojson=geojson,
                locations="oblast",
                featureidkey="properties.name",
                color="historical_share_pct",
                custom_data=["oblast"],
                hover_name="oblast",
                hover_data={
                    "historical_share_pct": ":.1f",
                    "attack_events": True,
                    "active_days": True,
                    "uav_events": True,
                    "missile_events": True,
                    "guided_bomb_events": True,
                },
                labels={
                    "historical_share_pct": "Історична частка, %",
                    "attack_events": "Подій",
                    "active_days": "Активних днів",
                    "uav_events": "БпЛА",
                    "missile_events": "Ракети",
                    "guided_bomb_events": "Керовані авіабомби",
                },
            )
            map_fig.update_geos(fitbounds="locations", visible=False)
            map_fig.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                coloraxis_colorbar_title="Частка, %",
                clickmode="event+select",
            )
            map_event = st.plotly_chart(
                map_fig,
                use_container_width=True,
                key="oblast_map",
                on_select="rerun",
                selection_mode="points",
            )
            clicked_oblast = selected_map_oblast(map_event)
        else:
            st.info("Карта тимчасово недоступна — показую рейтинг областей.")
            if not period_summary.empty:
                st.bar_chart(
                    period_summary.set_index("oblast")["historical_share_pct"]
                )

    with right:
        st.subheader("Рейтинг областей")
        table = period_summary[
            [
                "oblast",
                "attack_events",
                "historical_share_pct",
                "active_days",
            ]
        ].copy()
        table["historical_share_pct"] = table["historical_share_pct"].map(
            fmt_pct_points
        )
        table = table.rename(
            columns={
                "oblast": "Область",
                "attack_events": "Історичних подій",
                "historical_share_pct": "Частка, %",
                "active_days": "Днів із подіями",
            }
        )
        table_event = st.dataframe(
            table.head(15),
            use_container_width=True,
            hide_index=True,
            key="oblast_table",
            on_select="rerun",
            selection_mode="single-row",
        )

    table_oblast: str | None = None
    try:
        selected_rows = table_event.selection.rows
    except (AttributeError, TypeError):
        selected_rows = []
    if selected_rows:
        row_index = int(selected_rows[0])
        visible_table = table.head(15).reset_index(drop=True)
        if 0 <= row_index < len(visible_table):
            table_oblast = str(visible_table.iloc[row_index]["Область"])

    detail_oblast = clicked_oblast or table_oblast
    if detail_oblast is None and selected_oblast != "Усі області":
        detail_oblast = selected_oblast

    if detail_oblast:
        detail = period_oblast_daily[
            period_oblast_daily["oblast"] == detail_oblast
        ].copy()
        row = period_summary[period_summary["oblast"] == detail_oblast]

        st.divider()
        st.subheader(f"{detail_oblast}: історична статистика")

        if not row.empty:
            row = row.iloc[0]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Історичних подій", fmt_int(row["attack_events"]))
            d2.metric(
                "Частка серед розмічених подій",
                fmt_pct_points(row["historical_share_pct"]),
            )
            d3.metric("Днів із подіями", fmt_int(row["active_days"]))
            d4.metric(
                "БпЛА / ракети",
                f"{fmt_int(row['uav_events'])} / {fmt_int(row['missile_events'])}",
            )

        if not detail.empty:
            region_fig = px.bar(
                detail,
                x="day",
                y=["uav_events", "missile_events", "guided_bomb_events"],
                labels={
                    "value": "Кількість історичних подій",
                    "day": "Дата",
                    "variable": "Категорія",
                },
            )
            region_fig.for_each_trace(
                lambda trace: trace.update(
                    name={
                        "uav_events": "БпЛА",
                        "missile_events": "Ракети",
                        "guided_bomb_events": "Керовані авіабомби",
                    }.get(trace.name, trace.name)
                )
            )
            st.plotly_chart(region_fig, use_container_width=True)

with risk_tab:
    st.subheader("Відсотковий розподіл за областями")
    st.caption(
        "Це історична частка розмічених подій у вибраному періоді, "
        "а не твердження про те, де відбудеться наступний удар."
    )

    risk_summary = (
        period_oblast_daily.groupby("oblast", as_index=False)["attack_events"]
        .sum()
        .sort_values("attack_events", ascending=False)
    )
    total = float(risk_summary["attack_events"].sum())
    risk_summary["share_pct"] = (
        risk_summary["attack_events"] / total * 100 if total else 0.0
    )

    if not risk_summary.empty:
        risk_chart = px.bar(
            risk_summary.head(15),
            x="share_pct",
            y="oblast",
            orientation="h",
            labels={
                "share_pct": "Історична частка, %",
                "oblast": "Область",
            },
        )
        risk_chart.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(risk_chart, use_container_width=True)

    if learning.get("model_ready_for_serving"):
        st.success(
            "Модель пройшла поточні пороги якості даних. "
            "Агреговану модельну оцінку для області/доби можна публікувати "
            "окремо від історичної частки."
        )
    else:
        st.warning(
            "Модель поки не показує майбутні відсотки як надійний прогноз: "
            "регіональна розмітка даних недостатньо повна. "
            "Показувати точні відсотки зараз означало б створити хибну точність."
        )
        for reason in learning.get("model_readiness_reasons") or []:
            st.write(f"• {translate_reason(reason)}")

    st.info(
        "Рівень міста не використовується для прогнозування майбутнього удару. "
        "Проєкт працює з агрегованою аналітикою на рівні області та широких "
        "часових вікон."
    )

with quality_tab:
    st.subheader("Контроль якості даних")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Критичні помилки", fmt_int(quality.get("blocking_errors")))
    q2.metric("Попередження", fmt_int(quality.get("warnings")))
    q3.metric(
        "Покриття областями — загалом",
        fmt_pct(quality.get("region_coverage_rate")),
    )
    q4.metric(
        "Покриття областями — 90 днів",
        fmt_pct(quality.get("recent_region_coverage_90d")),
    )

    ready = bool(quality.get("model_ready_for_serving", False))
    if ready:
        st.success("Дані відповідають поточним порогам готовності моделі.")
    else:
        st.warning(
            "Регіональні дані поки недостатньо повні для публікації "
            "модельного прогнозу як надійного."
        )

    reasons = quality.get("model_readiness_reasons") or []
    if reasons:
        st.write("Причини:")
        for reason in reasons:
            st.write(f"• {translate_reason(reason)}")

    quality_rows = []
    for key, value in quality.items():
        if key == "model_readiness_reasons":
            continue
        label = QUALITY_LABELS_UA.get(key, key)
        if key in {"region_coverage_rate", "recent_region_coverage_90d"}:
            value = fmt_pct(value)
        elif key == "model_ready_for_serving":
            value = "Так" if value else "Ні"
        quality_rows.append({"Показник": label, "Значення": value})

    st.dataframe(
        pd.DataFrame(quality_rows),
        use_container_width=True,
        hide_index=True,
    )

with ml_tab:
    st.subheader("Стан самонавчання моделі")
    st.caption(
        "Модель автоматично перенавчається на нових історичних даних і "
        "замінює чинну модель лише тоді, коли кандидат проходить контроль якості."
    )

    candidate = learning.get("candidate_metrics") or {}
    baseline = learning.get("baseline_metrics") or {}
    champion = learning.get("champion_metrics_on_same_holdout") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "ROC AUC кандидата",
        f"{candidate.get('roc_auc'):.3f}".replace(".", ",")
        if candidate.get("roc_auc") is not None
        else "—",
    )
    m2.metric(
        "Середня точність кандидата",
        f"{candidate.get('average_precision'):.3f}".replace(".", ",")
        if candidate.get("average_precision") is not None
        else "—",
    )
    m3.metric(
        "Показник Брієра кандидата",
        f"{candidate.get('brier_score'):.3f}".replace(".", ",")
        if candidate.get("brier_score") is not None
        else "—",
    )
    m4.metric(
        "Показник Брієра базової моделі",
        f"{baseline.get('brier_score'):.3f}".replace(".", ",")
        if baseline.get("brier_score") is not None
        else "—",
    )

    promoted = learning.get("promoted")
    if promoted is True:
        st.success("Нова модель стала чинною моделлю.")
    elif promoted is False:
        st.info(
            "Нова модель не замінила чинну: покращення було недостатнім."
        )

    decision = str(learning.get("decision_reason", "—"))
    st.write("Рішення системи:", DECISION_UA.get(decision, decision))
    st.write(
        "Готовність до публікації модельної оцінки:",
        "так" if learning.get("model_ready_for_serving") else "ні",
    )

    comparison_rows = []
    for name, metrics in (
        ("Кандидат", candidate),
        ("Базова модель", baseline),
        ("Чинна модель", champion),
    ):
        if metrics:
            comparison_rows.append(
                {
                    "Модель": name,
                    "ROC AUC": metrics.get("roc_auc"),
                    "Середня точність": metrics.get("average_precision"),
                    "Показник Брієра": metrics.get("brier_score"),
                    "Збалансована точність": metrics.get("balanced_accuracy"),
                }
            )
    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "**Що система робить сама:** завантажує нові історичні дані, "
        "перевіряє якість, формує ознаки, навчає кандидата, порівнює його "
        "з базовою та чинною моделями й автоматично підвищує лише кращу модель."
    )
    st.markdown(
        "**Чого система не робить сама:** не вигадує відсутні факти, "
        "не переписує сирі дані без правил і не перетворює неповні дані "
        "на точний прогноз."
    )

with improve_tab:
    st.subheader("Що система рекомендує покращити")
    st.caption(
        "Це автоматичний аудит самого проєкту. Він аналізує якість даних, "
        "свіжість джерела та результати моделі, але не змінює сирі факти самостійно."
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Технічна готовність", f"{int(technical_readiness_score)} / 100")
    r2.metric(
        "Високих пріоритетів",
        fmt_int(
            sum(
                1
                for item in improvement_recommendations
                if item.get("level") in {"Критично", "Високий"}
            )
        ),
    )
    r3.metric(
        "Автоматизованих дій",
        fmt_int(
            sum(
                1
                for item in improvement_recommendations
                if item.get("automatic") is True
            )
        ),
    )
    st.progress(min(1.0, max(0.0, float(technical_readiness_score) / 100.0)))

    st.markdown("#### Пріоритетний план")
    for index, item in enumerate(improvement_recommendations, start=1):
        level = item.get("level", "—")
        title = item.get("title", "Рекомендація")
        area = item.get("area", "Система")
        automatic = (
            "може контролювати автоматично"
            if item.get("automatic")
            else "потребує розвитку або нового джерела"
        )
        with st.expander(f"{index}. [{level}] {title}", expanded=index <= 3):
            st.write(f"**Напрям:** {area}")
            st.write(f"**Що виявлено:** {item.get('finding', '—')}")
            st.write(f"**Що робити:** {item.get('action', '—')}")
            st.caption(f"Статус дії: {automatic}.")

    st.markdown("#### Як проєкт сам себе покращує")
    st.write(
        "Після кожного циклу система перевіряє якість даних, формує список "
        "слабких місць, навчає модель-кандидата, порівнює її з базовою та "
        "чинною моделями й не замінює чинну, якщо кандидат не кращий."
    )

st.divider()
st.caption(
    "Навчальний Data Science проєкт. Показники залежать від повноти "
    "відкритих джерел; відсутність запису не означає відсутність події."
)
