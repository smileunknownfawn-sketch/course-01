"""Streamlit dashboard for historical aggregated attack analytics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.analysis.consensus import build_oblast_consensus

ROOT_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/darmat1/ukraine-geo-data/"
    "main/geodata/Ukraine.geojson"
)

UKRAINE_ADMIN1_FALLBACK = [
    "Автономна Республіка Крим",
    "Вінницька область",
    "Волинська область",
    "Дніпропетровська область",
    "Донецька область",
    "Житомирська область",
    "Закарпатська область",
    "Запорізька область",
    "Івано-Франківська область",
    "Київ",
    "Київська область",
    "Кіровоградська область",
    "Луганська область",
    "Львівська область",
    "Миколаївська область",
    "Одеська область",
    "Полтавська область",
    "Рівненська область",
    "Севастополь",
    "Сумська область",
    "Тернопільська область",
    "Харківська область",
    "Херсонська область",
    "Хмельницька область",
    "Черкаська область",
    "Чернівецька область",
    "Чернігівська область",
]

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
viina_daily = parse_dates(load_csv("viina_oblast_daily.csv"), ["day"])
siren_daily = parse_dates(load_csv("siren_oblast_daily.csv"), ["day"])

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
    period_mode = st.selectbox(
        "Швидкий період",
        ["Останні 30 днів", "Останні 90 днів", "Увесь період", "Власний період"],
        index=2,
    )

    if period_mode == "Останні 30 днів":
        date_range = (
            max(min_day, max_day - pd.Timedelta(days=29)).date(),
            max_day.date(),
        )
    elif period_mode == "Останні 90 днів":
        date_range = (
            max(min_day, max_day - pd.Timedelta(days=89)).date(),
            max_day.date(),
        )
    elif period_mode == "Власний період":
        date_range = st.date_input(
            "Період",
            value=(min_day.date(), max_day.date()),
            min_value=min_day.date(),
            max_value=max_day.date(),
        )
    else:
        date_range = (min_day.date(), max_day.date())

    oblast_options = ["Усі області"]
    if not oblast_summary.empty:
        oblast_options += sorted(oblast_summary["oblast"].dropna().unique().tolist())
    selected_oblast = st.selectbox("Область", oblast_options)

    st.divider()
    st.caption("Джерела даних")
    st.write("• Kaggle — історія ракетних атак і БпЛА")
    st.write("• VIINA — незалежні геокодовані повітряні інциденти")
    st.write("• eTryvoga — історія повітряних тривог (контекст)")
    st.caption(
        "Джерела мають різні визначення події. Обстріли та тривоги "
        "не змішуються як один тип факту."
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

period_viina_daily = viina_daily[
    (viina_daily["day"] >= start_day)
    & (viina_daily["day"] < end_day)
].copy() if not viina_daily.empty else viina_daily.copy()

period_siren_daily = siren_daily[
    (siren_daily["day"] >= start_day)
    & (siren_daily["day"] < end_day)
].copy() if not siren_daily.empty else siren_daily.copy()

if selected_oblast == "Усі області":
    overview_daily = filtered_national.copy()
    overview_daily["all_events"] = overview_daily["attack_records"]
    scope_title = "Україна"
else:
    overview_daily = period_oblast_daily[
        period_oblast_daily["oblast"] == selected_oblast
    ].copy()
    overview_daily["all_events"] = overview_daily["attack_events"]
    scope_title = selected_oblast

overview_tab, regions_tab, risk_tab, quality_tab, ml_tab = st.tabs(
    [
        "Огляд",
        "Області",
        "Відсотки та оцінка",
        "Якість даних",
        "Модель",
    ]
)

with overview_tab:
    st.subheader(f"Огляд: {scope_title}")

    if selected_oblast == "Усі області":
        current_30 = national_daily[
            (national_daily["day"] >= max_day - pd.Timedelta(days=29))
            & (national_daily["day"] <= max_day)
        ].copy()
        previous_30 = national_daily[
            (national_daily["day"] >= max_day - pd.Timedelta(days=59))
            & (national_daily["day"] < max_day - pd.Timedelta(days=29))
        ].copy()
        current_count = float(current_30["attack_records"].sum())
        previous_count = float(previous_30["attack_records"].sum())
    else:
        current_30 = oblast_daily[
            (oblast_daily["oblast"] == selected_oblast)
            & (oblast_daily["day"] >= max_day - pd.Timedelta(days=29))
            & (oblast_daily["day"] <= max_day)
        ].copy()
        previous_30 = oblast_daily[
            (oblast_daily["oblast"] == selected_oblast)
            & (oblast_daily["day"] >= max_day - pd.Timedelta(days=59))
            & (oblast_daily["day"] < max_day - pd.Timedelta(days=29))
        ].copy()
        current_count = float(current_30["attack_events"].sum())
        previous_count = float(previous_30["attack_events"].sum())

    change_pct = (
        (current_count - previous_count) / previous_count * 100
        if previous_count
        else None
    )

    st.subheader("Що змінило за останні 30 днів")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Історичних подій", fmt_int(current_count))
    s2.metric(
        "Зміна до попередніх 30 днів",
        "—" if change_pct is None else fmt_pct_points(change_pct),
    )
    s3.metric(
        "БпЛА за 30 днів",
        fmt_int(current_30["uav_events"].sum()) if not current_30.empty else "0",
    )
    s4.metric(
        "Ракет за 30 днів",
        fmt_int(current_30["missile_events"].sum()) if not current_30.empty else "0",
    )

    st.caption(
        "Показники описують історичні записи у відкритому джерелі "
        "та не є прогнозом майбутніх подій."
    )

    k1, k2, k3, k4 = st.columns(4)
    if selected_oblast == "Усі області":
        k1.metric(
            "Записів атак за період",
            fmt_int(overview_daily["all_events"].sum()),
        )
        k2.metric(
            "Повідомлено запущено",
            fmt_int(overview_daily["launched_reported"].sum()),
        )
        k3.metric(
            "Повідомлено перехоплено",
            fmt_int(overview_daily["intercepted_reported"].sum()),
        )
        launched = overview_daily["launched_reported"].sum()
        intercepted = overview_daily["intercepted_reported"].sum()
        interception_rate = intercepted / launched if launched else None
        k4.metric("Частка перехоплень", fmt_pct(interception_rate))
    else:
        k1.metric(
            "Історичних подій за період",
            fmt_int(overview_daily["all_events"].sum()),
        )
        k2.metric(
            "Днів із подіями",
            fmt_int(overview_daily["day"].nunique()),
        )
        k3.metric(
            "Подій БпЛА",
            fmt_int(overview_daily["uav_events"].sum()),
        )
        k4.metric(
            "Ракетних подій",
            fmt_int(overview_daily["missile_events"].sum()),
        )

    st.subheader("Динаміка історичних записів")
    if overview_daily.empty:
        st.info("За вибраний період для цієї області немає розмічених записів.")
    else:
        timeline = overview_daily.melt(
            id_vars=["day"],
            value_vars=[
                "all_events",
                "uav_events",
                "missile_events",
                "guided_bomb_events",
            ],
            var_name="series",
            value_name="count",
        )
        labels = {
            "all_events": "Усі події",
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

    st.subheader("Структура подій за типом")
    structure = pd.DataFrame(
        {
            "Категорія": ["БпЛА", "Ракети", "Керовані авіабомби"],
            "Кількість": [
                float(overview_daily["uav_events"].sum()) if not overview_daily.empty else 0,
                float(overview_daily["missile_events"].sum()) if not overview_daily.empty else 0,
                float(overview_daily["guided_bomb_events"].sum()) if not overview_daily.empty else 0,
            ],
        }
    )
    structure = structure[structure["Кількість"] > 0].copy()

    if structure.empty:
        st.info("Для вибраного фільтра немає даних для структури типів.")
    else:
        structure["Частка, %"] = (
            structure["Кількість"] / structure["Кількість"].sum() * 100
        )
        structure_chart = px.pie(
            structure,
            names="Категорія",
            values="Кількість",
            hole=0.45,
        )
        structure_chart.update_traces(
            textposition="inside",
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.0f} подій (%{percent})<extra></extra>",
        )
        structure_chart.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="Категорія",
        )
        st.plotly_chart(structure_chart, use_container_width=True)

        structure_table = structure.copy()
        structure_table["Частка, %"] = structure_table["Частка, %"].map(fmt_pct_points)
        st.dataframe(
            structure_table,
            use_container_width=True,
            hide_index=True,
        )

with regions_tab:
    st.subheader("Карта України за областями")
    st.caption(
        "Карта завжди показує всі адміністративні регіони. Наведи курсор на "
        "область, щоб побачити дані з окремих джерел та узгоджену історичну частку."
    )

    geojson = load_geojson()
    if geojson:
        geo_oblasts = [
            feature.get("properties", {}).get("name")
            for feature in geojson.get("features", [])
            if str(feature.get("properties", {}).get("admin_level")) == "4"
            and feature.get("properties", {}).get("name")
        ]
        all_oblasts = sorted(set(geo_oblasts))
    else:
        all_oblasts = UKRAINE_ADMIN1_FALLBACK

    consensus = build_oblast_consensus(
        all_oblasts,
        period_oblast_daily,
        period_viina_daily,
        period_siren_daily,
    )

    kaggle_details = (
        period_oblast_daily.groupby("oblast", as_index=False)
        .agg(
            active_days=("day", "nunique"),
            uav_events=("uav_events", "sum"),
            missile_events=("missile_events", "sum"),
            guided_bomb_events=("guided_bomb_events", "sum"),
        )
        if not period_oblast_daily.empty
        else pd.DataFrame(
            columns=[
                "oblast",
                "active_days",
                "uav_events",
                "missile_events",
                "guided_bomb_events",
            ]
        )
    )
    consensus = consensus.merge(kaggle_details, on="oblast", how="left")
    for column in [
        "active_days",
        "uav_events",
        "missile_events",
        "guided_bomb_events",
    ]:
        consensus[column] = consensus[column].fillna(0).astype("int64")

    left, right = st.columns([1.55, 1.0])
    clicked_oblast: str | None = None

    with left:
        if geojson:
            map_fig = px.choropleth(
                consensus,
                geojson=geojson,
                locations="oblast",
                featureidkey="properties.name",
                color="consensus_share_pct",
                custom_data=["oblast"],
                hover_name="oblast",
                hover_data={
                    "consensus_share_pct": ":.1f",
                    "kaggle_events": True,
                    "viina_events": True,
                    "alert_count": True,
                    "evidence_sources": True,
                },
                labels={
                    "consensus_share_pct": "Узгоджена історична частка, %",
                    "kaggle_events": "Записів атак (Kaggle)",
                    "viina_events": "Повітряних інцидентів (VIINA)",
                    "alert_count": "Повітряних тривог",
                    "evidence_sources": "Джерел з подіями",
                },
            )
            map_fig.update_traces(marker_line_width=0.8)
            map_fig.update_geos(
                fitbounds="locations",
                visible=False,
            )
            map_fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar_title="Консенсус, %",
                clickmode="event+select",
                height=680,
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
            st.warning(
                "GeoJSON карти тимчасово недоступний. Дані по областях "
                "залишаються доступними у таблиці."
            )

    with right:
        st.subheader("Узгоджений рейтинг")
        table = consensus[
            [
                "oblast",
                "consensus_share_pct",
                "kaggle_events",
                "viina_events",
                "alert_count",
            ]
        ].copy()
        table["consensus_share_pct"] = table["consensus_share_pct"].map(
            fmt_pct_points
        )
        table = table.rename(
            columns={
                "oblast": "Область",
                "consensus_share_pct": "Консенсус, %",
                "kaggle_events": "Kaggle",
                "viina_events": "VIINA",
                "alert_count": "Тривоги",
            }
        )
        table_event = st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            key="oblast_table",
            on_select="rerun",
            selection_mode="single-row",
            height=620,
        )

    table_oblast: str | None = None
    try:
        selected_rows = table_event.selection.rows
    except (AttributeError, TypeError):
        selected_rows = []
    if selected_rows:
        row_index = int(selected_rows[0])
        visible_table = table.reset_index(drop=True)
        if 0 <= row_index < len(visible_table):
            table_oblast = str(visible_table.iloc[row_index]["Область"])

    detail_oblast = clicked_oblast or table_oblast
    if detail_oblast is None and selected_oblast != "Усі області":
        detail_oblast = selected_oblast

    if detail_oblast:
        st.divider()
        st.subheader(f"{detail_oblast}: детальна історична аналітика")
        row = consensus[consensus["oblast"] == detail_oblast]

        if not row.empty:
            row = row.iloc[0]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric(
                "Узгоджена історична частка",
                fmt_pct_points(row["consensus_share_pct"]),
            )
            d2.metric("Записів атак (Kaggle)", fmt_int(row["kaggle_events"]))
            d3.metric("Інцидентів (VIINA)", fmt_int(row["viina_events"]))
            d4.metric("Повітряних тривог", fmt_int(row["alert_count"]))

            d5, d6, d7, d8 = st.columns(4)
            d5.metric("Днів із записами атак", fmt_int(row["active_days"]))
            d6.metric("Подій БпЛА", fmt_int(row["uav_events"]))
            d7.metric("Ракетних подій", fmt_int(row["missile_events"]))
            d8.metric(
                "Джерел із подіями",
                f"{fmt_int(row['evidence_sources'])} / {fmt_int(row['active_attack_sources'])}",
            )

        kaggle_region = period_oblast_daily[
            period_oblast_daily["oblast"] == detail_oblast
        ][["day", "attack_events"]].copy()
        if not kaggle_region.empty:
            kaggle_region = kaggle_region.rename(
                columns={"attack_events": "Kaggle: записи атак"}
            )

        viina_region = period_viina_daily[
            period_viina_daily["oblast"] == detail_oblast
        ][["day", "viina_events"]].copy() if not period_viina_daily.empty else pd.DataFrame()
        if not viina_region.empty:
            viina_region = viina_region.rename(
                columns={"viina_events": "VIINA: повітряні інциденти"}
            )

        if not kaggle_region.empty or not viina_region.empty:
            if kaggle_region.empty:
                detail_daily = viina_region
            elif viina_region.empty:
                detail_daily = kaggle_region
            else:
                detail_daily = kaggle_region.merge(
                    viina_region,
                    on="day",
                    how="outer",
                )
            detail_daily = detail_daily.fillna(0).sort_values("day")
            detail_long = detail_daily.melt(
                id_vars=["day"],
                var_name="Джерело",
                value_name="Кількість",
            )
            detail_fig = px.line(
                detail_long,
                x="day",
                y="Кількість",
                color="Джерело",
                labels={"day": "Дата"},
            )
            st.plotly_chart(detail_fig, use_container_width=True)

        st.caption(
            "Kaggle та VIINA мають різні методики збору, тому їхні сирі "
            "кількості не додаються. Консенсус — середнє нормалізованих "
            "часток кожного незалежного джерела."
        )

with risk_tab:
    st.subheader("Узгоджений відсотковий розподіл за областями")
    st.caption(
        "Цей показник усереднює нормалізовані історичні частки незалежних "
        "джерел атак/повітряних інцидентів. Тривоги показуються окремо "
        "та не рахуються як факт обстрілу."
    )

    risk_geojson = load_geojson()
    if risk_geojson:
        risk_oblasts = [
            feature.get("properties", {}).get("name")
            for feature in risk_geojson.get("features", [])
            if str(feature.get("properties", {}).get("admin_level")) == "4"
            and feature.get("properties", {}).get("name")
        ]
    else:
        risk_oblasts = UKRAINE_ADMIN1_FALLBACK

    risk_summary = build_oblast_consensus(
        risk_oblasts,
        period_oblast_daily,
        period_viina_daily,
        period_siren_daily,
    )

    if not risk_summary.empty:
        risk_chart = px.bar(
            risk_summary.head(15),
            x="consensus_share_pct",
            y="oblast",
            orientation="h",
            hover_data=["kaggle_events", "viina_events", "alert_count"],
            labels={
                "consensus_share_pct": "Узгоджена історична частка, %",
                "oblast": "Область",
                "kaggle_events": "Kaggle",
                "viina_events": "VIINA",
                "alert_count": "Тривоги",
            },
        )
        risk_chart.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(risk_chart, use_container_width=True)

    st.info(
        "Це історична мультиджерельна аналітика, а не твердження про місце "
        "майбутнього удару. Модельний прогноз не публікується, доки якість "
        "регіональної розмітки не проходить задані пороги."
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

st.divider()
st.caption(
    "Навчальний Data Science проєкт. Показники залежать від повноти "
    "відкритих джерел; відсутність запису не означає відсутність події."
)
