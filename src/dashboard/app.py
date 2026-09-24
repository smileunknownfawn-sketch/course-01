"""Streamlit dashboard for historical aggregated attack analytics."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.analysis.consensus import build_oblast_consensus

ROOT_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"
GEOJSON_PATH = ROOT_DIR / "data" / "geo" / "ukraine_admin1.geojson"
HERO_IMAGE_PATH = ROOT_DIR / "assets" / "dashboard-hero-v3.webp"

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

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#187caf", "#e59a3b", "#237c83", "#8258a6", "#5c7392"
]

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
    "Data-quality gate blocks promotion":
        "Оновлення моделі зупинено: регіональні дані недостатньо повні.",
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
    "Training blocked: unverified oblast/day outcomes":
        "Навчання зупинено: невідомі дні не можна вважати днями без атак.",
    "Champion trained on or beyond holdout; independent comparison required":
        "Порівняння заблоковано: чинна модель уже бачила період перевірки.",
}

st.set_page_config(
    page_title="Історична аналітика атак по Україні",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """<style>
    :root { color-scheme: light; }
    [data-testid="stAppViewContainer"] {
      background: linear-gradient(180deg, #edf5fb 0, #f7f9fc 520px, #f3f7fc 100%);
      color: #17283c;
    }
    [data-testid="stHeader"] { background: rgba(243,247,252,.88); backdrop-filter: blur(12px); }
    .block-container { max-width: 1480px; padding-top: 1.35rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #123253; letter-spacing: -.025em; line-height: 1.2; }
    [data-testid="stMarkdownContainer"] h2 { font-size: clamp(1.7rem, 2.1vw, 2.15rem); }
    [data-testid="stMarkdownContainer"] h3 { font-size: clamp(1.35rem, 1.7vw, 1.7rem); padding-top: .7rem; }
    .dashboard-hero {
      text-align: center; padding: 2.55rem 1.5rem 2.25rem; margin-bottom: .8rem;
      border: 1px solid #caddec; border-radius: 26px 26px 12px 12px;
      background: linear-gradient(125deg, rgba(224,241,251,.98), rgba(251,253,255,.98) 62%, rgba(233,245,241,.98));
      box-shadow: 0 14px 45px rgba(19,66,103,.09);
    }
    .dashboard-hero .eyebrow {
      color: #126d85; font-size: 1.05rem; font-weight: 800;
      letter-spacing: .11em; text-transform: uppercase;
    }
    .dashboard-hero h1 {
      margin: .65rem auto 1rem; max-width: 950px;
      color: #123253; font-size: clamp(2.3rem, 4.2vw, 4.15rem);
      line-height: 1.12; font-weight: 800;
    }
    .dashboard-hero p {
      margin: 0 auto; max-width: 820px;
      color: #35516c; font-size: clamp(1.07rem, 1.55vw, 1.3rem);
      line-height: 1.55;
    }
    [data-testid="stImage"] {
      margin-bottom: 1.15rem;
      border: 1px solid #caddec; border-radius: 12px 12px 24px 24px;
      overflow: hidden; box-shadow: 0 14px 45px rgba(19,66,103,.09);
    }
    [data-testid="stImage"] img { display: block; }
    .reading-guide {
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: .85rem; margin: 1rem 0 1.45rem;
    }
    .guide-card {
      display: grid; grid-template-columns: 52px 1fr; align-items: center;
      gap: .8rem; min-height: 106px; padding: 1rem 1.05rem;
      background: linear-gradient(145deg, #ffffff, #f6fafe);
      border: 1px solid #d5e3ef; border-radius: 18px;
      box-shadow: 0 5px 18px rgba(20,62,94,.055);
    }
    .guide-icon {
      display: grid; place-items: center; width: 52px; height: 52px;
      color: #fff; background: linear-gradient(145deg, #1479ad, #11577f);
      border-radius: 15px; font-size: 1.45rem;
      box-shadow: 0 6px 15px rgba(18,107,158,.18);
    }
    .guide-title { color: #173b5a; font-size: 1.12rem; font-weight: 800; line-height: 1.3; }
    .guide-note { margin-top: .22rem; color: #526981; font-size: 1rem; line-height: 1.45; }
    .selection-strip {
      display: flex; flex-wrap: wrap; justify-content: center; gap: .65rem;
      margin: .3rem 0 1.35rem;
    }
    .selection-chip {
      display: inline-flex; align-items: center; gap: .45rem;
      padding: .5rem .82rem; border-radius: 999px; background: #fff;
      color: #294965; font-size: 1.08rem; font-weight: 700;
      border: 1px solid #d4e3ef; box-shadow: 0 3px 12px rgba(20,62,94,.05);
    }
    .selection-chip b { color: #0f6d9d; }
    [data-testid="stMetric"] {
      background: #fff; border: 1px solid #dce6f1; border-radius: 16px;
      padding: 1.1rem 1.25rem; box-shadow: 0 5px 22px rgba(22,53,85,.045);
      min-height: 125px;
    }
    [data-testid="stMetric"]:hover {
      border-color: #b9d2e5; box-shadow: 0 9px 28px rgba(22,53,85,.09);
      transform: translateY(-1px); transition: .18s ease;
    }
    [data-testid="stMetricLabel"] { color: #536981; font-weight: 600; }
    [data-testid="stMetricLabel"] p { font-size: 1.12rem !important; line-height: 1.45; }
    [data-testid="stMetricValue"] { color: #113e68; font-weight: 750; letter-spacing: -.035em; }
    [data-testid="stMetricValue"] div { font-size: clamp(1.8rem, 2.25vw, 2.55rem); }
    .stTabs [data-baseweb="tab-list"] {
      display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: .65rem; padding: .2rem 0 1.2rem; overflow: visible;
    }
    .stTabs [data-baseweb="tab"] {
      display: flex; align-items: center; justify-content: center;
      width: 100%; min-height: 64px; padding: .7rem 1rem;
      color: #294965; background: #fff; font-size: 1.13rem;
      line-height: 1.3; font-weight: 700; white-space: normal;
      border: 1px solid #d8e4ef; border-radius: 14px;
      box-shadow: 0 3px 10px rgba(22,53,85,.04);
    }
    .stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab"]:focus-visible {
      border-color: #2287b8; background: #edf7fc;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
      color: #fff; background: #126b9e; border-color: #126b9e;
      box-shadow: 0 5px 16px rgba(18,107,158,.18);
    }
    .stTabs [data-baseweb="tab-border"], .stTabs [data-baseweb="tab-highlight"] {
      display: none;
    }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
      background: #fff; border: 1px solid #dce6f1; border-radius: 16px;
      padding: .8rem; overflow: hidden; box-shadow: 0 5px 22px rgba(22,53,85,.045);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(255,255,255,.86); border-color: #d7e4ef !important;
      border-radius: 18px; box-shadow: 0 7px 28px rgba(20,62,94,.055);
    }
    .visual-equation {
      display: grid; grid-template-columns: 1fr auto 1fr auto 1fr;
      align-items: stretch; gap: .7rem; margin: .8rem 0 1.25rem;
    }
    .equation-card {
      display: flex; flex-direction: column; justify-content: center;
      min-height: 118px; padding: 1rem; text-align: center;
      background: #fff; border: 1px solid #d7e4ef; border-radius: 16px;
    }
    .equation-value { color: #123f69; font-size: clamp(1.8rem, 2.7vw, 2.65rem); font-weight: 800; }
    .equation-label { color: #536981; font-size: 1.08rem; line-height: 1.35; }
    .equation-sign { align-self: center; color: #6c8095; font-size: 2rem; font-weight: 800; }
    .process-flow {
      display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: .65rem; margin: 1rem 0 1.25rem;
    }
    .process-step {
      position: relative; min-height: 132px; padding: 1rem;
      background: #fff; border: 1px solid #d7e4ef; border-radius: 16px;
    }
    .process-step:not(:last-child)::after {
      content: "→"; position: absolute; right: -.58rem; top: 43%; z-index: 2;
      color: #2a7ca6; font-size: 1.25rem; font-weight: 900;
    }
    .process-step.blocked { background: #fff7e8; border-color: #efc77e; }
    .process-number { color: #1479aa; font-weight: 800; font-size: 1.02rem; text-transform: uppercase; }
    .process-title { margin: .3rem 0; color: #173b5a; font-size: 1.12rem; font-weight: 800; }
    .process-note { color: #5b6f83; font-size: 1.05rem; line-height: 1.45; }
    [data-testid="stCaptionContainer"] { color: #526981; }
    [data-testid="stCaptionContainer"] p { font-size: 1.08rem !important; line-height: 1.6; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
      font-size: 1.1rem; line-height: 1.65;
    }
    [data-testid="stMarkdownContainer"] .dashboard-hero p {
      font-size: clamp(1.07rem, 1.55vw, 1.3rem); line-height: 1.55;
    }
    [data-testid="stWidgetLabel"] p {
      font-size: 1.12rem !important; font-weight: 700;
    }
    [data-baseweb="select"] *, [data-baseweb="input"] input {
      font-size: 1.12rem !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] li {
      font-size: 1.1rem !important; line-height: 1.55;
    }
    [data-testid="stDataFrame"] { font-size: 1.08rem; }
    @media (max-width: 1050px) {
      .stTabs [data-baseweb="tab-list"] { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .reading-guide { grid-template-columns: 1fr; }
      .process-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .process-step::after { display: none; }
    }
    @media (max-width: 760px) {
      .block-container { padding: .8rem .7rem 2rem; }
      .dashboard-hero { padding: 1.8rem 1rem; border-radius: 18px; }
      .dashboard-hero h1 { font-size: clamp(2.2rem, 8vw, 3rem); }
      [data-testid="stMetric"] { min-height: 110px; padding: .85rem; }
      .stTabs [data-baseweb="tab"] { min-height: 68px; padding: .7rem .65rem; font-size: 1.08rem; }
      .visual-equation { grid-template-columns: 1fr; }
      .equation-sign { transform: rotate(90deg); line-height: .7; }
      .process-flow { grid-template-columns: 1fr; }
    }
    @media (max-width: 410px) {
      .stTabs [data-baseweb="tab-list"] { grid-template-columns: 1fr; }
    }
    </style>""",
    unsafe_allow_html=True,
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


@st.cache_data
def load_geojson() -> dict[str, object] | None:
    try:
        geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
        if geojson.get("type") != "FeatureCollection" or not geojson.get("features"):
            return None
        return geojson
    except (OSError, ValueError):
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


def chart_frequency(start: pd.Timestamp, end: pd.Timestamp) -> str:
    days = (end - start).days
    return "MS" if days > 180 else "W-MON" if days > 45 else "D"


def chart_frequency_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    frequency = chart_frequency(start, end)
    return {"D": "днями", "W-MON": "тижнями", "MS": "місяцями"}[frequency]


def style_chart(fig: go.Figure, *, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=18, color="#233c58"),
        hoverlabel=dict(font_size=18, bgcolor="#ffffff", font_color="#17334e"),
        margin=dict(l=22, r=26, t=32, b=42),
        legend=dict(
            font_size=17, orientation="h", y=1.16,
            bgcolor="rgba(255,255,255,.86)", bordercolor="#dce6f1", borderwidth=1,
        ),
        uniformtext=dict(minsize=15, mode="hide"),
    )
    fig.update_xaxes(tickfont_size=17, title_font_size=18, showgrid=False,
                     linecolor="#ccdbe9", zeroline=False)
    fig.update_yaxes(tickfont_size=17, title_font_size=18,
                     gridcolor="#e6edf5", zeroline=False)
    return fig


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
        value = text.split("only", 1)[-1].strip().replace(".", ",")
        return f"Загальне покриття подій регіональними мітками становить лише {value}."
    if "Recent 90-day region-label coverage is below" in text:
        value = text.split("below", 1)[-1].strip().replace(".", ",")
        return (
            "Покриття регіональними мітками за останні 90 днів "
            f"нижче {value}."
        )
    if text == "Unverified oblast/day outcomes cannot be used as negative labels":
        return "Немає підтверджень повноти спостережень для днів без записів про атаку."
    return "Система не розпізнала причину; подробиці є у звіті навчання."


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
        customdata = point.get("customdata")
        if isinstance(customdata, (list, tuple)) and customdata:
            return str(customdata[0])

        location = point.get("location")
        if location:
            return str(location)

    return None


metadata = load_metadata()
national_daily = parse_dates(load_csv("national_daily.csv"), ["day"])
oblast_daily = parse_dates(load_csv("oblast_daily.csv"), ["day"])
oblast_summary = parse_dates(
    load_csv("oblast_summary.csv"), ["first_seen", "last_seen"]
)
weapon_summary = load_csv("weapon_summary.csv")
interception_daily = parse_dates(load_csv("interception_by_type_daily.csv"), ["day"])
interception_coverage = load_csv("interception_coverage.csv")
viina_daily = parse_dates(load_csv("viina_oblast_daily.csv"), ["day"])
siren_daily = parse_dates(load_csv("siren_oblast_daily.csv"), ["day"])

if "attack_type" in weapon_summary.columns:
    weapon_summary["Категорія"] = (
        weapon_summary["attack_type"].map(ATTACK_TYPE_UA).fillna(
            weapon_summary["attack_type"]
        )
    )

st.markdown(
    """<section class="dashboard-hero" aria-labelledby="dashboard-title">
      <div class="eyebrow">Відкриті історичні дані</div>
      <h1 id="dashboard-title">Аналітика атак по Україні</h1>
      <p>Досліджуйте записи про атаки, заявлені збиття та дані за областями.
      Оберіть період і область нижче, щоб побачити потрібну інформацію.</p>
    </section>""",
    unsafe_allow_html=True,
)
if HERO_IMAGE_PATH.exists():
    st.image(
        str(HERO_IMAGE_PATH),
        width="stretch",
        caption="Декоративна ілюстрація: історичні дані та аналітика. Не є оперативною картою.",
    )

if not metadata or national_daily.empty:
    st.error(
        "Дані для панелі ще не підготовлено. Запустіть оновлення джерел "
        "за інструкцією проєкту та відкрийте сторінку знову."
    )
    st.stop()

generated_at = pd.to_datetime(metadata.get("generated_at"), utc=True, errors="coerce")
latest_source = pd.to_datetime(
    metadata.get("latest_source_event_at"), utc=True, errors="coerce"
)

quality = metadata.get("quality") or {}
learning = metadata.get("learning") or {}

geojson = load_geojson()
map_oblasts = [
    feature["properties"]["name"]
    for feature in (geojson or {}).get("features", [])
    if feature.get("properties", {}).get("name")
]
known_oblasts = set(map_oblasts or UKRAINE_ADMIN1_FALLBACK)
for frame in (oblast_daily, viina_daily, siren_daily):
    if "oblast" in frame.columns:
        known_oblasts.update(frame["oblast"].dropna().unique())

pending_oblast = st.session_state.pop("pending_oblast", None)
if pending_oblast in known_oblasts:
    st.session_state["selected_oblast"] = pending_oblast

min_day = min(
    frame["day"].min()
    for frame in (national_daily, viina_daily, siren_daily)
    if not frame.empty
)
max_day = national_daily["day"].max()

with st.container(border=True):
    st.subheader("Налаштуйте перегляд")
    filter_cols = st.columns(2, gap="large")
    with filter_cols[0]:
        period_mode = st.selectbox(
            "1. Період даних",
            ["Увесь період", "Останні 30 днів", "Останні 90 днів", "Власний період"],
            help="Останні 30 або 90 днів відраховуються від найновішого запису, а не від сьогодні.",
        )
    with filter_cols[1]:
        oblast_options = ["Усі області", *sorted(known_oblasts)]
        selected_oblast = st.selectbox(
            "2. Область", oblast_options, key="selected_oblast",
            help="Оберіть область тут або натисніть на неї на карті в розділі «Карта областей».",
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
            "Дати: від і до", value=(min_day.date(), max_day.date()),
            min_value=min_day.date(), max_value=max_day.date(),
        )
    else:
        date_range = (min_day.date(), max_day.date())

with st.expander("ℹ️ Джерела та дати останніх записів"):
    st.write("**Kaggle** — історичні записи про ракетні атаки й БпЛА")
    st.write("**VIINA** — окремі геокодовані повітряні інциденти")
    st.write("**eTryvoga** — історія повітряних тривог, лише для контексту")
    st.caption(
        "Джерела мають різні визначення події. Обстріли та тривоги "
        "не змішуються як один тип факту."
    )
    if not viina_daily.empty:
        st.caption("VIINA: останній запис " + viina_daily["day"].max().strftime("%d.%m.%Y"))
    if pd.notna(latest_source):
        st.caption("Kaggle: останній запис " + latest_source.strftime("%d.%m.%Y"))

    source_periods = []
    for source_name, frame, color in (
        ("Kaggle · атаки", national_daily, "#147bb3"),
        ("VIINA · інциденти", viina_daily, "#168c84"),
        ("eTryvoga · тривоги", siren_daily, "#dc9a42"),
    ):
        if not frame.empty and frame["day"].notna().any():
            source_periods.append(
                (source_name, frame["day"].min(), frame["day"].max(), color)
            )
    if source_periods:
        coverage_chart = go.Figure()
        for source_name, first_day, last_day, color in source_periods:
            coverage_chart.add_trace(go.Scatter(
                x=[first_day, last_day], y=[source_name, source_name],
                mode="lines+markers", showlegend=False,
                line=dict(color=color, width=14),
                marker=dict(color="#ffffff", size=13, line=dict(color=color, width=4)),
                customdata=[[first_day, last_day], [first_day, last_day]],
                hovertemplate=(
                    "%{y}<br>Від: %{customdata[0]|%d.%m.%Y}<br>"
                    "До: %{customdata[1]|%d.%m.%Y}<extra></extra>"
                ),
            ))
            coverage_chart.add_annotation(
                x=last_day, y=source_name,
                text=last_day.strftime("%d.%m.%Y"), showarrow=False,
                xanchor="left", xshift=12, font=dict(size=16, color="#38536d"),
            )
        style_chart(coverage_chart, height=285)
        coverage_chart.update_layout(margin=dict(l=20, r=125, t=24, b=45))
        coverage_chart.update_xaxes(title_text="Фактичний часовий діапазон у знімку")
        coverage_chart.update_yaxes(title_text=None, showgrid=False)
        st.plotly_chart(
            coverage_chart, width="stretch", key="source_coverage",
            config={"displayModeBar": False},
        )

if period_mode == "Власний період" and (
    not isinstance(date_range, tuple) or len(date_range) != 2
):
    st.info("Оберіть другу дату, щоб показати результати за власний період.")
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_day = pd.Timestamp(date_range[0], tz="UTC")
    end_day = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
else:
    start_day = min_day
    end_day = max_day + pd.Timedelta(days=1)

st.markdown(
    '<div class="selection-strip">'
    f'<span class="selection-chip">📅 Період: <b>{start_day:%d.%m.%Y} — '
    f'{end_day - pd.Timedelta(days=1):%d.%m.%Y}</b></span>'
    f'<span class="selection-chip">📍 Територія: <b>{escape(selected_oblast)}</b></span>'
    '</div>',
    unsafe_allow_html=True,
)

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

period_interception = interception_daily[
    (interception_daily["day"] >= start_day)
    & (interception_daily["day"] < end_day)
].copy() if not interception_daily.empty else interception_daily.copy()

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

scope_viina = period_viina_daily if selected_oblast == "Усі області" or period_viina_daily.empty else (
    period_viina_daily[period_viina_daily["oblast"] == selected_oblast]
)
scope_sirens = period_siren_daily if selected_oblast == "Усі області" or period_siren_daily.empty else (
    period_siren_daily[period_siren_daily["oblast"] == selected_oblast]
)
viina_has_period_coverage = (
    not viina_daily.empty
    and start_day <= viina_daily["day"].max()
    and end_day > viina_daily["day"].min()
)
st.subheader(
    f"{scope_title} · {start_day:%d.%m.%Y} — "
    f"{end_day - pd.Timedelta(days=1):%d.%m.%Y}"
)
status_cols = st.columns(3)
status_cols[0].metric("Записів атак · Kaggle", fmt_int(overview_daily["all_events"].sum()))
status_cols[1].metric(
    "Повітряних інцидентів · VIINA",
    fmt_int(scope_viina["viina_events"].sum()) if viina_has_period_coverage else "—",
)
status_cols[2].metric("Тривог · окремий контекст", fmt_int(scope_sirens["alert_count"].sum()) if not scope_sirens.empty else "0")
if not viina_has_period_coverage:
    st.info("VIINA не містить даних за вибраний період. Прочерк означає відсутність даних, а не відсутність інцидентів.")
elif not viina_daily.empty and (max_day - viina_daily["day"].max()).days > 30:
    st.caption(
        "VIINA охоплює лише частину вибраного періоду; останній запис: "
        + viina_daily["day"].max().strftime("%d.%m.%Y") + "."
    )
if pd.notna(generated_at):
    st.caption("Знімок даних оновлено: " + generated_at.strftime("%d.%m.%Y %H:%M UTC"))

st.markdown(
    """<div class="reading-guide" aria-label="Як користуватися панеллю">
      <div class="guide-card">
        <div class="guide-icon" aria-hidden="true">1</div>
        <div><div class="guide-title">Оберіть зріз</div>
        <div class="guide-note">Період і область керують усіма показниками нижче.</div></div>
      </div>
      <div class="guide-card">
        <div class="guide-icon" aria-hidden="true">2</div>
        <div><div class="guide-title">Порівняйте факти</div>
        <div class="guide-note">Кількості, відсотки та динаміка показані окремо.</div></div>
      </div>
      <div class="guide-card">
        <div class="guide-icon" aria-hidden="true">3</div>
        <div><div class="guide-title">Перевірте надійність</div>
        <div class="guide-note">Прочерк означає відсутність даних, а не нуль подій.</div></div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown("### Оберіть розділ")
overview_tab, interception_tab, regions_tab, risk_tab, quality_tab, ml_tab = st.tabs(
    [
        "📈 Загальна картина",
        "🛡️ Запуски та збиття",
        "🗺️ Карта областей",
        "📊 Порівняння областей",
        "🔎 Надійність даних",
        "⚙️ Стан моделі",
    ]
)

with overview_tab:
    st.subheader(f"Огляд: {scope_title}")

    source_viina = scope_viina
    st.caption(
        (
            f"За вибраний період у VIINA є {fmt_int(source_viina['viina_events'].sum())} "
            "геокодованих повітряних інцидентів. "
            if viina_has_period_coverage else
            "За вибраний період дані VIINA відсутні. "
        )
        + "Це окреме джерело, його події не додаються до записів атак Kaggle."
    )

    current_period = overview_daily
    current_count = float(current_period["all_events"].sum())
    change_pct = None
    if period_mode != "Увесь період":
        comparison_days = end_day - start_day
        comparison_start = start_day - comparison_days
        if selected_oblast == "Усі області":
            previous_records = national_daily[
                (national_daily["day"] >= comparison_start)
                & (national_daily["day"] < start_day)
            ]["attack_records"]
        else:
            previous_records = oblast_daily[
                (oblast_daily["oblast"] == selected_oblast)
                & (oblast_daily["day"] >= comparison_start)
                & (oblast_daily["day"] < start_day)
            ]["attack_events"]
        previous_count = float(previous_records.sum())
        if previous_count:
            change_pct = (current_count - previous_count) / previous_count * 100

    st.subheader("Показники Kaggle за вибраний період")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Записів атак", fmt_int(current_count))
    s2.metric(
        "Зміна до попереднього періоду",
        "—" if change_pct is None else fmt_pct_points(change_pct),
        help=(
            "Для всього періоду порівняння немає; якщо попередній такий "
            "період не містить записів, відсоток не обчислюється."
        ),
    )
    s3.metric(
        "Записів про БпЛА",
        fmt_int(current_period["uav_events"].sum()) if not current_period.empty else "0",
    )
    s4.metric(
        "Записів про ракети",
        fmt_int(current_period["missile_events"].sum()) if not current_period.empty else "0",
    )

    st.caption(
        "Показники описують історичні записи у відкритому джерелі "
        "та не є прогнозом майбутніх подій."
    )

    st.subheader("Як змінювалася кількість записів")
    if overview_daily.empty and source_viina.empty:
        st.info("За вибраний період для цієї області немає розмічених записів.")
    else:
        frequency = chart_frequency(start_day, end_day)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=.18,
            subplot_titles=("Записи атак · Kaggle", "Повітряні інциденти · VIINA"),
        )
        for row, source, column, color, fill in (
            (1, overview_daily, "all_events", "#156fa6", "rgba(21,111,166,.19)"),
            (2, source_viina, "viina_events", "#158f88", "rgba(21,143,136,.18)"),
        ):
            if source.empty:
                fig.add_annotation(
                    text="Немає записів за вибраний період", row=row, col=1,
                    showarrow=False, font=dict(size=17, color="#657a91"),
                )
                continue
            series = (
                source.set_index("day")[column]
                .resample(frequency).sum().reset_index()
            )
            fig.add_trace(go.Scatter(
                x=series["day"], y=series[column], mode="lines",
                line=dict(color=color, width=3, shape="spline", smoothing=.45),
                fill="tozeroy", fillcolor=fill, showlegend=False,
                hovertemplate="%{x|%d.%m.%Y}<br>Записів: %{y:,.0f}<extra></extra>",
            ), row=row, col=1)
        style_chart(fig, height=510)
        fig.update_annotations(font=dict(size=18, color="#23425e"))
        fig.update_yaxes(title_text="Кількість", row=1, col=1)
        fig.update_yaxes(title_text="Кількість", row=2, col=1)
        fig.update_xaxes(title_text="Дата", row=2, col=1)
        st.caption(
            "Кожне джерело має власну шкалу. Для читабельності дані "
            f"згруповано {chart_frequency_label(start_day, end_day)}; "
            "точні значення доступні при наведенні."
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

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
        structure = structure.sort_values("Частка, %")
        structure_chart = go.Figure(go.Bar(
            x=structure["Частка, %"], y=structure["Категорія"],
            orientation="h", width=.57,
            marker=dict(color=[
                {"БпЛА": "#147bb3", "Ракети": "#e4a04b",
                 "Керовані авіабомби": "#755eaa"}.get(category, "#5786a5")
                for category in structure["Категорія"]
            ], line_width=0),
            text=[
                f"{fmt_pct_points(share)} · {fmt_int(count)}"
                for share, count in zip(structure["Частка, %"], structure["Кількість"])
            ],
            textposition="outside", textfont=dict(size=17, color="#213b55"),
            customdata=structure["Кількість"],
            hovertemplate="%{y}<br>Частка: %{x:.1f}%<br>Записів: %{customdata:,.0f}<extra></extra>",
        ))
        style_chart(structure_chart, height=max(230, 75 * len(structure) + 85))
        structure_chart.update_layout(margin=dict(l=22, r=125, t=20, b=40))
        structure_chart.update_xaxes(range=[0, 108], title_text="Частка записів, %", showgrid=True)
        structure_chart.update_yaxes(title_text=None, showgrid=False)
        st.plotly_chart(structure_chart, width="stretch", config={"displayModeBar": False})

        structure_table = structure.copy()
        structure_table["Частка, %"] = structure_table["Частка, %"].map(fmt_pct_points)
        st.dataframe(
            structure_table,
            width="stretch",
            hide_index=True,
            row_height=48,
        )

with interception_tab:
    st.subheader("Запущено та заявлено збитими · Україна")
    if selected_oblast != "Усі області":
        st.info(
            "Тут показано загальноукраїнські підсумки за вибраний період. "
            "Джерело не вказує область збиття кожної цілі, тому ці числа "
            "не приписуються вибраній області."
        )
    st.caption(
        "Порівнюються лише записи Kaggle, у яких наведено обидві коректні "
        "кількості: запущено та збито."
    )

    if period_interception.empty:
        st.info("За цей період немає повних пар даних про запуски та збиття.")
    else:
        launched = period_interception["launched"].sum()
        intercepted = period_interception["intercepted"].sum()
        not_confirmed = period_interception["not_confirmed_intercepted"].sum()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Повідомлено запущено", fmt_int(launched))
        metric_cols[1].metric("Заявлено збито", fmt_int(intercepted))
        metric_cols[2].metric("Без підтвердженого збиття", fmt_int(not_confirmed))
        metric_cols[3].metric("Частка заявлених збиттів", fmt_pct(intercepted / launched) if launched else "—")

        st.warning(
            "«Без підтвердженого збиття» = запущено − збито. "
            "Це не кількість влучань: джерело не містить перевірених даних "
            "про наслідок кожної незбитої цілі."
        )

        st.markdown(
            '<div class="visual-equation">'
            f'<div class="equation-card"><span class="equation-value">{fmt_int(launched)}</span>'
            '<span class="equation-label">повідомлено запущено</span></div>'
            '<span class="equation-sign">=</span>'
            f'<div class="equation-card"><span class="equation-value">{fmt_int(intercepted)}</span>'
            '<span class="equation-label">заявлено збито</span></div>'
            '<span class="equation-sign">+</span>'
            f'<div class="equation-card"><span class="equation-value">{fmt_int(not_confirmed)}</span>'
            '<span class="equation-label">без підтвердженого збиття</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        composition_chart = go.Figure(go.Pie(
            values=[intercepted, not_confirmed],
            labels=["Заявлено збито", "Без підтвердженого збиття"],
            hole=.68, sort=False, direction="clockwise",
            marker=dict(colors=["#168c84", "#e3a044"], line=dict(color="#ffffff", width=3)),
            textinfo="percent", textfont=dict(size=18, color="#ffffff"),
            hovertemplate="%{label}<br>%{value:,.0f} · %{percent}<extra></extra>",
        ))
        style_chart(composition_chart, height=350)
        composition_chart.update_layout(
            margin=dict(l=15, r=15, t=25, b=15),
            legend=dict(orientation="h", y=-.04, x=.5, xanchor="center", font_size=16),
            annotations=[dict(
                text=f"{fmt_pct(intercepted / launched) if launched else '—'}<br><span style='font-size:14px'>заявлено збито</span>",
                x=.5, y=.5, showarrow=False, font=dict(size=25, color="#173b5a"),
            )],
        )
        st.plotly_chart(
            composition_chart, width="stretch", key="interception_composition",
            config={"displayModeBar": False},
        )

        by_type = (
            period_interception.groupby("category", as_index=False)
            .agg(
                launched=("launched", "sum"),
                intercepted=("intercepted", "sum"),
                not_confirmed_intercepted=("not_confirmed_intercepted", "sum"),
                source_records=("source_records", "sum"),
            )
        )
        by_type = by_type[by_type["launched"] > 0].copy()
        by_type["Тип"] = by_type["category"].map(ATTACK_TYPE_UA).fillna("Інші")
        by_type["Частка збиття, %"] = by_type["intercepted"] / by_type["launched"] * 100
        by_type = by_type.sort_values("launched", ascending=True)

        if not by_type.empty:
            st.subheader("Розподіл за типом цілей")
            bar = go.Figure()
            for column, title, color in (
                ("intercepted", "Заявлено збито", "#158f88"),
                ("not_confirmed_intercepted", "Без підтвердженого збиття", "#e4a04b"),
            ):
                shares = by_type[column] / by_type["launched"] * 100
                bar.add_trace(go.Bar(
                    y=by_type["Тип"], x=shares, name=title,
                    orientation="h", marker=dict(color=color, line_width=0),
                    customdata=by_type[column],
                    text=[fmt_pct_points(value) if value >= 7 else "" for value in shares],
                    textfont=dict(size=17, color="#ffffff"),
                    textposition="inside",
                    hovertemplate="%{y}<br>" + title
                    + ": %{customdata:,.0f} (%{x:.1f}%)<extra></extra>",
                ))
            style_chart(bar, height=max(270, 105 * len(by_type) + 100))
            bar.update_layout(barmode="stack", bargap=.44,
                              legend=dict(orientation="h", y=1.26))
            bar.update_xaxes(range=[0, 100], title_text="Частка від запущених, %")
            bar.update_yaxes(title_text=None, showgrid=False)
            st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

            table = by_type[["Тип", "launched", "intercepted", "not_confirmed_intercepted", "Частка збиття, %"]].copy()
            table["Частка збиття, %"] = table["Частка збиття, %"].map(fmt_pct_points)
            table = table.rename(columns={
                "launched": "Запущено", "intercepted": "Збито",
                "not_confirmed_intercepted": "Без підтвердженого збиття",
            })
            st.dataframe(table, width="stretch", hide_index=True, row_height=48)

        st.subheader("Як змінювалися повідомлені кількості")
        period_days = (end_day - start_day).days
        frequency = chart_frequency(start_day, end_day)
        history = (
            period_interception.set_index("day")
            .resample(frequency)[["launched", "intercepted"]]
            .sum()
            .reset_index()
        )
        history["remaining"] = history["launched"] - history["intercepted"]
        history_chart = go.Figure()
        for column, title, line_color, fill_color in (
            ("intercepted", "Заявлено збито", "#158f88", "rgba(21,143,136,.76)"),
            ("remaining", "Без підтвердженого збиття", "#da953d", "rgba(228,160,75,.68)"),
        ):
            history_chart.add_trace(go.Scatter(
                x=history["day"], y=history[column], name=title,
                mode="lines", stackgroup="кількість",
                line=dict(color=line_color, width=1.8), fillcolor=fill_color,
                hovertemplate="%{x|%d.%m.%Y}<br>" + title
                + ": %{y:,.0f}<extra></extra>",
            ))
        style_chart(history_chart, height=360)
        history_chart.update_layout(hovermode="x unified")
        history_chart.update_xaxes(title_text="Дата")
        history_chart.update_yaxes(title_text="Кількість цілей")
        st.plotly_chart(history_chart, width="stretch", config={"displayModeBar": False})

        with st.expander("Переглянути записи за датою і типом"):
            daily_table = period_interception[[
                "day", "category", "launched", "intercepted", "not_confirmed_intercepted"
            ]].copy().sort_values("day", ascending=False)
            daily_table["Дата"] = daily_table["day"].dt.strftime("%d.%m.%Y")
            daily_table["Тип"] = daily_table["category"].map(ATTACK_TYPE_UA).fillna("Інші")
            daily_table = daily_table.rename(columns={
                "launched": "Запущено", "intercepted": "Збито",
                "not_confirmed_intercepted": "Без підтвердженого збиття",
            })
            st.dataframe(
                daily_table[["Дата", "Тип", "Запущено", "Збито", "Без підтвердженого збиття"]],
                width="stretch", hide_index=True, height=390, row_height=48,
            )

        if not interception_coverage.empty:
            coverage = interception_coverage.iloc[0]
            st.caption(
                "Повнота всієї історичної вибірки: "
                f"{fmt_int(coverage['comparable_records'])} із {fmt_int(coverage['total_records'])} "
                "записів мають порівнювані значення; "
                f"{fmt_int(coverage['excluded_records'])} неповних або некоректних "
                "записів не враховано в показниках збиття."
            )

with regions_tab:
    st.subheader("Карта України за областями")
    st.caption(
        "Карта завжди показує всі адміністративні регіони. Наведи курсор на "
        "область, щоб побачити дані з окремих джерел та узгоджену історичну частку."
    )
    st.caption("Межі: OpenStreetMap (ODbL), набір ukraine-geo-data. Міста Київ і Севастополь не мають окремих контурів у цьому наборі.")

    consensus = build_oblast_consensus(
        sorted(known_oblasts),
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
                consensus[consensus["oblast"].isin(map_oblasts)],
                geojson=geojson,
                locations="oblast",
                featureidkey="properties.name",
                color="consensus_share_pct",
                color_continuous_scale=["#e3eff9", "#72afcf", "#145c90"],
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
            map_fig.update_traces(marker_line_width=0.9, marker_line_color="#f3f7fc")
            map_fig.update_geos(
                fitbounds="locations",
                visible=False,
                bgcolor="rgba(0,0,0,0)",
            )
            map_fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar_title="Консенсус, %",
                clickmode="event+select",
                height=680,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial, sans-serif", size=17, color="#233c58"),
                hoverlabel=dict(font_size=17, bgcolor="#ffffff"),
                coloraxis_colorbar=dict(tickfont_size=16, title_font_size=16),
            )
            map_event = st.plotly_chart(
                map_fig,
                width="stretch",
                key="oblast_map",
                on_select="rerun",
                selection_mode="points",
                config={"displayModeBar": False},
            )
            clicked_oblast = selected_map_oblast(map_event)
            if clicked_oblast != st.session_state.get("last_map_selection"):
                st.session_state["last_map_selection"] = clicked_oblast
                if clicked_oblast in known_oblasts:
                    st.session_state["pending_oblast"] = clicked_oblast
                    st.rerun()
        else:
            st.warning(
                "Межі областей тимчасово недоступні. Дані по областях "
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
            width="stretch",
            hide_index=True,
            key="oblast_table",
            on_select="rerun",
            selection_mode="single-row",
            height=620,
            row_height=48,
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

    if table_oblast != st.session_state.get("last_table_selection"):
        st.session_state["last_table_selection"] = table_oblast
        if table_oblast in known_oblasts:
            st.session_state["pending_oblast"] = table_oblast
            st.rerun()

    detail_oblast = selected_oblast if selected_oblast != "Усі області" else (
        clicked_oblast or table_oblast
    )

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
            detail_fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=.2,
                subplot_titles=("Записи атак · Kaggle", "Повітряні інциденти · VIINA"),
            )
            for row_number, frame, column, color, fill in (
                (1, kaggle_region, "Kaggle: записи атак", "#156fa6", "rgba(21,111,166,.18)"),
                (2, viina_region, "VIINA: повітряні інциденти", "#158f88", "rgba(21,143,136,.18)"),
            ):
                if frame.empty:
                    detail_fig.add_annotation(
                        text="Немає записів за вибраний період",
                        row=row_number, col=1, showarrow=False,
                        font=dict(size=17, color="#657a91"),
                    )
                    continue
                series = frame.set_index("day")[column].resample(
                    chart_frequency(start_day, end_day)
                ).sum().reset_index()
                detail_fig.add_trace(go.Scatter(
                    x=series["day"], y=series[column], mode="lines",
                    line=dict(color=color, width=3, shape="spline", smoothing=.4),
                    fill="tozeroy", fillcolor=fill, showlegend=False,
                    hovertemplate="%{x|%d.%m.%Y}<br>Записів: %{y:,.0f}<extra></extra>",
                ), row=row_number, col=1)
            style_chart(detail_fig, height=460)
            detail_fig.update_annotations(font=dict(size=16, color="#23425e"))
            detail_fig.update_yaxes(title_text="Кількість", row=1, col=1)
            detail_fig.update_yaxes(title_text="Кількість", row=2, col=1)
            detail_fig.update_xaxes(title_text="Дата", row=2, col=1)
            st.plotly_chart(detail_fig, width="stretch", config={"displayModeBar": False})
            st.caption(
                f"Дані згруповано {chart_frequency_label(start_day, end_day)}. "
                "Порожня панель означає відсутність записів джерела за цей період."
            )

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

    risk_summary = build_oblast_consensus(
        sorted(known_oblasts),
        period_oblast_daily,
        period_viina_daily,
        period_siren_daily,
    )

    if not risk_summary.empty:
        ranked = risk_summary.head(15).sort_values("consensus_share_pct")
        max_share = float(ranked["consensus_share_pct"].max())
        colors = [
            "#e4a04b" if name == selected_oblast else
            px.colors.sample_colorscale(
                "Blues", .32 + .55 * float(share) / max(max_share, 1)
            )[0]
            for name, share in zip(ranked["oblast"], ranked["consensus_share_pct"])
        ]
        risk_chart = go.Figure(go.Bar(
            x=ranked["consensus_share_pct"], y=ranked["oblast"],
            orientation="h", marker=dict(color=colors, line_width=0),
            customdata=ranked[["kaggle_events", "viina_events", "alert_count"]],
            text=[fmt_pct_points(value) for value in ranked["consensus_share_pct"]],
            textposition="outside", textfont=dict(size=16, color="#243e58"),
            hovertemplate=(
                "%{y}<br>Узгоджена частка: %{x:.1f}%<br>"
                "Kaggle: %{customdata[0]:,.0f}<br>VIINA: %{customdata[1]:,.0f}<br>"
                "Тривоги: %{customdata[2]:,.0f}<extra></extra>"
            ),
        ))
        style_chart(risk_chart, height=650)
        risk_chart.update_layout(margin=dict(l=20, r=70, t=20, b=42), bargap=.3)
        risk_chart.update_xaxes(
            range=[0, max(1, max_share * 1.22)], title_text="Узгоджена частка, %"
        )
        risk_chart.update_yaxes(title_text=None, showgrid=False)
        st.plotly_chart(risk_chart, width="stretch", config={"displayModeBar": False})

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

    coverage_values = [
        100 * float(quality.get("region_coverage_rate") or 0),
        100 * float(quality.get("recent_region_coverage_90d") or 0),
    ]
    quality_chart = go.Figure(go.Bar(
        x=coverage_values,
        y=["За весь період", "За останні 90 днів"],
        orientation="h", width=.46,
        marker=dict(color=["#147bb3", "#168c84"], line_width=0),
        text=[fmt_pct_points(value) for value in coverage_values],
        textposition="outside", textfont=dict(size=17, color="#243e58"),
        hovertemplate="%{y}<br>Покриття: %{x:.1f}%<extra></extra>",
    ))
    style_chart(quality_chart, height=265)
    quality_chart.add_vline(
        x=25, line_width=2, line_dash="dash", line_color="#d68d2f",
        annotation_text="Мінімальний поріг 25%",
        annotation_position="top right",
        annotation_font_size=16,
    )
    quality_chart.update_layout(margin=dict(l=20, r=85, t=45, b=45), showlegend=False)
    quality_chart.update_xaxes(range=[0, 100], title_text="Записи з визначеною областю, %")
    quality_chart.update_yaxes(title_text=None, showgrid=False)
    st.plotly_chart(
        quality_chart, width="stretch", key="region_coverage_quality",
        config={"displayModeBar": False},
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
        if key not in QUALITY_LABELS_UA:
            continue
        label = QUALITY_LABELS_UA[key]
        if key in {"region_coverage_rate", "recent_region_coverage_90d"}:
            value = fmt_pct(value)
        elif key == "model_ready_for_serving":
            value = "Так" if value else "Ні"
        quality_rows.append({"Показник": label, "Значення": str(value)})

    st.dataframe(
        pd.DataFrame(quality_rows),
        width="stretch",
        hide_index=True,
        row_height=48,
    )

with ml_tab:
    st.subheader("Стан самонавчання моделі")
    st.caption(
        "Щотижня система перевіряє нові дані. Навчання і заміна моделі "
        "дозволені лише після підтвердження повноти регіональних спостережень."
    )
    gate_blocked = not bool(quality.get("model_ready_for_serving", False))
    process_steps = [
        ("Крок 1", "Оновлення даних", "Джерела завантажуються щотижня.", False),
        (
            "Крок 2", "Контроль якості",
            "Зупинено: регіональних міток недостатньо."
            if gate_blocked else "Перевірки пройдено.",
            gate_blocked,
        ),
        ("Крок 3", "Навчання кандидата", "Очікує підтверджених результатів.", False),
        ("Крок 4", "Чесне порівняння", "Лише на періоді, якого модель не бачила.", False),
        ("Крок 5", "Заміна моделі", "Лише після покращення метрик.", False),
    ]
    process_html = '<div class="process-flow">'
    for number, title, note, blocked in process_steps:
        process_html += (
            f'<div class="process-step{" blocked" if blocked else ""}">'
            f'<div class="process-number">{escape(number)}</div>'
            f'<div class="process-title">{escape(title)}</div>'
            f'<div class="process-note">{escape(note)}</div></div>'
        )
    process_html += '</div>'
    st.markdown(process_html, unsafe_allow_html=True)

    if learning.get("status") == "blocked_unverified_outcomes":
        st.warning(
            "Навчання призупинено: відсутність запису про атаку не доводить, "
            "що атаки не було. Для перевірки моделі потрібні підтверджені "
            "дні спостереження для кожної області."
        )
        st.metric("Днів з невідомим результатом", fmt_int(learning.get("unknown_rows")))
    elif not quality.get("model_ready_for_serving", False):
        st.warning(
            "Автоматичне підвищення моделі заблоковано, доки регіональні "
            "дані не стануть достатньо повними. Навчання й порівняння "
            "кандидатів тривають для перевірки методики."
        )

    candidate = learning.get("candidate_metrics") or {}
    baseline = learning.get("baseline_metrics") or {}
    champion = learning.get("champion_metrics_on_same_holdout") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Якість розрізнення кандидата",
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
    elif promoted is False and learning.get("status") != "blocked_unverified_outcomes":
        st.info(
            "Нова модель не замінила чинну; причину наведено нижче."
        )

    decision = str(learning.get("decision_reason", "—"))
    st.write("Рішення системи:", DECISION_UA.get(
        decision, "Причину рішення не розпізнано; перевірте звіт моделі."
    ))
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
                    "Якість розрізнення": metrics.get("roc_auc"),
                    "Середня точність": metrics.get("average_precision"),
                    "Показник Брієра": metrics.get("brier_score"),
                    "Збалансована точність": metrics.get("balanced_accuracy"),
                }
            )
    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            width="stretch",
            hide_index=True,
            row_height=48,
        )

    st.caption(
        "Якість розрізнення та середня точність: більше — краще. "
        "Показник Брієра: менше — краще. За низького покриття областей "
        "ці метрики не підтверджують готовність до прогнозу."
    )

    st.markdown(
        "**Що система робить сама:** оновлює історичні дані, перевіряє "
        "якість і формує ознаки. Навчає й порівнює моделі тільки тоді, "
        "коли результати спостережень підтверджено."
    )
    st.markdown(
        "**Чого система не робить сама:** не вигадує відсутні факти, "
        "не переписує сирі дані без правил і не перетворює неповні дані "
        "на точний прогноз."
    )

st.divider()
st.caption(
    "Навчальний проєкт з аналізу даних. Показники залежать від повноти "
    "відкритих джерел; відсутність запису не означає відсутність події."
)
