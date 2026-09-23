from __future__ import annotations

import pandas as pd

# Canonical administrative names used throughout the project.
OBLAST_ALIASES = {
    "вінницька": "Вінницька область",
    "вінницька область": "Вінницька область",
    "волинська": "Волинська область",
    "волинська область": "Волинська область",
    "дніпропетровська": "Дніпропетровська область",
    "дніпропетровська область": "Дніпропетровська область",
    "донецька": "Донецька область",
    "донецька область": "Донецька область",
    "житомирська": "Житомирська область",
    "житомирська область": "Житомирська область",
    "закарпатська": "Закарпатська область",
    "закарпатська область": "Закарпатська область",
    "запорізька": "Запорізька область",
    "запорізька область": "Запорізька область",
    "івано-франківська": "Івано-Франківська область",
    "івано-франківська область": "Івано-Франківська область",
    "київська": "Київська область",
    "київська область": "Київська область",
    "кіровоградська": "Кіровоградська область",
    "кіровоградська область": "Кіровоградська область",
    "львівська": "Львівська область",
    "львівська область": "Львівська область",
    "миколаївська": "Миколаївська область",
    "миколаївська область": "Миколаївська область",
    "одеська": "Одеська область",
    "одеська область": "Одеська область",
    "полтавська": "Полтавська область",
    "полтавська область": "Полтавська область",
    "рівненська": "Рівненська область",
    "рівненська область": "Рівненська область",
    "сумська": "Сумська область",
    "сумська область": "Сумська область",
    "тернопільська": "Тернопільська область",
    "тернопільська область": "Тернопільська область",
    "харківська": "Харківська область",
    "харківська область": "Харківська область",
    "херсонська": "Херсонська область",
    "херсонська область": "Херсонська область",
    "хмельницька": "Хмельницька область",
    "хмельницька область": "Хмельницька область",
    "черкаська": "Черкаська область",
    "черкаська область": "Черкаська область",
    "чернівецька": "Чернівецька область",
    "чернівецька область": "Чернівецька область",
    "чернігівська": "Чернігівська область",
    "чернігівська область": "Чернігівська область",
    "луганська": "Луганська область",
    "луганська область": "Луганська область",
    "київ": "Київ",
    "м. київ": "Київ",
    "місто київ": "Київ",
}


OBLAST_ALIASES.update({
    "vinnytsia oblast": "Вінницька область",
    "volyn oblast": "Волинська область",
    "dnipropetrovsk oblast": "Дніпропетровська область",
    "donetsk oblast": "Донецька область",
    "zhytomyr oblast": "Житомирська область",
    "zakarpattia oblast": "Закарпатська область",
    "zaporizhzhia oblast": "Запорізька область",
    "ivano-frankivsk oblast": "Івано-Франківська область",
    "kyiv oblast": "Київська область",
    "kirovohrad oblast": "Кіровоградська область",
    "luhansk oblast": "Луганська область",
    "lviv oblast": "Львівська область",
    "mykolaiv oblast": "Миколаївська область",
    "odesa oblast": "Одеська область",
    "odessa oblast": "Одеська область",
    "poltava oblast": "Полтавська область",
    "rivne oblast": "Рівненська область",
    "sumy oblast": "Сумська область",
    "ternopil oblast": "Тернопільська область",
    "kharkiv oblast": "Харківська область",
    "kherson oblast": "Херсонська область",
    "khmelnytskyi oblast": "Хмельницька область",
    "cherkasy oblast": "Черкаська область",
    "chernivtsi oblast": "Чернівецька область",
    "chernihiv oblast": "Чернігівська область",
    "kyiv": "Київ",
    "crimea": "Автономна Республіка Крим",
    "autonomous republic of crimea": "Автономна Республіка Крим",
    "sevastopol": "Севастополь",
,
    # Source target labels sometimes use settlement names; collapse only
    # well-known deterministic mappings to the containing oblast.
    "vinnytsia": "Вінницька область",
    "lutsk": "Волинська область",
    "lutsk oblast": "Волинська область",
    "dnipro": "Дніпропетровська область",
    "kryvyi rih": "Дніпропетровська область",
    "dnipropetrovsk oblas": "Дніпропетровська область",
    "donetsk": "Донецька область",
    "zhytomyr": "Житомирська область",
    "uzhhorod": "Закарпатська область",
    "zaporizhzhia": "Запорізька область",
    "ivano-frankivsk": "Івано-Франківська область",
    "ivano-frankivsk obкlast": "Івано-Франківська область",
    "kropyvnytskyi": "Кіровоградська область",
    "luhansk": "Луганська область",
    "lviv": "Львівська область",
    "mykolaiv": "Миколаївська область",
    "odesa": "Одеська область",
    "odessa": "Одеська область",
    "poltava": "Полтавська область",
    "kremenchuk": "Полтавська область",
    "rivne": "Рівненська область",
    "sumy": "Сумська область",
    "ternopil": "Тернопільська область",
    "kharkiv": "Харківська область",
    "kherson": "Херсонська область",
    "khmelnytskyi": "Хмельницька область",
    "starokostiantyniv": "Хмельницька область",
    "cherkasy": "Черкаська область",
    "chernivtsi": "Чернівецька область",
    "chernihiv": "Чернігівська область",
})


def normalize_oblast(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return OBLAST_ALIASES.get(text, str(value).strip())


def add_event_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["started_at"] = pd.to_datetime(result["started_at"], utc=True, errors="coerce")
    result["oblast_normalized"] = result["oblast"].map(normalize_oblast)
    result["event_key"] = (
        result["started_at"].dt.floor("h").astype("string")
        + "|" + result["oblast_normalized"].astype("string")
        + "|" + result["attack_type"].astype("string").str.lower()
    )
    return result


def find_possible_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    result = add_event_key(df)
    counts = result.groupby("event_key", dropna=False)["attack_id"].transform("size")
    return result[counts > 1].sort_values("event_key")
