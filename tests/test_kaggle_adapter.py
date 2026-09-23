import pandas as pd

from src.data.adapters.kaggle_attacks import (
    parse_affected_regions,
    transform_kaggle_attacks,
)


def test_parse_affected_regions_supports_english_kaggle_names():
    value = "['Kyiv oblast', 'Odesa oblast']"
    assert parse_affected_regions(value) == ["Київська область", "Одеська область"]


def test_transform_keeps_attack_counts_separate_from_regions():
    source = pd.DataFrame([
        {
            "time_start": "2026-01-01 20:00",
            "time_end": "2026-01-02 08:00",
            "model": "Shahed-136/131",
            "launched": 10,
            "destroyed": 8,
            "affected region": "['Kyiv oblast', 'Odesa oblast']",
            "source": "official/source/1",
        },
        {
            "time_start": "2026-01-02 20:00",
            "time_end": "2026-01-03 08:00",
            "model": "Iskander-M",
            "launched": 3,
            "destroyed": 1,
            "affected region": "['Kharkiv oblast', 'south']",
            "source": "official/source/2",
        },
    ])

    tables = transform_kaggle_attacks(source)

    assert len(tables["attacks"]) == 2
    assert len(tables["attack_regions"]) == 3
    assert tables["weapons"]["quantity"].sum() == 13
    assert tables["unmapped_regions"]["raw_region"].tolist() == ["south"]

    multi_region_attack = tables["attacks"].loc[
        tables["attacks"]["attack_type"] == "uav"
    ].iloc[0]
    assert pd.isna(multi_region_attack["oblast"])

    single_region_attack = tables["attacks"].loc[
        tables["attacks"]["attack_type"] == "missile"
    ].iloc[0]
    assert single_region_attack["oblast"] == "Харківська область"


def test_transform_accepts_mixed_date_and_datetime_formats():
    source = pd.DataFrame([
        {
            "time_start": "2026-01-01 20:00",
            "time_end": "2026-01-02 08:00",
            "model": "Shahed-136/131",
            "launched": 2,
            "destroyed": 1,
            "affected region": "['Odesa oblast']",
            "source": "official/source/1",
        },
        {
            "time_start": "2026-01-03",
            "time_end": "2026-01-03",
            "model": "Unknown UAV",
            "launched": 1,
            "destroyed": 1,
            "affected region": "['Odesa oblast']",
            "source": "official/source/2",
        },
    ])

    tables = transform_kaggle_attacks(source)

    assert len(tables["attacks"]) == 2
    assert tables["attacks"]["started_at"].isna().sum() == 0
