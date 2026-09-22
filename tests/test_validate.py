import pandas as pd

from src.data.validate import normalize_attacks, validate_attacks


def test_valid_attack_table_has_no_errors():
    df = pd.DataFrame([{
        "attack_id": 1,
        "started_at": "2026-01-01T10:00:00Z",
        "oblast": "Одеська область",
        "attack_type": "UAV",
        "confidence": "reported",
    }])
    assert validate_attacks(df) == []


def test_invalid_confidence_is_reported():
    df = pd.DataFrame([{
        "attack_id": 1,
        "started_at": "2026-01-01",
        "oblast": "Одеська область",
        "attack_type": "UAV",
        "confidence": "guess",
    }])
    errors = validate_attacks(df)
    assert any("Invalid confidence" in error for error in errors)


def test_normalize_attacks_converts_datetime_to_utc():
    df = pd.DataFrame([{
        "attack_id": 1,
        "started_at": "2026-01-01 12:00",
        "oblast": " Одеська область ",
        "attack_type": " UAV ",
        "confidence": " REPORTED ",
    }])
    result = normalize_attacks(df)
    assert str(result.loc[0, "attack_type"]) == "uav"
    assert str(result.loc[0, "confidence"]) == "reported"
    assert str(result.loc[0, "oblast"]) == "Одеська область"
    assert str(result.loc[0, "started_at"].tz) == "UTC"
