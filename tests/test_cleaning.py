import pandas as pd

from src.data.cleaning import find_possible_duplicates, normalize_oblast


def test_normalize_oblast_aliases():
    assert normalize_oblast("Одеська") == "Одеська область"
    assert normalize_oblast("м. Київ") == "Київ"


def test_find_possible_duplicates_does_not_delete_rows():
    df = pd.DataFrame([
        {"attack_id": 1, "started_at": "2026-01-01T10:15:00Z", "oblast": "Одеська", "attack_type": "UAV"},
        {"attack_id": 2, "started_at": "2026-01-01T10:40:00Z", "oblast": "Одеська область", "attack_type": "uav"},
        {"attack_id": 3, "started_at": "2026-01-02T10:40:00Z", "oblast": "Одеська область", "attack_type": "UAV"},
    ])
    result = find_possible_duplicates(df)
    assert set(result["attack_id"]) == {1, 2}
    assert len(result) == 2
