import pandas as pd

from src.analysis.consensus import build_oblast_consensus


def test_consensus_keeps_zero_event_oblasts_and_balances_sources():
    all_oblasts = [
        "Одеська область",
        "Київська область",
        "Львівська область",
    ]
    kaggle = pd.DataFrame(
        [
            {"oblast": "Одеська область", "attack_events": 30},
            {"oblast": "Київська область", "attack_events": 10},
        ]
    )
    viina = pd.DataFrame(
        [
            {"oblast": "Одеська область", "viina_events": 20},
            {"oblast": "Київська область", "viina_events": 20},
        ]
    )
    sirens = pd.DataFrame(
        [
            {
                "oblast": "Одеська область",
                "alert_count": 5,
                "alert_minutes": 100,
            }
        ]
    )

    result = build_oblast_consensus(all_oblasts, kaggle, viina, sirens)
    by_oblast = result.set_index("oblast")

    assert len(result) == 3
    assert by_oblast.loc["Львівська область", "kaggle_events"] == 0
    assert by_oblast.loc["Львівська область", "viina_events"] == 0

    # Kaggle share Odesa = 75%, VIINA share Odesa = 50%; mean = 62.5%.
    assert by_oblast.loc["Одеська область", "consensus_share_pct"] == 62.5

    # Alerts remain context and do not change attack-source consensus.
    assert by_oblast.loc["Одеська область", "alert_count"] == 5
    assert by_oblast.loc["Одеська область", "active_attack_sources"] == 2
