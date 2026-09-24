"""Automatic project-improvement recommendations from quality and ML reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _days_old(value: Any, now: datetime | None = None) -> int | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0, (current - timestamp.astimezone(timezone.utc)).days)


def build_improvement_recommendations(
    quality: dict[str, Any] | None,
    learning: dict[str, Any] | None,
    latest_source_event_at: Any = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return prioritized, explainable improvement suggestions.

    The engine is intentionally rule-based: it can recommend what to improve,
    but it never changes raw facts or model rules by itself.
    """
    quality = quality or {}
    learning = learning or {}
    recommendations: list[dict[str, Any]] = []

    def add(
        priority: int,
        level: str,
        area: str,
        title: str,
        finding: str,
        action: str,
        automatic: bool,
    ) -> None:
        recommendations.append(
            {
                "priority": priority,
                "level": level,
                "area": area,
                "title": title,
                "finding": finding,
                "action": action,
                "automatic": automatic,
            }
        )

    blocking = _as_int(quality.get("blocking_errors"))
    coverage = _as_float(quality.get("region_coverage_rate"))
    recent_coverage = _as_float(quality.get("recent_region_coverage_90d"))
    unmapped = _as_int(quality.get("unmapped_region_rows"))
    warnings = _as_int(quality.get("warnings"))

    if blocking > 0:
        add(
            100,
            "Критично",
            "Дані",
            "Виправити критичні помилки даних",
            f"Контроль якості виявив {blocking} критичних помилок.",
            "Не запускати нове навчання, доки помилки не будуть усунені.",
            True,
        )

    if learning.get("status") == "blocked_unverified_outcomes":
        unknown_rows = _as_int(learning.get("unknown_rows"))
        add(
            99,
            "Високий",
            "Дані для навчання",
            "Підтвердити дні без атак",
            f"Для {unknown_rows} пар область–день результат невідомий.",
            "Додати перевірене джерело повноти спостережень із посиланням "
            "на кожен день; до цього не навчати модель на невідомих днях.",
            False,
        )

    if coverage < 0.25:
        add(
            95,
            "Високий",
            "Дані",
            "Збільшити регіональне покриття",
            f"Область визначена лише для {coverage:.1%} історичних записів.",
            "Підключити додаткове історичне джерело з надійним полем області "
            "та не вважати нерозмічені події негативними прикладами.",
            False,
        )

    if recent_coverage < 0.25:
        add(
            92,
            "Високий",
            "Дані",
            "Покращити свіжу регіональну розмітку",
            f"За останні 90 днів регіональне покриття становить {recent_coverage:.1%}.",
            "Пріоритетно додавати джерела, що стабільно містять область для нових подій.",
            False,
        )

    if unmapped > 0:
        add(
            75,
            "Середній",
            "Нормалізація",
            "Розібрати нерозпізнані регіони",
            f"Залишилося {unmapped} значень, які не можна однозначно прив'язати до області.",
            "Автоматично формувати список невідомих значень для ручної перевірки; "
            "додавати alias лише коли відповідність однозначна.",
            True,
        )

    source_age = _days_old(latest_source_event_at, now=now)
    if source_age is None:
        add(
            70,
            "Середній",
            "Оновлення",
            "Контролювати свіжість джерела",
            "Не вдалося визначити дату останньої події у джерелі.",
            "Додати перевірку свіжості до щотижневого циклу.",
            True,
        )
    elif source_age > 7:
        add(
            85,
            "Високий",
            "Оновлення",
            "Джерело відстає",
            f"Остання подія у джерелі має вік приблизно {source_age} днів.",
            "Перевірити доступність основного джерела або додати резервне джерело.",
            True,
        )

    candidate = learning.get("candidate_metrics") or {}
    baseline = learning.get("baseline_metrics") or {}
    candidate_ap = candidate.get("average_precision")
    baseline_ap = baseline.get("average_precision")

    if learning.get("beats_baseline") is False:
        add(
            90,
            "Високий",
            "Модель",
            "Модель не перевершує базовий рівень",
            "Кандидат не показав переваги над простою базовою моделлю.",
            "Не публікувати модельну оцінку; спочатку покращити дані та ознаки.",
            True,
        )
    elif candidate_ap is not None and baseline_ap is not None:
        if _as_float(candidate_ap) < 0.25:
            add(
                68,
                "Середній",
                "Модель",
                "Підсилити корисні ознаки",
                "Модель знаходить сигнал, але якість ранжування ще обмежена.",
                "Додати безпечні історичні ознаки: частоту за 7/30/90 днів, "
                "типи засобів та інтервал від попередньої події.",
                False,
            )
        else:
            add(
                25,
                "Добре",
                "Модель",
                "Модель перевершує базовий рівень",
                "Кандидат демонструє корисний історичний сигнал.",
                "Продовжувати контрольоване champion/challenger-порівняння.",
                True,
            )

    if not learning.get("model_ready_for_serving", False):
        add(
            88,
            "Високий",
            "Публікація",
            "Не показувати модель як готовий прогноз",
            "Поточні пороги готовності не пройдені.",
            "Показувати лише історичні частки та експериментальні метрики, "
            "доки якість даних не стане достатньою.",
            True,
        )

    if warnings > 0:
        add(
            55,
            "Середній",
            "Якість",
            "Зменшити кількість попереджень",
            f"Звіт якості містить {warnings} попереджень.",
            "Групувати попередження за типом і відстежувати їх динаміку між циклами.",
            True,
        )

    if not recommendations:
        add(
            10,
            "Добре",
            "Система",
            "Критичних напрямів покращення не виявлено",
            "Поточні автоматичні перевірки не знайшли суттєвих проблем.",
            "Продовжувати регулярне оновлення, тестування та моніторинг дрейфу даних.",
            True,
        )

    return sorted(recommendations, key=lambda item: item["priority"], reverse=True)


def project_readiness_score(
    quality: dict[str, Any] | None,
    learning: dict[str, Any] | None,
    latest_source_event_at: Any = None,
    now: datetime | None = None,
) -> int:
    """Compute a transparent technical-readiness score, not forecast accuracy."""
    quality = quality or {}
    learning = learning or {}

    blocking = _as_int(quality.get("blocking_errors"))
    coverage = min(1.0, _as_float(quality.get("region_coverage_rate")) / 0.40)
    recent = min(1.0, _as_float(quality.get("recent_region_coverage_90d")) / 0.40)

    source_age = _days_old(latest_source_event_at, now=now)
    if source_age is None:
        freshness = 0.0
    elif source_age <= 3:
        freshness = 1.0
    elif source_age <= 7:
        freshness = 0.75
    elif source_age <= 14:
        freshness = 0.4
    else:
        freshness = 0.1

    integrity = 1.0 if blocking == 0 else 0.0
    baseline = 1.0 if learning.get("beats_baseline") else 0.0

    score = (
        30 * integrity
        + 25 * coverage
        + 20 * recent
        + 15 * freshness
        + 10 * baseline
    )
    return int(round(max(0.0, min(100.0, score))))
