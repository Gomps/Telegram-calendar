"""Детерминированный санитайзер действий LLM.

Ответ модели — только черновик. Всё, что можно проверить и вычислить кодом,
перепроверяется здесь по исходному тексту сообщения:
- классификация разовое/периодическое — жёстко по маркерам периодичности;
- дни недели, чётность, шаг повторения — извлекаются из текста и
  перезаписывают выдумки модели;
- уточняющие вопросы, не связанные со словами пользователя (утечки из
  примеров промпта, бессмыслица), заменяются детерминированными.
"""

import logging
import re
from datetime import datetime
from typing import Optional

from .timeparse import (
    extract_interval_minutes,
    extract_parity,
    extract_weekdays,
    parse_time_expression,
)

log = logging.getLogger(__name__)

# Явные признаки периодичности в тексте
PERIODIC_RE = re.compile(
    r"\b(кажд\w+|ежедневн\w*|еженедельн\w*|ежемесячн\w*|ежечасн\w*|раз\s+в\b|"
    r"по\s+будням|по\s+выходным|"
    r"по\s+(понедельник|вторник|сред|четверг|пятниц|суббот|воскресень)\w*)"
)

# Служебные слова, не влияющие на смысл напоминания
FILLER_WORDS = {
    "напомни", "напоминай", "напомнить", "напоминание", "напоминания",
    "пожалуйста", "поставь", "создай", "сделай", "чтобы", "нужно", "надо",
    "мне", "меня", "потом", "если", "когда", "завтра", "сегодня", "через",
    "каждый", "каждую", "каждое", "каждые", "минут", "минуты", "часов", "часа",
}


def stems(text: str) -> set[str]:
    words = re.findall(r"[а-яa-z0-9]+", str(text).lower().replace("ё", "е"))
    return {w[:4] for w in words if len(w) >= 4 and w not in FILLER_WORDS}


def text_matches_source(candidate: str, source: str, threshold: float = 0.5) -> bool:
    """Доля 4-буквенных основ значимых слов candidate, найденных в source."""
    c, s = stems(candidate), stems(source)
    if not c:
        return True
    return len(c & s) / len(c) >= threshold


def is_periodic_text(text: str) -> bool:
    return PERIODIC_RE.search(str(text).lower().replace("ё", "е")) is not None


def sanitize_action(action: dict, text: str, now_local: datetime) -> dict:
    """Жёсткая детерминированная правка черновика от LLM по исходному тексту."""
    kind = action.get("action")

    if kind == "multi" and isinstance(action.get("actions"), list):
        action["actions"] = [
            sanitize_action(sub, text, now_local) if isinstance(sub, dict) else sub
            for sub in action["actions"]
        ]
        return action

    if kind == "create_recurring" and not is_periodic_text(text):
        action = _demote_to_reminder(action, text, now_local)
    elif kind == "create_reminder" and is_periodic_text(text):
        action = _promote_to_recurring(action, text)

    kind = action.get("action")
    if kind == "create_recurring":
        _fix_rule_from_text(action.get("recurring"), text)
    elif kind == "ask_clarification":
        _fix_clarification(action, text)
    return action


def _demote_to_reminder(action: dict, text: str, now_local: datetime) -> dict:
    """В тексте нет периодичности, а модель вернула серию — делаем разовое."""
    log.info("Санитайзер: в тексте нет периодичности — create_recurring -> create_reminder")
    rule = action.get("recurring") or {}
    fire_at = None
    # время: дословное выражение -> текст целиком -> time из правила
    expr = str(action.get("time_expression") or "").strip()
    resolved = parse_time_expression(expr, now_local) if expr else None
    if resolved is None:
        resolved = parse_time_expression(text, now_local)
        if resolved is not None:
            expr = text
    if resolved is not None:
        fire_at = resolved.strftime("%Y-%m-%dT%H:%M")
    elif rule.get("time"):
        expr = expr or f"в {rule['time']}"
    return {
        "action": "create_reminder",
        "reminder_text": action.get("reminder_text", ""),
        "time_expression": expr,
        **({"fire_at": fire_at} if fire_at else {}),
        **({"context_updates": action["context_updates"]} if action.get("context_updates") else {}),
    }


def _promote_to_recurring(action: dict, text: str) -> dict:
    """В тексте есть периодичность, а модель вернула разовое — строим правило."""
    interval = extract_interval_minutes(text)
    weekdays = extract_weekdays(text)
    parity = extract_parity(text)

    if interval is not None and action.get("recurring"):
        pass  # правило уже есть — оставим и поправим ниже
    rule: Optional[dict] = None
    time_s = None
    fire_at = str(action.get("fire_at") or "")
    if "T" in fire_at:
        time_s = fire_at.split("T", 1)[1][:5]
    if interval is None:
        # weekly/daily по времени из fire_at
        if time_s:
            rule = {
                "type": "weekly" if weekdays else "daily",
                "days_of_week": weekdays or None,
                "time": time_s,
            }
    if rule is None and interval is not None:
        # интервальная серия без якорей достроена быть не может — оставляем
        # разовое, дальше сработает обычный поток уточнений
        log.info("Санитайзер: периодичность с интервалом, но нет границ — оставляю как есть")
        return action
    if rule is None:
        log.info("Санитайзер: периодичность в тексте, но время не извлечь — оставляю как есть")
        return action

    if parity:
        rule["day_parity"] = parity
    log.info("Санитайзер: create_reminder -> create_recurring %s", rule)
    return {
        "action": "create_recurring",
        "reminder_text": action.get("reminder_text", ""),
        "recurring": rule,
        **({"context_updates": action["context_updates"]} if action.get("context_updates") else {}),
    }


def _fix_rule_from_text(rule, text: str) -> None:
    """Дни недели / чётность / шаг в правиле серии — по тексту, не по модели."""
    if not isinstance(rule, dict):
        return
    weekdays = extract_weekdays(text)
    if weekdays and set(rule.get("days_of_week") or []) != set(weekdays):
        log.info(
            "Санитайзер: days_of_week %s -> %s (по тексту)",
            rule.get("days_of_week"), weekdays,
        )
        rule["days_of_week"] = weekdays
        if rule.get("type") == "daily":
            rule["type"] = "weekly"
    parity = extract_parity(text)
    if parity and rule.get("day_parity") != parity:
        log.info("Санитайзер: day_parity %s -> %s (по тексту)", rule.get("day_parity"), parity)
        rule["day_parity"] = parity
    interval = extract_interval_minutes(text)
    if (
        interval is not None
        and rule.get("type") == "interval"
        and int(rule.get("interval_minutes") or 0) != interval
    ):
        log.info(
            "Санитайзер: interval_minutes %s -> %s (по тексту)",
            rule.get("interval_minutes"), interval,
        )
        rule["interval_minutes"] = interval


def _fix_clarification(action: dict, text: str) -> None:
    """Вопрос модели обязан быть про слова пользователя, иначе заменяем."""
    question = str(action.get("clarification_question") or "")
    if text_matches_source(question, text, threshold=0.34):
        return
    subject = str(action.get("reminder_text") or "").strip()
    if not subject or not text_matches_source(subject, text):
        subject = " ".join(text.split()[:8])
    new_q = f"Когда напомнить «{subject}»? Укажи дату и время — например «во вторник в 12:00»."
    log.info("Санитайзер: вопрос «%s» не про сообщение пользователя — заменён", question)
    action["clarification_question"] = new_q


def enforce_weekday(fire_local: datetime, text: str, now_local: datetime) -> datetime:
    """День вычисленного времени обязан совпадать с днём недели из текста.

    При расхождении: пересчёт по всему тексту детерминированным парсером,
    иначе сдвиг на ближайший упомянутый день с сохранением времени суток.
    """
    weekdays = extract_weekdays(text)
    if not weekdays or fire_local.weekday() in weekdays:
        return fire_local
    reparsed = parse_time_expression(text, now_local)
    if reparsed is not None and reparsed.weekday() in weekdays:
        log.info("Санитайзер: день %s не из текста — пересчитано: %s", fire_local, reparsed)
        return reparsed
    from datetime import timedelta
    best = None
    for wd in weekdays:
        days_ahead = (wd - now_local.weekday()) % 7
        candidate = (now_local + timedelta(days=days_ahead)).replace(
            hour=fire_local.hour, minute=fire_local.minute, second=0, microsecond=0
        )
        if candidate <= now_local:
            candidate += timedelta(days=7)
        if best is None or candidate < best:
            best = candidate
    log.info("Санитайзер: день %s не из текста — сдвинуто на %s", fire_local, best)
    return best