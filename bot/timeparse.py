"""Детерминированный разбор простых русских выражений времени.

Двухэтапная обработка запроса: LLM разбирает сообщение на составляющие
(суть напоминания, выражение времени дословно, факты о распорядке), а
вычисление времени по возможности делает этот модуль — маленькие модели
плохо считают «через 5 минут», детерминированный код не ошибается.
Контекстные выражения («после работы», «за два часа до сна») этот модуль
не берёт — их вычисляет LLM по сохранённому распорядку.
"""

import re
from datetime import date, datetime, timedelta
from typing import Optional

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "среда": 2, "среду": 2, "четверг": 3,
    "пятница": 4, "пятницу": 4, "суббота": 5, "субботу": 5, "воскресенье": 6,
}


def parse_time_expression(expr: str, now: datetime) -> Optional[datetime]:
    """Выражение времени -> aware datetime (в поясе `now`) или None.

    `now` — aware-время в часовом поясе пользователя.
    """
    s = " ".join(str(expr).lower().replace("ё", "е").split())
    if not s:
        return None

    delta = _parse_relative(s)
    if delta is not None:
        return now + delta

    clock = _parse_clock(s)
    if clock is None:
        return None  # без времени суток детерминированно не решить

    hour, minute = clock
    day, is_weekday = _parse_day(s, now)
    if day is None:
        # только время: ближайшее будущее вхождение
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    result = datetime(day.year, day.month, day.day, hour, minute, tzinfo=now.tzinfo)
    if is_weekday and result <= now:
        result += timedelta(days=7)  # «в среду в 9», а среда сегодня и время прошло
    return result


def _parse_relative(s: str) -> Optional[timedelta]:
    """«через 5 минут», «через час», «через полчаса», «через 2 часа 15 минут»."""
    if "через" not in s:
        return None
    tail = s.split("через", 1)[1]
    total = timedelta()
    found = False
    for m in re.finditer(r"(\d+)\s*(час\w*|ч\b|минут\w*|мин\b|м\b|дн\w*|день|сут\w*)", tail):
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith(("час", "ч")):
            total += timedelta(hours=n)
        elif unit.startswith(("мин", "м")):
            total += timedelta(minutes=n)
        else:
            total += timedelta(days=n)
        found = True
    if not found:
        if "полчаса" in tail:
            return timedelta(minutes=30)
        if re.match(r"\s*час\b", tail):
            return timedelta(hours=1)
        if re.match(r"\s*минуту\b", tail):
            return timedelta(minutes=1)
        if re.match(r"\s*(день|сутки)\b", tail):
            return timedelta(days=1)
        if re.match(r"\s*недел", tail):
            return timedelta(weeks=1)
        return None
    return total


def _parse_day(s: str, now: datetime) -> tuple[Optional[date], bool]:
    """Дата из выражения: (date, это_день_недели). (None, False) — день не указан."""
    if "послезавтра" in s:
        return (now + timedelta(days=2)).date(), False
    if "завтра" in s:
        return (now + timedelta(days=1)).date(), False
    if "сегодня" in s:
        return now.date(), False

    m = re.search(r"(\d{1,2})\s+(" + "|".join(MONTHS) + r")", s)
    if m:
        day_num, month = int(m.group(1)), MONTHS[m.group(2)]
        try:
            d = date(now.year, month, day_num)
        except ValueError:
            return None, False
        if d < now.date():
            d = date(now.year + 1, month, day_num)
        return d, False

    for name, idx in WEEKDAYS.items():
        if re.search(r"\b" + name + r"\b", s):
            if re.search(r"следующ", s):
                # «в следующий вторник» = вторник на следующей календарной неделе
                days_ahead = (7 - now.weekday()) + idx
            else:
                days_ahead = (idx - now.weekday()) % 7
            return (now + timedelta(days=days_ahead)).date(), True

    return None, False


def _parse_clock(s: str) -> Optional[tuple[int, int]]:
    """Время суток: «в 19:30», «в 16.30», «в 7», «в 8 вечера» -> (час, минута)."""
    m = re.search(
        r"\bв\s+(\d{1,2})(?:[:.\-]([0-5]\d))?(?:\s*(утра|дня|вечера|ночи))?\b", s
    )
    if m is None:
        m = re.search(r"\b(\d{1,2})[:.]([0-5]\d)\b(?:\s*(утра|дня|вечера|ночи))?", s)
        if m is None:
            return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    hint = m.group(3)
    if hint in ("вечера", "дня") and hour < 12:
        hour += 12
    elif hint == "ночи" and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return hour, minute
