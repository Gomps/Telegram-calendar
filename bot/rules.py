"""Вычисление срабатываний периодических серий.

Серия хранится как символическое правило (JSON), а не как список моментов:

{
  "type": "interval" | "daily" | "weekly",
  "days_of_week": [0..6] | null,          # 0 = понедельник; null = каждый день
  "time": "HH:MM",                        # для daily / weekly
  "interval_minutes": 60,                 # для interval
  "start_anchor": {"kind": "context", "key": "work_end", "offset_minutes": 0}
                | {"kind": "time", "time": "18:00", "offset_minutes": 0},
  "end_anchor":   {...}                   # для interval
}

Якоря вида {"kind": "context", "key": "work_end"} разрешаются в конкретное
время в момент вычисления — из актуального контекста пользователя. Поэтому
при изменении распорядка (например, новый work_end) вся серия автоматически
пересчитывается без изменения записи в БД.
"""

import re
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

# Люди (и LLM вслед за ними) пишут время как «16.30», «16-30», «7», «22» —
# принимаем всё это и приводим к каноническому «HH:MM».
HOUR_ONLY_RE = re.compile(r"^([01]?\d|2[0-3])$")
TIME_LENIENT_RE = re.compile(r"^([01]?\d|2[0-3])(?:\s*[:.\-чh]\s*|\s+)?([0-5]\d)$")

# Ключи контекста, которые считаем «временем» и подставляем в якоря
CONTEXT_TIME_HINT = ("work_start", "work_end", "sleep_start", "sleep_end")


def normalize_hhmm(value) -> Optional[str]:
    """«16.30» / «16-30» / «7» / «16:30» -> «16:30» / «07:00». None — не время."""
    s = str(value).strip().lower().rstrip(".")
    m = HOUR_ONLY_RE.match(s)
    if m:
        return f"{int(m.group(1)):02d}:00"
    m = TIME_LENIENT_RE.match(s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return None


def parse_hhmm(value) -> Optional[time]:
    s = normalize_hhmm(value)
    if s is None:
        return None
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def when_matches(when: dict, day: date) -> bool:
    """Проверка условий «when»: чётность числа месяца и дни недели (логическое И)."""
    if not isinstance(when, dict):
        return False
    parity = when.get("day_parity")
    if parity == "even" and day.day % 2 != 0:
        return False
    if parity == "odd" and day.day % 2 != 1:
        return False
    days = when.get("days_of_week")
    if days and day.weekday() not in days:
        return False
    return True


def resolve_context_value(ctx: dict, key: str, day: date):
    """Значение ключа контекста для конкретного дня.

    Значение может быть простым («16:30») или списком вариантов с условиями:
    [{"value": "16:30", "when": {"day_parity": "even"}}, {"value": "18:00"}].
    Берётся первый вариант, чьё «when» подходит дню (без when = по умолчанию).
    """
    raw = ctx.get(key)
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and when_matches(entry.get("when") or {}, day):
                return entry.get("value")
        return None
    return raw


def resolve_anchor(anchor: dict, ctx: dict, day: date, tz: ZoneInfo) -> Optional[datetime]:
    """Якорь -> конкретный datetime в этот день (aware, локальный TZ). None — нет данных."""
    if not isinstance(anchor, dict):
        return None
    kind = anchor.get("kind")
    if kind == "context":
        raw = resolve_context_value(ctx, anchor.get("key", ""), day)
        if raw is None:
            return None
        t = parse_hhmm(raw)
    elif kind == "time":
        t = parse_hhmm(anchor.get("time", ""))
    else:
        return None
    if t is None:
        return None
    dt = datetime.combine(day, t, tzinfo=tz)
    return dt + timedelta(minutes=int(anchor.get("offset_minutes", 0) or 0))


def occurrences_for_day(rule: dict, ctx: dict, day: date, tz: ZoneInfo) -> list[datetime]:
    days = rule.get("days_of_week")
    if days and day.weekday() not in days:
        return []
    parity = rule.get("day_parity")
    if parity == "even" and day.day % 2 != 0:
        return []
    if parity == "odd" and day.day % 2 != 1:
        return []

    rtype = rule.get("type")
    occs: list[datetime] = []
    if rtype in ("daily", "weekly"):
        t = parse_hhmm(rule.get("time", ""))
        if t:
            occs = [datetime.combine(day, t, tzinfo=tz)]
    elif rtype == "interval":
        start = resolve_anchor(rule.get("start_anchor", {}), ctx, day, tz)
        end = resolve_anchor(rule.get("end_anchor", {}), ctx, day, tz)
        step = int(rule.get("interval_minutes", 0) or 0)
        if start is not None and end is not None and step > 0 and end >= start:
            cur = start
            while cur <= end:
                occs.append(cur)
                cur += timedelta(minutes=step)

    # окна исключения («но не в обеденное время»): срабатывания внутри
    # любого окна отбрасываются
    exclude = rule.get("exclude") or []
    if occs and exclude:
        windows = []
        for win in exclude:
            if not isinstance(win, dict):
                continue
            ws = resolve_anchor(win.get("start_anchor", {}), ctx, day, tz)
            we = resolve_anchor(win.get("end_anchor", {}), ctx, day, tz)
            if ws is not None and we is not None:
                windows.append((ws, we))
        occs = [o for o in occs if not any(ws <= o <= we for ws, we in windows)]
    return occs


def next_occurrence(
    rule: dict, ctx: dict, after: datetime, tz: ZoneInfo, horizon_days: int = 14
) -> Optional[datetime]:
    """Первое срабатывание серии строго позже `after` (aware datetime)."""
    after_local = after.astimezone(tz)
    for offset in range(horizon_days + 1):
        day = after_local.date() + timedelta(days=offset)
        for occ in occurrences_for_day(rule, ctx, day, tz):
            if occ > after_local:
                return occ
    return None


def validate_rule(rule: dict) -> list[str]:
    """Список ошибок правила (пустой — правило корректно)."""
    errors: list[str] = []
    if not isinstance(rule, dict):
        return ["recurring должен быть объектом"]

    rtype = rule.get("type")
    if rtype not in ("interval", "daily", "weekly"):
        errors.append("recurring.type должен быть interval | daily | weekly")
        return errors

    days = rule.get("days_of_week")
    if days is not None:
        if not isinstance(days, list) or not all(isinstance(d, int) and 0 <= d <= 6 for d in days):
            errors.append("days_of_week — список чисел 0..6 (0 = понедельник) или null")

    if rule.get("day_parity") not in (None, "even", "odd"):
        errors.append("day_parity должен быть even | odd или null")

    if rtype in ("daily", "weekly"):
        if parse_hhmm(rule.get("time", "")) is None:
            errors.append("для daily/weekly нужно поле time в формате HH:MM")
        if rtype == "weekly" and not days:
            errors.append("для weekly нужен непустой days_of_week")

    if rtype == "interval":
        if int(rule.get("interval_minutes", 0) or 0) <= 0:
            errors.append("interval_minutes должен быть положительным числом")
        for name in ("start_anchor", "end_anchor"):
            errors.extend(_validate_anchor(rule.get(name), name))

    exclude = rule.get("exclude")
    if exclude is not None:
        if not isinstance(exclude, list):
            errors.append("exclude должен быть списком окон {start_anchor, end_anchor}")
        else:
            for i, win in enumerate(exclude):
                if not isinstance(win, dict):
                    errors.append(f"exclude[{i}] должен быть объектом")
                    continue
                errors.extend(_validate_anchor(win.get("start_anchor"), f"exclude[{i}].start_anchor"))
                errors.extend(_validate_anchor(win.get("end_anchor"), f"exclude[{i}].end_anchor"))
    return errors


def _validate_anchor(anchor, name: str) -> list[str]:
    if not isinstance(anchor, dict):
        return [f"{name} должен быть объектом"]
    kind = anchor.get("kind")
    if kind == "context":
        if not anchor.get("key"):
            return [f"{name}: для kind=context нужно поле key"]
    elif kind == "time":
        if parse_hhmm(anchor.get("time", "")) is None:
            return [f"{name}: для kind=time нужно поле time в формате HH:MM"]
    else:
        return [f"{name}.kind должен быть context | time"]
    return []


def context_time_usable(ctx: dict, key: str) -> bool:
    """Есть ли у ключа хотя бы одно пригодное значение-время (простое или в вариантах)."""
    raw = ctx.get(key)
    if isinstance(raw, list):
        return any(
            isinstance(e, dict) and parse_hhmm(str(e.get("value", ""))) is not None
            for e in raw
        )
    return parse_hhmm(str(raw or "")) is not None


def missing_context_keys(rule: dict, ctx: dict) -> list[str]:
    """Ключи контекста, на которые ссылается правило, но которых нет / они не HH:MM."""
    missing = []
    anchors = [rule.get("start_anchor"), rule.get("end_anchor")]
    for win in rule.get("exclude") or []:
        if isinstance(win, dict):
            anchors += [win.get("start_anchor"), win.get("end_anchor")]
    for anchor in anchors:
        if isinstance(anchor, dict) and anchor.get("kind") == "context":
            key = anchor.get("key", "")
            if not context_time_usable(ctx, key) and key not in missing:
                missing.append(key)
    return missing


DOW_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def describe_when(when: dict) -> str:
    """«when» человеческим языком: «по чётным числам», «по пт», их сочетание."""
    parts = []
    days = when.get("days_of_week") if isinstance(when, dict) else None
    if days:
        parts.append("по " + ", ".join(DOW_SHORT[d] for d in sorted(days) if 0 <= d <= 6))
    parity = when.get("day_parity") if isinstance(when, dict) else None
    if parity == "even":
        parts.append("по чётным числам")
    elif parity == "odd":
        parts.append("по нечётным числам")
    return ", ".join(parts) if parts else "по умолчанию"


def describe_anchor(anchor: dict, ctx: dict) -> str:
    kind = anchor.get("kind")
    off = int(anchor.get("offset_minutes", 0) or 0)
    if kind == "context":
        key = anchor.get("key", "?")
        raw = ctx.get(key)
        if isinstance(raw, list):
            base = f"{key} (по условиям)"
        else:
            base = f"{key} ({raw if raw is not None else '—'})"
    else:
        base = anchor.get("time", "?")
    if off:
        sign = "+" if off > 0 else "−"
        base += f" {sign} {abs(off) // 60}ч{abs(off) % 60:02d}м" if abs(off) % 60 else f" {sign} {abs(off) // 60}ч"
    return base


def describe_rule(rule: dict, ctx: dict) -> str:
    days = rule.get("days_of_week")
    days_s = "ежедневно" if not days else "по " + ", ".join(DOW_SHORT[d] for d in sorted(days))
    parity = rule.get("day_parity")
    if parity == "even":
        days_s += ", по чётным числам"
    elif parity == "odd":
        days_s += ", по нечётным числам"
    rtype = rule.get("type")
    if rtype in ("daily", "weekly"):
        return f"{days_s} в {rule.get('time')}"
    start = describe_anchor(rule.get("start_anchor", {}), ctx)
    end = describe_anchor(rule.get("end_anchor", {}), ctx)
    step = int(rule.get("interval_minutes", 0) or 0)
    step_s = f"каждые {step} мин" if step % 60 else (f"каждый час" if step == 60 else f"каждые {step // 60} ч")
    text = f"{days_s}, с {start} до {end}, {step_s}"
    for win in rule.get("exclude") or []:
        if isinstance(win, dict):
            ws = describe_anchor(win.get("start_anchor", {}), ctx)
            we = describe_anchor(win.get("end_anchor", {}), ctx)
            text += f", кроме {ws} – {we}"
    return text
