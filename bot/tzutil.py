"""Часовые пояса: IANA-имена, смещения вида «UTC+03:00» и города.

Пользователь может задать пояс тремя способами: /timezone Europe/Minsk,
«Я из Минска» (город -> IANA) или «У меня сейчас 16:45» (вычисляем смещение
от UTC и храним как «UTC+03:00»). get_tz понимает оба формата хранения.
"""

import re
from datetime import timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

UTC_OFFSET_RE = re.compile(r"^UTC([+-])(\d{2}):(\d{2})$")


def get_tz(name: str):
    """tzinfo по имени: «UTC+03:00» или IANA («Europe/Minsk»). Бросает при неизвестном."""
    m = UTC_OFFSET_RE.match(str(name).strip())
    if m:
        sign = 1 if m.group(1) == "+" else -1
        delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
        return timezone(sign * delta, name)
    return ZoneInfo(name)


def offset_tz_name(offset_minutes: int) -> str:
    sign = "+" if offset_minutes >= 0 else "-"
    off = abs(offset_minutes)
    return f"UTC{sign}{off // 60:02d}:{off % 60:02d}"


# Частые города; чего нет в списке — LLM обычно сама даёт IANA-имя в поле timezone
CITY_TZ = {
    "минск": "Europe/Minsk", "брест": "Europe/Minsk", "гомель": "Europe/Minsk",
    "гродно": "Europe/Minsk", "витебск": "Europe/Minsk", "могилев": "Europe/Minsk",
    "москва": "Europe/Moscow", "санкт-петербург": "Europe/Moscow", "питер": "Europe/Moscow",
    "спб": "Europe/Moscow", "казань": "Europe/Moscow", "сочи": "Europe/Moscow",
    "калининград": "Europe/Kaliningrad", "самара": "Europe/Samara",
    "екатеринбург": "Asia/Yekaterinburg", "омск": "Asia/Omsk",
    "новосибирск": "Asia/Novosibirsk", "красноярск": "Asia/Krasnoyarsk",
    "иркутск": "Asia/Irkutsk", "владивосток": "Asia/Vladivostok",
    "киев": "Europe/Kyiv", "харьков": "Europe/Kyiv", "одесса": "Europe/Kyiv",
    "варшава": "Europe/Warsaw", "вильнюс": "Europe/Vilnius", "рига": "Europe/Riga",
    "таллин": "Europe/Tallinn", "алматы": "Asia/Almaty", "астана": "Asia/Almaty",
    "ташкент": "Asia/Tashkent", "бишкек": "Asia/Bishkek", "душанбе": "Asia/Dushanbe",
    "тбилиси": "Asia/Tbilisi", "ереван": "Asia/Yerevan", "баку": "Asia/Baku",
    "берлин": "Europe/Berlin", "лондон": "Europe/London", "париж": "Europe/Paris",
    "стамбул": "Europe/Istanbul", "дубай": "Asia/Dubai", "нью-йорк": "America/New_York",
}


def city_to_tz(city: str) -> Optional[str]:
    return CITY_TZ.get(str(city).strip().lower().replace("ё", "е"))


def offset_from_current_time(user_hh: int, user_mm: int, now_utc) -> int:
    """Смещение (мин) от UTC по названному пользователем текущему времени.

    Округляется до 15 минут (реальные пояса кратны 15), приводится к
    диапазону UTC-12..UTC+14.
    """
    diff = (user_hh * 60 + user_mm) - (now_utc.hour * 60 + now_utc.minute)
    while diff > 14 * 60:
        diff -= 24 * 60
    while diff < -12 * 60:
        diff += 24 * 60
    return round(diff / 15) * 15
