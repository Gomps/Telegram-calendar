"""Построение промпта для LLM (qwen3.5:4b через Ollama).

Модель получает: текущие дату/время и часовой пояс, сохранённый контекст
пользователя, при необходимости — незавершённый уточняющий диалог, и обязана
вернуть строго один JSON-объект по схеме ниже.
"""

import json
from datetime import datetime, timedelta

DOW_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

SCHEMA = """\
Отвечай ТОЛЬКО одним JSON-объектом без пояснений и без markdown. Схема:
{
  "action": "save_context" | "create_reminder" | "create_recurring" | "ask_clarification" | "not_a_reminder",
  "context_updates": {"ключ": "HH:MM", ...},     // необязательно; новые факты о распорядке — можно вместе с ЛЮБЫМ action
  "reminder_text": "текст напоминания",           // для create_reminder / create_recurring
  "fire_at": "YYYY-MM-DDTHH:MM",                  // для create_reminder; локальное время пользователя, строго в будущем
  "recurring": {                                  // для create_recurring
    "type": "interval" | "daily" | "weekly",
    "days_of_week": [0..6] | null,                // 0 = понедельник; null = каждый день
    "time": "HH:MM",                              // для daily / weekly
    "interval_minutes": 60,                       // для interval
    "start_anchor": {"kind": "context", "key": "work_end", "offset_minutes": 0}
                  | {"kind": "time", "time": "18:00", "offset_minutes": 0},
    "end_anchor": { так же }                      // для interval
  },
  "clarification_question": "один короткий вопрос", // для ask_clarification
  "missing_fields": ["work_end"]                    // для ask_clarification
}

Стандартные ключи контекста: work_start, work_end, sleep_start, sleep_end (формат HH:MM).
Можно добавлять свои ключи латиницей (например gym_time, lunch_time)."""

RULES = """\
Правила:
1. Пользователь сообщает факты о распорядке («работаю с 9 до 18», «ложусь в 23») -> action=save_context, заполни context_updates.
2. Разовое напоминание -> action=create_reminder. Относительное время («после работы», «когда приду на работу», «за час до сна») вычисляй по контексту. fire_at всегда в будущем: если время сегодня уже прошло и день не указан — бери ближайший подходящий день.
3. Периодическое напоминание -> action=create_recurring. Если границы серии заданы распорядком («после работы», «до сна»), используй якоря kind=context с offset_minutes (например «за два часа до сна» = key=sleep_start, offset_minutes=-120) — НЕ подставляй конкретное время из контекста, серия должна пересчитываться при смене распорядка.
4. Если для вычисления времени не хватает данных (нет нужного ключа в контексте, не указано время/дата) -> action=ask_clarification: один короткий вопрос и missing_fields. НЕ выдумывай время.
5. Сообщение не про напоминания и не про распорядок -> action=not_a_reminder.
6. Если пользователь в одном сообщении и сообщает факт, и просит напоминание — верни create_reminder/create_recurring и заполни context_updates этим фактом.
7. reminder_text — короткая суть действия («Позвонить клиенту», «Выпить воду»), без слов «напомни»."""

EXAMPLES = """\
Примеры (сегодня четверг 2026-07-02, контекст: {"work_start": "09:00", "work_end": "18:00", "sleep_start": "23:00"}):

Сообщение: «Я работаю с 9 до 18, сплю с 23 до 7»
Ответ: {"action": "save_context", "context_updates": {"work_start": "09:00", "work_end": "18:00", "sleep_start": "23:00", "sleep_end": "07:00"}}

Сообщение: «Когда приду на работу завтра — напомни позвонить клиенту»
Ответ: {"action": "create_reminder", "reminder_text": "Позвонить клиенту", "fire_at": "2026-07-03T09:00"}

Сообщение: «Сегодня в 19:30 напомни выключить духовку»
Ответ: {"action": "create_reminder", "reminder_text": "Выключить духовку", "fire_at": "2026-07-02T19:30"}

Сообщение: «Напомни 15 августа поздравить маму»
Ответ: {"action": "ask_clarification", "clarification_question": "Во сколько 15 августа напомнить?", "missing_fields": ["time"]}

Сообщение: «Напоминай мне после работы каждый час пить воду, и так до двух часов до сна»
Ответ: {"action": "create_recurring", "reminder_text": "Выпить воду", "recurring": {"type": "interval", "days_of_week": null, "interval_minutes": 60, "start_anchor": {"kind": "context", "key": "work_end", "offset_minutes": 0}, "end_anchor": {"kind": "context", "key": "sleep_start", "offset_minutes": -120}}}

Сообщение: «Каждый понедельник в 10:00 напоминай про планёрку»
Ответ: {"action": "create_recurring", "reminder_text": "Планёрка", "recurring": {"type": "weekly", "days_of_week": [0], "time": "10:00"}}

Сообщение (контекст без work_end): «Напомни после работы забрать посылку»
Ответ: {"action": "ask_clarification", "clarification_question": "Во сколько ты заканчиваешь работу?", "missing_fields": ["work_end"]}

Сообщение: «Как у тебя дела?»
Ответ: {"action": "not_a_reminder"}"""


def build_system_prompt(
    now_local: datetime,
    tz_name: str,
    context: dict,
    pending: dict | None,
) -> str:
    ctx_s = json.dumps(context, ensure_ascii=False) if context else "пока пуст"
    tomorrow = now_local + timedelta(days=1)
    parts = [
        "Ты — модуль разбора сообщений Telegram-бота напоминаний. "
        "Извлекаешь из сообщения пользователя суть напоминания и время.",
        f"Сейчас: {DOW_RU[now_local.weekday()]}, {now_local.strftime('%Y-%m-%d %H:%M')} "
        f"(часовой пояс {tz_name}). Завтра: {tomorrow.strftime('%Y-%m-%d')} "
        f"({DOW_RU[tomorrow.weekday()]}).",
        f"Сохранённый контекст пользователя (распорядок): {ctx_s}",
    ]
    if pending:
        parts.append(
            "Ранее бот задал пользователю уточняющий вопрос: "
            f"«{pending['question']}» по исходному запросу: «{pending['original_request']}». "
            "Текущее сообщение — скорее всего ответ на этот вопрос: сохрани ответ в "
            "context_updates и, если данных теперь достаточно, выполни исходный запрос "
            "(create_reminder / create_recurring). Если данных всё ещё не хватает — задай "
            "следующий вопрос через ask_clarification."
        )
    parts += [SCHEMA, RULES, EXAMPLES]
    return "\n\n".join(parts)
