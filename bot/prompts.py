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
  "action": "save_context" | "create_reminder" | "create_recurring" | "ask_clarification" | "not_a_reminder" | "multi",
  "actions": [{...}, {...}],                      // только для multi: список действий из перечисленных выше (без multi и ask_clarification)
  "context_updates": {"ключ": "HH:MM", ...},     // необязательно; новые факты о распорядке — можно вместе с ЛЮБЫМ action
  "reminder_text": "текст напоминания",           // для create_reminder / create_recurring
  "time_expression": "через 5 минут",             // для create_reminder: выражение времени из сообщения ДОСЛОВНО
  "fire_at": "YYYY-MM-DDTHH:MM",                  // для create_reminder; локальное время пользователя, строго в будущем
  "recurring": {                                  // для create_recurring
    "type": "interval" | "daily" | "weekly",
    "days_of_week": [0..6] | null,                // 0 = понедельник; null = каждый день
    "day_parity": "even" | "odd" | null,          // только по чётным/нечётным числам месяца
    "time": "HH:MM",                              // для daily / weekly
    "interval_minutes": 60,                       // для interval
    "start_anchor": {"kind": "context", "key": "work_end", "offset_minutes": 0}
                  | {"kind": "time", "time": "18:00", "offset_minutes": 0},
    "end_anchor": { так же },                     // для interval
    "exclude": [{"start_anchor": {...}, "end_anchor": {...}}]  // окна-исключения («но не в обед»)
  },
  "clarification_question": "один короткий вопрос", // для ask_clarification
  "missing_fields": ["work_end"],                   // для ask_clarification
  "condition_question": "Ты не спишь?",             // для create_conditional: вопрос, который бот задаст в момент проверки
  "check_time_expression": "завтра в 11:30",        // для create_conditional: КОГДА задать вопрос (дословно из сообщения)
  "check_at": "YYYY-MM-DDTHH:MM",                   // для create_conditional: вычисленное время вопроса
  "timezone": "Europe/Minsk",                       // для set_timezone: IANA-имя, если знаешь его для города
  "city": "Минск",                                  // для set_timezone: город из сообщения
  "current_time": "16:45"                           // для set_timezone: текущее время пользователя из сообщения
}

Дополнительное действие: "set_timezone" — пользователь сообщил, откуда он («Я из Минска», «живу в Берлине») или сколько у него сейчас времени («у меня сейчас 16:45»).

Стандартные ключи контекста: work_start, work_end, sleep_start, sleep_end (формат HH:MM).
Можно добавлять свои ключи латиницей (например gym_time, lunch_time).
Значение ключа — «HH:MM» ЛИБО список вариантов с условиями (первый подходящий побеждает, вариант без when — по умолчанию):
[{"value": "16:30", "when": {"day_parity": "even"}}, {"value": "18:00", "when": {"day_parity": "odd"}}]
Условия when: day_parity: "even"|"odd" (чётное/нечётное число месяца), days_of_week: [0..6]. Несколько условий в одном when действуют вместе (логическое И)."""

RULES = """\
Правила:
1. Пользователь сообщает факты о распорядке («работаю с 9 до 18», «ложусь в 23») -> action=save_context, заполни context_updates.
2. Разовое напоминание -> action=create_reminder. Относительное время («после работы», «когда приду на работу», «за час до сна») вычисляй по контексту. fire_at всегда в будущем: если время сегодня уже прошло и день не указан — бери ближайший подходящий день.
3. Периодическое напоминание -> action=create_recurring. Если границы серии заданы распорядком («после работы», «до сна»), используй якоря kind=context с offset_minutes (например «за два часа до сна» = key=sleep_start, offset_minutes=-120) — НЕ подставляй конкретное время из контекста, серия должна пересчитываться при смене распорядка.
4. Если для вычисления времени не хватает данных (нет нужного ключа в контексте, не указано время/дата) -> action=ask_clarification: один короткий вопрос и missing_fields. НЕ выдумывай время.
5. Сообщение не про напоминания и не про распорядок -> action=not_a_reminder.
6. Если пользователь в одном сообщении и сообщает факт, и просит напоминание — верни create_reminder/create_recurring и заполни context_updates этим фактом.
7. reminder_text — короткая суть действия («Позвонить клиенту», «Выпить воду»), без слов «напомни».
8. Все значения времени пиши СТРОГО как HH:MM с двоеточием и ведущим нулём: «16:30», а не «16.30»; «07:00», а не «7». Пользовательское «16.30» означает 16:30, «с 7» — 07:00.
9. Разбирай сообщение на составляющие: служебные слова («напомни», «поставь», «пожалуйста») отбрасывай; выражение времени («через 5 минут», «завтра в 9», «после работы») копируй ДОСЛОВНО в time_expression; оставшуюся суть — в reminder_text. fire_at всё равно вычисляй — бот перепроверит время по time_expression.
10. НЕ задавай уточняющий вопрос о данных, которые уже есть в контексте (например, work_end известен) — просто используй их.
11. create_recurring — ТОЛЬКО при явных признаках повторения: «каждый/каждые», «ежедневно», «по понедельникам», «раз в …». Без таких слов («через 5 минут», «завтра в 9») — всегда РАЗОВОЕ create_reminder, даже если дел несколько.
12. Если просят и разовое, и периодическое («напомни через час, а потом каждый день») — верни action=multi со списком actions: сначала create_reminder, затем create_recurring. Время серии бери из разового (напоминание в 15:00 → серия daily в 15:00).
13. Условное напоминание «Если <условие в момент X> — напомни <дело> в Y» -> action=create_conditional: в check_* — время проверки условия, в condition_question — короткий вопрос пользователю по условию (от второго лица), reminder_text/time_expression/fire_at — само напоминание. Время напоминания должно быть ПОЗЖЕ времени проверки.
14. clarification_question всегда включает суть напоминания, а не общий вопрос: не «Во сколько напомнить?», а «Когда напомнить полить цветы? Укажи дату и время».
15. Пользователь пишет, откуда он или сколько у него времени -> action=set_timezone (заполни timezone, если знаешь IANA-имя для города, плюс city; или current_time).
16. Сначала раздели сообщение на смысловые части: факты о распорядке | разовое напоминание | периодическое | условие «если…» | город/текущее время. Каждую часть обработай своим action (несколько частей -> multi, факты — через context_updates).
17. reminder_text составляй ТОЛЬКО из слов пользователя (можно менять форму слова: «полей» -> «полить»). НЕ заменяй слова синонимами и не придумывай новые — иначе исказится смысл.
18. Дни недели вычисляй по календарю ниже. «в субботу» = ближайшая суббота; «в следующую субботу» = суббота следующей недели.
19. В условных напоминаниях время без указания дня относится к дню условия: «Если проснусь в субботу в 12:00 — напомни в 14:00 поесть» -> напоминание в СУББОТУ в 14:00.
20. «По чётным/нечётным дням», «только по пятницам» и их сочетания: в контексте — списком вариантов с when, в recurring — полями day_parity и days_of_week (действуют вместе).
21. offset_minutes задавай ТОЛЬКО если пользователь явно сказал «за N часов/минут до». «С начала рабочего дня до конца рабочего дня» = start work_start offset 0, end work_end offset 0 — БЕЗ смещений. Якоря выбирай точно по словам пользователя: «когда я на работе» = work_start..work_end (НЕ sleep_*).
22. «Но не в обеденное время / кроме обеда» -> recurring.exclude с окном lunch_start..lunch_end (если этих ключей нет в контексте — задай уточняющий вопрос).
23. Чтобы УДАЛИТЬ ключ из контекста («я больше не хожу в спортзал», «забудь про обед»), передай в context_updates значение null: {"gym_time": null}.
24. clarification_question строй ТОЛЬКО из слов текущего сообщения пользователя. НИКОГДА не упоминай примеры из этого промпта (машину, цветы, планёрку и т.п.), если их нет в сообщении.
25. НЕ вычисляй день недели сам — скопируй выражение времени дословно в time_expression («на ближайший вторник на 12:00»), бот посчитает дату кодом; fire_at заполни как подсказку."""

EXAMPLES = """\
Примеры (сегодня четверг 2026-07-02, контекст: {"work_start": "09:00", "work_end": "18:00", "sleep_start": "23:00", "lunch_end": "13:00"}):

Сообщение: «Я работаю с 9 до 18, сплю с 23 до 7»
Ответ: {"action": "save_context", "context_updates": {"work_start": "09:00", "work_end": "18:00", "sleep_start": "23:00", "sleep_end": "07:00"}}

Сообщение: «Я работаю с 7 до 16.30, сплю с 22 до 6»
Ответ: {"action": "save_context", "context_updates": {"work_start": "07:00", "work_end": "16:30", "sleep_start": "22:00", "sleep_end": "06:00"}}

Сообщение: «Когда приду на работу завтра — напомни позвонить клиенту»
Ответ: {"action": "create_reminder", "reminder_text": "Позвонить клиенту", "time_expression": "когда приду на работу завтра", "fire_at": "2026-07-03T09:00"}

Сообщение: «Сегодня в 19:30 напомни выключить духовку»
Ответ: {"action": "create_reminder", "reminder_text": "Выключить духовку", "time_expression": "сегодня в 19:30", "fire_at": "2026-07-02T19:30"}

Сообщение (сейчас 14:00): «Напомни через 5 минут создать напоминания для учёбы английского и русского на работе»
Ответ: {"action": "create_reminder", "reminder_text": "Создать напоминания для учёбы английского и русского на работе", "time_expression": "через 5 минут", "fire_at": "2026-07-02T14:05"}

Сообщение (сейчас 14:00): «Сделать зарядку напомни через час»
Ответ: {"action": "create_reminder", "reminder_text": "Сделать зарядку", "time_expression": "через час", "fire_at": "2026-07-02T15:00"}

Сообщение (сейчас 14:00): «Напомни через час полить цветы, а потом каждый день»
Ответ: {"action": "multi", "actions": [{"action": "create_reminder", "reminder_text": "Полить цветы", "time_expression": "через час", "fire_at": "2026-07-02T15:00"}, {"action": "create_recurring", "reminder_text": "Полить цветы", "recurring": {"type": "daily", "days_of_week": null, "time": "15:00"}}]}

Сообщение: «Напомни 15 августа поздравить маму»
Ответ: {"action": "ask_clarification", "clarification_question": "Во сколько 15 августа напомнить поздравить маму?", "missing_fields": ["time"]}

Сообщение: «Напомни полить цветы»
Ответ: {"action": "ask_clarification", "clarification_question": "Когда напомнить полить цветы? Укажи дату и время.", "missing_fields": ["time"]}

Сообщение: «Если в 11.30 завтра не буду спать — напомни после обеда выпить таблетку»
Ответ: {"action": "create_conditional", "condition_question": "Ты не спишь?", "check_time_expression": "завтра в 11:30", "check_at": "2026-07-03T11:30", "reminder_text": "Выпить таблетку", "time_expression": "после обеда", "fire_at": "2026-07-03T13:00"}

Сообщение: «Если проснусь в субботу в 12.00, то напомни в 14.00 поесть» (ближайшая суббота = 2026-07-04)
Ответ: {"action": "create_conditional", "condition_question": "Ты уже проснулся?", "check_time_expression": "в субботу в 12:00", "check_at": "2026-07-04T12:00", "reminder_text": "Поесть", "time_expression": "в 14:00", "fire_at": "2026-07-04T14:00"}

Сообщение: «Напоминай мне после работы каждый час пить воду, и так до двух часов до сна»
Ответ: {"action": "create_recurring", "reminder_text": "Выпить воду", "recurring": {"type": "interval", "days_of_week": null, "interval_minutes": 60, "start_anchor": {"kind": "context", "key": "work_end", "offset_minutes": 0}, "end_anchor": {"kind": "context", "key": "sleep_start", "offset_minutes": -120}}}

Сообщение: «Каждый понедельник в 10:00 напоминай про планёрку»
Ответ: {"action": "create_recurring", "reminder_text": "Планёрка", "recurring": {"type": "weekly", "days_of_week": [0], "time": "10:00"}}

Сообщение (контекст без work_end): «Напомни после работы забрать посылку»
Ответ: {"action": "ask_clarification", "clarification_question": "Во сколько ты заканчиваешь работу?", "missing_fields": ["work_end"]}

Сообщение: «С начала рабочего дня до конца рабочего дня напоминай мне раз в час учить русский»
Ответ: {"action": "create_recurring", "reminder_text": "Учить русский", "recurring": {"type": "interval", "days_of_week": null, "interval_minutes": 60, "start_anchor": {"kind": "context", "key": "work_start", "offset_minutes": 0}, "end_anchor": {"kind": "context", "key": "work_end", "offset_minutes": 0}}}

Сообщение (в контексте есть lunch_start и lunch_end): «Когда я на работе напоминай мне раз в час учить английский, но не в обеденное время»
Ответ: {"action": "create_recurring", "reminder_text": "Учить английский", "recurring": {"type": "interval", "days_of_week": null, "interval_minutes": 60, "start_anchor": {"kind": "context", "key": "work_start", "offset_minutes": 0}, "end_anchor": {"kind": "context", "key": "work_end", "offset_minutes": 0}, "exclude": [{"start_anchor": {"kind": "context", "key": "lunch_start", "offset_minutes": 0}, "end_anchor": {"kind": "context", "key": "lunch_end", "offset_minutes": 0}}]}}

Сообщение: «Я больше не хожу в спортзал, забудь об этом»
Ответ: {"action": "save_context", "context_updates": {"gym_time": null}}

Сообщение: «По чётным дням я работаю до 16:30, по нечётным — до 18»
Ответ: {"action": "save_context", "context_updates": {"work_end": [{"value": "16:30", "when": {"day_parity": "even"}}, {"value": "18:00", "when": {"day_parity": "odd"}}]}}

Сообщение: «Напоминай принимать витамины по чётным дням в 9:00»
Ответ: {"action": "create_recurring", "reminder_text": "Принять витамины", "recurring": {"type": "daily", "days_of_week": null, "day_parity": "even", "time": "09:00"}}

Сообщение: «Каждую пятницу по чётным числам в 18:30 напоминай сдать отчёт»
Ответ: {"action": "create_recurring", "reminder_text": "Сдать отчёт", "recurring": {"type": "weekly", "days_of_week": [4], "day_parity": "even", "time": "18:30"}}

Сообщение: «Я из Минска»
Ответ: {"action": "set_timezone", "timezone": "Europe/Minsk", "city": "Минск"}

Сообщение: «У меня сейчас 16:45»
Ответ: {"action": "set_timezone", "current_time": "16:45"}

Сообщение: «В следующую субботу в 12 напомни помыть машину» (сегодня чт 2026-07-02, суббота след. недели = 2026-07-11)
Ответ: {"action": "create_reminder", "reminder_text": "Помыть машину", "time_expression": "в следующую субботу в 12", "fire_at": "2026-07-11T12:00"}

Сообщение: «Как у тебя дела?»
Ответ: {"action": "not_a_reminder"}"""


SPLIT_PROMPT = """\
Ты — разметчик сообщений для бота-напоминалки. Разбей сообщение пользователя на смысловые блоки.

Отвечай ТОЛЬКО одним JSON-объектом: {"blocks": [{"type": "...", "text": "..."}]}

Типы блоков:
- task — что нужно сделать / о чём напомнить (без слов «напомни», «поставь напоминание»)
- time — когда (день, дата, время суток, «через час», «после работы»); если время разбито на куски — несколько блоков time
- repeat — признак повторения («каждый день», «раз в час», «по чётным дням»)
- condition — условие («если не буду спать»)
- fact — факт о распорядке («я работаю с 9 до 18»)
- location_time — откуда пользователь или сколько у него сейчас времени
- other — всё прочее

ГЛАВНОЕ ПРАВИЛО: поле text — ДОСЛОВНАЯ копия фрагмента сообщения. Не перефразируй, не исправляй ошибки, не добавляй и не выбрасывай слова внутри фрагмента. Служебные слова («напомни», «поставь напоминания») можно не относить ни к одному блоку.

Примеры:

Сообщение: «Поставь нам поминания на ближайший вторник. На 13.00 сделай Даши выписку и направление.»
Ответ: {"blocks": [{"type": "time", "text": "на ближайший вторник"}, {"type": "time", "text": "На 13.00"}, {"type": "task", "text": "сделай Даши выписку и направление"}]}

Сообщение: «Напомни через 5 минут выпить чай»
Ответ: {"blocks": [{"type": "time", "text": "через 5 минут"}, {"type": "task", "text": "выпить чай"}]}

Сообщение: «Каждый вторник в 10:00 напоминай про планёрку»
Ответ: {"blocks": [{"type": "repeat", "text": "Каждый вторник"}, {"type": "time", "text": "в 10:00"}, {"type": "task", "text": "планёрку"}]}

Сообщение: «Я работаю с 9 до 18, сплю с 23 до 7»
Ответ: {"blocks": [{"type": "fact", "text": "Я работаю с 9 до 18"}, {"type": "fact", "text": "сплю с 23 до 7"}]}

Сообщение: «Если проснусь в субботу в 12:00, то напомни в 14:00 поесть»
Ответ: {"blocks": [{"type": "condition", "text": "Если проснусь"}, {"type": "time", "text": "в субботу в 12:00"}, {"type": "time", "text": "в 14:00"}, {"type": "task", "text": "поесть"}]}

Сообщение: «Напоминай после работы каждый час пить воду»
Ответ: {"blocks": [{"type": "time", "text": "после работы"}, {"type": "repeat", "text": "каждый час"}, {"type": "task", "text": "пить воду"}]}

Сообщение: «Я из Минска»
Ответ: {"blocks": [{"type": "location_time", "text": "Я из Минска"}]}

Сообщение: «Как у тебя дела?»
Ответ: {"blocks": [{"type": "other", "text": "Как у тебя дела?"}]}"""


def build_split_prompt() -> str:
    return SPLIT_PROMPT


def build_system_prompt(
    now_local: datetime,
    tz_name: str,
    context: dict,
    pending: dict | None,
    blocks: list | None = None,
) -> str:
    ctx_s = json.dumps(context, ensure_ascii=False) if context else "пока пуст"
    tomorrow = now_local + timedelta(days=1)
    calendar = "; ".join(
        f"{DOW_RU[(now_local + timedelta(days=i)).weekday()]} — "
        f"{(now_local + timedelta(days=i)).strftime('%Y-%m-%d')}"
        for i in range(8)
    )
    parts = [
        "Ты — модуль разбора сообщений Telegram-бота напоминаний. "
        "Извлекаешь из сообщения пользователя суть напоминания и время.",
        f"Сейчас: {DOW_RU[now_local.weekday()]}, {now_local.strftime('%Y-%m-%d %H:%M')} "
        f"(часовой пояс {tz_name}). Завтра: {tomorrow.strftime('%Y-%m-%d')} "
        f"({DOW_RU[tomorrow.weekday()]}).",
        f"Календарь ближайших дней (сегодня — первый): {calendar}.",
        f"Сохранённый контекст пользователя (распорядок): {ctx_s}",
    ]
    if blocks:
        parts.append(
            "Сообщение уже размечено на составляющие (используй эту разметку как основу): "
            + json.dumps(blocks, ensure_ascii=False)
        )
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
