"""Обработчики Telegram: команды, текстовые и голосовые сообщения.

Зависимости (db, llm, transcriber, cfg) внедряются aiogram'ом из
workflow_data диспетчера (см. main.py).
"""

import html as html_lib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from .config import Config
from .db import Database
from .llm import LLMBadResponse, LLMUnavailable, OllamaClient
from .logbuffer import MemoryLogHandler, build_pages
from .prompts import build_system_prompt
from .rules import describe_rule, missing_context_keys, next_occurrence, parse_hhmm
from .timeparse import parse_time_expression
from .tzutil import city_to_tz, get_tz, offset_from_current_time, offset_tz_name
from .transcribe import Transcriber, TranscriptionError

log = logging.getLogger(__name__)
router = Router()

HELP_TEXT = (
    "Я — бот умных напоминаний. Пиши (или наговаривай голосом) обычным языком:\n\n"
    "📋 Распорядок:\n"
    "• «Я работаю с 9 до 18, сплю с 23 до 7» — запомню и буду использовать\n"
    "• «Я из Минска» или «У меня сейчас 16:45» — сам настрою часовой пояс\n\n"
    "🔔 Разовые напоминания:\n"
    "• «Напомни через 5 минут выпить чай», «Сделать зарядку напомни через час»\n"
    "• «Когда приду на работу завтра — напомни позвонить клиенту»\n"
    "• «Сегодня в 19:30 — выключить духовку», «15 августа в 12:00 — поздравить маму»\n"
    "• «В следующую субботу в 12 — помыть машину»\n\n"
    "🔁 Периодические:\n"
    "• «Напоминай после работы каждый час пить воду, и так до двух часов до сна»\n"
    "• «Каждый понедельник в 10:00 — планёрка»\n"
    "• «Напомни через час полить цветы, а потом каждый день» — разовое + серия\n\n"
    "❓ Условные:\n"
    "• «Если в 11:30 завтра не буду спать — напомни после обеда выпить таблетку» — "
    "в 11:30 задам вопрос; подтвердишь — напомню\n\n"
    "Команды:\n"
    "/list — активные напоминания и серии\n"
    "/delete — удалить напоминание или серию\n"
    "/context — сохранённый распорядок\n"
    "/timezone Europe/Minsk — сменить часовой пояс\n"
    "/id — узнать свой Telegram ID\n\n"
    "Администратору: /adduser <id>, /removeuser <id>, /users — управление доступом, "
    "/log — логи бота"
)


def fmt_local(dt_utc: datetime, tz: ZoneInfo) -> str:
    local = dt_utc.astimezone(tz)
    dow = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"][local.weekday()]
    return f"{dow}, {local.strftime('%d.%m.%Y %H:%M')}"


class StatusMessage:
    """Статусное сообщение о ходе обработки.

    Бот сразу отвечает «⏳ …», по мере прохождения этапов редактирует это
    сообщение, а в конце заменяет его результирующим ответом (если
    отредактировать нельзя — удаляет и отправляет новое).
    """

    def __init__(self, origin: Message):
        self._origin = origin
        self._msg: Message | None = None

    async def set(self, text: str) -> None:
        try:
            if self._msg is None:
                self._msg = await self._origin.answer(text)
            else:
                await self._msg.edit_text(text)
        except Exception:
            log.debug("Не удалось обновить статусное сообщение", exc_info=True)

    async def finish(self, text: str, reply_markup=None) -> None:
        """Заменяет статусное сообщение результирующим ответом."""
        if self._msg is not None:
            try:
                await self._msg.edit_text(text, reply_markup=reply_markup)
                return
            except Exception:
                try:
                    await self._msg.delete()
                except Exception:
                    log.debug("Не удалось удалить статусное сообщение", exc_info=True)
                self._msg = None
        await self._origin.answer(text, reply_markup=reply_markup)


# --- администрирование: белый список по ID -----------------------------------


def _parse_target_id(message: Message) -> int | None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].strip())
    except ValueError:
        return None


@router.message(Command("adduser"))
async def cmd_adduser(message: Message, db: Database, cfg: Config) -> None:
    if message.from_user.id not in cfg.admin_ids:
        await message.answer("⛔ Команда доступна только администратору.")
        return
    target = _parse_target_id(message)
    if target is None:
        await message.answer("Использование: /adduser <telegram_id>\nНапример: /adduser 123456789")
        return
    added = await db.add_allowed_user(target, message.from_user.id)
    log.info("Админ %d добавил пользователя %d в белый список", message.from_user.id, target)
    await message.answer(
        f"✅ Пользователь {target} добавлен." if added else f"Пользователь {target} уже в списке."
    )


@router.message(Command("removeuser"))
async def cmd_removeuser(message: Message, db: Database, cfg: Config) -> None:
    if message.from_user.id not in cfg.admin_ids:
        await message.answer("⛔ Команда доступна только администратору.")
        return
    target = _parse_target_id(message)
    if target is None:
        await message.answer("Использование: /removeuser <telegram_id>")
        return
    removed = await db.remove_allowed_user(target)
    if removed:
        log.info("Админ %d удалил пользователя %d из белого списка", message.from_user.id, target)
        await message.answer(f"✅ Пользователь {target} удалён из списка.")
    elif target in cfg.allowed_ids:
        await message.answer(
            f"Пользователь {target} задан в ALLOWED_USER_IDS (.env) — убери его оттуда и перезапусти бота."
        )
    else:
        await message.answer(f"Пользователя {target} нет в списке.")


@router.message(Command("users"))
async def cmd_users(message: Message, db: Database, cfg: Config) -> None:
    if message.from_user.id not in cfg.admin_ids:
        await message.answer("⛔ Команда доступна только администратору.")
        return
    lines = ["👑 Администраторы (.env):"] + [f"  • {i}" for i in cfg.admin_ids]
    if cfg.allowed_ids:
        lines.append("📄 Белый список (.env):")
        lines += [f"  • {i}" for i in cfg.allowed_ids]
    dynamic = await db.list_allowed_users()
    if dynamic:
        lines.append("➕ Добавлены через /adduser:")
        lines += [f"  • {u['user_id']} (добавил {u['added_by']})" for u in dynamic]
    else:
        lines.append("➕ Через /adduser пока никто не добавлен.")
    await message.answer("\n".join(lines))


@router.message(Command("id"))
async def cmd_id(message: Message) -> None:
    await message.answer(f"Твой Telegram ID: `{message.from_user.id}`", parse_mode="Markdown")


# --- просмотр логов бота (только админ) ---------------------------------------


def _render_log_page(logbuffer: MemoryLogHandler, page: int):
    """(текст, клавиатура) для страницы логов; None — логов нет.

    Страницы хронологические: последняя — самые свежие записи. Стрелка «⬅️»
    (старее) и «➡️» (новее) показываются только когда есть куда листать.
    """
    pages = build_pages(logbuffer.records)
    if not pages:
        return None
    page = max(0, min(page, len(pages) - 1))
    text = (
        f"📋 Логи бота — стр. {page + 1}/{len(pages)} (свежие в конце)\n"
        f"<pre>{html_lib.escape(pages[page])}</pre>"
    )
    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton(text="⬅️ старее", callback_data=f"log:{page - 1}"))
    if page < len(pages) - 1:
        buttons.append(InlineKeyboardButton(text="новее ➡️", callback_data=f"log:{page + 1}"))
    markup = InlineKeyboardMarkup(inline_keyboard=[buttons]) if buttons else None
    return text, markup


@router.message(Command("log"))
async def cmd_log(message: Message, cfg: Config, logbuffer: MemoryLogHandler) -> None:
    if message.from_user.id not in cfg.admin_ids:
        await message.answer("⛔ Команда доступна только администратору.")
        return
    view = _render_log_page(logbuffer, page=10**9)  # последняя страница — свежие логи
    if view is None:
        await message.answer("Логов пока нет.")
        return
    text, markup = view
    await message.answer(text, reply_markup=markup, parse_mode="HTML")


@router.callback_query(F.data.startswith("log:"))
async def cb_log(callback: CallbackQuery, cfg: Config, logbuffer: MemoryLogHandler) -> None:
    if callback.from_user.id not in cfg.admin_ids:
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    page = int(callback.data.split(":")[1])
    view = _render_log_page(logbuffer, page)
    if view is None:
        await callback.answer("Логов пока нет")
        return
    text, markup = view
    try:
        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
    except Exception:
        # например, «message is not modified» — просто гасим «часики»
        log.debug("Не удалось обновить страницу логов", exc_info=True)
    await callback.answer()


# --- команды ---------------------------------------------------------------


@router.message(CommandStart())
async def cmd_start(message: Message, db: Database, cfg: Config) -> None:
    await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    await message.answer("Привет! 👋\n\n" + HELP_TEXT)


@router.message(Command("context"))
async def cmd_context(message: Message, db: Database, cfg: Config) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    ctx = user["context"]
    if not ctx:
        await message.answer(
            "Я пока ничего не знаю о твоём распорядке. Расскажи, например:\n"
            "«Я работаю с 9 до 18, сплю с 23 до 7»"
        )
        return
    lines = [f"• {k}: {v}" for k, v in sorted(ctx.items())]
    await message.answer(
        "📋 Твой распорядок:\n" + "\n".join(lines) + f"\n\nЧасовой пояс: {user['timezone']}"
    )


@router.message(Command("timezone"))
async def cmd_timezone(message: Message, db: Database, cfg: Config) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(
            f"Текущий часовой пояс: {user['timezone']}\n"
            "Сменить: /timezone Europe/Minsk (имя из базы IANA)"
        )
        return
    tz_name = parts[1].strip()
    try:
        get_tz(tz_name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        await message.answer(
            f"Не знаю пояс «{tz_name}». Пример: Europe/Minsk, Europe/Moscow или UTC+03:00.\n"
            "Можно проще: напиши «Я из <город>» или «У меня сейчас 16:45» — настрою сам."
        )
        return
    await db.set_timezone(message.from_user.id, tz_name)
    await message.answer(f"✅ Часовой пояс: {tz_name}")


def render_list(
    user: dict, reminders: list[dict], series: list[dict], tz: ZoneInfo,
    conditionals: list[dict] = (),
) -> str:
    """Текст /list. Ошибка описания одной серии не прячет остальные."""
    if not reminders and not series and not conditionals:
        return "Активных напоминаний нет."
    lines = []
    if reminders:
        lines.append("🔔 Разовые:")
        for r in reminders:
            fire_at = datetime.fromisoformat(r["fire_at"])
            lines.append(f"  R{r['id']} · {fmt_local(fire_at, tz)} — {r['text']}")
    if series:
        lines.append("🔁 Периодические:")
        now = datetime.now(timezone.utc)
        for s in series:
            try:
                desc = describe_rule(s["rule"], user["context"])
                nxt = next_occurrence(s["rule"], user["context"], now, tz)
                nxt_s = f", ближайшее: {fmt_local(nxt.astimezone(timezone.utc), tz)}" if nxt else ""
                lines.append(f"  S{s['id']} · {s['text']} ({desc}{nxt_s})")
            except Exception:
                log.exception("Не удалось описать серию #%s (правило: %s)", s.get("id"), s.get("rule"))
                lines.append(f"  S{s['id']} · {s['text']} (⚠️ правило не читается — удали через /delete)")
    if conditionals:
        lines.append("❓ Условные:")
        for c in conditionals:
            check_at = datetime.fromisoformat(c["check_at"])
            fire_at = datetime.fromisoformat(c["fire_at"])
            state = "вопрос задан" if c["status"] == "asked" else f"вопрос {fmt_local(check_at, tz)}"
            lines.append(
                f"  C{c['id']} · «{c['question']}» ({state}) → "
                f"«{c['reminder_text']}» {fmt_local(fire_at, tz)}"
            )
    return "\n".join(lines)


@router.message(Command("list"))
async def cmd_list(message: Message, db: Database, cfg: Config) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    tz = get_tz(user["timezone"])
    reminders = await db.list_pending_reminders(message.from_user.id)
    series = await db.list_user_series(message.from_user.id)
    conditionals = await db.list_user_conditionals(message.from_user.id)
    log.info(
        "/list пользователя %d: %d разовых, %d серий, %d условных",
        message.from_user.id, len(reminders), len(series), len(conditionals),
    )
    await message.answer(render_list(user, reminders, series, tz, conditionals))


@router.message(Command("delete"))
async def cmd_delete(message: Message, db: Database, cfg: Config) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    tz = get_tz(user["timezone"])
    reminders = await db.list_pending_reminders(message.from_user.id)
    series = await db.list_user_series(message.from_user.id)
    conditionals = await db.list_user_conditionals(message.from_user.id)
    if not reminders and not series and not conditionals:
        await message.answer("Удалять нечего — активных напоминаний нет.")
        return
    buttons = []
    for r in reminders:
        fire_at = datetime.fromisoformat(r["fire_at"])
        label = f"🔔 {fmt_local(fire_at, tz)} — {r['text']}"
        buttons.append([InlineKeyboardButton(text=label[:60], callback_data=f"del:r:{r['id']}")])
    for s in series:
        buttons.append([InlineKeyboardButton(text=f"🔁 {s['text']}"[:60], callback_data=f"del:s:{s['id']}")])
    for c in conditionals:
        label = f"❓ {c['question']} → {c['reminder_text']}"
        buttons.append([InlineKeyboardButton(text=label[:60], callback_data=f"del:c:{c['id']}")])
    await message.answer(
        "Что удалить?", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
    )


@router.callback_query(F.data.startswith("del:"))
async def cb_delete(callback: CallbackQuery, db: Database) -> None:
    _, kind, raw_id = callback.data.split(":")
    item_id = int(raw_id)
    user_id = callback.from_user.id
    tz_name, ctx = await db.get_user_tz_and_context(user_id)
    tz = get_tz(tz_name)

    # читаем запись до удаления, чтобы показать, ЧТО именно удалено
    if kind == "r":
        item = await db.get_reminder(user_id, item_id)
        ok = await db.cancel_reminder(user_id, item_id)
        if ok and item:
            fire_at = datetime.fromisoformat(item["fire_at"])
            result = f"🗑 Удалено напоминание: «{item['text']}»\n🕐 было на {fmt_local(fire_at, tz)}"
            log.info("Пользователь %d удалил напоминание #%d («%s»)", user_id, item_id, item["text"])
        else:
            result = "Уже неактуально."
    elif kind == "c":
        item = await db.get_conditional(user_id, item_id)
        ok = await db.cancel_conditional(user_id, item_id)
        if ok and item:
            result = (
                f"🗑 Удалено условное напоминание: «{item['question']}» → «{item['reminder_text']}»"
            )
            log.info("Пользователь %d удалил условное #%d", user_id, item_id)
        else:
            result = "Уже неактуально."
    else:
        item = await db.get_series(user_id, item_id)
        ok = await db.deactivate_series(user_id, item_id)
        if ok and item:
            try:
                desc = describe_rule(item["rule"], ctx)
            except Exception:
                desc = "правило не читается"
            result = f"🗑 Удалена серия: «{item['text']}»\n📅 была: {desc}"
            log.info("Пользователь %d удалил серию #%d («%s»)", user_id, item_id, item["text"])
        else:
            result = "Уже неактуально."

    await callback.answer("Удалено ✅" if ok else "Уже неактуально")
    if callback.message:
        try:
            await callback.message.edit_text(result)
        except Exception:
            log.debug("Не удалось отредактировать сообщение удаления", exc_info=True)


# --- текст и голос ----------------------------------------------------------


@router.message(F.voice)
async def on_voice(
    message: Message,
    bot: Bot,
    db: Database,
    llm: OllamaClient,
    transcriber: Transcriber,
    cfg: Config,
) -> None:
    status = StatusMessage(message)
    await status.set("⏳ Этап 1/3: распознаю голосовое сообщение…")
    tmp_path = None
    try:
        file = await bot.get_file(message.voice.file_id)
        fd, tmp_path = tempfile.mkstemp(suffix=".ogg")
        os.close(fd)
        await bot.download_file(file.file_path, destination=tmp_path)
        text = await transcriber.transcribe(tmp_path)
    except TranscriptionError as e:
        log.warning("Транскрибация не удалась: %s", e)
        await status.finish(f"😔 Не удалось распознать голосовое сообщение: {e}")
        return
    except Exception:
        log.exception("Ошибка обработки голосового сообщения")
        await status.finish("😔 Не удалось обработать голосовое сообщение.")
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                # Windows: файл может быть ещё занят (антивирус/индексатор)
                log.debug("Не удалось удалить временный файл %s", tmp_path, exc_info=True)
    await process_text(message, text, db, llm, cfg, bot, status, prefix=f"🎙 «{text}»\n\n")


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(
    message: Message, bot: Bot, db: Database, llm: OllamaClient, cfg: Config
) -> None:
    status = StatusMessage(message)
    await process_text(message, message.text, db, llm, cfg, bot, status)


@router.message()
async def on_other(message: Message) -> None:
    """Фолбэк: бот всегда отвечает, даже на неподдерживаемый тип сообщения."""
    await message.answer(
        "Я понимаю текстовые и голосовые сообщения 🙂 Напиши, о чём напомнить, — см. /start"
    )


# --- основной конвейер -------------------------------------------------------

# Явные признаки периодичности в тексте — для сверки классификации LLM
PERIODIC_RE = re.compile(
    r"\b(кажд\w+|ежедневн\w*|еженедельн\w*|ежемесячн\w*|ежечасн\w*|раз\s+в\b|"
    r"по\s+будням|по\s+выходным|"
    r"по\s+(понедельник|вторник|сред|четверг|пятниц|суббот|воскресень)\w*)"
)


async def _corrective_retry(
    llm: OllamaClient, system_prompt: str, hint: str, text: str
) -> dict | None:
    """Один повторный запрос к LLM с подсказкой; None — не получилось."""
    try:
        return await llm.parse_message(system_prompt + hint, text)
    except (LLMUnavailable, LLMBadResponse):
        return None


# Служебные слова, не влияющие на смысл напоминания
_FILLER_WORDS = {
    "напомни", "напоминай", "напомнить", "напоминание", "напоминания",
    "пожалуйста", "поставь", "создай", "сделай", "чтобы", "нужно", "надо",
    "мне", "меня", "потом", "если", "когда", "завтра", "сегодня", "через",
    "каждый", "каждую", "каждое", "каждые", "минут", "минуты", "часов", "часа",
}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[а-яa-z0-9]+", str(text).lower().replace("ё", "е"))
    return {w[:4] for w in words if len(w) >= 4 and w not in _FILLER_WORDS}


def text_matches_source(reminder_text: str, source: str) -> bool:
    """Проверка, что LLM не подменила слова напоминания.

    Сравниваются 4-буквенные основы значимых слов (морфологию русского
    «полей/полить» это переживает). Если меньше половины основ текста
    напоминания встречается в исходном сообщении — модель что-то придумала,
    и надо переспросить пользователя.
    """
    r, s = _stems(reminder_text), _stems(source)
    if not r:
        return True
    return len(r & s) / len(r) >= 0.5


async def _do_set_timezone(action: dict, user_id: int, db: Database) -> str:
    """Настройка пояса по городу/IANA-имени от LLM или по текущему времени."""
    tz_name = str(action.get("timezone") or "").strip()
    city = str(action.get("city") or "").strip()
    current = str(action.get("current_time") or "").strip()

    resolved = None
    if tz_name:
        try:
            get_tz(tz_name)
            resolved = tz_name
        except Exception:
            resolved = None
    if resolved is None and city:
        resolved = city_to_tz(city)
    if resolved is None and current:
        t = parse_hhmm(current)
        if t is not None:
            offset = offset_from_current_time(t.hour, t.minute, datetime.now(timezone.utc))
            resolved = offset_tz_name(offset)
    if resolved is None:
        return (
            "Не смог определить часовой пояс 🤔 Напиши, сколько у тебя сейчас "
            "времени (например «у меня сейчас 16:45») — вычислю пояс по нему."
        )

    await db.set_timezone(user_id, resolved)
    now_local = datetime.now(get_tz(resolved))
    log.info("Пользователь %d: часовой пояс установлен %s", user_id, resolved)
    return (
        f"🌍 Часовой пояс: {resolved}.\n"
        f"У тебя сейчас {now_local.strftime('%H:%M')}, верно? Если нет — напиши, "
        "сколько у тебя времени, и я пересчитаю."
    )


async def process_text(
    message: Message,
    text: str,
    db: Database,
    llm: OllamaClient,
    cfg: Config,
    bot: Bot,
    status: StatusMessage,
    prefix: str = "",
) -> None:
    """Конвейер обработки. Любой исход завершается заменой статусного
    сообщения на результирующий ответ — бот никогда не молчит."""
    try:
        await _process_text(message, text, db, llm, cfg, bot, status, prefix, depth=0)
    except Exception:
        log.exception("Необработанная ошибка конвейера (пользователь %d)", message.from_user.id)
        await status.finish(
            prefix + "⚠️ Внутренняя ошибка при обработке сообщения. "
            "Попробуй ещё раз; подробности — в логах бота."
        )


async def _process_text(
    message: Message,
    text: str,
    db: Database,
    llm: OllamaClient,
    cfg: Config,
    bot: Bot,
    status: StatusMessage,
    prefix: str,
    depth: int,
) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    user_id = user["user_id"]
    tz = get_tz(user["timezone"])
    now_local = datetime.now(timezone.utc).astimezone(tz)
    pending = await db.get_pending_clarification(user_id)

    await status.set(prefix + "⏳ Этап 2/3: разбираю запрос (LLM)…")
    system_prompt = build_system_prompt(now_local, user["timezone"], user["context"], pending)
    try:
        action = await llm.parse_message(system_prompt, text)
    except LLMUnavailable as e:
        log.error("Ollama недоступна: %s", e)
        await status.finish(
            prefix + "⚠️ Языковая модель сейчас недоступна (Ollama не отвечает). "
            "Проверь, что Ollama запущена и модель скачана, и повтори сообщение."
        )
        return
    except LLMBadResponse as e:
        log.error("LLM не вернула валидный JSON: %s", e)
        await status.finish(
            prefix + "😕 Не смог разобрать запрос. Попробуй сформулировать иначе — "
            "например, укажи время с двоеточием: «в 16:30»."
        )
        return

    log.info("Пользователь %d: action=%s", user_id, action.get("action"))
    kind = action["action"]

    # Сверка классификации: признаки периодичности в тексте против типа
    # действия — модель путает «через 5 минут» с серией и наоборот
    if depth == 0 and kind in ("create_reminder", "create_recurring"):
        periodic = PERIODIC_RE.search(text.lower().replace("ё", "е")) is not None
        hint = None
        if kind == "create_recurring" and not periodic:
            hint = (
                "\n\nВАЖНО: в сообщении НЕТ признаков периодичности («каждый», «ежедневно», "
                "«по понедельникам», «раз в…»). Это РАЗОВОЕ напоминание — верни create_reminder."
            )
        elif kind == "create_reminder" and periodic:
            hint = (
                "\n\nВАЖНО: в сообщении ЕСТЬ признак периодичности. Вероятно, нужен "
                "create_recurring или multi (разовое + серия), см. правило 12."
            )
        if hint:
            log.info("Классификация «%s» не согласуется с текстом — корректирующий повтор", kind)
            await status.set(prefix + "⏳ Этап 2/3: перепроверяю разбор…")
            corrected = await _corrective_retry(llm, system_prompt, hint, text)
            if corrected is not None:
                action, kind = corrected, corrected["action"]
                log.info("Пользователь %d: action после сверки=%s", user_id, kind)

    await status.set(prefix + "⏳ Этап 3/3: сохраняю…")

    # Новые факты о распорядке применяем при любом действии
    ctx = user["context"]
    updates = action.get("context_updates") or {}
    if updates:
        ctx = await db.update_context(user_id, updates)
        log.info("Пользователь %d: контекст обновлён %s", user_id, updates)

    # Модель переспрашивает то, что уже есть в контексте, — один
    # корректирующий повтор с явной подсказкой
    if kind == "ask_clarification" and depth == 0:
        missing = [str(k) for k in (action.get("missing_fields") or [])]
        if missing and all(ctx.get(k) for k in missing):
            known = {k: ctx[k] for k in missing}
            log.info("LLM переспрашивает известные ключи %s — корректирующий повтор", missing)
            hint = (
                "\n\nВАЖНО: эти данные УЖЕ есть в контексте: "
                + json.dumps(known, ensure_ascii=False)
                + ". Не задавай уточняющий вопрос — выполни запрос, используя эти значения."
            )
            corrected = await _corrective_retry(llm, system_prompt, hint, text)
            if corrected is not None:
                action, kind = corrected, corrected["action"]
                new_updates = corrected.get("context_updates") or {}
                if new_updates:
                    ctx = await db.update_context(user_id, new_updates)
                    updates = {**updates, **new_updates}

    if kind == "save_context":
        await db.clear_pending_clarification(user_id)
        if pending and depth == 0:
            # Ответ на уточнение сохранён — возвращаемся к исходному запросу,
            # иначе он терялся бы («запомнил», а напоминание не создал)
            saved_note = "; ".join(f"{k}: {v}" for k, v in updates.items())
            new_prefix = prefix + (f"💾 Запомнил: {saved_note}\n\n" if saved_note else "")
            await status.set(new_prefix + "⏳ Возвращаюсь к исходному запросу…")
            log.info("Пользователь %d: возврат к исходному запросу «%s»",
                     user_id, pending["original_request"])
            await _process_text(
                message, pending["original_request"], db, llm, cfg, bot,
                status, new_prefix, depth=1,
            )
            return
        lines = [f"• {k}: {v}" for k, v in updates.items()]
        await status.finish(prefix + "✅ Запомнил:\n" + "\n".join(lines))
        return

    if kind == "set_timezone":
        msg = await _do_set_timezone(action, user_id, db)
        await status.finish(prefix + msg)
        return

    # Модель заменила слова напоминания (не просто опечатки) — переспрашиваем
    if kind in ("create_reminder", "create_recurring", "create_conditional"):
        reminder_text = str(action.get("reminder_text") or "").strip()
        if reminder_text and not text_matches_source(reminder_text, text):
            await db.set_pending_action(user_id, json.dumps(action, ensure_ascii=False), text)
            kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="✅ Да, верно", callback_data="confirm:yes"),
                InlineKeyboardButton(text="✏️ Нет, не то", callback_data="confirm:no"),
            ]])
            log.info(
                "Пользователь %d: текст напоминания «%s» не совпадает с сообщением — прошу подтвердить",
                user_id, reminder_text,
            )
            await status.finish(
                prefix + "🤔 Хочу убедиться, что понял правильно.\n"
                f"Ты написал: «{text}»\n"
                f"Я понял напоминание как: «{reminder_text}»\n\nВсё верно?",
                reply_markup=kb,
            )
            return

    if kind == "create_reminder":
        msg, _stop, _fire = await _do_create_reminder(action, user, ctx, tz, db, text)
        await status.finish(prefix + msg)
        return

    if kind == "create_recurring":
        msg, _stop = await _do_create_recurring(action, user, ctx, tz, db, text)
        await status.finish(prefix + msg)
        return

    if kind == "create_conditional":
        msg, _stop = await _do_create_conditional(action, user, ctx, tz, db, text)
        await status.finish(prefix + msg)
        return

    if kind == "multi":
        # Комбинированный запрос («напомни через час, а потом каждый день»):
        # выполняем действия по очереди, ответы склеиваем в одно сообщение
        parts: list[str] = []
        last_fire_utc = None
        for sub in action.get("actions", []):
            sub_updates = sub.get("context_updates") or {}
            if sub_updates:
                ctx = await db.update_context(user_id, sub_updates)
                log.info("Пользователь %d: контекст обновлён %s", user_id, sub_updates)
            skind = sub.get("action")
            if skind == "save_context" and sub_updates:
                parts.append(
                    "✅ Запомнил: " + "; ".join(f"{k}: {v}" for k, v in sub_updates.items())
                )
            elif skind == "create_reminder":
                msg, stop, fire_utc = await _do_create_reminder(sub, user, ctx, tz, db, text)
                parts.append(msg)
                if fire_utc is not None:
                    last_fire_utc = fire_utc
                if stop:
                    break
            elif skind == "create_recurring":
                # серия не должна дублировать только что созданное разовое —
                # её отсчёт начинается после него
                msg, stop = await _do_create_recurring(
                    sub, user, ctx, tz, db, text, skip_until=last_fire_utc
                )
                parts.append(msg)
                if stop:
                    break
            elif skind == "create_conditional":
                msg, stop = await _do_create_conditional(sub, user, ctx, tz, db, text)
                parts.append(msg)
                if stop:
                    break
        await status.finish(
            prefix
            + ("\n\n".join(parts) if parts else "Не понял запрос — попробуй сформулировать иначе.")
        )
        return

    if kind == "ask_clarification":
        question = action["clarification_question"].strip()
        # исходный запрос сохраняем сквозь цепочку уточнений
        original = pending["original_request"] if pending else text
        if pending and pending["original_request"] != text:
            original = f"{pending['original_request']} (уточнение: {text})"
        await db.set_pending_clarification(user_id, original, question)
        await status.finish(prefix + "❓ " + question)
        return

    # not_a_reminder
    if pending:
        # пользователь сменил тему — не держим устаревший вопрос
        await db.clear_pending_clarification(user_id)
    await status.finish(
        prefix + "Это, кажется, не про напоминания 🙂 Я умею запоминать распорядок и "
        "ставить напоминания — см. /start"
    )


def _resolve_time(expr: str, iso_value, now_local: datetime, tz: ZoneInfo) -> datetime | None:
    """Время из составляющих: сначала детерминированный парсер по дословному
    выражению («через 5 минут», «завтра в 9») — арифметике кода доверяем
    больше, чем маленькой модели; затем значение, вычисленное LLM
    (контекстные «после работы»)."""
    dt = None
    if expr:
        dt = parse_time_expression(expr, now_local)
        if dt is not None:
            log.info("Время из «%s» вычислено детерминированно: %s", expr, dt)
    if dt is None and iso_value:
        dt = datetime.fromisoformat(str(iso_value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
    return dt


async def _do_create_reminder(
    action: dict, user: dict, ctx: dict, tz: ZoneInfo, db: Database, raw_text: str
) -> tuple[str, bool, datetime | None]:
    """Создаёт разовое напоминание.

    Возвращает (текст ответа, прервать_ли_дальнейшую_обработку, время UTC).
    """
    now_local = datetime.now(timezone.utc).astimezone(tz)
    expr = str(action.get("time_expression") or "").strip()
    fire_local = _resolve_time(expr, action.get("fire_at"), now_local, tz)
    if fire_local is None:
        what = str(action.get("reminder_text") or "").strip() or "об этом"
        question = (
            f"Когда напомнить «{what}»? Укажи дату и время — "
            "например «завтра в 9», «в 16:30» или «через 20 минут»."
        )
        await db.set_pending_clarification(user["user_id"], raw_text, question)
        return "❓ " + question, True, None

    fire_utc = fire_local.astimezone(timezone.utc)
    if fire_utc <= datetime.now(timezone.utc):
        return (
            f"Похоже, это время уже прошло ({fire_local.strftime('%d.%m.%Y %H:%M')}). "
            "Уточни, когда напомнить?",
            True,
            None,
        )
    reminder_id = await db.add_reminder(
        user["user_id"], user["chat_id"], action["reminder_text"].strip(), fire_utc
    )
    await db.clear_pending_clarification(user["user_id"])
    log.info("Создано напоминание #%d на %s", reminder_id, fire_utc)
    return (
        f"✅ Напомню: «{action['reminder_text'].strip()}»\n🕐 {fmt_local(fire_utc, tz)}",
        False,
        fire_utc,
    )


async def _do_create_conditional(
    action: dict, user: dict, ctx: dict, tz: ZoneInfo, db: Database, raw_text: str
) -> tuple[str, bool]:
    """Условное напоминание: в check_at бот задаст condition_question с
    кнопками Да/Нет; «Да» до fire_at создаёт напоминание, иначе условие
    истекает. Возвращает (текст ответа, прервать_ли_обработку)."""
    now_local = datetime.now(timezone.utc).astimezone(tz)
    question = str(action.get("condition_question") or "").strip()
    reminder_text = str(action.get("reminder_text") or "").strip()

    check_local = _resolve_time(
        str(action.get("check_time_expression") or "").strip(),
        action.get("check_at"), now_local, tz,
    )
    if check_local is None:
        q = f"Когда проверить условие «{question}»? Укажи дату и время."
        await db.set_pending_clarification(user["user_id"], raw_text, q)
        return "❓ " + q, True

    fire_local = _resolve_time(
        str(action.get("time_expression") or "").strip(),
        action.get("fire_at"), now_local, tz,
    )
    if fire_local is None:
        q = (
            f"Когда напомнить «{reminder_text}», если условие подтвердится? "
            "Укажи дату и время."
        )
        await db.set_pending_clarification(user["user_id"], raw_text, q)
        return "❓ " + q, True

    if check_local <= now_local:
        return (
            f"Время проверки условия уже прошло ({check_local.strftime('%d.%m.%Y %H:%M')}). "
            "Уточни, когда задать вопрос?",
            True,
        )
    if fire_local <= check_local:
        return (
            "Время напоминания должно быть позже времени проверки условия "
            f"(вопрос в {check_local.strftime('%H:%M')}, напоминание в "
            f"{fire_local.strftime('%H:%M')}). Уточни времена?",
            True,
        )

    check_utc = check_local.astimezone(timezone.utc)
    fire_utc = fire_local.astimezone(timezone.utc)
    cond_id = await db.add_conditional(
        user["user_id"], user["chat_id"], question, reminder_text, check_utc, fire_utc
    )
    await db.clear_pending_clarification(user["user_id"])
    log.info("Создано условное #%d: вопрос %s, напоминание %s", cond_id, check_utc, fire_utc)
    return (
        f"⏳ Принято. {fmt_local(check_utc, tz)} спрошу: «{question}»\n"
        f"Если подтвердишь до {fmt_local(fire_utc, tz)} — напомню: «{reminder_text}». "
        "Без ответа напоминание не создаётся.",
        False,
    )


@router.callback_query(F.data.startswith("confirm:"))
async def cb_confirm(callback: CallbackQuery, db: Database, cfg: Config) -> None:
    """Подтверждение действия, в котором LLM могла исказить текст напоминания."""
    user_id = callback.from_user.id
    pa = await db.get_pending_action(user_id)
    if pa is None:
        await callback.answer("Уже неактуально")
        return
    await db.clear_pending_action(user_id)

    if callback.data.split(":")[1] == "no":
        await callback.answer("Ок")
        if callback.message:
            try:
                await callback.message.edit_text(
                    "Ок, отменил. Сформулируй, пожалуйста, ещё раз — что и когда напомнить?"
                )
            except Exception:
                log.debug("Не удалось отредактировать подтверждение", exc_info=True)
        return

    action = json.loads(pa["action_json"])
    raw_text = pa["original_text"]
    chat_id = callback.message.chat.id if callback.message else user_id
    user = await db.get_or_create_user(user_id, chat_id, cfg.default_tz)
    tz = get_tz(user["timezone"])
    ctx = user["context"]

    kind = action.get("action")
    if kind == "create_reminder":
        result, _stop, _fire = await _do_create_reminder(action, user, ctx, tz, db, raw_text)
    elif kind == "create_recurring":
        result, _stop = await _do_create_recurring(action, user, ctx, tz, db, raw_text)
    elif kind == "create_conditional":
        result, _stop = await _do_create_conditional(action, user, ctx, tz, db, raw_text)
    else:
        result = "Не смог выполнить подтверждённое действие — попробуй сформулировать заново."
    await callback.answer("Подтверждено ✅")
    if callback.message:
        try:
            await callback.message.edit_text(result)
        except Exception:
            log.debug("Не удалось отредактировать подтверждение", exc_info=True)


@router.callback_query(F.data.startswith("cond:"))
async def cb_conditional(callback: CallbackQuery, db: Database) -> None:
    _, answer, raw_id = callback.data.split(":")
    user_id = callback.from_user.id
    cond = await db.get_conditional(user_id, int(raw_id))
    if cond is None or cond["status"] != "asked":
        await callback.answer("Уже неактуально")
        return

    tz_name, _ = await db.get_user_tz_and_context(user_id)
    tz = get_tz(tz_name)
    fire_at = datetime.fromisoformat(cond["fire_at"])
    now = datetime.now(timezone.utc)

    if answer == "no":
        await db.set_conditional_status(cond["id"], "declined")
        log.info("Условное #%d: пользователь ответил «нет»", cond["id"])
        result = f"Ок, напоминание «{cond['reminder_text']}» не создаю."
        await callback.answer("Принято")
    elif now >= fire_at:
        await db.set_conditional_status(cond["id"], "expired")
        log.info("Условное #%d: подтверждение пришло после срока", cond["id"])
        result = (
            f"Увы, время напоминания уже прошло ({fmt_local(fire_at, tz)}) — "
            f"«{cond['reminder_text']}» не создано."
        )
        await callback.answer("Слишком поздно")
    else:
        await db.add_reminder(user_id, cond["chat_id"], cond["reminder_text"], fire_at)
        await db.set_conditional_status(cond["id"], "confirmed")
        log.info("Условное #%d подтверждено — напоминание на %s", cond["id"], fire_at)
        result = f"✅ Напомню: «{cond['reminder_text']}»\n🕐 {fmt_local(fire_at, tz)}"
        await callback.answer("Напоминание создано ✅")

    if callback.message:
        try:
            await callback.message.edit_text(result)
        except Exception:
            log.debug("Не удалось отредактировать сообщение условия", exc_info=True)


async def _do_create_recurring(
    action: dict, user: dict, ctx: dict, tz: ZoneInfo, db: Database, raw_text: str,
    skip_until: datetime | None = None,
) -> tuple[str, bool]:
    """Создаёт периодическую серию.

    skip_until — не срабатывать до этого момента (UTC): в multi серия не
    должна дублировать только что созданное разовое напоминание.
    Возвращает (текст ответа, прервать_ли_дальнейшую_обработку).
    """
    rule = action["recurring"]
    # Правило ссылается на распорядок, которого нет, — доспрашиваем, а не создаём пустышку
    missing = missing_context_keys(rule, ctx)
    if missing:
        questions = {
            "work_start": "Во сколько ты начинаешь работу?",
            "work_end": "Во сколько ты заканчиваешь работу?",
            "sleep_start": "Во сколько ты ложишься спать?",
            "sleep_end": "Во сколько ты просыпаешься?",
        }
        question = questions.get(missing[0], f"Уточни время «{missing[0]}» (HH:MM)?")
        await db.set_pending_clarification(user["user_id"], raw_text, question)
        return "❓ " + question, True

    series_id = await db.add_series(
        user["user_id"], user["chat_id"], action["reminder_text"].strip(), rule
    )
    if skip_until is not None:
        await db.set_series_cursor(series_id, skip_until)
    await db.clear_pending_clarification(user["user_id"])
    log.info("Создана серия #%d: %s", series_id, rule)

    after = max(datetime.now(timezone.utc), skip_until or datetime.min.replace(tzinfo=timezone.utc))
    nxt = next_occurrence(rule, ctx, after, tz)
    nxt_s = f"\n🕐 Ближайшее: {fmt_local(nxt.astimezone(timezone.utc), tz)}" if nxt else ""
    return (
        f"✅ Серия создана: «{action['reminder_text'].strip()}»\n"
        f"📅 {describe_rule(rule, ctx)}{nxt_s}\n"
        "При изменении распорядка расписание пересчитается автоматически.",
        False,
    )
