"""Обработчики Telegram: команды, текстовые и голосовые сообщения.

Зависимости (db, llm, transcriber, cfg) внедряются aiogram'ом из
workflow_data диспетчера (см. main.py).
"""

import html as html_lib
import json
import logging
import os
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
from .rules import describe_rule, missing_context_keys, next_occurrence
from .timeparse import parse_time_expression
from .transcribe import Transcriber, TranscriptionError

log = logging.getLogger(__name__)
router = Router()

HELP_TEXT = (
    "Я — бот умных напоминаний. Пиши (или наговаривай голосом) обычным языком:\n\n"
    "• «Я работаю с 9 до 18, сплю с 23 до 7» — запомню твой распорядок\n"
    "• «Когда приду на работу завтра — напомни позвонить клиенту»\n"
    "• «Напоминай после работы каждый час пить воду, и так до двух часов до сна»\n"
    "• «Каждый понедельник в 10:00 — планёрка», «15 августа в 12:00 — поздравить маму»\n\n"
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

    async def finish(self, text: str) -> None:
        """Заменяет статусное сообщение результирующим ответом."""
        if self._msg is not None:
            try:
                await self._msg.edit_text(text)
                return
            except Exception:
                try:
                    await self._msg.delete()
                except Exception:
                    log.debug("Не удалось удалить статусное сообщение", exc_info=True)
                self._msg = None
        await self._origin.answer(text)


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
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        await message.answer(f"Не знаю пояс «{tz_name}». Пример: Europe/Minsk, Europe/Moscow, Asia/Almaty.")
        return
    await db.set_timezone(message.from_user.id, tz_name)
    await message.answer(f"✅ Часовой пояс: {tz_name}")


@router.message(Command("list"))
async def cmd_list(message: Message, db: Database, cfg: Config) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    tz = ZoneInfo(user["timezone"])
    reminders = await db.list_pending_reminders(message.from_user.id)
    series = await db.list_user_series(message.from_user.id)
    if not reminders and not series:
        await message.answer("Активных напоминаний нет.")
        return
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
            desc = describe_rule(s["rule"], user["context"])
            nxt = next_occurrence(s["rule"], user["context"], now, tz)
            nxt_s = f", ближайшее: {fmt_local(nxt.astimezone(timezone.utc), tz)}" if nxt else ""
            lines.append(f"  S{s['id']} · {s['text']} ({desc}{nxt_s})")
    await message.answer("\n".join(lines))


@router.message(Command("delete"))
async def cmd_delete(message: Message, db: Database, cfg: Config) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    tz = ZoneInfo(user["timezone"])
    reminders = await db.list_pending_reminders(message.from_user.id)
    series = await db.list_user_series(message.from_user.id)
    if not reminders and not series:
        await message.answer("Удалять нечего — активных напоминаний нет.")
        return
    buttons = []
    for r in reminders:
        fire_at = datetime.fromisoformat(r["fire_at"])
        label = f"🔔 {fmt_local(fire_at, tz)} — {r['text']}"
        buttons.append([InlineKeyboardButton(text=label[:60], callback_data=f"del:r:{r['id']}")])
    for s in series:
        buttons.append([InlineKeyboardButton(text=f"🔁 {s['text']}"[:60], callback_data=f"del:s:{s['id']}")])
    await message.answer(
        "Что удалить?", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
    )


@router.callback_query(F.data.startswith("del:"))
async def cb_delete(callback: CallbackQuery, db: Database) -> None:
    _, kind, raw_id = callback.data.split(":")
    item_id = int(raw_id)
    if kind == "r":
        ok = await db.cancel_reminder(callback.from_user.id, item_id)
    else:
        ok = await db.deactivate_series(callback.from_user.id, item_id)
    await callback.answer("Удалено ✅" if ok else "Уже неактуально")
    if ok and callback.message:
        await callback.message.edit_text("Удалено ✅")


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
    tz = ZoneInfo(user["timezone"])
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
    await status.set(prefix + "⏳ Этап 3/3: сохраняю…")

    # Новые факты о распорядке применяем при любом действии
    ctx = user["context"]
    updates = action.get("context_updates") or {}
    if updates:
        ctx = await db.update_context(user_id, updates)
        log.info("Пользователь %d: контекст обновлён %s", user_id, updates)

    kind = action["action"]

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
            try:
                action = await llm.parse_message(system_prompt + hint, text)
                kind = action["action"]
                new_updates = action.get("context_updates") or {}
                if new_updates:
                    ctx = await db.update_context(user_id, new_updates)
                    updates = {**updates, **new_updates}
            except (LLMUnavailable, LLMBadResponse):
                pass  # остаёмся с исходным уточняющим вопросом

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

    if kind == "create_reminder":
        await handle_create_reminder(action, user, ctx, tz, db, status, prefix, text)
        return

    if kind == "create_recurring":
        await handle_create_recurring(action, user, ctx, tz, db, text, status, prefix)
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


async def handle_create_reminder(
    action: dict, user: dict, ctx: dict, tz: ZoneInfo, db: Database,
    status: StatusMessage, prefix: str, raw_text: str,
) -> None:
    now_local = datetime.now(timezone.utc).astimezone(tz)

    # Этап «время»: сначала детерминированный парсер по дословному выражению
    # («через 5 минут», «завтра в 9») — арифметику времени коду доверяем
    # больше, чем маленькой модели; затем fire_at, вычисленный LLM
    # (контекстные «после работы»)
    fire_local = None
    expr = str(action.get("time_expression") or "").strip()
    if expr:
        fire_local = parse_time_expression(expr, now_local)
        if fire_local is not None:
            log.info("Время из «%s» вычислено детерминированно: %s", expr, fire_local)
    if fire_local is None and action.get("fire_at"):
        fire_local = datetime.fromisoformat(str(action["fire_at"]))
        if fire_local.tzinfo is None:
            fire_local = fire_local.replace(tzinfo=tz)
    if fire_local is None:
        question = (
            f"Не понял, когда напомнить («{expr}»). "
            "Укажи время, например «в 16:30» или «через 20 минут»."
        )
        await db.set_pending_clarification(user["user_id"], raw_text, question)
        await status.finish(prefix + "❓ " + question)
        return

    fire_utc = fire_local.astimezone(timezone.utc)
    if fire_utc <= datetime.now(timezone.utc):
        await status.finish(
            prefix + f"Похоже, это время уже прошло ({fire_local.strftime('%d.%m.%Y %H:%M')}). "
            "Уточни, когда напомнить?"
        )
        return
    reminder_id = await db.add_reminder(
        user["user_id"], user["chat_id"], action["reminder_text"].strip(), fire_utc
    )
    await db.clear_pending_clarification(user["user_id"])
    log.info("Создано напоминание #%d на %s", reminder_id, fire_utc)
    await status.finish(
        prefix + f"✅ Напомню: «{action['reminder_text'].strip()}»\n🕐 {fmt_local(fire_utc, tz)}"
    )


async def handle_create_recurring(
    action: dict, user: dict, ctx: dict, tz: ZoneInfo, db: Database, raw_text: str,
    status: StatusMessage, prefix: str,
) -> None:
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
        await status.finish(prefix + "❓ " + question)
        return

    series_id = await db.add_series(
        user["user_id"], user["chat_id"], action["reminder_text"].strip(), rule
    )
    await db.clear_pending_clarification(user["user_id"])
    log.info("Создана серия #%d: %s", series_id, rule)

    nxt = next_occurrence(rule, ctx, datetime.now(timezone.utc), tz)
    nxt_s = f"\n🕐 Ближайшее: {fmt_local(nxt.astimezone(timezone.utc), tz)}" if nxt else ""
    await status.finish(
        prefix + f"✅ Серия создана: «{action['reminder_text'].strip()}»\n"
        f"📅 {describe_rule(rule, ctx)}{nxt_s}\n"
        "При изменении распорядка расписание пересчитается автоматически."
    )
