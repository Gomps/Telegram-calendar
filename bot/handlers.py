"""Обработчики Telegram: команды, текстовые и голосовые сообщения.

Зависимости (db, llm, transcriber, cfg) внедряются aiogram'ом из
workflow_data диспетчера (см. main.py).
"""

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
from .prompts import build_system_prompt
from .rules import describe_rule, missing_context_keys, next_occurrence
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
    "/timezone Europe/Minsk — сменить часовой пояс"
)


def fmt_local(dt_utc: datetime, tz: ZoneInfo) -> str:
    local = dt_utc.astimezone(tz)
    dow = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"][local.weekday()]
    return f"{dow}, {local.strftime('%d.%m.%Y %H:%M')}"


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
    await bot.send_chat_action(message.chat.id, "typing")
    tmp_path = None
    try:
        file = await bot.get_file(message.voice.file_id)
        fd, tmp_path = tempfile.mkstemp(suffix=".ogg")
        os.close(fd)
        await bot.download_file(file.file_path, destination=tmp_path)
        text = await transcriber.transcribe(tmp_path)
    except TranscriptionError as e:
        log.warning("Транскрибация не удалась: %s", e)
        await message.answer("😔 Не удалось распознать голосовое сообщение. Попробуй ещё раз или напиши текстом.")
        return
    except Exception:
        log.exception("Ошибка обработки голосового сообщения")
        await message.answer("😔 Не удалось обработать голосовое сообщение.")
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    await message.answer(f"🎙 Распознал: «{text}»")
    await process_text(message, text, db, llm, cfg, bot)


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(
    message: Message, bot: Bot, db: Database, llm: OllamaClient, cfg: Config
) -> None:
    await process_text(message, message.text, db, llm, cfg, bot)


# --- основной конвейер -------------------------------------------------------


async def process_text(
    message: Message, text: str, db: Database, llm: OllamaClient, cfg: Config, bot: Bot
) -> None:
    user = await db.get_or_create_user(message.from_user.id, message.chat.id, cfg.default_tz)
    user_id = user["user_id"]
    tz = ZoneInfo(user["timezone"])
    now_local = datetime.now(timezone.utc).astimezone(tz)
    pending = await db.get_pending_clarification(user_id)

    await bot.send_chat_action(message.chat.id, "typing")
    system_prompt = build_system_prompt(now_local, user["timezone"], user["context"], pending)
    try:
        action = await llm.parse_message(system_prompt, text)
    except LLMUnavailable as e:
        log.error("Ollama недоступна: %s", e)
        await message.answer(
            "⚠️ Языковая модель сейчас недоступна (Ollama не отвечает). "
            "Проверь, что Ollama запущена, и повтори сообщение."
        )
        return
    except LLMBadResponse as e:
        log.error("LLM не вернула валидный JSON: %s", e)
        await message.answer("😕 Не смог разобрать запрос. Попробуй сформулировать иначе.")
        return

    log.info("Пользователь %d: action=%s", user_id, action.get("action"))

    # Новые факты о распорядке применяем при любом действии
    ctx = user["context"]
    updates = action.get("context_updates") or {}
    if updates:
        ctx = await db.update_context(user_id, updates)
        log.info("Пользователь %d: контекст обновлён %s", user_id, updates)

    kind = action["action"]
    if kind == "save_context":
        await db.clear_pending_clarification(user_id)
        lines = [f"• {k}: {v}" for k, v in updates.items()]
        await message.answer("✅ Запомнил:\n" + "\n".join(lines))
        return

    if kind == "create_reminder":
        await handle_create_reminder(message, action, user, ctx, tz, db)
        return

    if kind == "create_recurring":
        await handle_create_recurring(message, action, user, ctx, tz, db, text)
        return

    if kind == "ask_clarification":
        question = action["clarification_question"].strip()
        # исходный запрос сохраняем сквозь цепочку уточнений
        original = pending["original_request"] if pending else text
        if pending and pending["original_request"] != text:
            original = f"{pending['original_request']} (уточнение: {text})"
        await db.set_pending_clarification(user_id, original, question)
        await message.answer("❓ " + question)
        return

    # not_a_reminder
    if pending:
        # пользователь сменил тему — не держим устаревший вопрос
        await db.clear_pending_clarification(user_id)
    await message.answer(
        "Это, кажется, не про напоминания 🙂 Я умею запоминать распорядок и ставить "
        "напоминания — см. /start"
    )


async def handle_create_reminder(
    message: Message, action: dict, user: dict, ctx: dict, tz: ZoneInfo, db: Database
) -> None:
    fire_local = datetime.fromisoformat(str(action["fire_at"]))
    if fire_local.tzinfo is None:
        fire_local = fire_local.replace(tzinfo=tz)
    fire_utc = fire_local.astimezone(timezone.utc)
    if fire_utc <= datetime.now(timezone.utc):
        await message.answer(
            f"Похоже, это время уже прошло ({fire_local.strftime('%d.%m.%Y %H:%M')}). "
            "Уточни, когда напомнить?"
        )
        return
    reminder_id = await db.add_reminder(
        user["user_id"], user["chat_id"], action["reminder_text"].strip(), fire_utc
    )
    await db.clear_pending_clarification(user["user_id"])
    log.info("Создано напоминание #%d на %s", reminder_id, fire_utc)
    await message.answer(
        f"✅ Напомню: «{action['reminder_text'].strip()}»\n🕐 {fmt_local(fire_utc, tz)}"
    )


async def handle_create_recurring(
    message: Message, action: dict, user: dict, ctx: dict, tz: ZoneInfo, db: Database, raw_text: str
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
        await message.answer("❓ " + question)
        return

    series_id = await db.add_series(
        user["user_id"], user["chat_id"], action["reminder_text"].strip(), rule
    )
    await db.clear_pending_clarification(user["user_id"])
    log.info("Создана серия #%d: %s", series_id, rule)

    nxt = next_occurrence(rule, ctx, datetime.now(timezone.utc), tz)
    nxt_s = f"\n🕐 Ближайшее: {fmt_local(nxt.astimezone(timezone.utc), tz)}" if nxt else ""
    await message.answer(
        f"✅ Серия создана: «{action['reminder_text'].strip()}»\n"
        f"📅 {describe_rule(rule, ctx)}{nxt_s}\n"
        "При изменении распорядка расписание пересчитается автоматически."
    )
