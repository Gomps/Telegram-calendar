"""Обработка текста без привязки к Telegram — используется мини-приложением.

Логика намеренно зеркалит `handlers._process_text` (это единственный способ
провести сообщение через LLM + детерминированный санитайзер + запись в БД
без объектов aiogram Message/CallbackQuery). Дублирование небольшое и
осознанное: сама бизнес-логика (парсинг времени, валидация правил, запись
в БД) не копируется — переиспользуются `handlers._do_create_*`,
`postprocess.sanitize_action` и остальные уже протестированные функции;
здесь только маршрутизация результата в JSON-совместимый вид вместо
Telegram-сообщений с кнопками.
"""

import logging
from datetime import datetime, timezone

from . import handlers
from .config import Config
from .db import Database
from .llm import LLMBadResponse, LLMClient, LLMUnavailable
from .postprocess import action_from_blocks, is_periodic_text, sanitize_action, text_matches_source
from .prompts import build_split_prompt, build_system_prompt
from .tzutil import get_tz

log = logging.getLogger(__name__)


async def process_message(
    user_id: int, chat_id: int, text: str, db: Database, llm: LLMClient, cfg: Config,
    depth: int = 0,
) -> dict:
    """Возвращает структурированный результат обработки одного сообщения.

    {kind, message, created: {type, id} | None, text_mismatch: {...} | None}
    """
    # update_chat=False: вызов приходит из мини-аппа, где настоящий chat_id
    # неизвестен — сохранённый Telegram'ом чат затирать нельзя
    user = await db.get_or_create_user(user_id, chat_id, cfg.default_tz, update_chat=False)
    tz = get_tz(user["timezone"])
    now_local = datetime.now(timezone.utc).astimezone(tz)
    pending = await db.get_pending_clarification(user_id)

    blocks = None
    action = None
    if cfg.split_stage and pending is None:
        try:
            blocks = await llm.split_message(build_split_prompt(), text)
        except LLMUnavailable as e:
            return _llm_error(e, cfg)
        except LLMBadResponse as e:
            log.warning("Разметка на блоки не удалась (%s) — одновызовный путь", e)
    if blocks:
        assembled = action_from_blocks(blocks, text, now_local)
        if assembled is not None:
            action = assembled

    system_prompt = build_system_prompt(
        now_local, user["timezone"], user["context"], pending, blocks=blocks
    )
    if action is None:
        try:
            action = await llm.parse_message(system_prompt, text)
        except LLMUnavailable as e:
            return _llm_error(e, cfg)
        except LLMBadResponse as e:
            return {
                "kind": "error", "created": None, "text_mismatch": None,
                "message": "Не смог разобрать запрос. Попробуй сформулировать иначе — "
                           "например, укажи время с двоеточием: «в 16:30».",
            }

    action = sanitize_action(action, text, now_local)
    kind = action["action"]

    if depth == 0 and kind == "create_reminder" and is_periodic_text(text):
        corrected = await handlers._corrective_retry(
            llm, system_prompt,
            "\n\nВАЖНО: в сообщении ЕСТЬ признак периодичности. Вероятно, нужен "
            "create_recurring (с якорями start/end для интервала) или multi, см. правило 12.",
            text,
        )
        if corrected is not None:
            action = sanitize_action(corrected, text, now_local)
            kind = action["action"]

    ctx = user["context"]
    updates = action.get("context_updates") or {}
    if updates:
        ctx = await db.update_context(user_id, updates)

    if kind == "ask_clarification" and depth == 0:
        missing = [str(k) for k in (action.get("missing_fields") or [])]
        if missing and all(ctx.get(k) for k in missing):
            import json as _json
            hint = (
                "\n\nВАЖНО: эти данные УЖЕ есть в контексте: "
                + _json.dumps({k: ctx[k] for k in missing}, ensure_ascii=False)
                + ". Не задавай уточняющий вопрос — выполни запрос, используя эти значения."
            )
            corrected = await handlers._corrective_retry(llm, system_prompt, hint, text)
            if corrected is not None:
                action = sanitize_action(corrected, text, now_local)
                kind = action["action"]
                new_updates = corrected.get("context_updates") or {}
                if new_updates:
                    ctx = await db.update_context(user_id, new_updates)
                    updates = {**updates, **new_updates}

    if kind == "save_context":
        await db.clear_pending_clarification(user_id)
        if pending and depth == 0:
            saved_note = "; ".join(
                f"{handlers.context_label(k)}: "
                f"{'удалено' if v is None else handlers.fmt_context_value(v)}"
                for k, v in updates.items()
            )
            inner = await process_message(user_id, chat_id, pending["original_request"],
                                           db, llm, cfg, depth=1)
            prefix = f"💾 Запомнил: {saved_note}\n\n" if saved_note else ""
            inner["message"] = prefix + inner["message"]
            return inner
        lines = [
            f"• {handlers.context_label(k)}: "
            f"{'удалено 🗑' if v is None else handlers.fmt_context_value(v)}"
            for k, v in updates.items()
        ]
        return {
            "kind": "save_context", "created": None, "text_mismatch": None,
            "message": "✅ Запомнил:\n" + "\n".join(lines),
        }

    if kind == "set_timezone":
        msg = await handlers._do_set_timezone(action, user_id, db)
        return {"kind": "set_timezone", "message": msg, "created": None, "text_mismatch": None}

    if kind in ("create_reminder", "create_recurring", "create_conditional"):
        if kind == "create_reminder":
            msg, _stop, _fire, item_id = await handlers._do_create_reminder(
                action, user, ctx, tz, db, text
            )
            created = {"type": "reminder", "id": item_id} if item_id is not None else None
        elif kind == "create_recurring":
            msg, _stop, item_id = await handlers._do_create_recurring(action, user, ctx, tz, db, text)
            created = {"type": "series", "id": item_id} if item_id is not None else None
        else:
            msg, _stop, item_id = await handlers._do_create_conditional(action, user, ctx, tz, db, text)
            created = {"type": "conditional", "id": item_id} if item_id is not None else None

        reminder_text = str(action.get("reminder_text") or "").strip()
        mismatch = None
        if item_id is not None and reminder_text and not text_matches_source(reminder_text, text):
            mismatch = {"got": text, "understood": reminder_text}
        return {"kind": kind, "message": msg, "created": created, "text_mismatch": mismatch}

    if kind == "multi":
        parts: list[str] = []
        last_fire_utc = None
        created_items: list[dict] = []
        for sub in action.get("actions", []):
            sub_updates = sub.get("context_updates") or {}
            if sub_updates:
                ctx = await db.update_context(user_id, sub_updates)
            skind = sub.get("action")
            if skind == "save_context" and sub_updates:
                parts.append(
                    "✅ Запомнил: " + "; ".join(f"{k}: {v}" for k, v in sub_updates.items())
                )
            elif skind == "create_reminder":
                msg, stop, fire_utc, rid = await handlers._do_create_reminder(
                    sub, user, ctx, tz, db, text
                )
                parts.append(msg)
                if rid is not None:
                    created_items.append({"type": "reminder", "id": rid})
                if fire_utc is not None:
                    last_fire_utc = fire_utc
                if stop:
                    break
            elif skind == "create_recurring":
                msg, stop, sid = await handlers._do_create_recurring(
                    sub, user, ctx, tz, db, text, skip_until=last_fire_utc
                )
                parts.append(msg)
                if sid is not None:
                    created_items.append({"type": "series", "id": sid})
                if stop:
                    break
            elif skind == "create_conditional":
                msg, stop, cid = await handlers._do_create_conditional(sub, user, ctx, tz, db, text)
                parts.append(msg)
                if cid is not None:
                    created_items.append({"type": "conditional", "id": cid})
                if stop:
                    break
            elif skind == "set_timezone":
                parts.append(await handlers._do_set_timezone(sub, user_id, db))
        return {
            "kind": "multi",
            "message": "\n\n".join(parts) if parts else "Не понял запрос — попробуй сформулировать иначе.",
            "created": created_items or None,
            "text_mismatch": None,
        }

    if kind == "ask_clarification":
        question = action["clarification_question"].strip()
        original = pending["original_request"] if pending else text
        if pending and pending["original_request"] != text:
            original = f"{pending['original_request']} (уточнение: {text})"
        await db.set_pending_clarification(user_id, original, question)
        return {
            "kind": "ask_clarification", "created": None, "text_mismatch": None,
            "message": "❓ " + question,
        }

    if pending:
        await db.clear_pending_clarification(user_id)
    return {
        "kind": "not_a_reminder", "created": None, "text_mismatch": None,
        "message": "Это, кажется, не про напоминания 🙂",
    }


def _llm_error(e: LLMUnavailable, cfg: Config) -> dict:
    log.error("LLM API недоступен: %s", e)
    return {
        "kind": "error", "created": None, "text_mismatch": None,
        "message": "Языковая модель сейчас недоступна. Попробуй ещё раз чуть позже.",
    }
