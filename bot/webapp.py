"""HTTP-бэкенд Telegram Mini App: тот же функционал, что в чате, но в виде
REST API для веб-интерфейса, открываемого внутри Telegram.

Аутентификация — Telegram WebApp initData (см. telegram_auth.py): фронтенд
шлёт заголовок `Authorization: tma <initData>` на каждый запрос, сервер
проверяет подпись бот-токеном и достаёт user_id. Тот же белый список
доступа (access.user_has_access), что и у самого бота.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aiohttp import web

from . import handlers, nlpipe
from .access import user_has_access
from .config import Config
from .db import Database
from .llm import LLMClient, normalize_action, validate_action
from .rules import next_occurrence
from .telegram_auth import validate_init_data
from .tzutil import get_tz, resolve_timezone_input

log = logging.getLogger(__name__)

WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"

PUBLIC_PATHS = {"/", "/healthz"}


def _err(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    if request.path in PUBLIC_PATHS or request.path.startswith("/static/"):
        return await handler(request)

    cfg: Config = request.app["cfg"]
    db: Database = request.app["db"]

    user_id: Optional[int] = None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("tma "):
        data = validate_init_data(
            auth[4:], cfg.bot_token, max_age=cfg.webapp_init_data_max_age
        )
        if data and isinstance(data.get("user"), dict):
            try:
                user_id = int(data["user"]["id"])
            except (KeyError, TypeError, ValueError):
                user_id = None
    if user_id is None and cfg.webapp_allow_dev_auth:
        dev_id = request.headers.get("X-Dev-User-Id")
        if dev_id and dev_id.lstrip("-").isdigit():
            user_id = int(dev_id)

    if user_id is None:
        return _err(401, "unauthorized")
    if not await user_has_access(user_id, cfg, db):
        return _err(403, "forbidden")
    request["user_id"] = user_id
    return await handler(request)


def _reminder_json(r: dict, tz) -> dict:
    fire_at = datetime.fromisoformat(r["fire_at"])
    return {
        "id": r["id"],
        "text": r["text"],
        "fire_at": fire_at.isoformat(),
        "fire_at_human": handlers.fmt_local(fire_at, tz),
    }


def _series_json(s: dict, ctx: dict, tz, now: datetime) -> dict:
    try:
        description = handlers.describe_rule(s["rule"], ctx)
    except Exception:
        description = "правило не читается"
    nxt = None
    try:
        occ = next_occurrence(s["rule"], ctx, now, tz)
        if occ:
            nxt = handlers.fmt_local(occ.astimezone(timezone.utc), tz)
    except Exception:
        pass
    return {
        "id": s["id"],
        "text": s["text"],
        "description": description,
        "next_occurrence": nxt,
    }


def _conditional_json(c: dict, tz) -> dict:
    return {
        "id": c["id"],
        "question": c["question"],
        "reminder_text": c["reminder_text"],
        "status": c["status"],
        "check_at_human": handlers.fmt_local(datetime.fromisoformat(c["check_at"]), tz),
        "fire_at_human": handlers.fmt_local(datetime.fromisoformat(c["fire_at"]), tz),
    }


async def get_state(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    cfg: Config = request.app["cfg"]
    db: Database = request.app["db"]
    user = await db.get_or_create_user(user_id, user_id, cfg.default_tz, update_chat=False)
    tz = get_tz(user["timezone"])
    now = datetime.now(timezone.utc)
    now_local = now.astimezone(tz)

    reminders = await db.list_pending_reminders(user_id)
    series = await db.list_user_series(user_id)
    conditionals = await db.list_user_conditionals(user_id)

    return web.json_response({
        "user_id": user_id,
        "timezone": user["timezone"],
        "now_local": now_local.strftime("%d.%m.%Y %H:%M"),
        "context": [
            {"key": k, "label": handlers.context_label(k), "value": handlers.fmt_context_value(v)}
            for k, v in sorted(user["context"].items())
        ],
        "reminders": [_reminder_json(r, tz) for r in reminders],
        "series": [_series_json(s, user["context"], tz, now) for s in series],
        "conditionals": [_conditional_json(c, tz) for c in conditionals],
    })


import re as _re

CONTEXT_KEY_RE = _re.compile(r"^[a-zA-Z0-9_]{1,64}$")


async def post_context(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    cfg: Config = request.app["cfg"]
    db: Database = request.app["db"]
    try:
        body = await request.json()
    except Exception:
        return _err(400, "invalid json")
    updates = body.get("updates")
    if not isinstance(updates, dict) or not updates:
        return _err(400, "updates must be a non-empty object")
    # ключи — латиница/цифры/подчёркивание; значения — строка, список
    # вариантов или null (удаление). Прочие типы в контексте не нужны и
    # только ломали бы отображение.
    for key, value in updates.items():
        if not isinstance(key, str) or not CONTEXT_KEY_RE.match(key):
            return _err(400, f"недопустимый ключ «{key}» (латиница, цифры, _, до 64 символов)")
        if value is not None and not isinstance(value, (str, list)):
            return _err(400, f"значение «{key}» должно быть строкой, списком или null")
        if isinstance(value, str) and len(value) > 256:
            return _err(400, f"значение «{key}» слишком длинное")

    await db.get_or_create_user(user_id, user_id, cfg.default_tz, update_chat=False)
    action = {"action": "save_context", "context_updates": updates}
    normalize_action(action)
    errors = validate_action(action)
    if errors:
        return _err(400, "; ".join(errors))
    await db.update_context(user_id, action["context_updates"])
    log.info("Mini App: пользователь %d обновил контекст %s", user_id, updates)
    return web.json_response({"ok": True})


async def delete_context_key(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    key = request.match_info["key"]
    cfg: Config = request.app["cfg"]
    db: Database = request.app["db"]
    # без get_or_create update_context падал бы 500-й для пользователя,
    # который открыл мини-апп раньше, чем написал боту
    await db.get_or_create_user(user_id, user_id, cfg.default_tz, update_chat=False)
    await db.update_context(user_id, {key: None})
    log.info("Mini App: пользователь %d удалил ключ контекста «%s»", user_id, key)
    return web.json_response({"ok": True})


async def post_timezone(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    cfg: Config = request.app["cfg"]
    db: Database = request.app["db"]
    try:
        body = await request.json()
    except Exception:
        return _err(400, "invalid json")
    value = str(body.get("value") or "").strip()
    if not value:
        return _err(400, "value is required")
    await db.get_or_create_user(user_id, user_id, cfg.default_tz, update_chat=False)
    resolved = resolve_timezone_input(value)
    if resolved is None:
        return _err(400, f"не понял «{value}»")
    await db.set_timezone(user_id, resolved)
    log.info("Mini App: пользователь %d установил часовой пояс %s", user_id, resolved)
    return web.json_response({"ok": True, "timezone": resolved})


async def delete_reminder(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    db: Database = request.app["db"]
    try:
        rid = int(request.match_info["id"])
    except ValueError:
        return _err(400, "invalid id")
    ok = await db.cancel_reminder(user_id, rid)
    if not ok:
        return _err(404, "not found")
    log.info("Mini App: пользователь %d удалил напоминание #%d", user_id, rid)
    return web.json_response({"ok": True})


async def delete_series(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    db: Database = request.app["db"]
    try:
        sid = int(request.match_info["id"])
    except ValueError:
        return _err(400, "invalid id")
    ok = await db.deactivate_series(user_id, sid)
    if not ok:
        return _err(404, "not found")
    log.info("Mini App: пользователь %d удалил серию #%d", user_id, sid)
    return web.json_response({"ok": True})


async def delete_conditional(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    db: Database = request.app["db"]
    try:
        cid = int(request.match_info["id"])
    except ValueError:
        return _err(400, "invalid id")
    ok = await db.cancel_conditional(user_id, cid)
    if not ok:
        return _err(404, "not found")
    log.info("Mini App: пользователь %d удалил условное #%d", user_id, cid)
    return web.json_response({"ok": True})


async def post_message(request: web.Request) -> web.Response:
    user_id = request["user_id"]
    cfg: Config = request.app["cfg"]
    db: Database = request.app["db"]
    llm: LLMClient = request.app["llm"]
    try:
        body = await request.json()
    except Exception:
        return _err(400, "invalid json")
    text = str(body.get("text") or "").strip()
    if not text:
        return _err(400, "text is required")
    if len(text) > 4096:  # лимит Telegram-сообщения; заодно защита LLM-бюджета
        return _err(400, "text too long (max 4096)")
    log.info("Mini App: сообщение от %d: %s", user_id, text)
    try:
        result = await nlpipe.process_message(user_id, user_id, text, db, llm, cfg)
    except Exception:
        log.exception("Mini App: ошибка обработки сообщения пользователя %d", user_id)
        return _err(500, "внутренняя ошибка обработки")
    return web.json_response(result)


async def index(request: web.Request) -> web.Response:
    return web.FileResponse(WEBAPP_DIR / "index.html")


async def healthz(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def build_app(db: Database, llm: LLMClient, cfg: Config) -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    app["db"] = db
    app["llm"] = llm
    app["cfg"] = cfg

    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/api/state", get_state)
    app.router.add_post("/api/context", post_context)
    app.router.add_delete("/api/context/{key}", delete_context_key)
    app.router.add_post("/api/timezone", post_timezone)
    app.router.add_delete("/api/reminders/{id}", delete_reminder)
    app.router.add_delete("/api/series/{id}", delete_series)
    app.router.add_delete("/api/conditionals/{id}", delete_conditional)
    app.router.add_post("/api/message", post_message)
    if WEBAPP_DIR.is_dir():
        app.router.add_static("/static/", WEBAPP_DIR, name="static")
    return app


async def start_webapp(db: Database, llm: LLMClient, cfg: Config) -> web.AppRunner:
    app = build_app(db, llm, cfg)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, cfg.webapp_host, cfg.webapp_port)
    await site.start()
    log.info("Mini App слушает на %s:%d", cfg.webapp_host, cfg.webapp_port)
    return runner
