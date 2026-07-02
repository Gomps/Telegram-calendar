"""Слой хранения: SQLite (aiosqlite).

База — единственный источник истины. Планировщик каждый тик читает её,
поэтому перезапуск бота ничего не теряет: все напоминания, серии,
контекст пользователей и незавершённые уточнения переживают рестарт.

Все моменты времени храним в UTC (ISO 8601 со смещением +00:00),
локальное время вычисляется по часовому поясу пользователя.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    chat_id     INTEGER NOT NULL,
    timezone    TEXT    NOT NULL,
    context_json TEXT   NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    chat_id     INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    fire_at     TEXT    NOT NULL,           -- UTC ISO
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | sent | cancelled
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (status, fire_at);

CREATE TABLE IF NOT EXISTS series (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    chat_id       INTEGER NOT NULL,
    text          TEXT    NOT NULL,
    rule_json     TEXT    NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    last_fired_at TEXT,                     -- UTC ISO: курсор последнего обработанного срабатывания
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS conditionals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    chat_id       INTEGER NOT NULL,
    question      TEXT    NOT NULL,   -- вопрос условия («Ты не спишь?»)
    reminder_text TEXT    NOT NULL,
    check_at      TEXT    NOT NULL,   -- UTC: когда задать вопрос
    fire_at       TEXT    NOT NULL,   -- UTC: когда напомнить при подтверждении
    -- pending -> asked -> confirmed | declined | expired | cancelled
    status        TEXT    NOT NULL DEFAULT 'pending',
    ask_message_id INTEGER,           -- id сообщения-вопроса (для удаления по истечении)
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS allowed_users (
    user_id    INTEGER PRIMARY KEY,
    added_by   INTEGER NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_actions (
    user_id       INTEGER PRIMARY KEY,
    action_json   TEXT NOT NULL,   -- действие LLM, ожидающее подтверждения пользователя
    original_text TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_clarifications (
    user_id          INTEGER PRIMARY KEY,
    original_request TEXT NOT NULL,
    question         TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str):
        self.path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._migrate()
        await self._db.commit()

    async def _migrate(self) -> None:
        """Догоняющие миграции для баз, созданных старыми версиями схемы."""
        cur = await self._db.execute("PRAGMA table_info(conditionals)")
        columns = {row["name"] for row in await cur.fetchall()}
        if "ask_message_id" not in columns:
            await self._db.execute(
                "ALTER TABLE conditionals ADD COLUMN ask_message_id INTEGER"
            )

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database.connect() не вызван"
        return self._db

    # --- пользователи и контекст -------------------------------------------

    async def get_or_create_user(self, user_id: int, chat_id: int, default_tz: str) -> dict:
        cur = await self.db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        if row is None:
            await self.db.execute(
                "INSERT INTO users (user_id, chat_id, timezone, context_json, created_at) "
                "VALUES (?, ?, ?, '{}', ?)",
                (user_id, chat_id, default_tz, utcnow_iso()),
            )
            await self.db.commit()
            return {"user_id": user_id, "chat_id": chat_id, "timezone": default_tz, "context": {}}
        if row["chat_id"] != chat_id:
            await self.db.execute(
                "UPDATE users SET chat_id = ? WHERE user_id = ?", (chat_id, user_id)
            )
            await self.db.commit()
        return {
            "user_id": row["user_id"],
            "chat_id": chat_id,
            "timezone": row["timezone"],
            "context": json.loads(row["context_json"]),
        }

    async def update_context(self, user_id: int, updates: dict[str, Any]) -> dict:
        cur = await self.db.execute(
            "SELECT context_json FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cur.fetchone()
        if row is None:
            # get_or_create_user всегда должен вызываться раньше; молчаливый
            # no-op здесь означал бы «✅ Запомнил» без реального сохранения
            raise RuntimeError(f"update_context: пользователь {user_id} не зарегистрирован")
        ctx = json.loads(row["context_json"])
        ctx.update(updates)
        await self.db.execute(
            "UPDATE users SET context_json = ? WHERE user_id = ?",
            (json.dumps(ctx, ensure_ascii=False), user_id),
        )
        await self.db.commit()
        return ctx

    async def set_timezone(self, user_id: int, tz: str) -> None:
        await self.db.execute("UPDATE users SET timezone = ? WHERE user_id = ?", (tz, user_id))
        await self.db.commit()

    async def get_user_tz_and_context(self, user_id: int) -> tuple[str, dict]:
        cur = await self.db.execute(
            "SELECT timezone, context_json FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cur.fetchone()
        if row is None:
            return "UTC", {}
        return row["timezone"], json.loads(row["context_json"])

    # --- разовые напоминания -------------------------------------------------

    async def add_reminder(self, user_id: int, chat_id: int, text: str, fire_at_utc: datetime) -> int:
        cur = await self.db.execute(
            "INSERT INTO reminders (user_id, chat_id, text, fire_at, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, chat_id, text, fire_at_utc.isoformat(timespec="seconds"), utcnow_iso()),
        )
        await self.db.commit()
        return cur.lastrowid

    async def due_reminders(self, now_utc: datetime) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM reminders WHERE status = 'pending' AND fire_at <= ? ORDER BY fire_at",
            (now_utc.isoformat(timespec="seconds"),),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def mark_reminder_sent(self, reminder_id: int) -> None:
        await self.db.execute(
            "UPDATE reminders SET status = 'sent' WHERE id = ?", (reminder_id,)
        )
        await self.db.commit()

    async def list_pending_reminders(self, user_id: int) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM reminders WHERE user_id = ? AND status = 'pending' ORDER BY fire_at",
            (user_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_reminder(self, user_id: int, reminder_id: int) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, user_id)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def cancel_reminder(self, user_id: int, reminder_id: int) -> bool:
        cur = await self.db.execute(
            "UPDATE reminders SET status = 'cancelled' "
            "WHERE id = ? AND user_id = ? AND status = 'pending'",
            (reminder_id, user_id),
        )
        await self.db.commit()
        return cur.rowcount > 0

    # --- периодические серии ---------------------------------------------------

    async def add_series(self, user_id: int, chat_id: int, text: str, rule: dict) -> int:
        cur = await self.db.execute(
            "INSERT INTO series (user_id, chat_id, text, rule_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, chat_id, text, json.dumps(rule, ensure_ascii=False), utcnow_iso()),
        )
        await self.db.commit()
        return cur.lastrowid

    async def active_series(self) -> list[dict]:
        cur = await self.db.execute("SELECT * FROM series WHERE active = 1")
        rows = [dict(r) for r in await cur.fetchall()]
        for r in rows:
            r["rule"] = json.loads(r["rule_json"])
        return rows

    async def list_user_series(self, user_id: int) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM series WHERE user_id = ? AND active = 1", (user_id,)
        )
        rows = [dict(r) for r in await cur.fetchall()]
        for r in rows:
            r["rule"] = json.loads(r["rule_json"])
        return rows

    async def get_series(self, user_id: int, series_id: int) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM series WHERE id = ? AND user_id = ?", (series_id, user_id)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        s = dict(row)
        s["rule"] = json.loads(s["rule_json"])
        return s

    async def set_series_cursor(self, series_id: int, fired_at_utc: datetime) -> None:
        await self.db.execute(
            "UPDATE series SET last_fired_at = ? WHERE id = ?",
            (fired_at_utc.isoformat(timespec="seconds"), series_id),
        )
        await self.db.commit()

    async def deactivate_series(self, user_id: int, series_id: int) -> bool:
        cur = await self.db.execute(
            "UPDATE series SET active = 0 WHERE id = ? AND user_id = ? AND active = 1",
            (series_id, user_id),
        )
        await self.db.commit()
        return cur.rowcount > 0

    # --- условные напоминания ------------------------------------------------------

    async def add_conditional(
        self, user_id: int, chat_id: int, question: str, reminder_text: str,
        check_at_utc: datetime, fire_at_utc: datetime,
    ) -> int:
        cur = await self.db.execute(
            "INSERT INTO conditionals (user_id, chat_id, question, reminder_text, "
            "check_at, fire_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, chat_id, question, reminder_text,
             check_at_utc.isoformat(timespec="seconds"),
             fire_at_utc.isoformat(timespec="seconds"), utcnow_iso()),
        )
        await self.db.commit()
        return cur.lastrowid

    async def due_conditionals(self, now_utc: datetime) -> list[dict]:
        """Условия, по которым пора задать вопрос."""
        cur = await self.db.execute(
            "SELECT * FROM conditionals WHERE status = 'pending' AND check_at <= ?",
            (now_utc.isoformat(timespec="seconds"),),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def expired_conditionals(self, now_utc: datetime) -> list[dict]:
        """Заданные вопросы, на которые не ответили до целевого времени."""
        cur = await self.db.execute(
            "SELECT * FROM conditionals WHERE status = 'asked' AND fire_at <= ?",
            (now_utc.isoformat(timespec="seconds"),),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_conditional(self, user_id: int, cond_id: int) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM conditionals WHERE id = ? AND user_id = ?", (cond_id, user_id)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def set_conditional_status(self, cond_id: int, status: str) -> None:
        await self.db.execute(
            "UPDATE conditionals SET status = ? WHERE id = ?", (status, cond_id)
        )
        await self.db.commit()

    async def set_conditional_ask_message(self, cond_id: int, message_id: int) -> None:
        await self.db.execute(
            "UPDATE conditionals SET ask_message_id = ? WHERE id = ?", (message_id, cond_id)
        )
        await self.db.commit()

    async def list_user_conditionals(self, user_id: int) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM conditionals WHERE user_id = ? AND status IN ('pending', 'asked')",
            (user_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def cancel_conditional(self, user_id: int, cond_id: int) -> bool:
        cur = await self.db.execute(
            "UPDATE conditionals SET status = 'cancelled' "
            "WHERE id = ? AND user_id = ? AND status IN ('pending', 'asked')",
            (cond_id, user_id),
        )
        await self.db.commit()
        return cur.rowcount > 0

    # --- белый список пользователей ----------------------------------------------

    async def add_allowed_user(self, user_id: int, added_by: int) -> bool:
        """True — добавлен, False — уже был в списке."""
        cur = await self.db.execute(
            "INSERT OR IGNORE INTO allowed_users (user_id, added_by, created_at) VALUES (?, ?, ?)",
            (user_id, added_by, utcnow_iso()),
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def remove_allowed_user(self, user_id: int) -> bool:
        cur = await self.db.execute(
            "DELETE FROM allowed_users WHERE user_id = ?", (user_id,)
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def is_user_allowed(self, user_id: int) -> bool:
        cur = await self.db.execute(
            "SELECT 1 FROM allowed_users WHERE user_id = ?", (user_id,)
        )
        return await cur.fetchone() is not None

    async def list_allowed_users(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM allowed_users ORDER BY created_at"
        )
        return [dict(r) for r in await cur.fetchall()]

    # --- действия, ожидающие подтверждения ----------------------------------------

    async def set_pending_action(self, user_id: int, action_json: str, original_text: str) -> None:
        await self.db.execute(
            "INSERT INTO pending_actions (user_id, action_json, original_text, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET action_json = excluded.action_json, "
            "original_text = excluded.original_text, created_at = excluded.created_at",
            (user_id, action_json, original_text, utcnow_iso()),
        )
        await self.db.commit()

    async def get_pending_action(self, user_id: int) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM pending_actions WHERE user_id = ?", (user_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def clear_pending_action(self, user_id: int) -> None:
        await self.db.execute("DELETE FROM pending_actions WHERE user_id = ?", (user_id,))
        await self.db.commit()

    # --- незавершённые уточнения -------------------------------------------------

    async def get_pending_clarification(self, user_id: int) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM pending_clarifications WHERE user_id = ?", (user_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def set_pending_clarification(
        self, user_id: int, original_request: str, question: str
    ) -> None:
        await self.db.execute(
            "INSERT INTO pending_clarifications (user_id, original_request, question, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET original_request = excluded.original_request, "
            "question = excluded.question, created_at = excluded.created_at",
            (user_id, original_request, question, utcnow_iso()),
        )
        await self.db.commit()

    async def clear_pending_clarification(self, user_id: int) -> None:
        await self.db.execute(
            "DELETE FROM pending_clarifications WHERE user_id = ?", (user_id,)
        )
        await self.db.commit()
