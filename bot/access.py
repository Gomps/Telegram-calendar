"""Контроль доступа: белый список пользователей по Telegram ID.

Правила:
- Если ADMIN_USER_IDS в .env пуст — проверка выключена, бот открыт для всех.
- Иначе доступ есть у: администраторов, ID из ALLOWED_USER_IDS (.env)
  и пользователей, добавленных администратором командой /adduser (хранятся в БД).
- Остальным бот отвечает отказом и показывает их ID, чтобы его можно было
  передать администратору.
"""

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

log = logging.getLogger(__name__)


async def user_has_access(user_id: int, cfg, db) -> bool:
    if not cfg.admin_ids:
        return True
    if user_id in cfg.admin_ids or user_id in cfg.allowed_ids:
        return True
    return await db.is_user_allowed(user_id)


class AccessMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is None:
            return await handler(event, data)
        if await user_has_access(user.id, data["cfg"], data["db"]):
            return await handler(event, data)

        log.info("Отказ в доступе пользователю %d (@%s)", user.id, user.username)
        if isinstance(event, Message):
            await event.answer(
                "⛔ У тебя нет доступа к этому боту.\n"
                f"Твой Telegram ID: `{user.id}` — передай его администратору, "
                "чтобы он добавил тебя командой /adduser.",
                parse_mode="Markdown",
            )
        elif isinstance(event, CallbackQuery):
            await event.answer("⛔ Нет доступа", show_alert=True)
        return None
