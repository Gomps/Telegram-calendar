"""Точка входа: инициализация БД, LLM-клиента, Whisper, планировщика и поллинга."""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher

from .access import AccessMiddleware
from .config import load_config
from .db import Database
from .handlers import router
from .llm import OllamaClient
from .scheduler import ReminderScheduler
from .transcribe import Transcriber

log = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config()

    os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
    db = Database(cfg.db_path)
    await db.connect()
    log.info("БД подключена: %s", cfg.db_path)

    llm = OllamaClient(cfg.ollama_url, cfg.ollama_model, retries=cfg.llm_retries)
    if await llm.healthcheck():
        log.info("Ollama доступна: %s (модель %s)", cfg.ollama_url, cfg.ollama_model)
    else:
        log.warning(
            "Ollama недоступна на %s — бот запустится, но разбор сообщений не будет "
            "работать, пока Ollama не поднимется.",
            cfg.ollama_url,
        )

    bot = Bot(token=cfg.bot_token)
    dp = Dispatcher()
    dp.include_router(router)
    # Внедрение зависимостей в обработчики по имени параметра
    dp["db"] = db
    dp["llm"] = llm
    dp["transcriber"] = Transcriber(cfg.whisper_model)
    dp["cfg"] = cfg

    # Белый список: если ADMIN_USER_IDS задан, доступ только у админов и добавленных
    dp.message.outer_middleware(AccessMiddleware())
    dp.callback_query.outer_middleware(AccessMiddleware())
    if cfg.admin_ids:
        log.info(
            "Контроль доступа включён: админы %s, статический белый список %s",
            list(cfg.admin_ids), list(cfg.allowed_ids) or "—",
        )
    else:
        log.info("ADMIN_USER_IDS не задан — бот открыт для всех пользователей")

    scheduler = ReminderScheduler(bot, db, cfg)
    scheduler.start()

    try:
        log.info("Запускаю поллинг Telegram…")
        await dp.start_polling(bot)
    finally:
        await scheduler.stop()
        await db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
