"""Точка входа: инициализация БД, LLM-клиента, Whisper, планировщика и поллинга."""

import asyncio
import logging
import logging.handlers
import os
import shutil
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot, Dispatcher

from .access import AccessMiddleware
from .config import load_config
from .db import Database
from .handlers import router
from .llm import LLMClient, LLMProvider
from .logbuffer import MemoryLogHandler
from .scheduler import ReminderScheduler
from .transcribe import Transcriber

log = logging.getLogger(__name__)


async def main() -> None:
    log_format = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    cfg = load_config()

    # Кольцевой буфер последних записей — отдаётся администратору командой /log
    logbuffer = MemoryLogHandler(capacity=1000)
    logbuffer.setFormatter(logging.Formatter(log_format, datefmt="%d.%m %H:%M:%S"))

    if cfg.log_file:
        os.makedirs(os.path.dirname(cfg.log_file) or ".", exist_ok=True)
        # история из файла доступна в /log сразу после перезапуска
        seeded = logbuffer.seed_from_file(cfg.log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            cfg.log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        log.info("Логи пишутся в %s (загружено %d строк истории)", cfg.log_file, seeded)

    logging.getLogger().addHandler(logbuffer)

    # Fail-fast: без базы часовых поясов бот бесполезен (на Windows её нет
    # в системе — нужна pip-зависимость tzdata)
    try:
        ZoneInfo(cfg.default_tz)
    except ZoneInfoNotFoundError:
        raise RuntimeError(
            f"Часовой пояс «{cfg.default_tz}» не найден. База поясов IANA "
            "недоступна — установите её: pip install tzdata (входит в requirements.txt)."
        ) from None

    os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
    db = Database(cfg.db_path)
    await db.connect()
    log.info("БД подключена: %s", cfg.db_path)

    if not cfg.asr_api_base and shutil.which("ffmpeg") is None:
        log.warning(
            "ffmpeg не найден в PATH — локальное распознавание голосовых не будет "
            "работать (Windows: winget install Gyan.FFmpeg; Linux: sudo apt install ffmpeg). "
            "Либо настрой облачное: ASR_API_BASE в .env."
        )
    if cfg.asr_api_base:
        log.info("Распознавание речи: облачное API %s (модель %s, язык %s)",
                 cfg.asr_api_base, cfg.asr_model, cfg.asr_language)
        if Transcriber.local_available():
            log.info("При недоступности облачного ASR — автоматический откат на локальный Whisper")
    else:
        log.info("Распознавание речи: локальный Whisper «%s»", cfg.whisper_model)

    providers = [
        LLMProvider(cfg.llm_api_base, cfg.llm_api_key, m) for m in cfg.llm_models
    ]
    if cfg.llm_fallback_api_base and cfg.llm_fallback_models:
        providers += [
            LLMProvider(cfg.llm_fallback_api_base, cfg.llm_fallback_api_key, m)
            for m in cfg.llm_fallback_models
        ]
    llm = LLMClient(
        providers=providers, retries=cfg.llm_retries, timeout=cfg.llm_timeout,
        connect_timeout=cfg.llm_connect_timeout,
        attempts_per_model=cfg.llm_attempts_per_model, cooldown=cfg.llm_cooldown,
    )
    log.info("Цепочка LLM (по %d попытки на модель): %s",
             cfg.llm_attempts_per_model, llm.describe())
    server_ok, model_ok = await llm.healthcheck()
    if not server_ok:
        log.warning(
            "Ни один LLM-провайдер не отвечает — бот запустится, но разбор сообщений "
            "не будет работать. Проверь LLM_API_BASE/LLM_API_KEY; если ты в регионе, "
            "где NVIDIA API заблокирован (например, Беларусь/Россия), нужен "
            "VPN/прокси или запасной провайдер (LLM_FALLBACK_API_BASE)."
        )
    elif not model_ok:
        log.warning("LLM API отвечает, но модель не найдена в списке — проверь LLM_MODELS.")
    else:
        log.info("LLM API доступен")

    bot = Bot(token=cfg.bot_token)
    dp = Dispatcher()
    dp.include_router(router)
    # Внедрение зависимостей в обработчики по имени параметра
    dp["db"] = db
    dp["llm"] = llm
    dp["transcriber"] = Transcriber(
        cfg.whisper_model,
        api_base=cfg.asr_api_base,
        api_key=cfg.asr_api_key,
        api_model=cfg.asr_model,
        language=cfg.asr_language,
        connect_timeout=cfg.llm_connect_timeout,
    )
    dp["cfg"] = cfg
    dp["logbuffer"] = logbuffer

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # штатная остановка по Ctrl+C (на Windows иначе сыплется трейсбек)
        logging.getLogger(__name__).info("Бот остановлен")
