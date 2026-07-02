"""Конфигурация приложения: всё берётся из переменных окружения / .env."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _parse_ids(raw: str) -> tuple[int, ...]:
    ids = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                ids.append(int(part))
            except ValueError:
                raise RuntimeError(f"Некорректный Telegram ID в конфиге: «{part}»") from None
    return tuple(ids)


@dataclass(frozen=True)
class Config:
    bot_token: str
    # Администраторы: управляют белым списком (/adduser, /removeuser, /users).
    # Если список пуст — бот открыт для всех (доступ не проверяется).
    admin_ids: tuple[int, ...]
    # Статический белый список из .env (в дополнение к добавленным через /adduser)
    allowed_ids: tuple[int, ...]
    ollama_url: str
    ollama_model: str
    whisper_model: str
    default_tz: str
    db_path: str
    # Файл логов с ротацией; пустая строка — писать только в stdout
    log_file: str
    poll_interval: int
    llm_retries: int
    # Насколько поздно (сек) напоминание ещё считается «вовремя», а не просроченным
    overdue_threshold: int
    # Пропущенное срабатывание серии старше этого (сек) не отправляется, а считается пропущенным
    series_grace: int


def load_config() -> Config:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Скопируйте .env.example в .env и впишите токен бота."
        )
    return Config(
        bot_token=token,
        admin_ids=_parse_ids(os.getenv("ADMIN_USER_IDS", "")),
        allowed_ids=_parse_ids(os.getenv("ALLOWED_USER_IDS", "")),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        whisper_model=os.getenv("WHISPER_MODEL", "medium"),
        default_tz=os.getenv("DEFAULT_TZ", "Europe/Minsk"),
        db_path=os.getenv("DB_PATH", "data/bot.db"),
        log_file=os.getenv("LOG_FILE", "data/bot.log").strip(),
        poll_interval=int(os.getenv("POLL_INTERVAL", "15")),
        llm_retries=int(os.getenv("LLM_RETRIES", "3")),
        overdue_threshold=int(os.getenv("OVERDUE_THRESHOLD", "120")),
        series_grace=int(os.getenv("SERIES_GRACE", "900")),
    )
