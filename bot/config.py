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
    # LLM: любой OpenAI-совместимый API (NVIDIA NIM, Ollama /v1, OpenRouter…)
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    # Цепочка моделей основного провайдера: недоступна одна — берём следующую
    llm_models: tuple[str, ...]
    # Запасной провайдер (например, локальная Ollama), пробуется после основного
    llm_fallback_api_base: str
    llm_fallback_api_key: str
    llm_fallback_models: tuple[str, ...]
    # Сколько попыток на каждую модель при недоступности
    llm_attempts_per_model: int
    # Предварительное LLM-разбиение сообщения на блоки (0 = LLM решает всё сама
    # одним вызовом, бот проверяет форму; 1 = включить этап разбиения)
    split_stage: bool
    # Распознавание речи: OpenAI-совместимый /audio/transcriptions
    # (например, Groq whisper-large-v3). Пустой asr_api_base = локальный Whisper.
    asr_api_base: str
    asr_api_key: str
    asr_model: str
    asr_language: str
    whisper_model: str
    default_tz: str
    db_path: str
    # Файл логов с ротацией; пустая строка — писать только в stdout
    log_file: str
    poll_interval: int
    llm_retries: int
    # Таймаут запроса к Ollama, сек. Холодная загрузка модели на CPU
    # (особенно под Windows) может занимать минуты — не занижайте.
    llm_timeout: float
    # Таймаут ТОЛЬКО на установку соединения, сек. Отдельно от llm_timeout:
    # заблокированный/недоступный хост должен отваливаться быстро (секунды),
    # а не ждать полный llm_timeout, как обычный медленный ответ модели.
    llm_connect_timeout: float
    # На сколько секунд провайдер помечается недоступным после исчерпания
    # попыток — следующие сообщения пропускают его без повторных попыток,
    # пока кулдаун не истечёт.
    llm_cooldown: float
    # Насколько поздно (сек) напоминание ещё считается «вовремя», а не просроченным
    overdue_threshold: int
    # Пропущенное срабатывание серии старше этого (сек) не отправляется, а считается пропущенным
    series_grace: int
    # Telegram Mini App: веб-интерфейс со всем функционалом бота
    webapp_enabled: bool
    webapp_host: str
    webapp_port: int
    # Публичный HTTPS-адрес мини-аппа (без него кнопка не показывается,
    # но сам HTTP-сервер всё равно поднимается — можно завести реверс-прокси)
    webapp_url: str
    # Сколько секунд считать initData свежим (защита от повторной отправки)
    webapp_init_data_max_age: int
    # Принимать заголовок X-Dev-User-Id вместо настоящей подписи Telegram —
    # ТОЛЬКО для локальной разработки без реального Telegram-клиента
    webapp_allow_dev_auth: bool


def load_config() -> Config:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Скопируйте .env.example в .env и впишите токен бота."
        )
    # Обратная совместимость: если заданы старые OLLAMA_URL/OLLAMA_MODEL,
    # используем OpenAI-совместимый эндпоинт Ollama (/v1)
    llm_api_base = os.getenv("LLM_API_BASE", "").strip().rstrip("/")
    llm_model = os.getenv("LLM_MODEL", "").strip()
    ollama_url = os.getenv("OLLAMA_URL", "").strip().rstrip("/")
    if not llm_api_base:
        if ollama_url:
            llm_api_base = ollama_url + "/v1"
        else:
            llm_api_base = "https://integrate.api.nvidia.com/v1"
    if not llm_model:
        llm_model = os.getenv("OLLAMA_MODEL", "").strip() or "qwen/qwen3.5-122b-a10b"

    models_raw = os.getenv(
        "LLM_MODELS",
        "qwen/qwen3.5-122b-a10b,meta/llama-3.3-70b-instruct,qwen/qwen3-next-80b-a3b-instruct",
    )
    llm_models = tuple(m.strip() for m in models_raw.split(",") if m.strip())
    if llm_model not in llm_models:
        llm_models = (llm_model,) + llm_models

    fb_models_raw = os.getenv("LLM_FALLBACK_MODELS", "")
    llm_fallback_models = tuple(m.strip() for m in fb_models_raw.split(",") if m.strip())

    return Config(
        bot_token=token,
        admin_ids=_parse_ids(os.getenv("ADMIN_USER_IDS", "")),
        allowed_ids=_parse_ids(os.getenv("ALLOWED_USER_IDS", "")),
        llm_api_base=llm_api_base,
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=llm_models[0],
        llm_models=llm_models,
        llm_fallback_api_base=os.getenv("LLM_FALLBACK_API_BASE", "").strip().rstrip("/"),
        llm_fallback_api_key=os.getenv("LLM_FALLBACK_API_KEY", "").strip(),
        llm_fallback_models=llm_fallback_models,
        llm_attempts_per_model=int(os.getenv("LLM_ATTEMPTS_PER_MODEL", "3")),
        split_stage=os.getenv("SPLIT_STAGE", "0").strip() in ("1", "true", "yes"),
        asr_api_base=os.getenv("ASR_API_BASE", "").strip().rstrip("/"),
        asr_api_key=os.getenv("ASR_API_KEY", "").strip(),
        asr_model=os.getenv("ASR_MODEL", "whisper-large-v3").strip(),
        asr_language=os.getenv("ASR_LANGUAGE", "ru").strip(),
        whisper_model=os.getenv("WHISPER_MODEL", "medium"),
        default_tz=os.getenv("DEFAULT_TZ", "Europe/Minsk"),
        db_path=os.getenv("DB_PATH", "data/bot.db"),
        log_file=os.getenv("LOG_FILE", "data/bot.log").strip(),
        poll_interval=int(os.getenv("POLL_INTERVAL", "15")),
        llm_retries=int(os.getenv("LLM_RETRIES", "3")),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "180")),
        llm_connect_timeout=float(os.getenv("LLM_CONNECT_TIMEOUT", "10")),
        llm_cooldown=float(os.getenv("LLM_COOLDOWN", "120")),
        overdue_threshold=int(os.getenv("OVERDUE_THRESHOLD", "120")),
        series_grace=int(os.getenv("SERIES_GRACE", "900")),
        webapp_enabled=os.getenv("WEBAPP_ENABLED", "0").strip() in ("1", "true", "yes"),
        webapp_host=os.getenv("WEBAPP_HOST", "0.0.0.0").strip(),
        webapp_port=int(os.getenv("WEBAPP_PORT", "8080")),
        webapp_url=os.getenv("WEBAPP_URL", "").strip().rstrip("/"),
        webapp_init_data_max_age=int(os.getenv("WEBAPP_INIT_DATA_MAX_AGE", "86400")),
        webapp_allow_dev_auth=os.getenv("WEBAPP_ALLOW_DEV_AUTH", "0").strip() in ("1", "true", "yes"),
    )
