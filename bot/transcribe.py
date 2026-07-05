"""Транскрибация голосовых сообщений.

Два бэкенда:
- облачный: любой OpenAI-совместимый /audio/transcriptions (например,
  бесплатный Groq whisper-large-v3) — включается, если задан ASR_API_BASE;
- локальный Whisper (фолбэк): модель загружается лениво при первом
  голосовом и держится в памяти; распознавание — блокирующая операция,
  выполняется в отдельном потоке и сериализовано (модель не потокобезопасна).
"""

import asyncio
import importlib.util
import logging
import os
import shutil
from typing import Optional

import httpx

log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """detail — для логов/админа; public — короткая категория для пользователя.

    network=True помечает сетевые сбои облачного ASR (не достучались до
    хоста — не путать с ответом сервиса вроде 401/400), при которых имеет
    смысл откатиться на локальный Whisper, если он настроен.
    """

    def __init__(self, detail: str, public: str = "ошибка сервиса", network: bool = False):
        super().__init__(detail)
        self.public = public
        self.network = network


class Transcriber:
    def __init__(
        self,
        model_name: str = "medium",
        api_base: str = "",
        api_key: str = "",
        api_model: str = "whisper-large-v3",
        language: str = "ru",
        connect_timeout: float = 10.0,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.api_model = api_model
        self.language = language
        self.connect_timeout = connect_timeout
        self._model = None
        self._lock = asyncio.Lock()
        # Модель Whisper не потокобезопасна — распознавания выполняются
        # по одному (сообщения разных пользователей ждут очереди только здесь)
        self._infer_lock = asyncio.Lock()

    @property
    def remote(self) -> bool:
        return bool(self.api_base)

    @staticmethod
    def local_available() -> bool:
        """Можно ли откатиться на локальный Whisper: есть ffmpeg и пакет установлен."""
        return shutil.which("ffmpeg") is not None and importlib.util.find_spec("whisper") is not None

    async def _get_model(self):
        async with self._lock:
            if self._model is None:
                log.info("Загружаю Whisper «%s» (первый запуск может занять минуты)…", self.model_name)
                try:
                    import whisper  # noqa: PLC0415 — тяжёлый импорт откладываем до первого использования
                    self._model = await asyncio.to_thread(whisper.load_model, self.model_name)
                except Exception as e:
                    raise TranscriptionError(
                        f"Не удалось загрузить Whisper: {e}", public="сервис недоступен"
                    ) from e
                log.info("Whisper «%s» загружен", self.model_name)
        return self._model

    async def transcribe(self, path: str, language: Optional[str] = None) -> str:
        if not self.remote:
            return await self._transcribe_local(path, language)
        try:
            return await self._transcribe_remote(path, language or self.language)
        except TranscriptionError as e:
            # Сетевой сбой облачного ASR (не ответ сервиса вроде 400/401) —
            # пробуем локальный Whisper вместо немедленного отказа
            if e.network and self.local_available():
                log.warning(
                    "Облачный ASR недоступен (%s) — переключаюсь на локальный Whisper", e
                )
                return await self._transcribe_local(path, language)
            raise

    async def _transcribe_remote(self, path: str, language: str) -> str:
        """OpenAI-совместимый POST /audio/transcriptions (multipart)."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = {"model": self.api_model, "response_format": "json"}
        if language:
            data["language"] = language
        # Раздельные таймауты: заблокированный/недоступный хост отваливается
        # за секунды на этапе соединения, а не ждёт полный read-таймаут
        timeout = httpx.Timeout(connect=self.connect_timeout, read=120.0, write=30.0, pool=10.0)
        try:
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f, "audio/ogg")}
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{self.api_base}/audio/transcriptions",
                        headers=headers, data=data, files=files,
                    )
            resp.raise_for_status()
            text = str(resp.json().get("text", "")).strip()
        except httpx.HTTPStatusError as e:
            # Сервис ответил (пусть и ошибкой) — это конфигурация (неверный
            # ключ/модель), а не недоступность сети; на локальный не откатываем
            raise TranscriptionError(
                f"ASR API вернул {e.response.status_code}: {e.response.text[:200]}",
                public="ошибка сервиса",
            ) from e
        except (httpx.HTTPError, ValueError) as e:
            raise TranscriptionError(
                f"ASR API недоступен: {e}", public="сервис недоступен", network=True
            ) from e
        if not text:
            raise TranscriptionError("Речь не распознана (пустой результат)", public="пустое сообщение")
        return text

    async def _transcribe_local(self, path: str, language: Optional[str]) -> str:
        # Whisper читает аудио через ffmpeg; без него ошибка была бы невнятной
        if shutil.which("ffmpeg") is None:
            raise TranscriptionError(
                "ffmpeg не найден в PATH. Установи его и перезапусти бота "
                "(Windows: winget install Gyan.FFmpeg; Linux: sudo apt install ffmpeg)",
                public="сервис не настроен",
            )
        model = await self._get_model()
        async with self._infer_lock:
            try:
                result = await asyncio.to_thread(
                    model.transcribe, path, language=language, fp16=False
                )
            except Exception as e:
                raise TranscriptionError(f"Ошибка распознавания: {e}") from e
        text = str(result.get("text", "")).strip()
        if not text:
            raise TranscriptionError("Речь не распознана (пустой результат)", public="пустое сообщение")
        return text
