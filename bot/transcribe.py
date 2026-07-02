"""Транскрибация голосовых сообщений через Whisper (модель medium).

Модель загружается лениво при первом голосовом сообщении и держится в памяти.
Распознавание — блокирующая CPU/GPU-операция, поэтому выполняется в отдельном
потоке, чтобы не останавливать event loop бота.
"""

import asyncio
import logging
import shutil
from typing import Optional

log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    pass


class Transcriber:
    def __init__(self, model_name: str = "medium"):
        self.model_name = model_name
        self._model = None
        self._lock = asyncio.Lock()

    async def _get_model(self):
        async with self._lock:
            if self._model is None:
                log.info("Загружаю Whisper «%s» (первый запуск может занять минуты)…", self.model_name)
                try:
                    import whisper  # noqa: PLC0415 — тяжёлый импорт откладываем до первого использования
                    self._model = await asyncio.to_thread(whisper.load_model, self.model_name)
                except Exception as e:
                    raise TranscriptionError(f"Не удалось загрузить Whisper: {e}") from e
                log.info("Whisper «%s» загружен", self.model_name)
        return self._model

    async def transcribe(self, path: str, language: Optional[str] = None) -> str:
        # Whisper читает аудио через ffmpeg; без него ошибка была бы невнятной
        if shutil.which("ffmpeg") is None:
            raise TranscriptionError(
                "ffmpeg не найден в PATH. Установи его и перезапусти бота "
                "(Windows: winget install Gyan.FFmpeg; Linux: sudo apt install ffmpeg)"
            )
        model = await self._get_model()
        try:
            result = await asyncio.to_thread(
                model.transcribe, path, language=language, fp16=False
            )
        except Exception as e:
            raise TranscriptionError(f"Ошибка распознавания: {e}") from e
        text = str(result.get("text", "")).strip()
        if not text:
            raise TranscriptionError("Речь не распознана (пустой результат)")
        return text
