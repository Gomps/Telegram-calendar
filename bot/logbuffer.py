"""Кольцевой буфер логов в памяти для команды /log.

Хэндлер вешается на корневой логгер и хранит последние N отформатированных
записей. Страницы для Telegram собираются на лету: несколько маленьких
записей группируются в одно сообщение, слишком длинная запись обрезается,
чтобы страница гарантированно влезала в лимит Telegram (4096 символов).
"""

import logging
import os
from collections import deque

# Максимум символов лога на страницу (плюс заголовок и <pre> — с запасом до 4096)
PAGE_CHARS = 3000


class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.records: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # логирование не должно ронять бота
            pass

    def seed_from_file(self, path: str, tail_bytes: int = 256 * 1024) -> int:
        """Наполняет буфер хвостом лог-файла, чтобы /log видел историю
        и после перезапуска бота. Возвращает число загруженных строк."""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - tail_bytes))
                data = f.read().decode("utf-8", errors="replace")
        except OSError:
            return 0
        lines = [line for line in data.splitlines() if line.strip()]
        if size > tail_bytes and lines:
            lines = lines[1:]  # первая строка может быть обрезана посередине
        self.records.extend(lines)
        return len(lines)


def build_pages(records) -> list[str]:
    """Группирует записи (в хронологическом порядке) в страницы <= PAGE_CHARS.

    Последняя страница — самые свежие логи.
    """
    pages: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for rec in records:
        if len(rec) > PAGE_CHARS:
            rec = rec[: PAGE_CHARS - 1] + "…"
        if cur and cur_len + len(rec) + 1 > PAGE_CHARS:
            pages.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(rec)
        cur_len += len(rec) + 1
    if cur:
        pages.append("\n".join(cur))
    return pages
