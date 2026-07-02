"""Кольцевой буфер логов в памяти для команды /log.

Хэндлер вешается на корневой логгер и хранит последние N отформатированных
записей. Страницы для Telegram собираются на лету: несколько маленьких
записей группируются в одно сообщение, слишком длинная запись обрезается,
чтобы страница гарантированно влезала в лимит Telegram (4096 символов).
"""

import logging
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
