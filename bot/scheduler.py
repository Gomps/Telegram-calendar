"""Планировщик: доставка напоминаний точно в срок и восстановление после простоя.

Принцип: БД — источник истины, планировщик не держит состояния в памяти.
Каждые POLL_INTERVAL секунд:
  1. разовые напоминания с fire_at <= now отправляются и помечаются sent
     (если опоздание больше порога — с пометкой «просрочено»);
  2. для каждой активной серии по её символическому правилу и АКТУАЛЬНОМУ
     контексту пользователя вычисляются все срабатывания позже курсора
     last_fired_at; свежие отправляются, сильно устаревшие (простой бота)
     сворачиваются в одно уведомление «пропущено N», курсор сдвигается.

Благодаря этому перезапуск бота не требует отдельного «восстановления»:
первый же тик после старта обрабатывает всё, что накопилось, а изменение
распорядка пользователя автоматически пересчитывает интервальные серии.
"""

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from aiogram import Bot

from .config import Config
from .db import Database
from .rules import next_occurrence

log = logging.getLogger(__name__)


def fmt_local(dt_utc: datetime, tz: ZoneInfo) -> str:
    return dt_utc.astimezone(tz).strftime("%d.%m.%Y %H:%M")


class ReminderScheduler:
    def __init__(self, bot: Bot, db: Database, cfg: Config):
        self.bot = bot
        self.db = db
        self.cfg = cfg
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="reminder-scheduler")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        log.info("Планировщик запущен (интервал %d с)", self.cfg.poll_interval)
        while True:
            try:
                await self._tick()
            except Exception:
                log.exception("Ошибка тика планировщика")
            await asyncio.sleep(self.cfg.poll_interval)

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        await self._process_one_off(now)
        await self._process_series(now)

    # --- разовые ---------------------------------------------------------

    async def _process_one_off(self, now: datetime) -> None:
        for r in await self.db.due_reminders(now):
            fire_at = datetime.fromisoformat(r["fire_at"])
            tz_name, _ = await self.db.get_user_tz_and_context(r["user_id"])
            tz = ZoneInfo(tz_name)
            late = (now - fire_at).total_seconds()
            if late > self.cfg.overdue_threshold:
                text = (
                    f"⏰ Просрочено (бот был недоступен, срок был "
                    f"{fmt_local(fire_at, tz)}):\n{r['text']}"
                )
            else:
                text = f"🔔 Напоминание: {r['text']}"
            try:
                await self.bot.send_message(r["chat_id"], text)
                await self.db.mark_reminder_sent(r["id"])
                log.info("Отправлено напоминание #%d пользователю %d", r["id"], r["user_id"])
            except Exception:
                log.exception("Не удалось отправить напоминание #%d, повтор на следующем тике", r["id"])

    # --- серии -----------------------------------------------------------

    async def _process_series(self, now: datetime) -> None:
        for s in await self.db.active_series():
            try:
                await self._process_one_series(s, now)
            except Exception:
                log.exception("Ошибка обработки серии #%d", s["id"])

    async def _process_one_series(self, s: dict, now: datetime) -> None:
        tz_name, ctx = await self.db.get_user_tz_and_context(s["user_id"])
        tz = ZoneInfo(tz_name)
        cursor = datetime.fromisoformat(s["last_fired_at"] or s["created_at"])

        missed = 0
        last_missed = None
        # Догоняем все срабатывания между курсором и «сейчас»
        for _ in range(500):  # предохранитель от бесконечного цикла
            occ = next_occurrence(s["rule"], ctx, cursor, tz)
            if occ is None:
                return
            occ_utc = occ.astimezone(timezone.utc)
            if occ_utc > now:
                break
            if (now - occ_utc).total_seconds() <= self.cfg.series_grace:
                await self.bot.send_message(s["chat_id"], f"🔁 Напоминание: {s['text']}")
                log.info("Серия #%d: отправлено срабатывание %s", s["id"], occ)
            else:
                missed += 1
                last_missed = occ
            cursor = occ_utc
            await self.db.set_series_cursor(s["id"], occ_utc)

        if missed:
            await self.bot.send_message(
                s["chat_id"],
                f"⏰ Пока бот был недоступен, пропущено {missed} напоминаний серии "
                f"«{s['text']}» (последнее — {last_missed.strftime('%d.%m.%Y %H:%M')}).",
            )
            log.info("Серия #%d: пропущено %d срабатываний за время простоя", s["id"], missed)
