"""Клиент Ollama: запрос к qwen3.5:4b, валидация JSON-ответа и повторные попытки.

При невалидном JSON (или JSON, не проходящем схему) модель получает свой
ответ обратно вместе со списком ошибок и просьбу исправить — до N попыток.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

import httpx

from .rules import normalize_hhmm, parse_hhmm, validate_rule

log = logging.getLogger(__name__)

ACTIONS = {"save_context", "create_reminder", "create_recurring", "ask_clarification", "not_a_reminder"}


class LLMUnavailable(Exception):
    """Ollama недоступна или вернула ошибку."""


class LLMBadResponse(Exception):
    """Модель так и не вернула валидный JSON после всех попыток."""


def _extract_json(text: str) -> Optional[dict]:
    """Достаёт первый JSON-объект из текста (модель может добавить мусор вокруг)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def normalize_action(data: dict) -> dict:
    """Приводит времена в ответе модели к канону до валидации.

    Модель нередко повторяет формат пользователя («16.30», «7» вместо «16:30»,
    «07:00») — не отбрасываем такой ответ, а чиним детерминированно.
    """
    if data.get("action") == "multi" and isinstance(data.get("actions"), list):
        for sub in data["actions"]:
            if isinstance(sub, dict):
                normalize_action(sub)

    updates = data.get("context_updates")
    if isinstance(updates, dict):
        for key, value in list(updates.items()):
            norm = normalize_hhmm(value)
            if norm is not None:
                updates[key] = norm

    rec = data.get("recurring")
    if isinstance(rec, dict):
        norm = normalize_hhmm(rec.get("time", ""))
        if norm is not None:
            rec["time"] = norm
        for name in ("start_anchor", "end_anchor"):
            anchor = rec.get(name)
            if isinstance(anchor, dict) and "time" in anchor:
                norm = normalize_hhmm(anchor["time"])
                if norm is not None:
                    anchor["time"] = norm

    fire_at = data.get("fire_at")
    if isinstance(fire_at, str):
        # «2026-07-03T16.30» / «2026-07-03 16-30» -> «2026-07-03T16:30».
        # Чинить надо ДО fromisoformat: Python разбирает «T16.30» как
        # 16:00:00.300000 (доли часа) — напоминание молча встало бы не на то время.
        fixed = re.sub(
            r"[T ]\s*(\d{1,2})[.\-](\d{2})\s*$",
            lambda m: f"T{int(m.group(1)):02d}:{m.group(2)}",
            fire_at.strip(),
        )
        try:
            datetime.fromisoformat(fixed)
            data["fire_at"] = fixed
        except ValueError:
            pass
    return data


def validate_action(data: dict, allow_multi: bool = True) -> list[str]:
    """Проверяет ответ модели по схеме; возвращает список ошибок."""
    errors: list[str] = []
    action = data.get("action")

    if action == "multi":
        if not allow_multi:
            return ["multi внутри multi запрещён"]
        subs = data.get("actions")
        if not isinstance(subs, list) or not subs:
            return ["для multi нужен непустой список actions"]
        for i, sub in enumerate(subs):
            if not isinstance(sub, dict):
                errors.append(f"actions[{i}] должен быть объектом")
                continue
            if sub.get("action") in ("multi", "ask_clarification"):
                errors.append(
                    f"actions[{i}]: {sub.get('action')} внутри multi запрещён — "
                    "если нужен уточняющий вопрос, верни один ask_clarification вместо multi"
                )
                continue
            errors.extend(f"actions[{i}]: {e}" for e in validate_action(sub, allow_multi=False))
        return errors

    if action not in ACTIONS:
        return [f"action должен быть одним из {sorted(ACTIONS | {'multi'})}"]

    updates = data.get("context_updates")
    if updates is not None and not isinstance(updates, dict):
        errors.append("context_updates должен быть объектом")

    if action == "save_context":
        if not isinstance(updates, dict) or not updates:
            errors.append("для save_context нужен непустой context_updates")

    if action == "create_reminder":
        if not str(data.get("reminder_text", "")).strip():
            errors.append("для create_reminder нужен reminder_text")
        fire_at = data.get("fire_at")
        expr = str(data.get("time_expression") or "").strip()
        if fire_at:
            try:
                datetime.fromisoformat(str(fire_at))
            except ValueError:
                errors.append(f"fire_at «{fire_at}» не разбирается как YYYY-MM-DDTHH:MM")
        elif not expr:
            errors.append(
                "для create_reminder нужен fire_at (YYYY-MM-DDTHH:MM) или time_expression"
            )

    if action == "create_recurring":
        if not str(data.get("reminder_text", "")).strip():
            errors.append("для create_recurring нужен reminder_text")
        errors.extend(validate_rule(data.get("recurring", {})))

    if action == "ask_clarification":
        if not str(data.get("clarification_question", "")).strip():
            errors.append("для ask_clarification нужен clarification_question")

    if isinstance(updates, dict):
        for key, value in updates.items():
            if key in ("work_start", "work_end", "sleep_start", "sleep_end") and parse_hhmm(str(value)) is None:
                errors.append(f"context_updates.{key} должен быть в формате HH:MM, получено «{value}»")

    return errors


class OllamaClient:
    def __init__(self, base_url: str, model: str, retries: int = 3, timeout: float = 120.0):
        self.base_url = base_url
        self.model = model
        self.retries = retries
        self.timeout = timeout

    async def parse_message(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        """Возвращает провалидированный dict-действие. Бросает LLMUnavailable / LLMBadResponse."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        last_errors: list[str] = []
        for attempt in range(1, self.retries + 1):
            raw = await self._chat(messages)
            log.debug("LLM raw (attempt %d): %s", attempt, raw)
            data = _extract_json(raw)
            if data is not None:
                data = normalize_action(data)
            errors = ["ответ не является JSON-объектом"] if data is None else validate_action(data)
            if not errors:
                return data  # type: ignore[return-value]
            last_errors = errors
            log.warning(
                "Невалидный ответ LLM (попытка %d/%d): %s; ответ: %.300r",
                attempt, self.retries, errors, raw,
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "Твой ответ не прошёл валидацию: "
                + "; ".join(errors)
                + ". Верни исправленный JSON-объект по схеме, без каких-либо пояснений.",
            })
        raise LLMBadResponse("; ".join(last_errors))

    async def _chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0.1},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "")
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise LLMUnavailable(str(e)) from e

    async def healthcheck(self) -> tuple[bool, bool]:
        """(сервер доступен, модель скачана)."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = {m.get("name", "") for m in resp.json().get("models", [])}
        except (httpx.HTTPError, json.JSONDecodeError):
            return False, False
        return True, self.model in models or f"{self.model}:latest" in models
