"""Клиент Ollama: запрос к qwen3.5:4b, валидация JSON-ответа и повторные попытки.

При невалидном JSON (или JSON, не проходящем схему) модель получает свой
ответ обратно вместе со списком ошибок и просьбу исправить — до N попыток.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from .rules import parse_hhmm, validate_rule

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


def validate_action(data: dict) -> list[str]:
    """Проверяет ответ модели по схеме; возвращает список ошибок."""
    errors: list[str] = []
    action = data.get("action")
    if action not in ACTIONS:
        return [f"action должен быть одним из {sorted(ACTIONS)}"]

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
        if not fire_at:
            errors.append("для create_reminder нужен fire_at (YYYY-MM-DDTHH:MM)")
        else:
            try:
                datetime.fromisoformat(str(fire_at))
            except ValueError:
                errors.append(f"fire_at «{fire_at}» не разбирается как YYYY-MM-DDTHH:MM")

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
            errors = ["ответ не является JSON-объектом"] if data is None else validate_action(data)
            if not errors:
                return data  # type: ignore[return-value]
            last_errors = errors
            log.warning("Невалидный ответ LLM (попытка %d/%d): %s", attempt, self.retries, errors)
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

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
