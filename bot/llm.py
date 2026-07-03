"""Клиент LLM: любой OpenAI-совместимый API (NVIDIA NIM, Ollama /v1, OpenRouter…).

Валидация JSON-ответа и повторные попытки: при невалидном JSON (или JSON,
не проходящем схему) модель получает свой ответ обратно вместе со списком
ошибок и просьбу исправить — до N попыток.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

import httpx

from .postprocess import is_verbatim_fragment
from .rules import normalize_hhmm, parse_hhmm, validate_rule

log = logging.getLogger(__name__)

ACTIONS = {
    "save_context", "create_reminder", "create_recurring", "create_conditional",
    "set_timezone", "ask_clarification", "not_a_reminder",
}


class LLMUnavailable(Exception):
    """LLM API недоступен или вернул ошибку."""


class LLMBadResponse(Exception):
    """Модель так и не вернула валидный JSON после всех попыток."""


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Убирает блоки размышлений (<think>…</think>) у reasoning-моделей."""
    return THINK_RE.sub("", text).strip()


def _extract_json(text: str) -> Optional[dict]:
    """Достаёт первый JSON-объект из текста (модель может добавить мусор вокруг)."""
    text = _strip_think(text)
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
            if value is None:  # null = удалить ключ из контекста
                continue
            if isinstance(value, list):
                # условные значения: [{"value": "16.30", "when": {...}}, ...]
                for entry in value:
                    if isinstance(entry, dict):
                        norm = normalize_hhmm(entry.get("value", ""))
                        if norm is not None:
                            entry["value"] = norm
            else:
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

    current = data.get("current_time")
    if current is not None:
        norm = normalize_hhmm(current)
        if norm is not None:
            data["current_time"] = norm

    for field in ("fire_at", "check_at"):
        value = data.get(field)
        if isinstance(value, str):
            # «2026-07-03T16.30» / «2026-07-03 16-30» -> «2026-07-03T16:30».
            # Чинить надо ДО fromisoformat: Python разбирает «T16.30» как
            # 16:00:00.300000 (доли часа) — время молча встало бы не то.
            fixed = re.sub(
                r"[T ]\s*(\d{1,2})[.\-](\d{2})\s*$",
                lambda m: f"T{int(m.group(1)):02d}:{m.group(2)}",
                value.strip(),
            )
            try:
                datetime.fromisoformat(fixed)
                data[field] = fixed
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

    if action == "create_conditional":
        if not str(data.get("condition_question", "")).strip():
            errors.append("для create_conditional нужен condition_question")
        if not str(data.get("reminder_text", "")).strip():
            errors.append("для create_conditional нужен reminder_text")
        for iso_field, expr_field in (
            ("check_at", "check_time_expression"),
            ("fire_at", "time_expression"),
        ):
            iso_value = data.get(iso_field)
            if iso_value:
                try:
                    datetime.fromisoformat(str(iso_value))
                except ValueError:
                    errors.append(f"{iso_field} «{iso_value}» не разбирается как YYYY-MM-DDTHH:MM")
            elif not str(data.get(expr_field) or "").strip():
                errors.append(f"для create_conditional нужен {iso_field} или {expr_field}")

    if action == "set_timezone":
        if not any(
            str(data.get(k) or "").strip() for k in ("timezone", "city", "current_time")
        ):
            errors.append("для set_timezone нужен timezone, city или current_time")

    if action == "ask_clarification":
        if not str(data.get("clarification_question", "")).strip():
            errors.append("для ask_clarification нужен clarification_question")

    if isinstance(updates, dict):
        for key, value in updates.items():
            if key not in ("work_start", "work_end", "sleep_start", "sleep_end"):
                continue
            if value is None:  # null = удалить ключ
                continue
            if isinstance(value, list):
                errors.extend(_validate_conditional_value(key, value))
            elif parse_hhmm(str(value)) is None:
                errors.append(f"context_updates.{key} должен быть в формате HH:MM, получено «{value}»")

    return errors


def _validate_conditional_value(key: str, variants: list) -> list[str]:
    """Проверка условного значения контекста: список вариантов с value/when."""
    errors = []
    for i, entry in enumerate(variants):
        where = f"context_updates.{key}[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} должен быть объектом с полями value и when")
            continue
        if parse_hhmm(str(entry.get("value", ""))) is None:
            errors.append(f"{where}.value должен быть в формате HH:MM")
        when = entry.get("when")
        if when is not None:
            if not isinstance(when, dict):
                errors.append(f"{where}.when должен быть объектом")
                continue
            if when.get("day_parity") not in (None, "even", "odd"):
                errors.append(f"{where}.when.day_parity должен быть even | odd")
            days = when.get("days_of_week")
            if days is not None and (
                not isinstance(days, list)
                or not all(isinstance(d, int) and 0 <= d <= 6 for d in days)
            ):
                errors.append(f"{where}.when.days_of_week — список чисел 0..6")
    return errors


SPLIT_TYPES = {"task", "time", "repeat", "condition", "fact", "location_time", "other"}


def validate_blocks(data, source: str) -> tuple[list[dict], list[str]]:
    """Проверка разметки: типы из списка, каждый text — ДОСЛОВНЫЙ кусок сообщения."""
    if not isinstance(data, dict):
        return [], ["ответ не является JSON-объектом"]
    blocks = data.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return [], ["нужен непустой список blocks"]
    out: list[dict] = []
    errors: list[str] = []
    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            errors.append(f"blocks[{i}] должен быть объектом")
            continue
        btype = b.get("type")
        btext = str(b.get("text") or "").strip()
        if btype not in SPLIT_TYPES:
            errors.append(f"blocks[{i}].type «{btype}» не из списка {sorted(SPLIT_TYPES)}")
        elif not btext:
            errors.append(f"blocks[{i}].text пуст")
        elif not is_verbatim_fragment(btext, source):
            errors.append(
                f"blocks[{i}].text «{btext}» не является дословным фрагментом сообщения — "
                "скопируй кусок сообщения без изменений, перефразировать нельзя"
            )
        else:
            out.append({"type": btype, "text": btext})
    return out, errors


class LLMClient:
    """OpenAI-совместимый chat-клиент (NVIDIA NIM, Ollama /v1, OpenRouter…)."""

    def __init__(
        self, base_url: str, model: str, api_key: str = "",
        retries: int = 3, timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.retries = retries
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

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

    async def split_message(self, system_prompt: str, text: str) -> list[dict]:
        """Разбиение сообщения на дословные блоки. Бросает LLMUnavailable/LLMBadResponse."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        last_errors: list[str] = []
        for attempt in range(1, self.retries + 1):
            raw = await self._chat(messages)
            data = _extract_json(raw)
            blocks, errors = ([], ["ответ не является JSON-объектом"]) if data is None \
                else validate_blocks(data, text)
            if not errors:
                return blocks
            last_errors = errors
            log.warning(
                "Невалидная разметка (попытка %d/%d): %s; ответ: %.300r",
                attempt, self.retries, errors, raw,
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "Разметка не прошла проверку: "
                + "; ".join(errors)
                + ". Верни исправленный JSON {\"blocks\": [...]} без пояснений.",
            })
        raise LLMBadResponse("; ".join(last_errors))

    async def _chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if "qwen3" in self.model.lower():
            # у Qwen3/3.5 на NIM размышления уходят в reasoning_content и
            # съедают лимит токенов — для наших структурных задач отключаем
            payload["chat_template_kwargs"] = {"thinking": False}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                )
                resp.raise_for_status()
                message = resp.json()["choices"][0]["message"]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raise LLMUnavailable(str(e)) from e
        # content может отсутствовать (напр., лимит токенов в thinking) —
        # пустая строка провалит валидацию и уйдёт в обычный ретрай
        return message.get("content") or ""

    async def healthcheck(self) -> tuple[bool, bool]:
        """(API доступен, модель существует)."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{self.base_url}/models", headers=self._headers())
                resp.raise_for_status()
                models = {m.get("id", "") for m in resp.json().get("data", [])}
        except (httpx.HTTPError, json.JSONDecodeError, AttributeError):
            return False, False
        # некоторые шлюзы не отдают полный список — отсутствие в списке не фатально
        return True, (not models) or self.model in models or f"{self.model}:latest" in models


# Обратная совместимость со старым именем
OllamaClient = LLMClient
