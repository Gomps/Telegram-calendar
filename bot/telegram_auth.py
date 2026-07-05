"""Проверка подлинности Telegram Mini App initData.

Алгоритм — официальный (https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app):
1. Все поля initData, кроме hash, сортируются по ключу и склеиваются в
   строку "key=value" через \n.
2. secret_key = HMAC_SHA256(key=b"WebAppData", msg=bot_token).
3. Ожидаемый hash = HMAC_SHA256(key=secret_key, msg=data_check_string).hexdigest().
4. Сравнение — только через hmac.compare_digest (защита от timing-атак).

Без этой проверки любой, кто узнает URL мини-аппа, мог бы прислать
произвольный user_id и действовать от чужого имени.
"""

import hashlib
import hmac
import json
import time
from typing import Optional
from urllib.parse import parse_qsl


def validate_init_data(
    init_data: str, bot_token: str, max_age: int = 86400
) -> Optional[dict]:
    """Возвращает разобранные данные (с ключом "user" -> dict) или None.

    max_age — сколько секунд с auth_date считается свежим; 0 — не проверять.
    """
    if not init_data or not bot_token:
        return None
    try:
        pairs = parse_qsl(init_data, strict_parsing=True, keep_blank_values=True)
    except ValueError:
        return None
    data = dict(pairs)
    received_hash = data.pop("hash", None)
    if not received_hash:
        return None

    check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key, check_string.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, received_hash):
        return None

    if max_age > 0:
        try:
            auth_date = int(data.get("auth_date", "0"))
        except ValueError:
            return None
        if time.time() - auth_date > max_age:
            return None

    if "user" in data:
        try:
            data["user"] = json.loads(data["user"])
        except (json.JSONDecodeError, TypeError):
            return None
    return data


def sign_init_data(fields: dict, bot_token: str) -> str:
    """Обратная операция — собрать валидный initData из полей (для тестов/dev)."""
    data = dict(fields)
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    data["hash"] = hmac.new(secret_key, check_string.encode(), hashlib.sha256).hexdigest()
    from urllib.parse import urlencode

    return urlencode(data)
