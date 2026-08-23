import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl

from fastapi import Header, HTTPException, status

from jumpbot.config import get_settings


@dataclass(frozen=True)
class TelegramIdentity:
    id: int
    username: str | None
    first_name: str | None
    language_code: str


def validate_init_data(
    init_data: str,
    bot_token: str,
    max_age_seconds: int,
    now: int | None = None,
) -> TelegramIdentity:
    if not init_data or not bot_token:
        raise ValueError("Telegram authentication is unavailable")
    try:
        fields = dict(parse_qsl(init_data, keep_blank_values=True, strict_parsing=True))
        supplied_hash = fields.pop("hash")
        auth_date = int(fields["auth_date"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid Telegram authentication data") from exc

    current_time = int(time.time()) if now is None else now
    if auth_date > current_time + 30 or current_time - auth_date > max_age_seconds:
        raise ValueError("Telegram authentication data has expired")

    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise ValueError("Invalid Telegram authentication signature")

    try:
        user = json.loads(fields["user"])
        user_id = int(user["id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Telegram user data is missing") from exc
    if user_id <= 0:
        raise ValueError("Telegram user data is invalid")
    return TelegramIdentity(
        id=user_id,
        username=user.get("username"),
        first_name=user.get("first_name"),
        language_code=user.get("language_code") or "ru",
    )


async def require_telegram_user(
    authorization: str | None = Header(default=None),
) -> TelegramIdentity:
    if not authorization or not authorization.startswith("tma "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Telegram authentication required",
        )
    settings = get_settings()
    try:
        return validate_init_data(
            authorization[4:],
            settings.telegram_bot_token,
            settings.telegram_init_data_ttl_seconds,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
