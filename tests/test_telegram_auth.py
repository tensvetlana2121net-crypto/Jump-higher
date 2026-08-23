import hashlib
import hmac
import json
from urllib.parse import urlencode

import pytest

from jumpbot.api.telegram_auth import validate_init_data


def signed_init_data(
    bot_token: str,
    auth_date: int = 1_700_000_000,
    user_id: int = 42,
) -> str:
    fields = {
        "auth_date": str(auth_date),
        "query_id": "AAHdF6IQAAAAAN0XohDhrOrc",
        "user": json.dumps(
            {
                "id": user_id,
                "first_name": "Анна",
                "username": "anna_jump",
                "language_code": "ru",
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    }
    check_string = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


def test_valid_init_data_returns_identity() -> None:
    payload = signed_init_data("123456:test-token")

    identity = validate_init_data(payload, "123456:test-token", 3600, now=1_700_000_100)

    assert identity.id == 42
    assert identity.first_name == "Анна"
    assert identity.username == "anna_jump"


def test_tampered_init_data_is_rejected() -> None:
    payload = signed_init_data("123456:test-token").replace("anna_jump", "other")

    with pytest.raises(ValueError, match="signature"):
        validate_init_data(payload, "123456:test-token", 3600, now=1_700_000_100)


def test_expired_init_data_is_rejected() -> None:
    payload = signed_init_data("123456:test-token")

    with pytest.raises(ValueError, match="expired"):
        validate_init_data(payload, "123456:test-token", 60, now=1_700_000_100)


def test_future_init_data_is_rejected() -> None:
    payload = signed_init_data("123456:test-token", auth_date=1_700_000_100)

    with pytest.raises(ValueError, match="expired"):
        validate_init_data(payload, "123456:test-token", 3600, now=1_700_000_000)


def test_missing_user_is_rejected() -> None:
    fields = {"auth_date": "1700000000"}
    check_string = "auth_date=1700000000"
    secret = hmac.new(b"WebAppData", b"token", hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()

    with pytest.raises(ValueError, match="user data"):
        validate_init_data(urlencode(fields), "token", 3600, now=1_700_000_100)
