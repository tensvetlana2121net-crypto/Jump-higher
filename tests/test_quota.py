from datetime import UTC, datetime
from unittest.mock import AsyncMock

from jumpbot.config import get_settings
from jumpbot.services.quota import consume_analysis, week_start


def test_week_starts_on_monday() -> None:
    value = week_start(datetime(2026, 8, 15, 12, 0, tzinfo=UTC))
    assert value == datetime(2026, 8, 10, tzinfo=UTC)


async def test_disabled_quota_allows_analysis_without_database_access(monkeypatch) -> None:
    monkeypatch.setenv("FREE_QUOTA_ENABLED", "false")
    get_settings.cache_clear()
    session = AsyncMock()

    try:
        assert await consume_analysis(session, user_id=1) is True
        session.scalar.assert_not_awaited()
    finally:
        get_settings.cache_clear()
