from datetime import UTC, datetime

from jumpbot.services.quota import week_start


def test_week_starts_on_monday() -> None:
    value = week_start(datetime(2026, 8, 15, 12, 0, tzinfo=UTC))
    assert value == datetime(2026, 8, 10, tzinfo=UTC)
