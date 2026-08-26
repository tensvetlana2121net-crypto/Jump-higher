from jumpbot.config import Settings


def test_admin_ids_are_explicitly_allowlisted() -> None:
    settings = Settings(admin_telegram_ids="123, 456")

    assert settings.is_admin(123)
    assert settings.is_admin(456)
    assert not settings.is_admin(789)


def test_empty_admin_list_denies_access() -> None:
    assert not Settings(admin_telegram_ids="").is_admin(123)
