from io import BytesIO

from PIL import Image

from jumpbot.bot.welcome_card import welcome_card_png


def test_welcome_card_is_valid_telegram_png() -> None:
    payload = welcome_card_png()

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.size == (1080, 1350)
        assert image.mode == "RGB"


def test_welcome_card_generation_is_deterministic() -> None:
    assert welcome_card_png() == welcome_card_png()
