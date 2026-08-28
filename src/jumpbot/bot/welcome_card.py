from __future__ import annotations

from functools import lru_cache
from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((1080 - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)


def _draw_skater(layer: Image.Image) -> None:
    draw = ImageDraw.Draw(layer)
    cyan = (60, 239, 255, 255)
    blue = (54, 117, 255, 255)

    # Rotation trails keep the figure readable while conveying movement.
    draw.arc((170, 95, 910, 655), 202, 510, fill=(39, 145, 255, 190), width=7)
    draw.arc((245, 155, 835, 600), 28, 330, fill=(42, 237, 255, 210), width=5)
    draw.arc((315, 205, 765, 550), 190, 478, fill=(94, 88, 255, 165), width=4)
    draw.polygon(((190, 371), (220, 356), (216, 390)), fill=cyan)
    draw.polygon(((830, 327), (860, 342), (831, 361)), fill=blue)

    # Stylised skater in a compact airborne rotation pose.
    draw.ellipse((500, 157, 577, 234), fill=(232, 250, 255, 255), outline=cyan, width=4)
    draw.polygon(
        ((493, 233), (587, 228), (613, 405), (535, 478), (461, 396)),
        fill=(15, 53, 106, 255),
        outline=cyan,
    )
    draw.line((493, 258, 397, 347, 527, 315), fill=cyan, width=24, joint="curve")
    draw.line((578, 255, 677, 330, 534, 316), fill=blue, width=24, joint="curve")
    draw.line((514, 445, 486, 555, 540, 650), fill=cyan, width=29, joint="curve")
    draw.line((559, 445, 588, 555, 540, 650), fill=blue, width=29, joint="curve")
    draw.line((492, 555, 433, 579), fill=(222, 250, 255, 255), width=11)
    draw.line((587, 555, 646, 579), fill=(222, 250, 255, 255), width=11)


@lru_cache(maxsize=1)
def welcome_card_png() -> bytes:
    """Build the reusable /start welcome card without external assets."""
    image = Image.new("RGB", (1080, 1350), "#030817")
    base = ImageDraw.Draw(image)
    for y in range(image.height):
        blend = y / image.height
        base.line((0, y, image.width, y), fill=(3, int(8 + 8 * blend), int(23 + 31 * blend)))

    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((145, -90, 935, 705), fill=(0, 117, 255, 95))
    glow_draw.ellipse((265, 90, 815, 650), fill=(0, 255, 239, 60))
    glow_draw.ellipse((-180, 960, 330, 1470), fill=(68, 37, 255, 65))
    glow = glow.filter(ImageFilter.GaussianBlur(90))
    image = Image.alpha_composite(image.convert("RGBA"), glow)

    skater_glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    _draw_skater(skater_glow)
    image = Image.alpha_composite(
        image, skater_glow.filter(ImageFilter.GaussianBlur(16))
    )
    _draw_skater(image)

    draw = ImageDraw.Draw(image)
    _centered_text(draw, 48, "JUMP HIGHER", _font(25, bold=True), "#50F1FF")
    _centered_text(
        draw,
        695,
        "НАУЧИЛСЯ ПАДАТЬ,",
        _font(56, bold=True),
        "#F7FBFF",
    )
    _centered_text(draw, 762, "НАУЧИСЬ ВЗЛЕТАТЬ!", _font(56, bold=True), "#66F3FF")

    draw.rounded_rectangle(
        (82, 862, 998, 1248), radius=38, fill="#071A36", outline="#198BD8", width=3
    )
    items = (
        ("01", "Загрузи видео до 10 сек."),
        ("02", '“Предварительное вращение” бот не считает!'),
        ("03", "Используй приложения для анализа статистики."),
    )
    for index, (number, text) in enumerate(items):
        top = 905 + index * 105
        draw.rounded_rectangle(
            (125, top, 191, top + 66), radius=22, fill="#0B315C", outline="#24D9EF", width=2
        )
        draw.text((142, top + 19), number, font=_font(20, bold=True), fill="#6EF7FF")
        draw.text((224, top + 13), text, font=_font(27, bold=True), fill="#E8F3FF")

    _centered_text(
        draw,
        1290,
        "ВИДЕОАНАЛИЗ ПРЫЖКОВ · 2D-ОЦЕНКА",
        _font(19, bold=True),
        "#668FB6",
    )
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()
