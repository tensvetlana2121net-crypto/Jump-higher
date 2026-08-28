from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

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


def _reference_skater() -> Image.Image:
    source_path = Path(__file__).resolve().parents[1] / "assets" / "welcome_skater_reference.webp"
    with Image.open(source_path) as source:
        # The upper "swallow" is used exactly as supplied; colour thresholding
        # removes the pale background and the disconnected diagram labels.
        crop = source.convert("RGB").crop((165, 32, 385, 198))
    pixels = crop.load()
    mask = Image.new("L", crop.size, 0)
    mask_pixels = mask.load()
    for y in range(crop.height):
        for x in range(crop.width):
            red, green, blue = pixels[x, y]
            if blue > red + 5 and blue > green + 5 and red < 145:
                mask_pixels[x, y] = 255

    # Keep only the largest connected component: the skater, not nearby words.
    visited: set[tuple[int, int]] = set()
    largest: list[tuple[int, int]] = []
    for y in range(mask.height):
        for x in range(mask.width):
            if mask_pixels[x, y] == 0 or (x, y) in visited:
                continue
            component: list[tuple[int, int]] = []
            stack = [(x, y)]
            visited.add((x, y))
            while stack:
                current_x, current_y = stack.pop()
                component.append((current_x, current_y))
                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if (
                        0 <= next_x < mask.width
                        and 0 <= next_y < mask.height
                        and mask_pixels[next_x, next_y]
                        and (next_x, next_y) not in visited
                    ):
                        visited.add((next_x, next_y))
                        stack.append((next_x, next_y))
            if len(component) > len(largest):
                largest = component

    clean_mask = Image.new("L", crop.size, 0)
    clean_pixels = clean_mask.load()
    for x, y in largest:
        clean_pixels[x, y] = 255
    bounds = clean_mask.getbbox()
    if bounds is None:
        raise RuntimeError("Reference skater could not be extracted")
    clean_mask = clean_mask.crop(bounds).resize((620, 470), Image.Resampling.LANCZOS)
    skater = Image.new("RGBA", clean_mask.size, (56, 225, 255, 0))
    skater.putalpha(clean_mask)
    return skater


def _draw_skater(layer: Image.Image) -> None:
    draw = ImageDraw.Draw(layer)
    draw.arc((170, 95, 910, 655), 202, 510, fill=(39, 145, 255, 190), width=7)
    draw.arc((245, 155, 835, 600), 28, 330, fill=(42, 237, 255, 210), width=5)
    draw.arc((315, 205, 765, 550), 190, 478, fill=(94, 88, 255, 165), width=4)
    draw.polygon(((190, 371), (220, 356), (216, 390)), fill=(60, 239, 255, 255))
    draw.polygon(((830, 327), (860, 342), (831, 361)), fill=(54, 117, 255, 255))
    layer.alpha_composite(_reference_skater(), (230, 118))


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
        (82, 852, 998, 1295), radius=38, fill="#071A36", outline="#198BD8", width=3
    )
    items = (
        ("01", ("Загрузи видео до 10 сек.",)),
        ("02", ("“Предварительное вращение”", "бот не считает!")),
        ("03", ("Используй приложения", "для анализа статистики.")),
    )
    row_tops = (890, 1000, 1150)
    for top, (number, lines) in zip(row_tops, items, strict=True):
        draw.rounded_rectangle(
            (125, top, 197, top + 72), radius=23, fill="#0B315C", outline="#24D9EF", width=2
        )
        draw.text((143, top + 20), number, font=_font(22, bold=True), fill="#6EF7FF")
        for line_index, text in enumerate(lines):
            draw.text(
                (226, top - 2 + line_index * 42),
                text,
                font=_font(34, bold=True),
                fill="#FFFFFF",
            )
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()
