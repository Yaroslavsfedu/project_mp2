"""Рендеринг текстовых документов (DOCX/TXT) в изображение страницы.

Нужен, чтобы Split-Screen верификации работал единообразно для всех форматов:
слева всегда изображение страницы, поверх которого рисуются Bounding Boxes,
справа — текст и сущности. Для DOCX координаты слов известны из раскладки.
"""
from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

PAGE_WIDTH = 1240
MARGIN = 70
LINE_HEIGHT = 34
FONT_SIZE = 22
MAX_LINES_PER_PAGE = 42
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/Library/Fonts/Arial.ttf",
)


@dataclass
class RenderedPage:
    png_bytes: bytes
    width: int
    height: int
    lines: list[dict]


def _font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_pages(text: str) -> list[RenderedPage]:
    font = _font()
    avg_char_width = max(font.getlength("щ") if hasattr(font, "getlength") else 11.0, 8.0)
    chars_per_line = max(int((PAGE_WIDTH - 2 * MARGIN) / avg_char_width), 40)

    wrapped: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(paragraph, width=chars_per_line) or [""])

    pages: list[RenderedPage] = []
    for start in range(0, max(len(wrapped), 1), MAX_LINES_PER_PAGE):
        block = wrapped[start : start + MAX_LINES_PER_PAGE]
        height = MARGIN * 2 + LINE_HEIGHT * max(len(block), 1)
        image = Image.new("RGB", (PAGE_WIDTH, height), "white")
        draw = ImageDraw.Draw(image)
        lines: list[dict] = []
        y = MARGIN
        for line in block:
            if line.strip():
                draw.text((MARGIN, y), line, fill=(20, 20, 20), font=font)
                # координаты отдельных слов — для точной подсветки на фронтенде
                x = MARGIN
                words = []
                for word in line.split(" "):
                    if not word:
                        x += avg_char_width
                        continue
                    width = font.getlength(word) if hasattr(font, "getlength") else len(word) * avg_char_width
                    words.append(
                        {
                            "text": word,
                            "bbox": [round(x, 1), round(y - 4, 1), round(x + width, 1), round(y + LINE_HEIGHT - 6, 1)],
                        }
                    )
                    x += width + avg_char_width * 0.35
                lines.append({"text": line, "bbox": [MARGIN, y - 4, PAGE_WIDTH - MARGIN, y + LINE_HEIGHT - 6], "words": words})
            y += LINE_HEIGHT
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        pages.append(
            RenderedPage(png_bytes=buffer.getvalue(), width=PAGE_WIDTH, height=height, lines=lines)
        )
    return pages
