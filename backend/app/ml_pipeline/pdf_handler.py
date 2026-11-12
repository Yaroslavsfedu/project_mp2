"""Разбор PDF: нативный текстовый слой + рендеринг страниц для просмотрщика."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Минимальный объём «осмысленного» текста на странице, при котором
# считаем PDF текстовым и не запускаем тяжёлый OCR.
NATIVE_TEXT_MIN_CHARS = 40


@dataclass
class PageData:
    page_number: int
    width: int
    height: int
    text: str = ""
    words: list[dict] = field(default_factory=list)
    is_native: bool = False
    png_bytes: bytes | None = None


def _page_words(page: fitz.Page) -> list[dict]:
    words = []
    for idx, (x0, y0, x1, y1, text, block_no, line_no, word_no) in enumerate(page.get_text("words")):
        if not text.strip():
            continue
        words.append(
            {
                "text": text,
                "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "block": block_no,
                "line": line_no,
            }
        )
    return words


def render_page_png(page: fitz.Page, dpi: int = 150) -> tuple[bytes, int, int]:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png"), pix.width, pix.height


def extract_pdf(path: str, dpi: int = 150) -> list[PageData]:
    """Возвращает страницы PDF: текст (если есть), слова с координатами и PNG-рендер.

    Координаты слов пересчитываются в пиксели отрендеренной страницы, чтобы
    фронтенд мог рисовать Bounding Boxes поверх картинки без пересчёта масштаба.
    """
    pages: list[PageData] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            png, width, height = render_page_png(page, dpi=dpi)
            scale = width / page.rect.width if page.rect.width else 1.0
            words = _page_words(page)
            text = page.get_text("text").strip()
            is_native = len(text) >= NATIVE_TEXT_MIN_CHARS
            scaled_words = [
                {
                    "text": w["text"],
                    "bbox": [round(c * scale, 1) for c in w["bbox"]],
                    "block": w["block"],
                    "line": w["line"],
                }
                for w in words
            ] if is_native else []
            pages.append(
                PageData(
                    page_number=index,
                    width=width,
                    height=height,
                    text=text if is_native else "",
                    words=scaled_words,
                    is_native=is_native,
                    png_bytes=png,
                )
            )
    return pages


def pdf_page_count(path: str) -> int:
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001
        return 0
