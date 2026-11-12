"""Обёртка над OCR-движками.

Порядок выбора (OCR_ENGINE=auto):
1. ``tesseract`` — если в системе доступен бинарник с языками rus+eng;
2. ``none`` — если OCR недоступен, страница помечается как требующая
   ручной верификации (в этом режиме приложение остаётся работоспособным:
   PDF с текстовым слоем и DOCX обрабатываются без OCR).

Структура класса позволяет подключить PaddleOCR/EasyOCR без изменения
остального пайплайна — достаточно реализовать ``_run`` и указать движок.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import pytesseract
    from PIL import Image
except Exception:  # noqa: BLE001
    pytesseract = None
    Image = None


@dataclass
class OcrWord:
    text: str
    bbox: list[float]
    confidence: float


@dataclass
class OcrPageResult:
    words: list[OcrWord]
    avg_confidence: float
    engine: str

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def _cppm_available() -> bool:
    return shutil.which("tesseract") is not None and pytesseract is not None


def available_engine() -> str:
    if settings.OCR_ENGINE == "native":
        return "native"
    if settings.OCR_ENGINE == "tesseract":
        return "tesseract" if _cppm_available() else "none"
    return "tesseract" if _cppm_available() else "none"


def recognize_image(image_bytes: bytes) -> OcrPageResult:
    """Распознаёт изображение (PNG/JPEG-байты) и возвращает слова с боксами."""
    engine = available_engine()
    if engine != "tesseract":
        logger.warning("OCR-движок недоступен: страница будет помечена как требующая OCR")
        return OcrPageResult(words=[], avg_confidence=0.0, engine="none")

    import io

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = pytesseract.image_to_data(
        image,
        lang=settings.OCR_LANGUAGES,
        output_type=pytesseract.Output.DICT,
        config="--psm 3",
    )
    words: list[OcrWord] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < 0:
            continue
        x, y = float(data["left"][i]), float(data["top"][i])
        w, h = float(data["width"][i]), float(data["height"][i])
        words.append(OcrWord(text=text, bbox=[x, y, x + w, y + h], confidence=round(conf / 100, 3)))

    avg = round(sum(w.confidence for w in words) / len(words), 3) if words else 0.0
    return OcrPageResult(words=words, avg_confidence=avg, engine="tesseract")


def words_to_boxes(words: list[OcrWord], start_index: int = 0) -> list[dict]:
    return [
        {
            "id": f"w{start_index + i}",
            "text": w.text,
            "confidence": w.confidence,
            "bbox": [round(v, 1) for v in w.bbox],
        }
        for i, w in enumerate(words)
    ]
