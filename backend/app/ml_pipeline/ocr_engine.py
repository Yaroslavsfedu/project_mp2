"""Выбор и вызов OCR-движка.

Цепочка (OCR_ENGINE=auto):
1. ``tesseract`` — быстрый CPU-OCR с координатами слов (rus+eng);
2. ``ollama-vision`` — локальная VLM (qwen2-vl / llava) для сканов и рукописи,
   когда Tesseract не установлен, а GPU доступна;
3. ``none`` — страница помечается как требующая OCR (текст и сущности не извлекаются).

OCR_ENGINE можно зафиксировать: ``tesseract`` | ``ollama`` | ``native``.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - зависит от окружения
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
    words: list[OcrWord] = field(default_factory=list)
    avg_confidence: float = 0.0
    engine: str = "none"
    text: str = ""
    boxes: list[dict] = field(default_factory=list)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None and pytesseract is not None


async def resolve_engine() -> str:
    """Активный OCR-движок с учётом настроек и доступности моделей."""
    from app.ml_pipeline.llm_service import vision_available

    configured = settings.OCR_ENGINE.lower()
    if configured == "native":
        return "native"
    if configured == "tesseract":
        return "tesseract" if tesseract_available() else "none"
    if configured in ("ollama", "vision", "vlm"):
        return "ollama-vision" if await vision_available() else "none"
    # auto
    if tesseract_available():
        return "tesseract"
    if await vision_available():
        return "ollama-vision"
    return "none"


def available_engine() -> str:
    """Быстрая синхронная проверка (без обращения к Ollama) — для статуса в API."""
    if settings.OCR_ENGINE.lower() == "native":
        return "native"
    return "tesseract" if tesseract_available() else "none"


def recognize_image(image_bytes: bytes) -> OcrPageResult:
    """Распознаёт изображение через Tesseract и возвращает слова с боксами."""
    if not tesseract_available():
        logger.warning("Tesseract недоступен: страница будет помечена как требующая OCR")
        return OcrPageResult(engine="none")

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

    if not words:
        # иногда Tesseract не отдаёт уровни слов — берём текст страницы
        raw_text = pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGES).strip()
        return OcrPageResult(words=[], avg_confidence=0.6 if raw_text else 0.0, engine="tesseract", text=raw_text)

    avg = round(sum(w.confidence for w in words) / len(words), 3)
    return OcrPageResult(words=words, avg_confidence=avg, engine="tesseract")


def detect_orientation(image) -> int:
    """Поворот страницы, при котором текст читается прямо: 0, 90, 180 или 270 градусов.

    Использует Tesseract OSD (Orientation and Script Detection). Если Tesseract
    не установлен или текста недостаточно для уверенного решения — возвращает 0.
    Угол — это поворот, который нужно компенсировать (по часовой стрелке).
    """
    if not tesseract_available():
        return 0
    try:
        osd = pytesseract.image_to_osd(image, config="--psm 0")
    except Exception as exc:  # noqa: BLE001 - мало текста / нет языковых данных OSD
        logger.info("OSD не определил ориентацию: %s", exc)
        return 0
    values: dict[str, str] = {}
    for line in osd.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip()
    try:
        rotate = int(float(values.get("Rotate", "0")))
        confidence = float(values.get("Orientation confidence", "0"))
    except ValueError:
        return 0
    if rotate in (90, 180, 270) and confidence >= 1.5:
        return rotate
    return 0


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


def text_to_line_boxes(text: str) -> list[dict]:
    """Псевдо-боксы для VLM/текстового OCR: по одной рамке на строку.

    VLM не возвращает координаты, поэтому используется «мягкая» разметка
    без подсветки на скане, но с сохранением структуры строк.
    """
    boxes = []
    for index, line in enumerate(line for line in text.splitlines() if line.strip()):
        boxes.append(
            {
                "id": f"t{index}",
                "text": line.strip(),
                "confidence": 0.0,
                "bbox": [0, 0, 0, 0],
            }
        )
    return boxes
