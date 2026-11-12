"""Оркестрация пайплайна обработки документа.

Этапы (см. раздел 7 спецификации):
    PENDING -> OCR/извлечение текста -> LLM/NLP -> индексация -> COMPLETED

В dev-режиме (QUEUE_BACKEND=inline) пайплайн выполняется как asyncio-задача
внутри API; в prod — тот же код вызывается из Celery-воркера.
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import SessionLocal
from app.ml_pipeline import pdf_handler, renderer
from app.ml_pipeline.llm_service import analyze_document, embed_texts
from app.ml_pipeline.ocr_engine import available_engine, recognize_image, words_to_boxes
from app.ml_pipeline.preprocessor import preprocess
from app.ml_pipeline.text_utils import chunk_text, normalize_text
from app.models.document import Document, DocumentChunk, DocumentStatus, ExtractedData, OcrPage
from app.services.events import bus
from app.services.storage_service import storage

logger = logging.getLogger(__name__)

WORD_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_EXT = {".txt", ".md"}
_task_registry: dict[str, asyncio.Task] = {}


async def publish_status(document: Document, status: str, progress: int, message: str | None = None) -> None:
    await bus.publish(
        {
            "document_id": document.id,
            "filename": document.filename,
            "status": status,
            "progress": progress,
            "message": message,
        }
    )


def _line_boxes_from_words(words: list[dict]) -> list[dict]:
    """Группирует слова в строки — так подсветка на скане читается лучше."""
    if not words:
        return []
    # слова идут в порядке чтения: группируем по близости по вертикали
    lines: list[list[dict]] = []
    for word in words:
        x0, y0, x1, y1 = word["bbox"]
        placed = False
        for line in lines:
            lx0, ly0, lx1, ly1 = line[0]["bbox"]
            center_y = (y0 + y1) / 2
            if ly0 - 2 <= center_y <= ly1 + 2:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    boxes: list[dict] = []
    for index, line in enumerate(lines):
        xs0 = min(w["bbox"][0] for w in line)
        ys0 = min(w["bbox"][1] for w in line)
        xs1 = max(w["bbox"][2] for w in line)
        ys1 = max(w["bbox"][3] for w in line)
        confidences = [w.get("confidence", 1.0) for w in line]
        boxes.append(
            {
                "id": f"l{index}",
                "text": " ".join(w["text"] for w in line),
                "confidence": round(sum(confidences) / len(confidences), 3),
                "bbox": [round(xs0, 1), round(ys0, 1), round(xs1, 1), round(ys1, 1)],
            }
        )
    return boxes


def _read_docx_text(path: str) -> str:
    from docx import Document as DocxDocument

    document = DocxDocument(path)
    blocks: list[str] = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(block for block in blocks if block.strip())


async def _extract_pages(document: Document) -> list[dict]:
    """Возвращает страницы: png, размеры, текст, боксы, confidence, источник."""
    path = await storage.local_path(document.s3_key_original)
    suffix = Path(document.filename).suffix.lower()
    pages: list[dict] = []

    if suffix == ".pdf":
        raw_pages = await asyncio.to_thread(pdf_handler.extract_pdf, path, settings.OCR_DPI)
        if not raw_pages:
            raise ValueError("Не удалось прочитать PDF: документ пуст")
        for raw in raw_pages:
            if raw.is_native:
                pages.append(
                    {
                        "number": raw.page_number,
                        "png": raw.png_bytes,
                        "width": raw.width,
                        "height": raw.height,
                        "text": raw.text,
                        "boxes": _line_boxes_from_words(
                            [{"text": w["text"], "bbox": w["bbox"], "confidence": 0.99} for w in raw.words]
                        ),
                        "confidence": 0.99,
                        "source": "native",
                    }
                )
                continue
            # скан: препроцессинг + OCR
            pages.append(await _ocr_page(raw.page_number, raw.png_bytes, raw.width, raw.height))
        return pages

    if suffix == ".docx":
        text = await asyncio.to_thread(_read_docx_text, path)
        if not text.strip():
            raise ValueError("DOCX не содержит текста")
        rendered = await asyncio.to_thread(renderer.render_text_pages, text)
        for index, page in enumerate(rendered, start=1):
            words: list[dict] = []
            for line in page.lines:
                for word in line["words"]:
                    words.append({**word, "confidence": 0.99})
            pages.append(
                {
                    "number": index,
                    "png": page.png_bytes,
                    "width": page.width,
                    "height": page.height,
                    "text": "\n".join(line["text"] for line in page.lines),
                    "boxes": _line_boxes_from_words(words),
                    "confidence": 0.99,
                    "source": "native",
                }
            )
        return pages

    if suffix in TEXT_EXT:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        rendered = await asyncio.to_thread(renderer.render_text_pages, text)
        for index, page in enumerate(rendered, start=1):
            words = [{**w, "confidence": 0.99} for line in page.lines for w in line["words"]]
            pages.append(
                {
                    "number": index,
                    "png": page.png_bytes,
                    "width": page.width,
                    "height": page.height,
                    "text": "\n".join(line["text"] for line in page.lines),
                    "boxes": _line_boxes_from_words(words),
                    "confidence": 0.99,
                    "source": "native",
                }
            )
        return pages

    # изображения (сканы/фото)
    data = await storage.get(document.s3_key_original)
    from PIL import Image

    import io

    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        png = data if suffix == ".png" else _to_png(image)
    return [await _ocr_page(1, png, width, height)]


def _to_png(image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


async def _ocr_page(number: int, png: bytes, width: int, height: int) -> dict:
    if available_engine() == "none":
        return {
            "number": number,
            "png": png,
            "width": width,
            "height": height,
            "text": "",
            "boxes": [],
            "confidence": 0.0,
            "source": "needs_ocr",
        }

    def _run() -> dict:
        import io

        from PIL import Image

        import numpy as np

        with Image.open(io.BytesIO(png)) as image:
            array = np.array(image.convert("RGB"))
        processed, meta = preprocess(array) if settings.PREPROCESS_ENABLED else (array, {})
        buffer = io.BytesIO()
        Image.fromarray(processed).save(buffer, format="PNG")
        result = recognize_image(buffer.getvalue())
        return {
            "number": number,
            "png": png,
            "width": width,
            "height": height,
            "text": result.text,
            "boxes": words_to_boxes(result.words),
            "confidence": result.avg_confidence,
            "source": result.engine,
            "preprocess": meta,
        }

    page = await asyncio.to_thread(_run)
    page["boxes"] = _line_boxes_from_words(
        [{"text": box["text"], "bbox": box["bbox"], "confidence": box["confidence"]} for box in page["boxes"]]
    )
    return page


async def _set_status(document_id: str, status: str, progress: int, error: str | None = None) -> Document | None:
    async with SessionLocal() as session:
        document = await session.get(Document, document_id)
        if document is None:
            return None
        document.status = status
        document.progress = progress
        document.error_message = error
        await session.commit()
        await session.refresh(document)
        await publish_status(document, status, progress, error)
        return document


async def process_document(document_id: str) -> None:
    """Полный цикл обработки одного документа."""
    try:
        async with SessionLocal() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return
            filename = document.filename

        await _set_status(document_id, DocumentStatus.OCR_PROCESSING, 10)

        # 1. Извлечение страниц (нативный слой / OCR)
        pages = await _extract_pages(document)
        await _set_status(document_id, DocumentStatus.OCR_PROCESSING, 45)

        async with SessionLocal() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return
            document.page_count = len(pages)
            document.mime_type = document.mime_type or (
                mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )
            # очищаем предыдущие результаты (reprocess)
            existing = await session.execute(select(OcrPage).where(OcrPage.document_id == document_id))
            for page in existing.scalars():
                await session.delete(page)
            existing_chunks = await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            for chunk in existing_chunks.scalars():
                await session.delete(chunk)
            await session.commit()

        recognized_text = ""
        for page in pages:
            image_key = f"pages/{document_id}/page-{page['number']}.png"
            await storage.put(image_key, page["png"])
            text = normalize_text(page["text"])
            recognized_text += f"\n\n{text}"
            async with SessionLocal() as session:
                session.add(
                    OcrPage(
                        document_id=document_id,
                        page_number=page["number"],
                        raw_text=text,
                        bounding_boxes=page["boxes"],
                        image_key=image_key,
                        image_width=page["width"],
                        image_height=page["height"],
                        avg_confidence=page["confidence"],
                        source=page["source"],
                    )
                )
                await session.commit()

        recognized_text = recognized_text.strip()
        if not recognized_text:
            raise ValueError(
                "Текст не распознан: для сканов требуется OCR-движок (Tesseract с языком rus) "
                "или загрузите PDF с текстовым слоем / DOCX."
            )

        # 2. Локальный ИИ: коррекция, классификация, NER, резюме
        await _set_status(document_id, DocumentStatus.LLM_PROCESSING, 65)
        outcome = await analyze_document(
            recognized_text, page_hint=f"Документ: {filename}, страниц: {len(pages)}"
        )
        result = outcome.result

        async with SessionLocal() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return
            document.doc_type = result.document_type
            extracted = await session.scalar(
                select(ExtractedData).where(ExtractedData.document_id == document_id)
            )
            if extracted is None:
                extracted = ExtractedData(document_id=document_id)
                session.add(extracted)
            extracted.summary = result.summary
            extracted.entities = result.entities.model_dump()
            extracted.keywords = result.keywords
            extracted.engine = outcome.engine

            # сохраняем исправленный текст постранично
            if result.corrected_text:
                page_rows = await session.execute(
                    select(OcrPage).where(OcrPage.document_id == document_id).order_by(OcrPage.page_number)
                )
                rows = list(page_rows.scalars())
                for row in rows:
                    row.corrected_text = normalize_text(row.raw_text)
            await session.commit()

        # 3. Индексация: чанки + эмбеддинги
        await _set_status(document_id, DocumentStatus.LLM_PROCESSING, 85)
        async with SessionLocal() as session:
            page_rows = await session.execute(
                select(OcrPage).where(OcrPage.document_id == document_id).order_by(OcrPage.page_number)
            )
            rows = list(page_rows.scalars())
        pairs: list[tuple[int, int, str]] = []
        for row in rows:
            page_text = row.corrected_text or row.raw_text
            for index, chunk in enumerate(chunk_text(page_text)):
                pairs.append((row.page_number, index, chunk))
        if not pairs:
            pairs = [(1, 0, recognized_text[:900])]

        vectors, embed_engine = await embed_texts([chunk for _, _, chunk in pairs])
        async with SessionLocal() as session:
            for (page_number, index, chunk), vector in zip(pairs, vectors, strict=False):
                session.add(
                    DocumentChunk(
                        document_id=document_id,
                        page_number=page_number,
                        chunk_index=index,
                        chunk_text=chunk,
                        embedding=vector,
                    )
                )
            await session.commit()
        logger.info("Документ %s проиндексирован (эмбеддинги: %s)", document_id, embed_engine)

        await _set_status(document_id, DocumentStatus.COMPLETED, 100)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка обработки документа %s", document_id)
        await _set_status(document_id, DocumentStatus.FAILED, 100, str(exc)[:900])


def enqueue(document_id: str) -> None:
    """Ставит документ в обработку (inline asyncio-задача либо Celery)."""
    if settings.QUEUE_BACKEND == "celery":
        from app.ml_pipeline.celery_worker import process_document_task

        process_document_task.delay(document_id)  # type: ignore[attr-defined]
        return

    task = asyncio.create_task(process_document(document_id))
    _task_registry[document_id] = task
    task.add_done_callback(lambda _: _task_registry.pop(document_id, None))


def is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False


async def load_document_full(document_id: str) -> Document | None:
    async with SessionLocal() as session:
        return await session.scalar(
            select(Document)
            .options(selectinload(Document.pages), selectinload(Document.extracted))
            .where(Document.id == document_id)
        )
