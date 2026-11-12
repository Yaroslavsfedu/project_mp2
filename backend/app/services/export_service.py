"""Экспорт результатов обработки: JSON, CSV, DOCX, Markdown, TXT (+XLSX батч)."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from docx import Document as DocxDocument
from openpyxl import Workbook

from app.models.document import Document

ENTITY_LABELS = {
    "persons": "ФИО",
    "organizations": "Организации",
    "dates": "Даты",
    "amounts": "Суммы",
    "inn_ogrn": "Реквизиты (ИНН/КПП/ОГРН)",
    "document_number": "Номер документа",
    "addresses": "Адреса",
}


def _document_payload(document: Document) -> dict:
    pages = sorted(document.pages, key=lambda p: p.page_number)
    extracted = document.extracted
    return {
        "id": document.id,
        "filename": document.filename,
        "mime_type": document.mime_type,
        "file_size": document.file_size,
        "page_count": document.page_count,
        "status": document.status,
        "doc_type": document.doc_type,
        "created_at": document.created_at.isoformat(),
        "updated_at": document.updated_at.isoformat(),
        "summary": extracted.summary if extracted else "",
        "keywords": extracted.keywords if extracted else [],
        "entities": extracted.entities if extracted else {},
        "is_verified": bool(extracted.is_verified) if extracted else False,
        "analysis_engine": extracted.engine if extracted else None,
        "pages": [
            {
                "page_number": page.page_number,
                "text": page.corrected_text or page.raw_text,
                "avg_confidence": page.avg_confidence,
                "source": page.source,
                "bounding_boxes": page.bounding_boxes,
            }
            for page in pages
        ],
    }


def export_json(document: Document) -> tuple[bytes, str, str]:
    payload = json.dumps(_document_payload(document), ensure_ascii=False, indent=2)
    return payload.encode("utf-8"), "application/json", ".json"


def export_txt(document: Document) -> tuple[bytes, str, str]:
    parts = []
    for page in sorted(document.pages, key=lambda p: p.page_number):
        parts.append(f"--- Страница {page.page_number} ---\n{page.corrected_text or page.raw_text}")
    return "\n\n".join(parts).encode("utf-8"), "text/plain; charset=utf-8", ".txt"


def export_markdown(document: Document) -> tuple[bytes, str, str]:
    extracted = document.extracted
    lines = [f"# {document.filename}", ""]
    lines.append(f"**Тип документа:** {document.doc_type or 'не определён'}  ")
    lines.append(f"**Дата загрузки:** {document.created_at:%d.%m.%Y %H:%M}  ")
    lines.append(f"**Страниц:** {document.page_count}  ")
    lines.append(f"**Статус:** {document.status}")
    if extracted and extracted.summary:
        lines += ["", "## Резюме", extracted.summary]
    if extracted and extracted.entities:
        lines += ["", "## Извлечённые сущности", ""]
        for key, label in ENTITY_LABELS.items():
            value = extracted.entities.get(key)
            if not value:
                continue
            rendered = value if isinstance(value, str) else ", ".join(value)
            lines.append(f"- **{label}:** {rendered}")
    if extracted and extracted.keywords:
        lines += ["", f"**Ключевые слова:** {', '.join(extracted.keywords)}"]
    lines += ["", "## Текст документа", ""]
    for page in sorted(document.pages, key=lambda p: p.page_number):
        lines.append(f"### Страница {page.page_number}")
        lines.append((page.corrected_text or page.raw_text) or "_(текст отсутствует)_")
        lines.append("")
    return "\n".join(lines).encode("utf-8"), "text/markdown; charset=utf-8", ".md"


def export_docx(document: Document) -> tuple[bytes, str, str]:
    doc = DocxDocument()
    doc.add_heading(document.filename, level=0)
    extracted = document.extracted

    meta = doc.add_paragraph()
    meta.add_run("Тип документа: ").bold = True
    meta.add_run(f"{document.doc_type or 'не определён'}\n")
    meta.add_run("Дата загрузки: ").bold = True
    meta.add_run(f"{document.created_at:%d.%m.%Y %H:%M}\n")
    meta.add_run("Статус: ").bold = True
    meta.add_run(document.status)

    if extracted and extracted.summary:
        doc.add_heading("Резюме", level=1)
        doc.add_paragraph(extracted.summary)

    if extracted and extracted.entities:
        doc.add_heading("Извлечённые сущности", level=1)
        table = doc.add_table(rows=1, cols=2)
        table.style = "Light Grid Accent 1"
        table.rows[0].cells[0].text = "Поле"
        table.rows[0].cells[1].text = "Значение"
        for key, label in ENTITY_LABELS.items():
            value = extracted.entities.get(key)
            if not value:
                continue
            cells = table.add_row().cells
            cells[0].text = label
            cells[1].text = value if isinstance(value, str) else "; ".join(value)

    doc.add_heading("Текст документа", level=1)
    for page in sorted(document.pages, key=lambda p: p.page_number):
        doc.add_heading(f"Страница {page.page_number}", level=2)
        doc.add_paragraph(page.corrected_text or page.raw_text or "")

    buffer = io.BytesIO()
    doc.save(buffer)
    return (
        buffer.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    )


def export_csv(document: Document) -> tuple[bytes, str, str]:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    payload = _document_payload(document)
    writer.writerow(["Документ", payload["filename"]])
    writer.writerow(["Тип", payload["doc_type"] or ""])
    writer.writerow(["Статус", payload["status"]])
    writer.writerow(["Страниц", payload["page_count"]])
    writer.writerow(["Резюме", payload["summary"]])
    writer.writerow([])
    writer.writerow(["Сущность", "Значение"])
    for key, label in ENTITY_LABELS.items():
        value = payload["entities"].get(key)
        if not value:
            continue
        writer.writerow([label, value if isinstance(value, str) else "; ".join(value)])
    writer.writerow([])
    writer.writerow(["Страница", "Текст"])
    for page in payload["pages"]:
        writer.writerow([page["page_number"], page["text"]])
    # BOM для корректного открытия в Excel
    return ("\ufeff" + buffer.getvalue()).encode("utf-8"), "text/csv; charset=utf-8", ".csv"


def export_batch_csv(documents: list[Document]) -> tuple[bytes, str, str]:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(
        ["Файл", "Тип", "Статус", "Страниц", "Дата загрузки", "Резюме", "ФИО", "Организации", "Суммы", "ИНН/ОГРН", "Проверен"]
    )
    for document in documents:
        extracted = document.extracted
        entities = extracted.entities if extracted else {}
        writer.writerow(
            [
                document.filename,
                document.doc_type or "",
                document.status,
                document.page_count,
                document.created_at.strftime("%d.%m.%Y %H:%M"),
                (extracted.summary if extracted else "").replace("\n", " "),
                "; ".join(entities.get("persons", [])),
                "; ".join(entities.get("organizations", [])),
                "; ".join(entities.get("amounts", [])),
                "; ".join(entities.get("inn_ogrn", [])),
                "да" if extracted and extracted.is_verified else "нет",
            ]
        )
    return ("\ufeff" + buffer.getvalue()).encode("utf-8"), "text/csv; charset=utf-8", ".csv"


def export_batch_xlsx(documents: list[Document]) -> tuple[bytes, str, str]:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Документы"
    headers = [
        "Файл", "Тип", "Статус", "Страниц", "Дата загрузки", "Резюме",
        "ФИО", "Организации", "Суммы", "ИНН/ОГРН", "Проверен",
    ]
    sheet.append(headers)
    for document in documents:
        extracted = document.extracted
        entities = extracted.entities if extracted else {}
        sheet.append(
            [
                document.filename,
                document.doc_type or "",
                document.status,
                document.page_count,
                document.created_at.strftime("%d.%m.%Y %H:%M"),
                extracted.summary if extracted else "",
                "; ".join(entities.get("persons", [])),
                "; ".join(entities.get("organizations", [])),
                "; ".join(entities.get("amounts", [])),
                "; ".join(entities.get("inn_ogrn", [])),
                "да" if extracted and extracted.is_verified else "нет",
            ]
        )
    for column, width in zip("ABCDEFGHIJK", (34, 20, 16, 10, 18, 60, 28, 34, 22, 26, 11), strict=False):
        sheet.column_dimensions[column].width = width
    buffer = io.BytesIO()
    workbook.save(buffer)
    return (
        buffer.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    )


EXPORTERS = {
    "json": export_json,
    "txt": export_txt,
    "md": export_markdown,
    "docx": export_docx,
    "csv": export_csv,
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")
