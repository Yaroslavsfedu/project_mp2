import json
from datetime import datetime, timezone

from app.models.document import Document, ExtractedData, OcrPage
from app.services import export_service


def build_document() -> Document:
    document = Document(
        id="00000000-0000-0000-0000-000000000001",
        user_id="user-1",
        filename="dogovor.pdf",
        s3_key_original="originals/1/dogovor.pdf",
        mime_type="application/pdf",
        file_size=1024,
        page_count=1,
        status="completed",
        doc_type="Договор",
        created_at=datetime(2024, 3, 12, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 3, 12, 10, 5, tzinfo=timezone.utc),
    )
    document.pages = [
        OcrPage(
            page_number=1,
            raw_text="Договор поставки № 47/2024. Стоимость 1 250 000 руб.",
            avg_confidence=0.99,
            source="native",
            bounding_boxes=[{"id": "l0", "text": "Договор", "confidence": 0.99, "bbox": [1, 2, 3, 4]}],
        )
    ]
    document.extracted = ExtractedData(
        document_id=document.id,
        summary="Договор поставки оборудования между ООО «ТехноСервис» и ЮФУ.",
        entities={
            "persons": ["Семенов В. Е."],
            "organizations": ["ООО «ТехноСервис»", "ФГБОУ ВО «ЮФУ»"],
            "dates": ["12.03.2024"],
            "amounts": ["1 250 000 руб."],
            "inn_ogrn": ["ИНН 6165048312"],
            "document_number": "47/2024",
            "addresses": [],
        },
        keywords=["договор", "поставка"],
        is_verified=True,
        engine="fallback",
    )
    return document


def test_export_json_contains_entities_and_boxes():
    content, media_type, extension = export_service.export_json(build_document())
    payload = json.loads(content)
    assert payload["entities"]["document_number"] == "47/2024"
    assert payload["pages"][0]["bounding_boxes"][0]["text"] == "Договор"
    assert payload["is_verified"] is True
    assert media_type == "application/json"
    assert extension == ".json"


def test_export_markdown_and_txt():
    document = build_document()
    md, _, _ = export_service.export_markdown(document)
    assert "## Резюме" in md.decode()
    assert "1 250 000" in md.decode()

    txt, _, _ = export_service.export_txt(document)
    assert "Договор поставки" in txt.decode()


def test_export_docx_is_zip_archive():
    content, media_type, extension = export_service.export_docx(build_document())
    assert content[:2] == b"PK"
    assert extension == ".docx"
    assert "wordprocessingml" in media_type


def test_export_csv_has_bom_and_values():
    content, _, _ = export_service.export_csv(build_document())
    text = content.decode("utf-8-sig")
    assert "Договор" in text
    assert "1 250 000 руб." in text


def test_batch_export_xlsx():
    content, _, extension = export_service.export_batch_xlsx([build_document()])
    assert content[:2] == b"PK"
    assert extension == ".xlsx"
