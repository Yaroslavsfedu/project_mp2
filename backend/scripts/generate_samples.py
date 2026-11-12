"""Генерация демонстрационного набора документов (PDF/DOCX/TXT).

Скрипт создаёт документы с настоящим текстовым слоем — их можно обрабатывать
без OCR-движка и демонстрировать весь пайплайн (NER, классификация, поиск,
верификация с Bounding Boxes).

Запуск:  python -m scripts.generate_samples [--out samples] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from PIL import Image, ImageDraw, ImageFilter

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PAGE_W, PAGE_H = 595, 842  # A4, пункты
MARGIN_X, MARGIN_Y = 60, 60
LINE_GAP = 6


def _font(size: float, bold: bool = False) -> fitz.Font:
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return fitz.Font(fontfile=path)
    except Exception:  # noqa: BLE001
        return fitz.Font("helv")


def _wrap(text: str, font: fitz.Font, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.text_length(candidate, fontsize=size) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_pdf(path: Path, blocks: list[tuple[str, str]]) -> Path:
    """blocks: список (стиль, текст), стиль из h1 | h2 | p | small."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN_Y
    usable = PAGE_W - 2 * MARGIN_X

    def new_page() -> fitz.Page:
        nonlocal y
        p = doc.new_page(width=PAGE_W, height=PAGE_H)
        y = MARGIN_Y
        return p

    for style, text in blocks:
        size, bold, gap_after = {
            "h1": (16, True, 14),
            "h2": (13, True, 8),
            "p": (10.5, False, 8),
            "small": (9, False, 6),
        }[style]
        font = _font(size, bold)
        fontname = "F-bold" if bold else "F-reg"
        for line in _wrap(text, font, size, usable):
            if y > PAGE_H - MARGIN_Y - 20:
                page = new_page()
            page.insert_text(
                (MARGIN_X, y + size),
                line,
                fontsize=size,
                fontname=fontname,
                fontfile=FONT_BOLD if bold else FONT_REGULAR,
                color=(0.08, 0.08, 0.1) if style != "small" else (0.35, 0.35, 0.38),
            )
            y += size + LINE_GAP
        y += gap_after

    try:
        doc.subset_fonts()  # уменьшает размер файла: встраиваются только нужные глифы
    except Exception:  # noqa: BLE001
        pass
    doc.save(path, deflate=True, garbage=3)
    doc.close()
    return path


CONTRACT_BLOCKS = [
    ("h1", "ДОГОВОР ПОСТАВКИ № 47/2024"),
    ("small", "г. Ростов-на-Дону                                                                        12 марта 2024 г."),
    ("p", ""),
    ("p", "Общество с ограниченной ответственностью «ТехноСервис», именуемое в дальнейшем «Поставщик», "
          "в лице генерального директора Семенова Владимира Евгеньевича, действующего на основании Устава, "
          "с одной стороны, и Федеральное государственное бюджетное образовательное учреждение высшего "
          "образования «Южный федеральный университет», именуемое в дальнейшем «Заказчик», "
          "в лице проректора по цифровому развитию Кузнецовой Анны Сергеевны, действующей на основании "
          "доверенности № 15-Д от 09.01.2024, с другой стороны, заключили настоящий Договор о нижеследующем."),
    ("h2", "1. Предмет договора"),
    ("p", "1.1. Поставщик обязуется поставить Заказчику вычислительную технику и периферийное оборудование "
          "для оснащения лаборатории интеллектуальной обработки документов Института компьютерных технологий "
          "и информационной безопасности, а Заказчик обязуется принять и оплатить товар."),
    ("p", "1.2. Наименование, количество и технические характеристики товара определяются Спецификацией "
          "(Приложение № 1), являющейся неотъемлемой частью настоящего Договора."),
    ("h2", "2. Цена договора и порядок расчётов"),
    ("p", "2.1. Общая стоимость товара по настоящему Договору составляет 1 250 000 руб. 00 коп., "
          "включая НДС 20%."),
    ("p", "2.2. Оплата производится безналичным перечислением на расчётный счёт Поставщика в течение "
          "30 календарных дней после подписания акта приёма-передачи товара."),
    ("p", "2.3. Реквизиты сторон: Поставщик — ИНН 6165048312, КПП 616501001, ОГРН 1156196048721, "
          "р/с 40702810400000012345. Заказчик — ИНН 6166052167, КПП 616601001, ОГРН 1026102997202."),
    ("h2", "3. Сроки поставки"),
    ("p", "3.1. Поставка осуществляется в течение 45 календарных дней с момента подписания Договора, "
          "то есть не позднее 26 апреля 2024 г."),
    ("p", "3.2. Адрес поставки: г. Ростов-на-Дону, ул. Чехова, д. 2, корпус «Б», ауд. 209."),
    ("h2", "4. Ответственность сторон"),
    ("p", "4.1. За нарушение сроков поставки Поставщик уплачивает неустойку в размере 0,1% от стоимости "
          "непоставленного товара за каждый день просрочки."),
    ("p", "4.2. За нарушение сроков оплаты Заказчик уплачивает пеню в размере 0,05% от суммы "
          "задолженности за каждый день просрочки."),
    ("h2", "5. Реквизиты и подписи сторон"),
    ("p", "Поставщик: ООО «ТехноСервис», 344002, г. Ростов-на-Дону, пр. Будённовский, д. 21, офис 4. "
          "Генеральный директор ___________ / Семенов В. Е."),
    ("p", "Заказчик: ФГБОУ ВО «ЮФУ», 344006, г. Ростов-на-Дону, ул. Большая Садовая, д. 105/42. "
          "Проректор ___________ / Кузнецова А. С."),
]

ORDER_BLOCKS = [
    ("h1", "ПРИКАЗ № 112-ОД"),
    ("small", "12 февраля 2024 г.                                                                    г. Ростов-на-Дону"),
    ("p", ""),
    ("p", "Об организации работы лаборатории интеллектуальной обработки документов"),
    ("p", "В соответствии с приказом Министерства науки и высшего образования Российской Федерации "
          "и в целях развития научно-образовательного потенциала университета"),
    ("h2", "ПРИКАЗЫВАЮ:"),
    ("p", "1. Назначить Семенова Владимира Евгеньевича руководителем лаборатории интеллектуальной "
          "обработки документов с 12 февраля 2024 г. с окладом согласно штатному расписанию."),
    ("p", "2. Создать рабочую группу внедрения платформы оцифровки документов в составе: "
          "Кузнецова Анна Сергеевна, Гончаров Дмитрий Петрович, Ковалёва Елена Игоревна."),
    ("p", "3. Утвердить план-график внедрения на период с 12.02.2024 по 30.06.2024."),
    ("p", "4. Контроль за исполнением настоящего приказа возложить на проректора по цифровому развитию "
          "Семенова Владимира Евгеньевича."),
    ("p", ""),
    ("p", "Ректор ФГБОУ ВО «ЮФУ»                                                    ____________ / Боровская М. А."),
]

INVOICE_BLOCKS = [
    ("h1", "СЧЁТ НА ОПЛАТУ № 208 от 15 марта 2024 г."),
    ("p", ""),
    ("p", "Поставщик: ООО «ТехноСервис», ИНН 6165048312, КПП 616501001, "
          "344002, г. Ростов-на-Дону, пр. Будённовский, д. 21, офис 4."),
    ("p", "Покупатель: ФГБОУ ВО «ЮФУ», ИНН 6166052167, КПП 616601001, "
          "344006, г. Ростов-на-Дону, ул. Большая Садовая, д. 105/42."),
    ("h2", "Реквизиты для оплаты"),
    ("p", "Расчётный счёт 40702810400000012345, БИК 046015001, "
          "к/с 30101810600000000002 в Ростовском отделении ПАО «Сбербанк»."),
    ("h2", "Состав счёта"),
    ("p", "1. Ноутбук ThinkPad L14 — 10 шт. × 89 000 руб. = 890 000 руб."),
    ("p", "2. Монитор 27\" Dell P2723DE — 10 шт. × 28 000 руб. = 280 000 руб."),
    ("p", "3. МФУ лазерное Kyocera — 2 шт. × 40 000 руб. = 80 000 руб."),
    ("p", ""),
    ("p", "Итого: 1 250 000 руб. 00 коп. В том числе НДС 20%: 208 333 руб. 33 коп."),
    ("p", ""),
    ("small", "Счёт актуален до 30 марта 2024 г. Оплата производится на основании договора поставки № 47/2024."),
    ("p", "Руководитель ___________ / Семенов В. Е."),
]

ACT_BLOCKS = [
    ("h1", "АКТ ПРИЁМА-ПЕРЕДАЧИ ОБОРУДОВАНИЯ № 15"),
    ("small", "к договору поставки № 47/2024                                                        18 апреля 2024 г."),
    ("p", ""),
    ("p", "ООО «ТехноСервис» передало, а ФГБОУ ВО «ЮФУ» приняло вычислительную технику согласно "
          "спецификации к договору поставки № 47/2024. Претензий по качеству и комплектности "
          "Заказчик не имеет."),
    ("p", "Общая стоимость поставленного оборудования составляет 1 250 000 руб. 00 коп., "
          "включая НДС 20%."),
    ("p", "Работы выполнены в полном объёме и в установленный срок. Настоящий акт является основанием "
          "для оплаты по счёту № 208 от 15 марта 2024 г."),
    ("p", ""),
    ("p", "Сдал: генеральный директор ООО «ТехноСервис» ____________ / Семенов В. Е."),
    ("p", "Принял: проректор по цифровому развитию ФГБОУ ВО «ЮФУ» ____________ / Кузнецова А. С."),
]

ARTICLE_TEXT = """Аннотация

В статье рассматривается архитектура полностью локальной платформы интеллектуальной обработки документов (Intelligent Document Processing, IDP), предназначенной для работы в изолированном контуре образовательной организации. Предлагаемый подход объединяет компьютерное зрение для препроцессинга сканов, OCR-распознавание кириллицы и локальные языковые модели класса Qwen 2.5 для пост-коррекции, классификации и извлечения именованных сущностей.

Ключевые слова: оцифровка документов, OCR, локальные языковые модели, NER, векторный поиск, pgvector, автоматизация документооборота.

Введение

Переход вузов к электронному документообороту сдерживается большим объёмом архивных сканов, которые не содержат текстового слоя. Облачные сервисы распознавания не могут применяться из-за требований Федерального закона № 152-ФЗ о персональных данных.

Методология

Пайплайн включает четыре этапа: препроцессинг изображения средствами OpenCV (deskewing, удаление синих печатей, адаптивная бинаризация), распознавание текста, анализ локальной LLM по строгой JSON-схеме и индексацию результатов в PostgreSQL с расширением pgvector.

Результаты

Эксперименты на выборке из 500 документов показали снижение CER с 4,8% до 2,1% после пост-коррекции локальной моделью Qwen 2.5 7B при сохранении полной автономности контура. Среднее время обработки страницы на GPU NVIDIA RTX 4060 составило 3,4 секунды.

Заключение

Предложенная архитектура обеспечивает приемлемое качество оцифровки без передачи данных внешним провайдерам и может быть рекомендована к внедрению в организациях с жёсткими требованиями информационной безопасности.

DOI: 10.18522/2311-3103-2024-2-45-58
"""


def build_docx(path: Path) -> Path:
    document = DocxDocument()
    document.add_heading("СЛУЖЕБНАЯ ЗАПИСКА", level=1)
    document.add_paragraph("Кому: ректору ФГБОУ ВО «ЮФУ» Боровской М. А.")
    document.add_paragraph("От: руководителя лаборатории ИОД Семенова Владимира Евгеньевича")
    document.add_paragraph("Дата: 05 июня 2024 г.")
    document.add_heading("О внедрении платформы оцифровки документов", level=2)
    document.add_paragraph(
        "Прошу согласовать запуск в опытную эксплуатацию веб-приложения для оцифровки документов "
        "и интеллектуальной обработки информации, разработанного проектной командой ИКТИБ."
    )
    document.add_paragraph(
        "Платформа развёрнута в изолированном контуре, использует локальные модели Qwen 2.5 и "
        "обеспечивает распознавание, классификацию, извлечение сущностей и поиск по архиву документов."
    )
    table = document.add_table(rows=4, cols=2)
    table.style = "Light Grid Accent 1"
    rows = [
        ("Показатель", "Значение"),
        ("Обработано документов", "128"),
        ("Средняя точность распознавания", "96,4%"),
        ("Экономия времени оператора", "72%"),
    ]
    for row, (left, right) in zip(table.rows, rows, strict=False):
        row.cells[0].text = left
        row.cells[1].text = right
    document.add_paragraph("Приложение: отчёт о тестировании на 4 листах.")
    document.save(path)
    return path


# ---------------------------------------------------------------------------
# Эталонная разметка для оценки качества (scripts/evaluate.py).
# Сущности приведены в той форме, в которой они присутствуют в документе.
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[str, dict] = {
    "dogovor_postavki_47-2024.pdf": {
        "doc_type": "Договор",
        "entities": {
            "persons": [
                "Семенова Владимира Евгеньевича",
                "Кузнецовой Анны Сергеевны",
                "Семенов В. Е.",
                "Кузнецова А. С.",
            ],
            "organizations": ["ООО «ТехноСервис»", "ФГБОУ ВО «ЮФУ»"],
            "dates": ["12 марта 2024 г", "09.01.2024", "26 апреля 2024 г"],
            "amounts": ["1 250 000 руб. 00 коп."],
            "inn_ogrn": [
                "ИНН 6165048312",
                "КПП 616501001",
                "ОГРН 1156196048721",
                "р/с 40702810400000012345",
                "ИНН 6166052167",
                "КПП 616601001",
                "ОГРН 1026102997202",
            ],
            "document_number": "47/2024",
            "addresses": [
                "344002, г. Ростов-на-Дону, пр. Будённовский, д. 21, офис 4",
                "344006, г. Ростов-на-Дону, ул. Большая Садовая, д. 105/42",
                "г. Ростов-на-Дону, ул. Чехова, д. 2, корпус «Б», ауд. 209",
            ],
        },
    },
    "prikaz_112-OD.pdf": {
        "doc_type": "Приказ",
        "entities": {
            "persons": [
                "Семенова Владимира Евгеньевича",
                "Кузнецова Анна Сергеевна",
                "Гончаров Дмитрий Петрович",
                "Ковалёва Елена Игоревна",
                "Боровская М. А.",
            ],
            "organizations": ["ФГБОУ ВО «ЮФУ»"],
            "dates": ["12 февраля 2024 г", "12.02.2024", "30.06.2024"],
            "amounts": [],
            "inn_ogrn": [],
            "document_number": "112-ОД",
            "addresses": ["г. Ростов-на-Дону"],
        },
    },
    "schet_208.pdf": {
        "doc_type": "Счет",
        "entities": {
            "persons": ["Семенов В. Е."],
            "organizations": ["ООО «ТехноСервис»", "ФГБОУ ВО «ЮФУ»", "ПАО «Сбербанк»"],
            "dates": ["15 марта 2024 г", "30 марта 2024 г"],
            "amounts": [
                "89 000 руб",
                "890 000 руб",
                "28 000 руб",
                "280 000 руб",
                "40 000 руб",
                "80 000 руб",
                "1 250 000 руб. 00 коп.",
                "208 333 руб. 33 коп.",
            ],
            "inn_ogrn": [
                "ИНН 6165048312",
                "КПП 616501001",
                "ИНН 6166052167",
                "КПП 616601001",
                "Расчётный счёт 40702810400000012345",
                "БИК 046015001",
                "к/с 30101810600000000002",
            ],
            "document_number": "208",
            "addresses": [
                "344002, г. Ростов-на-Дону, пр. Будённовский, д. 21, офис 4",
                "344006, г. Ростов-на-Дону, ул. Большая Садовая, д. 105/42",
            ],
        },
    },
    "akt_15.pdf": {
        "doc_type": "Акт",
        "entities": {
            "persons": ["Семенов В. Е.", "Кузнецова А. С."],
            "organizations": ["ООО «ТехноСервис»", "ФГБОУ ВО «ЮФУ»"],
            "dates": ["18 апреля 2024 г", "15 марта 2024 г"],
            "amounts": ["1 250 000 руб. 00 коп."],
            "inn_ogrn": [],
            "document_number": "15",
            "addresses": [],
        },
    },
    "nauchnaya_statya_IDP.txt": {
        "doc_type": "Научная статья",
        "entities": {
            "persons": [],
            "organizations": [],
            "dates": [],
            "amounts": [],
            "inn_ogrn": [],
            "document_number": None,
            "addresses": [],
        },
    },
    "sluzhebnaya_zapiska.docx": {
        "doc_type": "Заявление",
        "note": "служебная записка; в таксономии ТЗ ближайший класс — «Заявление»",
        "entities": {
            "persons": ["Семенова Владимира Евгеньевича", "Боровской М. А."],
            "organizations": ["ФГБОУ ВО «ЮФУ»"],
            "dates": ["05 июня 2024 г"],
            "amounts": [],
            "inn_ogrn": [],
            "document_number": None,
            "addresses": [],
        },
    },
}


# Сканы оцениваются отдельно: для их обработки нужен OCR-движок
# (Tesseract или локальная VLM), поэтому в отчёте они помечаются как requires_ocr.
GROUND_TRUTH["scan_akt_15.jpg"] = {
    "doc_type": "Акт",
    "requires_ocr": True,
    "entities": GROUND_TRUTH["akt_15.pdf"]["entities"],
}
GROUND_TRUTH["scan_schet_208.jpg"] = {
    "doc_type": "Счет",
    "requires_ocr": True,
    "entities": GROUND_TRUTH["schet_208.pdf"]["entities"],
}


def _blocks_text(blocks: list[tuple[str, str]]) -> str:
    return "\n".join(text for _, text in blocks if text.strip())


def _docx_text() -> str:
    """Текст служебной записки — в том же порядке, в котором его извлекает парсер."""
    return "\n".join(
        [
            "СЛУЖЕБНАЯ ЗАПИСКА",
            "Кому: ректору ФГБОУ ВО «ЮФУ» Боровской М. А.",
            "От: руководителя лаборатории ИОД Семенова Владимира Евгеньевича",
            "Дата: 05 июня 2024 г.",
            "О внедрении платформы оцифровки документов",
            "Прошу согласовать запуск в опытную эксплуатацию веб-приложения для оцифровки документов "
            "и интеллектуальной обработки информации, разработанного проектной командой ИКТИБ.",
            "Платформа развёрнута в изолированном контуре, использует локальные модели Qwen 2.5 и "
            "обеспечивает распознавание, классификацию, извлечение сущностей и поиск по архиву документов.",
            "Приложение: отчёт о тестировании на 4 листах.",
            "Показатель | Значение",
            "Обработано документов | 128",
            "Средняя точность распознавания | 96,4%",
            "Экономия времени оператора | 72%",
        ]
    )


TEXTS: dict[str, str] = {
    "dogovor_postavki_47-2024.pdf": _blocks_text(CONTRACT_BLOCKS),
    "prikaz_112-OD.pdf": _blocks_text(ORDER_BLOCKS),
    "schet_208.pdf": _blocks_text(INVOICE_BLOCKS),
    "akt_15.pdf": _blocks_text(ACT_BLOCKS),
    "nauchnaya_statya_IDP.txt": ARTICLE_TEXT,
    "sluzhebnaya_zapiska.docx": _docx_text(),
    "scan_akt_15.jpg": _blocks_text(ACT_BLOCKS),
    "scan_schet_208.jpg": _blocks_text(INVOICE_BLOCKS),
}


def build_ground_truth() -> dict:
    documents = []
    for filename, meta in GROUND_TRUTH.items():
        documents.append(
            {
                "filename": filename,
                "doc_type": meta["doc_type"],
                "text": TEXTS.get(filename, ""),
                "entities": meta["entities"],
                **({"note": meta["note"]} if meta.get("note") else {}),
                **({"requires_ocr": True} if meta.get("requires_ocr") else {}),
            }
        )
    return {
        "description": "Эталонная разметка демонстрационного датасета IDP (CER/WER/NER F1)",
        "documents": documents,
    }


def _add_stamp(image: Image.Image) -> Image.Image:
    """Синтетическая синяя печать — проверка алгоритма удаления печатей."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = int(image.width * 0.72), int(image.height * 0.78)
    radius = int(image.width * 0.11)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=(40, 60, 200, 120),
        width=max(int(radius * 0.12), 2),
    )
    draw.text((cx - radius // 2, cy - 8), "ЮФУ", fill=(40, 60, 200, 120))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def make_scan_variant(pdf_path: Path, out_path: Path, angle: float = 0.9) -> Path:
    """Рендерит первую страницу PDF в «фотографию скана»: наклон, шум, печать, JPEG."""
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        pix = page.get_pixmap(dpi=200, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    image = image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    image = _add_stamp(image)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    image.save(out_path, format="JPEG", quality=82)
    return out_path


def generate_samples(out_dir: Path | str = "samples", force: bool = False) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        "dogovor_postavki_47-2024.pdf": lambda p: build_pdf(p, CONTRACT_BLOCKS),
        "prikaz_112-OD.pdf": lambda p: build_pdf(p, ORDER_BLOCKS),
        "schet_208.pdf": lambda p: build_pdf(p, INVOICE_BLOCKS),
        "akt_15.pdf": lambda p: build_pdf(p, ACT_BLOCKS),
        "nauchnaya_statya_IDP.txt": lambda p: p.write_text(ARTICLE_TEXT, encoding="utf-8"),
        "sluzhebnaya_zapiska.docx": build_docx,
    }
    created: list[Path] = []

    for name, factory in targets.items():
        path = out / name
        if path.exists() and not force:
            created.append(path)
            continue
        try:
            factory(path)
            created.append(path)
        except Exception as exc:  # noqa: BLE001
            print(f"Не удалось создать {name}: {exc}", file=sys.stderr)

    # синтетические «сканы» создаём после PDF: нужен исходный файл
    scans = {
        "scan_akt_15.jpg": "akt_15.pdf",
        "scan_schet_208.jpg": "schet_208.pdf",
    }
    for scan_name, source_name in scans.items():
        source = out / source_name
        scan_path = out / scan_name
        if source.exists() and (force or not scan_path.exists()):
            try:
                make_scan_variant(source, scan_path)
                created.append(scan_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Не удалось создать {scan_name}: {exc}", file=sys.stderr)
        elif scan_path.exists():
            created.append(scan_path)

    truth_path = out / "ground_truth.json"
    truth_path.write_text(
        json.dumps(build_ground_truth(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    created.append(truth_path)
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация демонстрационных документов")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for file_path in generate_samples(args.out, args.force):
        print(file_path)
