from app.ml_pipeline.llm_service import classify, extract_entities, heuristic_analysis

CONTRACT = """ДОГОВОР ПОСТАВКИ № 47/2024
г. Ростов-на-Дону                12 марта 2024 г.
ООО «ТехноСервис», ИНН 6165048312, именуемое Поставщик, и ФГБОУ ВО «ЮФУ», ИНН 6166052167,
именуемое Заказчик, заключили договор о поставке оборудования.
Общая стоимость составляет 1 250 000 руб. 00 коп., включая НДС 20%.
Оплата производится в течение 30 календарных дней после подписания акта.
Адрес поставки: г. Ростов-на-Дону, ул. Чехова, д. 2, корпус «Б».
"""

ACT = """АКТ ПРИЁМА-ПЕРЕДАЧИ ОБОРУДОВАНИЯ № 15
к договору поставки № 47/2024 от 18 апреля 2024 г.
ООО «ТехноСервис» передало, а ФГБОУ ВО «ЮФУ» приняло оборудование.
Претензий по качеству Заказчик не имеет. Стоимость 1 250 000 руб.
"""

INVOICE = """СЧЁТ НА ОПЛАТУ № 208 от 15 марта 2024 г.
Поставщик: ООО «ТехноСервис», ИНН 6165048312.
Покупатель: ФГБОУ ВО «ЮФУ», ИНН 6166052167.
Итого к оплате: 1 250 000 руб. 00 коп. В том числе НДС 20%.
"""

ORDER = """ПРИКАЗ № 112-ОД
12 февраля 2024 г. г. Ростов-на-Дону
ПРИКАЗЫВАЮ:
1. Назначить Семенова Владимира Евгеньевича руководителем лаборатории.
"""


def test_classify_contract():
    assert classify(CONTRACT) == "Договор"


def test_classify_act_not_contract():
    assert classify(ACT) == "Акт"


def test_classify_invoice():
    assert classify(INVOICE) == "Счет"


def test_classify_order():
    assert classify(ORDER) == "Приказ"


def test_extract_entities_contract():
    entities = extract_entities(CONTRACT)
    assert any("1 250 000" in amount for amount in entities.amounts)
    assert any("ИНН 6165048312" in item for item in entities.inn_ogrn)
    assert entities.document_number == "47/2024"
    assert any("Ростов-на-Дону" in address for address in entities.addresses)
    assert any("ООО" in org for org in entities.organizations)


def test_extract_person_with_patronymic():
    entities = extract_entities(ORDER)
    assert any("Семенова" in person for person in entities.persons)


def test_no_false_document_number_from_english_word():
    text = "В статье используется модель Qwen 2.5 и библиотека NER для анализа."
    entities = extract_entities(text)
    assert entities.document_number is None


def test_heuristic_analysis_returns_keywords_and_summary():
    result = heuristic_analysis(CONTRACT)
    assert result.document_type == "Договор"
    assert result.summary
    assert len(result.keywords) >= 3
    assert result.corrected_text.strip()
