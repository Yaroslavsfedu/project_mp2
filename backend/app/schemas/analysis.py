"""Эталонные схемы структурированного ответа локальной LLM (Structured Outputs)."""
from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal[
    "Договор",
    "Приказ",
    "Счет",
    "Акт",
    "Заявление",
    "Научная статья",
    "Паспорт/Удостоверение",
    "Другое",
]

DOC_TYPES: list[str] = [
    "Договор",
    "Приказ",
    "Счет",
    "Акт",
    "Заявление",
    "Научная статья",
    "Паспорт/Удостоверение",
    "Другое",
]


class ExtractedEntities(BaseModel):
    persons: list[str] = Field(default_factory=list, description="ФИО физических лиц")
    organizations: list[str] = Field(default_factory=list, description="ООО, АО, ведомства, вузы, ИП")
    dates: list[str] = Field(default_factory=list, description="Значимые даты DD.MM.YYYY")
    amounts: list[str] = Field(default_factory=list, description="Денежные суммы с валютой")
    inn_ogrn: list[str] = Field(default_factory=list, description="ИНН, КПП, ОГРН, БИК, счета")
    document_number: str | None = Field(default=None, description="Номер документа")
    addresses: list[str] = Field(default_factory=list, description="Адреса")


class DocumentAnalysisResult(BaseModel):
    corrected_text: str = Field(default="", description="Исправленный текст документа")
    document_type: DocType = "Другое"
    summary: str = Field(default="", description="Резюме в 2-4 предложениях")
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    keywords: list[str] = Field(default_factory=list)
