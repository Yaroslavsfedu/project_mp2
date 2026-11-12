"""Сервис локального ИИ: Ollama (Qwen 2.5 / bge-m3) + эвристический fallback.

Все запросы идут только на локальный `OLLAMA_URL` — никаких облачных API.
Если Ollama недоступна (нет GPU/весов), включается fallback-движок на правилах и
регулярных выражениях: приложение остаётся полностью работоспособным,
а в интерфейсе честно отображается, какой движок обработал документ.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from app.core.config import settings
from app.ml_pipeline.text_utils import (
    chunk_text,
    cosine,
    hashing_embedding,
    keywords as extract_keywords,
    normalize_text,
    split_sentences,
    tokenize,
)
from app.schemas.analysis import DocumentAnalysisResult, ExtractedEntities

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — интеллектуальная система обработки документов (IDP). На вход ты получаешь сырой текст "
    "документа, извлечённый с помощью OCR (возможны опечатки, разорванные слова и артефакты).\n"
    "1. Восстанови и исправь текст, сохранив все исходные факты, цифры и структуру.\n"
    "2. Классифицируй тип документа.\n"
    "3. Напиши краткое резюме (2-4 предложения).\n"
    "4. Извлеки именованные сущности (ФИО, организации, даты, суммы, ИНН/ОГРН/КПП, номер, адреса).\n"
    "Ответ верни СТРОГО в формате валидного JSON по заданной схеме."
)

JSON_SAMPLE = {
    "corrected_text": "…",
    "document_type": "Договор",
    "summary": "…",
    "entities": {
        "persons": [],
        "organizations": [],
        "dates": [],
        "amounts": [],
        "inn_ogrn": [],
        "document_number": None,
        "addresses": [],
    },
    "keywords": [],
}

MONTHS = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
)

# --------------------------------- regex-слой ---------------------------------
# последовательность из 2-4 слов с заглавной буквы — кандидат в ФИО
RE_NAME_SEQUENCE = re.compile(
    r"\b[А-ЯЁ][а-яё]{2,}(?:-[А-ЯЁ][а-яё]{2,})?(?:\s+[А-ЯЁ][а-яё]{2,}(?:-[А-ЯЁ][а-яё]{2,})?){1,3}\b"
)
RE_PERSON_SHORT = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.(?:[а-яё]{1,3})?")
RE_ORG = re.compile(
    r"\b(?:ООО|ОАО|ЗАО|ПАО|АО|ИП|ФГБОУ ВО|ФГБНУ|ФГУП|ГБУ|МУП|АНО|НИИ|НПО|ЮФУ|ИКТИБ)\s*"
    r"(?:«[^»]{2,80}»|\"[^\"]{2,80}\"|[А-ЯЁ][^,;\n]{2,70})"
)
RE_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
RE_DATE_TEXT = re.compile(rf"\b\d{{1,2}}\s+(?:{MONTHS})\s+\d{{4}}\s*(?:г\.|года)?", re.IGNORECASE)
RE_AMOUNT = re.compile(
    r"\b\d[\d\s]{0,12}(?:[.,]\d{1,2})?\s*(?:руб(?:лей|ля|\.)?|₽|тыс\.?\s*руб|млн\.?\s*руб)"
    r"(?:\s*\d{2}\s*коп(?:\.|еек)?)?",
    re.IGNORECASE,
)
RE_INN = re.compile(r"\bИНН[\s:№]*\d{10,12}\b", re.IGNORECASE)
RE_KPP = re.compile(r"\bКПП[\s:№]*\d{9}\b", re.IGNORECASE)
RE_OGRN = re.compile(r"\bОГРН(?:ИП)?[\s:№]*\d{13,15}\b", re.IGNORECASE)
RE_BANK = re.compile(r"\b(?:БИК|р/с|к/с|расч[её]тный сч[её]т)[\s:№]*\d{9,20}\b", re.IGNORECASE)
RE_DOC_NUMBER = re.compile(
    r"(?:№|(?<![A-Za-zА-Яа-яЁё])N[оo]?\.?)(?![A-Za-zА-Яа-яЁё])\s*([0-9][0-9A-Za-zА-Яа-яЁё\-/]{0,24})"
    r"|(?:номер)\s*([0-9][0-9A-Za-zА-Яа-яЁё\-/]{0,24})",
    re.IGNORECASE,
)
RE_ADDRESS = re.compile(
    r"\b(?:г\.|город|гор\.)\s*[А-ЯЁ][А-Яа-яё]*(?:-[А-Яа-яёЁ][А-Яа-яёЁ]*)*,"
    r"\s*(?:ул\.|улица|пр\.|пр-т|просп\.|пер\.|переулок|д\.|шоссе|бульвар|наб\.)[^\n;]{0,90}"
)
# город без улицы принимаем только в «шапке» документа (первая строка адресного блока)
RE_CITY_ONLY = re.compile(r"\b(?:г\.|город|гор\.)\s*[А-ЯЁ][А-Яа-яё]*(?:-[А-Яа-яёЁ][А-Яа-яёЁ]*)*\b")

# Сильные маркеры типа документа (заголовок документа) и слабые (контекст)
STRONG_MARKERS: dict[str, tuple[str, ...]] = {
    "Договор": ("договор поставки", "договор №", "договор возмездного", "контракт №", "соглашение №"),
    "Приказ": ("приказ №", "приказ от", "приказываю"),
    "Счет": ("счёт на оплату", "счет на оплату", "счёт №", "счет №", "счёт-фактура", "счет-фактура", "invoice"),
    "Акт": ("акт приёма", "акт приема", "акт выполненных", "акт №", "акт оказанных"),
    "Заявление": ("заявление", "прошу", "прошение"),
    "Научная статья": ("аннотация", "abstract", "ключевые слова", "doi", "научная статья"),
    "Паспорт/Удостоверение": ("паспорт", "удостоверение", "код подразделения", "дата рождения"),
}

WEAK_MARKERS: dict[str, tuple[str, ...]] = {
    "Договор": ("договор", "контракт", "соглашение", "стороны заключили", "поставщик", "заказчик", "покупатель"),
    "Приказ": ("приказ", "распоряжение", "в соответствии с приказом"),
    "Счет": ("счёт", "счет", "к оплате", "платёжное поручение", "ндс", "итого"),
    "Акт": ("акт", "приёма-передачи", "приема-передачи", "оказанные услуги", "претензий нет"),
    "Заявление": ("заявитель", "рассмотреть", "согласовать"),
    "Научная статья": ("введение", "методология", "исследование", "результаты", "заключение"),
    "Паспорт/Удостоверение": ("выдан", "серия", "документ, удостоверяющий личность"),
}


@dataclass
class AnalysisOutcome:
    result: DocumentAnalysisResult
    engine: str


PERSON_STOPWORDS = {
    "назначить", "утвердить", "создать", "возложить", "согласовать", "прошу", "приказываю",
    "российская", "федерация", "акт", "приказа", "договор", "договора", "поставки", "почтовый",
    "юридический", "фактический", "адрес", "генеральный", "директор", "проректор", "ректор",
}
PATRONYMIC_SUFFIXES = (
    "ович", "евич", "ьич", "ич",
    "овн", "евн", "ичн", "иничн",
)

PERSON_LASTWORD_ENDINGS = (
    "ов", "ев", "ёв", "ин", "ын", "ова", "ева", "ёва", "ина", "ына", "ову", "еву", "ину",
    "овым", "евым", "иным", "овой", "евой", "иной", "ского", "цкого", "ская", "цкая",
    "енко", "ук", "юк", "ых", "их", "ко", "дзе", "швили",
)


def _looks_like_person(value: str) -> bool:
    """Отсекает «трёхсловные» ложные срабатывания (глаголы, должности)."""
    words = [w for w in re.split(r"\s+", value.strip()) if w]
    if len(words) < 2:
        return False
    if words[0].lower().strip(".,") in PERSON_STOPWORDS:
        return False
    if any(part in w.lower() for w in words for part in PATRONYMIC_SUFFIXES):
        return True
    last = words[-1].lower().strip(".,")
    return any(last.endswith(ending) for ending in PERSON_LASTWORD_ENDINGS)


def _extract_person_candidates(text: str) -> list[str]:
    """Скользящее окно по последовательностям слов: «Назначить Семенова Владимира
    Евгеньевича» -> «Семенова Владимира Евгеньевича»."""
    candidates: list[str] = []
    for sequence in RE_NAME_SEQUENCE.findall(text):
        words = sequence.split()
        accepted: list[str] = []
        for size in (4, 3, 2):
            if size > len(words):
                continue
            for start in range(0, len(words) - size + 1):
                candidate = " ".join(words[start : start + size])
                if not _looks_like_person(candidate):
                    continue
                if any(candidate in existing for existing in accepted):
                    continue
                accepted.append(candidate)
        # оставляем только максимальные совпадения
        candidates.extend(
            candidate for candidate in accepted if not any(candidate != other and candidate in other for other in accepted)
        )
    return candidates


def _uniq(values: list[str], limit: int = 12) -> list[str]:
    seen: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip(" ,;")
        if len(cleaned) < 3:
            continue
        if cleaned.lower() in {v.lower() for v in seen}:
            continue
        seen.append(cleaned)
        if len(seen) >= limit:
            break
    return seen


def classify(text: str) -> str:
    """Классификация документа по сильным (заголовок) и слабым (контекст) маркерам."""
    low = text.lower()
    head = low[:180]
    scores: dict[str, float] = {}
    for doc_type in STRONG_MARKERS:
        strong = STRONG_MARKERS[doc_type]
        weak = WEAK_MARKERS[doc_type]
        head_hits = sum(1 for marker in strong if marker in head)
        body_hits = sum(low.count(marker) for marker in strong)
        weak_hits = sum(low.count(marker) for marker in weak)
        scores[doc_type] = head_hits * 5.0 + body_hits * 1.0 + weak_hits * 0.3
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1.0 else "Другое"


ORG_PREFIXES = ("ООО", "ОАО", "ЗАО", "ПАО", "АО", "ИП", "ФГБОУ", "ФГУП", "ГБУ", "МУП", "АНО", "НИИ", "НПО")
CITY_STOPWORDS = {"поставщик", "покупатель", "заказчик", "исполнитель", "продавец", "оплата", "сдал", "принял", "о"}


def _clean_city_candidates(candidates: list[str]) -> list[str]:
    """Убирает из «город без улицы» служебные слова («г. Поставщик», «г. ООО»)."""
    cleaned: list[str] = []
    for candidate in candidates:
        value = re.sub(r"^(?:г\.|город|гор\.)\s*", "", candidate).strip(" ,;.")
        if len(value) < 4 or value.lower() in CITY_STOPWORDS:
            continue
        if value.upper().startswith(ORG_PREFIXES):
            continue
        cleaned.append(candidate.strip())
    return cleaned


def extract_entities(text: str) -> ExtractedEntities:
    persons = _uniq(_extract_person_candidates(text) + RE_PERSON_SHORT.findall(text))
    organizations = _uniq([re.sub(r"\s+$", "", o).rstrip(" ,;") for o in RE_ORG.findall(text)])
    dates = _uniq(RE_DATE.findall(text) + [d.strip() for d in RE_DATE_TEXT.findall(text)], limit=15)
    amounts = _uniq([re.sub(r"\s+", " ", a) for a in RE_AMOUNT.findall(text)], limit=10)
    requisites = _uniq(RE_INN.findall(text) + RE_KPP.findall(text) + RE_OGRN.findall(text) + RE_BANK.findall(text))
    number = None
    # номер документа ищем только в шапке (первые 500 символов), иначе ловятся
    # ссылки на другие документы в теле
    match = RE_DOC_NUMBER.search(text[:500])
    if match:
        number = (match.group(1) or match.group(2) or "").strip(".,")

    # адрес без улицы берём только из шапки (первые 400 символов)
    head_addresses = _clean_city_candidates(RE_CITY_ONLY.findall(text[:400]))
    addresses = _uniq(RE_ADDRESS.findall(text) + head_addresses, limit=6)

    # Отбрасываем «ФИО» из служебных шапок
    persons = [p for p in persons if p.lower() not in {"российская федерация", "ростов на"}]

    return ExtractedEntities(
        persons=persons,
        organizations=organizations,
        dates=dates,
        amounts=amounts,
        inn_ogrn=requisites,
        document_number=number,
        addresses=addresses,
    )


def build_summary(text: str, entities: ExtractedEntities, doc_type: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    interesting = [s for s in sentences if len(s) > 40]
    if not interesting:
        interesting = sentences

    def score(sentence: str) -> float:
        low = sentence.lower()
        value = 0.0
        for group in (
            entities.persons,
            entities.organizations,
            entities.amounts,
            entities.dates,
        ):
            value += sum(0.6 for item in group if item.lower() in low)
        markers = STRONG_MARKERS.get(doc_type, ()) + WEAK_MARKERS.get(doc_type, ())
        value += 0.4 * sum(1 for word in markers if word in low)
        value += min(len(sentence) / 400.0, 0.5)
        return value

    ranked = sorted(interesting[:40], key=score, reverse=True)[:3]
    ordered = [s for s in interesting if s in ranked][:3]
    summary = " ".join(ordered) if ordered else " ".join(interesting[:2])
    return summary.strip()[:1200]


def heuristic_analysis(text: str) -> DocumentAnalysisResult:
    text = normalize_text(text)
    doc_type = classify(text)
    entities = extract_entities(text)
    return DocumentAnalysisResult(
        corrected_text=text,
        document_type=doc_type,  # type: ignore[arg-type]
        summary=build_summary(text, entities, doc_type),
        entities=entities,
        keywords=extract_keywords(text),
    )


# --------------------------------- Ollama ---------------------------------
class OllamaClient:
    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self._available: bool | None = None

    async def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                self._available = response.status_code == 200
        except Exception:  # noqa: BLE001
            self._available = False
        return self._available

    async def generate_json(self, prompt: str, system: str = SYSTEM_PROMPT) -> dict | None:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": 8192},
        }
        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
                return json.loads(body.get("response", "{}"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama generate failed: %s", exc)
            self._available = False
            return None

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not await self.is_available():
            return None
        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": settings.EMBEDDING_MODEL, "input": texts},
                )
                response.raise_for_status()
                return response.json().get("embeddings")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama embed failed: %s", exc)
            return None


ollama = OllamaClient()


def _build_prompt(text: str, page_hint: str = "") -> str:
    truncated = text[:12000]
    return (
        f"{page_hint}\n\nСхема ответа (JSON):\n{json.dumps(JSON_SAMPLE, ensure_ascii=False, indent=2)}\n\n"
        f"Текст документа:\n<<<\n{truncated}\n>>>"
    )


async def analyze_document(text: str, *, page_hint: str = "") -> AnalysisOutcome:
    """Полный анализ документа: коррекция, классификация, NER, резюме."""
    if await ollama.is_available():
        raw = await ollama.generate_json(_build_prompt(text, page_hint))
        if raw:
            try:
                result = DocumentAnalysisResult.model_validate({**JSON_SAMPLE, **raw})
                if not result.corrected_text.strip():
                    result.corrected_text = normalize_text(text)
                return AnalysisOutcome(result=result, engine="ollama")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Не удалось разобрать ответ LLM: %s", exc)

    if not settings.LLM_FALLBACK_ENABLED:
        raise RuntimeError("Локальная LLM недоступна, fallback отключён")
    return AnalysisOutcome(result=heuristic_analysis(text), engine="fallback")


async def embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    if await ollama.is_available():
        vectors = await ollama.embed(texts)
        if vectors and len(vectors) == len(texts):
            return vectors, "ollama"
    return [hashing_embedding(t, settings.EMBEDDING_DIM) for t in texts], "fallback"


async def answer_question(question: str, context_chunks: list[str]) -> tuple[str, str, int]:
    """RAG-ответ по найденным чанкам. Возвращает (ответ, движок, индекс лучшего чанка)."""
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    if await ollama.is_available():
        prompt = (
            "Ответь на вопрос пользователя, опираясь ТОЛЬКО на фрагменты документа ниже. "
            "Если ответа в тексте нет — так и скажи. Отвечай кратко, по-русски.\n\n"
            f"Фрагменты:\n{context}\n\nВопрос: {question}\nОтвет:"
        )
        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{ollama.base_url}/api/generate",
                    json={"model": ollama.model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.2, "num_ctx": 8192}},
                )
                response.raise_for_status()
                answer = (response.json().get("response") or "").strip()
                if answer:
                    return answer, "ollama", 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG generate failed: %s", exc)

    # --- extractive fallback ---
    question_tokens = set(tokenize(question))
    best_index, best_score = 0, -1.0
    for index, chunk in enumerate(context_chunks):
        for sentence in split_sentences(chunk):
            tokens = set(tokenize(sentence))
            if not tokens:
                continue
            score = len(tokens & question_tokens) / (len(question_tokens) ** 0.5 or 1)
            if score > best_score:
                best_index, best_score = index, score
    if best_score <= 0:
        return (
            "В загруженных фрагментах документа нет прямого ответа на этот вопрос. "
            "Уточните формулировку или откройте полный текст страницы.",
            "fallback",
            best_index,
        )
    chunk_sentences = split_sentences(context_chunks[best_index])
    relevant = sorted(
        chunk_sentences,
        key=lambda s: len(set(tokenize(s)) & question_tokens),
        reverse=True,
    )[:2]
    answer = " ".join(relevant).strip()
    return (
        f"По тексту документа: {answer}",
        "fallback",
        best_index,
    )


async def llm_status() -> dict:
    available = await ollama.is_available()
    return {
        "provider": "ollama",
        "url": ollama.base_url,
        "model": ollama.model,
        "embedding_model": settings.EMBEDDING_MODEL,
        "available": available,
        "mode": "ollama" if available else "fallback",
    }
