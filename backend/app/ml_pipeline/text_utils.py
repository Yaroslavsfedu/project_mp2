"""Утилиты нормализации текста, чанкинга и лёгкого NLP (без внешних моделей)."""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

SPACE_RE = re.compile(r"[ \t\u00a0]+")
HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)", re.UNICODE)
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
    "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
    "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем",
    "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
    "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю",
    "между", "россии", "российская", "федерации", "является", "также", "данного", "данном",
}

# Расширение запроса синонимами — компенсирует отсутствие векторной модели
# в локальном (SQLite) режиме и заметно улучшает «семантический» поиск.
SYNONYMS: dict[str, list[str]] = {
    "договор": ["контракт", "соглашение"],
    "контракт": ["договор", "соглашение"],
    "соглашение": ["договор", "контракт"],
    "оплата": ["платеж", "перечисление", "счет"],
    "платеж": ["оплата", "перечисление"],
    "счет": ["счёт", "invoice", "оплата"],
    "счёт": ["счет", "invoice", "оплата"],
    "поставка": ["поставщик", "отгрузка", "товар"],
    "поставщик": ["поставка", "продавец"],
    "покупатель": ["заказчик", "клиент"],
    "заказчик": ["покупатель", "клиент"],
    "сотрудник": ["работник", "специалист"],
    "работник": ["сотрудник"],
    "приказ": ["распоряжение", "указ"],
    "штраф": ["неустойка", "пеня"],
    "неустойка": ["штраф", "пеня"],
    "срок": ["период", "дата", "дедлайн"],
    "сумма": ["стоимость", "цена"],
    "стоимость": ["сумма", "цена"],
    "университет": ["вуз", "юфу", "институт"],
    "институт": ["университет", "вуз"],
    "компьютер": ["пк", "ноутбук", "техника"],
    "техника": ["оборудование", "компьютер"],
    "оборудование": ["техника"],
    "акт": ["акт", "приемка"],
    "заявление": ["прошение", "обращение"],
}


def normalize_text(text: str) -> str:
    """Пост-OCR коррекция «дешёвого» уровня: склейка переносов, пробелы."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    # типичные ошибки распознавания в цифробуквенных реквизитах
    text = re.sub(r"(?<=\d)[ОO](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)[Il|](?=\d)", "1", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[А-ЯA-ZЁа-яa-z0-9«\"(])", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 3]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]{3,}", text.lower())


def keywords(text: str, limit: int = 8) -> list[str]:
    freq: dict[str, int] = {}
    for token in tokenize(text):
        if token in STOPWORDS or token.isdigit():
            continue
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [word for word, _ in ranked[:limit]]


def expand_query(query: str) -> list[str]:
    tokens = tokenize(query)
    expanded = set(tokens)
    for token in list(tokens):
        for syn in SYNONYMS.get(token, []):
            expanded.update(tokenize(syn))
    return sorted(expanded)


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Разбиение на смысловые чанки по абзацам с ограничением длины."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
            else:
                for i in range(0, len(paragraph), max_chars):
                    chunks.append(paragraph[i : i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def hashing_embedding(text: str, dim: int = 512) -> list[float]:
    """Детерминированный fallback-эмбеддинг (word/char-ngram hashing).

    Используется, когда Ollama/`bge-m3` недоступны: сохраняет
    чувствительность к словам и морфологии (за счёт 4-грамм).
    """
    vector = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    for token in tokens:
        weight = 1.0 / (1.0 + math.log(1 + len(token)))
        for feature in (token[:5], token[-4:] if len(token) > 4 else token):
            h = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
            vector[h % dim] += weight
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return [round(float(v), 5) for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def make_snippet(text: str, terms: list[str], radius: int = 140) -> tuple[str, bool]:
    """Возвращает фрагмент вокруг первого совпадения (для подсветки на фронте)."""
    low = text.lower()
    position = -1
    for term in sorted(terms, key=len, reverse=True):
        idx = low.find(term.lower())
        if idx != -1 and (position == -1 or idx < position):
            position = idx
    if position == -1:
        return text[: radius * 2].strip(), False
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}", True
