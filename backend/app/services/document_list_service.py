"""Список документов под миллионы строк.

Два режима пагинации:
* **keyset (курсор)** — основной для больших баз: ``WHERE (created_at, id) < (курсор) ORDER BY created_at DESC, id DESC
  LIMIT n``. Читает ровно одну страницу индекса (``ix_documents_created_id``), стоимость не зависит от номера страницы.
* **offset (page/limit)** — классический режим для совместимости и интерфейса с номерами страниц.
  Стоимость растёт с OFFSET, поэтому глубина ограничена ``LIST_OFFSET_MAX``.

Счётчик ``total``:
* без фильтров на PostgreSQL при большой таблице — оценка ``pg_class.reltuples`` (``total_estimated=True``);
* с фильтрами — точный ``COUNT(*)``, кэшируется на ``COUNT_CACHE_TTL_SECONDS``;
* в курсорном режиме ``total`` не считается (None): он не нужен для «следующая страница».

Поиск по тексту (``q``) на PostgreSQL идёт через GIN-индекс ``search_tsv`` и ограничен
``FTS_CANDIDATE_LIMIT`` страниц-кандидатов, поэтому не превращается в полное сканирование.
"""
from __future__ import annotations

import base64
import binascii
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import gin_only
from app.models.document import Document, OcrPage

SORT_ORDERINGS = {
    "created_at_asc": (Document.created_at.asc(), Document.id.asc()),
    "created_at_desc": (Document.created_at.desc(), Document.id.desc()),
    "filename": (Document.filename.asc(), Document.id.asc()),
    "status": (Document.status.asc(), Document.id.asc()),
}
KEYSET_SORTS = ("created_at_asc", "created_at_desc")


class _NullContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


@dataclass
class ListQuery:
    status: str | None = None
    doc_type: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    query: str | None = None
    sort: str = "created_at_desc"


@dataclass
class ListPage:
    items: list[Document]
    total: int | None
    total_estimated: bool
    next_cursor: str | None
    has_more: bool


# ---------------------------- курсор ----------------------------
def encode_cursor(created_at: datetime, document_id: str) -> str:
    raw = f"{created_at.isoformat()}|{document_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        created_raw, document_id = raw.split("|", 1)
        created_at = datetime.fromisoformat(created_raw)
        if len(document_id) != 36:
            raise ValueError("id")
        return created_at, document_id
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Некорректный курсор пагинации") from exc


# ---------------------------- фильтры ----------------------------
def _is_postgres(db: AsyncSession) -> bool:
    bind = db.get_bind() if hasattr(db, "get_bind") else None
    return getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"


def _conditions(db: AsyncSession, q: ListQuery) -> list[Any]:
    conditions: list[Any] = []
    if q.status:
        conditions.append(Document.status == q.status)
    if q.doc_type:
        conditions.append(Document.doc_type == q.doc_type)
    if q.date_from:
        conditions.append(Document.created_at >= q.date_from)
    if q.date_to:
        conditions.append(Document.created_at <= q.date_to)
    if q.query:
        conditions.append(_text_condition(db, q.query))
    return conditions


def _text_condition(db: AsyncSession, query: str) -> Any:
    """Документы, у которых есть страница с совпадением, либо имя файла содержит строку.

    Страницы — через GIN (search_tsv), не больше FTS_CANDIDATE_LIMIT кандидатов.
    Имя файла на PostgreSQL ищется через ILIKE только при индексе pg_trgm: без него это полное
    сканирование documents (O(N)), поэтому по имени там не ищем — остаётся поиск по тексту страниц.
    """
    from app.core import database as db_core  # noqa: PLC0415

    limit = settings.FTS_CANDIDATE_LIMIT
    if _is_postgres(db):
        pages = (
            select(OcrPage.document_id)
            .where(text("ocr_pages.search_tsv @@ plainto_tsquery('russian', :q)").bindparams(q=query))
            .limit(limit)
        )
        by_name = [Document.filename.ilike(f"%{query}%")] if db_core.trgm_filename_index_ready else []
        return or_(*by_name, Document.id.in_(pages))
    pattern = f"%{query.lower()}%"
    pages = (
        select(OcrPage.document_id)
        .where(or_(func.lower(OcrPage.raw_text).like(pattern), func.lower(OcrPage.corrected_text).like(pattern)))
        .limit(limit)
    )
    return or_(func.lower(Document.filename).like(pattern), Document.id.in_(pages))


# ---------------------------- счётчик ----------------------------
_count_cache: dict[tuple, tuple[float, int]] = {}
_COUNT_CACHE_MAX = 512


async def _estimated_total(db: AsyncSession) -> int | None:
    """Оценка числа строк по статистике PostgreSQL (обновляется ANALYZE/autovacuum, без full scan)."""
    row = await db.execute(text("SELECT reltuples::bigint FROM pg_class WHERE relname = 'documents'"))
    value = row.scalar()
    if value is None or value < 0:  # -1: таблица ещё не анализировалась
        return None
    return int(value)


async def count_documents(db: AsyncSession, q: ListQuery, conditions: list[Any]) -> tuple[int, bool]:
    if not conditions and _is_postgres(db):
        estimate = await _estimated_total(db)
        if estimate is not None and estimate >= settings.COUNT_ESTIMATE_THRESHOLD:
            return estimate, True

    key = (q.status, q.doc_type, q.date_from, q.date_to, q.query)
    now = time.monotonic()
    cached = _count_cache.get(key)
    if cached and now - cached[0] < settings.COUNT_CACHE_TTL_SECONDS:
        return cached[1], False

    stmt = select(func.count(Document.id))
    for condition in conditions:
        stmt = stmt.where(condition)
    async with (gin_only(db) if q.query else _NullContext()):
        total = int(await db.scalar(stmt) or 0)
    if len(_count_cache) >= _COUNT_CACHE_MAX:
        _count_cache.clear()
    _count_cache[key] = (now, total)
    return total, False


def clear_count_cache() -> None:
    _count_cache.clear()


# ---------------------------- выборка ----------------------------
async def list_documents_page(
    db: AsyncSession,
    q: ListQuery,
    *,
    page: int = 1,
    limit: int = 20,
    cursor: tuple[datetime, str] | None = None,
) -> ListPage:
    if q.sort not in SORT_ORDERINGS:
        q.sort = "created_at_desc"
    conditions = _conditions(db, q)
    keyset_mode = cursor is not None
    if keyset_mode and q.sort not in KEYSET_SORTS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Курсорная пагинация доступна только для сортировки по дате создания"
        )

    stmt = select(Document).options(selectinload(Document.extracted))
    for condition in conditions:
        stmt = stmt.where(condition)
    # полнотекстовая часть — только через GIN (см. gin_only); без q запрос не меняем
    guard = gin_only(db) if q.query else _NullContext()

    total: int | None = None
    total_estimated = False
    if keyset_mode:
        created_at, document_id = cursor
        if q.sort == "created_at_desc":
            stmt = stmt.where(tuple_(Document.created_at, Document.id) < tuple_(created_at, document_id))
        else:
            stmt = stmt.where(tuple_(Document.created_at, Document.id) > tuple_(created_at, document_id))
        order = SORT_ORDERINGS[q.sort]
        async with guard:
            rows = (await db.execute(stmt.order_by(*order).limit(limit + 1))).scalars().unique().all()
        has_more = len(rows) > limit
        items = rows[:limit]
        next_cursor = encode_cursor(items[-1].created_at, items[-1].id) if has_more and items else None
        return ListPage(items=items, total=None, total_estimated=False, next_cursor=next_cursor, has_more=has_more)

    total, total_estimated = await count_documents(db, q, conditions)
    order = SORT_ORDERINGS[q.sort]
    offset = (page - 1) * limit
    async with guard:
        rows = (await db.execute(stmt.order_by(*order).offset(offset).limit(limit))).scalars().unique().all()
    has_more = offset + len(rows) < total
    # курсор выдаём и из offset-режима: клиент может перейти на keyset с любой страницы
    next_cursor = None
    if has_more and rows and q.sort in KEYSET_SORTS:
        next_cursor = encode_cursor(rows[-1].created_at, rows[-1].id)
    return ListPage(
        items=rows, total=total, total_estimated=total_estimated, next_cursor=next_cursor, has_more=has_more
    )
