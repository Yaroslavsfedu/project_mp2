"""Гибридный поиск: полнотекстовый + семантический (векторный).

Prod-контур: PostgreSQL FTS (`tsvector` + `pg_trgm`) и `pgvector` (HNSW).
Локальный контур: префиксный LIKE по стемам + косинусная близость
эмбеддингов (Ollama `bge-m3` либо hashing-fallback).
"""
from __future__ import annotations

import time

from sqlalchemy import and_, or_, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_pipeline.llm_service import embed_texts
from app.services.vector_store import is_postgres, top_chunks_pg, top_chunks_sqlite
from app.ml_pipeline.text_utils import (
    expand_query,
    make_snippet,
    normalize_text,
    tokenize,
)
from app.models.document import Document, OcrPage
from app.schemas.document import SearchFilters, SearchHit, SearchResponse

STEM_LENGTH = 5


def _stem(term: str) -> str:
    return term[:STEM_LENGTH] if len(term) > STEM_LENGTH else term


def _apply_filters(query, filters: SearchFilters):
    clauses = []
    if filters.doc_type:
        clauses.append(Document.doc_type == filters.doc_type)
    if filters.status:
        clauses.append(Document.status == filters.status)
    if filters.date_from:
        clauses.append(Document.created_at >= filters.date_from)
    if filters.date_to:
        clauses.append(Document.created_at <= filters.date_to)
    # фильтр по организации применяется к тексту страницы/чанка (см. ниже),
    # так как организация — это извлечённая сущность, а не колонка таблицы
    return query.where(and_(*clauses)) if clauses else query


def _dialect(db: AsyncSession) -> str:
    bind = db.get_bind() if hasattr(db, "get_bind") else None
    return getattr(getattr(bind, "dialect", None), "name", "sqlite")


async def _postgres_fulltext_hits(
    db: AsyncSession, query: str, filters: SearchFilters, limit: int
) -> list[SearchHit]:
    """Полнотекстовый поиск средствами PostgreSQL: to_tsvector + ts_rank + ts_headline.

    Используется русская морфология ('russian'), поэтому запрос «неустойка»
    находит «неустойки», «неустойку» и т.д. Индекс GIN создаётся миграцией.
    """
    clauses = [
        "p.search_tsv @@ plainto_tsquery('russian', :q)"
    ]
    params: dict = {"q": query, "limit": limit}
    if filters.doc_type:
        clauses.append("d.doc_type = :doc_type")
        params["doc_type"] = filters.doc_type
    if filters.status:
        clauses.append("d.status = :status")
        params["status"] = filters.status
    if filters.date_from:
        clauses.append("d.created_at >= :date_from")
        params["date_from"] = filters.date_from
    if filters.date_to:
        clauses.append("d.created_at <= :date_to")
        params["date_to"] = filters.date_to
    if filters.organization:
        clauses.append(
            "(coalesce(p.corrected_text, '') || ' ' || coalesce(p.raw_text, '')) ILIKE :organization"
        )
        params["organization"] = f"%{filters.organization}%"

    # Сначала отбираем top-N страниц по рангу, и только для них строим ts_headline.
    # Иначе headline считается для всех совпадений (общее слово совпадает почти везде) — это было ~400 мс.
    sql = f"""
        WITH matched AS (
            SELECT p.id AS page_id,
                   p.document_id,
                   p.page_number,
                   d.filename,
                   d.doc_type,
                   d.status,
                   d.created_at,
                   coalesce(p.corrected_text, p.raw_text) AS body,
                   ts_rank(p.search_tsv, plainto_tsquery('russian', :q)) AS rank
            FROM ocr_pages p
            JOIN documents d ON d.id = p.document_id
            WHERE {' AND '.join(clauses)}
            ORDER BY rank DESC, d.created_at DESC
            LIMIT :limit
        )
        SELECT m.document_id,
               m.filename,
               m.doc_type,
               m.status,
               m.page_number,
               m.rank,
               ts_headline(
                   'russian',
                   m.body,
                   plainto_tsquery('russian', :q),
                   'StartSel=<b>, StopSel=</b>, MaxWords=45, MinWords=12'
               ) AS headline
        FROM matched m
        ORDER BY m.rank DESC, m.created_at DESC
    """
    rows = (await db.execute(sql_text(sql), params)).mappings().all()
    hits: list[SearchHit] = []
    for row in rows:
        headline = (row["headline"] or "").replace("<b>", "").replace("</b>", "")
        hits.append(
            SearchHit(
                document_id=row["document_id"],
                filename=row["filename"],
                doc_type=row["doc_type"],
                status=row["status"],
                page_number=row["page_number"],
                snippet=headline.strip() or "…",
                score=round(min(float(row["rank"] or 0) * 4, 1.0), 4),
                match_type="postgres-fts",
            )
        )
    return hits


async def _semantic_hits(
    db: AsyncSession, query_vector: list[float], engine: str, filters: SearchFilters, limit: int, terms: list[str]
) -> list[SearchHit]:
    """Семантический поиск: pgvector (HNSW) в PostgreSQL, косинус в Python — в SQLite."""
    found: list[tuple[str, str, str | None, str, int, str, float]] = []
    if is_postgres(db):
        extra, params = _pg_filter_sql(filters)
        try:
            for hit in await top_chunks_pg(
                db, query_vector, engine, limit=limit, extra_sql=extra, params=params
            ):
                found.append((hit.document_id, hit.filename, hit.doc_type, hit.status,
                              hit.page_number, hit.chunk_text, hit.similarity))
        except Exception:  # noqa: BLE001 - деградируем к поиску по JSON-векторам
            found = []
    if not found:
        hits_sql = await top_chunks_sqlite(
            db,
            query_vector,
            engine,
            limit=limit,
            doc_type=filters.doc_type,
            status=filters.status,
            date_from=filters.date_from,
            date_to=filters.date_to,
            organization=filters.organization,
        )
        found = [
            (h.document_id, h.filename, h.doc_type, h.status, h.page_number, h.chunk_text, h.similarity)
            for h in hits_sql
        ]

    hits: list[SearchHit] = []
    for doc_id, filename, doc_type, status, page_number, chunk_text, similarity in found:
        if similarity < 0.12:
            continue
        snippet, _ = make_snippet(normalize_text(chunk_text), terms)
        hits.append(
            SearchHit(
                document_id=doc_id,
                filename=filename,
                doc_type=doc_type,
                status=status,
                page_number=page_number,
                snippet=snippet,
                score=round(similarity, 4),
                match_type=f"semantic:{engine}",
            )
        )
    return hits


def _pg_filter_sql(filters: SearchFilters) -> tuple[str, dict]:
    clauses: list[str] = []
    params: dict = {}
    if filters.doc_type:
        clauses.append("d.doc_type = :f_doc_type")
        params["f_doc_type"] = filters.doc_type
    if filters.status:
        clauses.append("d.status = :f_status")
        params["f_status"] = filters.status
    if filters.date_from:
        clauses.append("d.created_at >= :f_date_from")
        params["f_date_from"] = filters.date_from
    if filters.date_to:
        clauses.append("d.created_at <= :f_date_to")
        params["f_date_to"] = filters.date_to
    if filters.organization:
        clauses.append("c.chunk_text ILIKE :f_org")
        params["f_org"] = f"%{filters.organization}%"
    return " AND ".join(clauses), params


async def rank_document_chunks(
    db: AsyncSession, document_id: str, query_vector: list[float], engine: str, limit: int
) -> list[tuple[int, str]]:
    """Наиболее релевантные чанки одного документа для RAG-ответа: [(страница, текст)]."""
    if is_postgres(db):
        try:
            hits = await top_chunks_pg(db, query_vector, engine, limit=limit, document_id=document_id)
            return [(h.page_number, h.chunk_text) for h in hits]
        except Exception:  # noqa: BLE001
            pass
    hits = await top_chunks_sqlite(db, query_vector, engine, limit=limit, document_id=document_id)
    return [(h.page_number, h.chunk_text) for h in hits]


async def search_documents(db: AsyncSession, query: str, filters: SearchFilters, mode: str, limit: int) -> SearchResponse:
    started = time.perf_counter()
    terms = expand_query(query)
    stems = sorted({_stem(t) for t in terms if len(t) >= 3})
    raw_terms = [t for t in tokenize(query)]

    hits: list[SearchHit] = []

    # ---------- полнотекстовая часть ----------
    dialect = _dialect(db)
    if mode in ("fulltext", "hybrid") and dialect == "postgresql":
        try:
            hits.extend(await _postgres_fulltext_hits(db, query, filters, limit * 3))
        except Exception:  # noqa: BLE001 - деградируем к LIKE-поиску
            dialect = "fallback"

    if mode in ("fulltext", "hybrid") and stems and dialect != "postgresql":
        conditions = [
            OcrPage.raw_text.ilike(f"%{stem}%") | OcrPage.corrected_text.ilike(f"%{stem}%")
            for stem in stems
        ]
        stmt = (
            select(OcrPage, Document)
            .join(Document, Document.id == OcrPage.document_id)
            .where(or_(*conditions))
            .order_by(Document.created_at.desc())
            .limit(300)
        )
        stmt = _apply_filters(stmt, filters)
        rows = (await db.execute(stmt)).all()
        for page, document in rows:
            text = page.corrected_text or page.raw_text or ""
            low = text.lower()
            matched = [stem for stem in stems if stem in low]
            exact = sum(low.count(term) for term in raw_terms)
            score = 0.55 + 0.1 * len(matched) + min(exact, 6) * 0.03
            if filters.organization and filters.organization.lower() not in low:
                continue
            snippet, found = make_snippet(text, raw_terms or stems)
            hits.append(
                SearchHit(
                    document_id=document.id,
                    filename=document.filename,
                    doc_type=document.doc_type,
                    status=document.status,
                    page_number=page.page_number,
                    snippet=snippet,
                    score=round(min(score, 1.0), 4),
                    match_type="fulltext" if found else "morphology",
                )
            )

    # ---------- семантическая часть ----------
    if mode in ("semantic", "hybrid"):
        vectors, engine = await embed_texts([query])
        semantic = await _semantic_hits(db, vectors[0], engine, filters, limit * 3, raw_terms or stems)
        hits.extend(semantic)

    # ---------- слияние (RRF-подобное) ----------
    merged: dict[tuple[str, int | None], SearchHit] = {}
    for hit in hits:
        key = (hit.document_id, hit.page_number)
        existing = merged.get(key)
        if existing is None or hit.score > existing.score:
            merged[key] = hit
        elif mode == "hybrid" and existing.match_type.startswith("fulltext"):
            # гибрид: бонус за совпадение обоими способами
            existing.score = round(min(existing.score + 0.15, 1.0), 4)
            existing.match_type = "fulltext+semantic"
    results = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:limit]
    took = (time.perf_counter() - started) * 1000

    return SearchResponse(
        query=query,
        mode=mode,
        took_ms=round(took, 1),
        total=len(results),
        hits=results,
    )
