"""Гибридный поиск: полнотекстовый + семантический (векторный).

Prod-контур: PostgreSQL FTS (`tsvector` + `pg_trgm`) и `pgvector` (HNSW).
Локальный контур: префиксный LIKE по стемам + косинусная близость
эмбеддингов (Ollama `bge-m3` либо hashing-fallback).
"""
from __future__ import annotations

import time

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_pipeline.llm_service import embed_texts
from app.ml_pipeline.text_utils import (
    cosine,
    expand_query,
    make_snippet,
    normalize_text,
    tokenize,
)
from app.models.document import Document, DocumentChunk, OcrPage
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


async def search_documents(db: AsyncSession, query: str, filters: SearchFilters, mode: str, limit: int) -> SearchResponse:
    started = time.perf_counter()
    terms = expand_query(query)
    stems = sorted({_stem(t) for t in terms if len(t) >= 3})
    raw_terms = [t for t in tokenize(query)]

    hits: list[SearchHit] = []

    # ---------- полнотекстовая часть ----------
    if mode in ("fulltext", "hybrid") and stems:
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
        query_vector = vectors[0]
        stmt = select(DocumentChunk, Document).join(Document, Document.id == DocumentChunk.document_id)
        stmt = _apply_filters(stmt, filters)
        rows = (await db.execute(stmt)).all()
        for chunk, document in rows:
            if filters.organization and filters.organization.lower() not in chunk.chunk_text.lower():
                continue
            similarity = cosine(query_vector, chunk.embedding or [])
            if similarity < 0.12:
                continue
            snippet, _ = make_snippet(normalize_text(chunk.chunk_text), raw_terms or stems)
            hits.append(
                SearchHit(
                    document_id=document.id,
                    filename=document.filename,
                    doc_type=document.doc_type,
                    status=document.status,
                    page_number=chunk.page_number,
                    snippet=snippet,
                    score=round(similarity, 4),
                    match_type=f"semantic:{engine}",
                )
            )

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
