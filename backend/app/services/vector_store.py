"""Векторное хранилище чанков: PostgreSQL + pgvector (HNSW) или JSON-вектора (SQLite).

Все функции принимают ``engine`` — название движка эмбеддингов ("ollama" | "fallback"),
которым получен запрос. Сравниваются только векторы того же движка.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_pipeline.text_utils import cosine
from app.models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)


def is_postgres(db: AsyncSession) -> bool:
    bind = db.get_bind() if hasattr(db, "get_bind") else None
    return getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"


async def save_pg_vectors(db: AsyncSession, rows: list[tuple[str, list[float]]]) -> None:
    """Пишет вектора в колонку pgvector (только PostgreSQL). rows: [(chunk_id, vector)]."""
    if not rows or not is_postgres(db):
        return
    try:
        await db.execute(
            sql_text("UPDATE document_chunks SET embedding_vec = CAST(:vec AS vector) WHERE id = :id"),
            [{"id": chunk_id, "vec": _vector_literal(vector)} for chunk_id, vector in rows],
        )
    except Exception as exc:  # noqa: BLE001 - нет расширения pgvector: остаётся JSON-поиск
        logger.warning("pgvector недоступен, векторы сохранены только в JSON: %s", exc)


@dataclass
class ChunkHit:
    document_id: str
    filename: str
    doc_type: str | None
    status: str
    page_number: int
    chunk_text: str
    similarity: float


async def top_chunks_pg(
    db: AsyncSession,
    query_vector: list[float],
    engine: str,
    *,
    limit: int,
    document_id: str | None = None,
    extra_sql: str = "",
    params: dict | None = None,
) -> list[ChunkHit]:
    """Ближайшие чанки через оператор косинусного расстояния pgvector ``<=>`` (HNSW)."""
    where = ["c.embedding_vec IS NOT NULL", "c.embedding_model = :engine"]
    args: dict = {"qv": _vector_literal(query_vector), "engine": engine, "limit": limit}
    if document_id:
        where.append("c.document_id = :doc_id")
        args["doc_id"] = document_id
    if extra_sql:
        where.append(extra_sql)
        args.update(params or {})
    sql = f"""
        SELECT c.document_id, d.filename, d.doc_type, d.status, c.page_number, c.chunk_text,
               1 - (c.embedding_vec <=> CAST(:qv AS vector)) AS similarity
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {' AND '.join(where)}
        ORDER BY c.embedding_vec <=> CAST(:qv AS vector)
        LIMIT :limit
    """
    rows = (await db.execute(sql_text(sql), args)).mappings().all()
    return [
        ChunkHit(
            document_id=row["document_id"],
            filename=row["filename"],
            doc_type=row["doc_type"],
            status=row["status"],
            page_number=row["page_number"],
            chunk_text=row["chunk_text"],
            similarity=float(row["similarity"]),
        )
        for row in rows
    ]


def top_chunks_python(chunks: list[DocumentChunk], query_vector: list[float], engine: str, limit: int) -> list[tuple[DocumentChunk, float]]:
    """Косинус в Python (SQLite / без pgvector). Только вектора того же движка."""
    scored = [
        (chunk, cosine(query_vector, chunk.embedding or []))
        for chunk in chunks
        if chunk.embedding and chunk.embedding_model == engine
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


# ---------- быстрый косинус для SQLite (без pgvector) ----------
# Раньше каждый запрос загружал колонку embedding (JSON, 1024 числа на чанк) для всех чанков
# и парсил её в ORM: на 1000 документов это ~320 мс. Теперь нормализованные вектора
# хранятся в памяти процесса (float32), а косинус — одно матричное умножение.
VECTOR_CACHE_LIMIT = 60_000  # чанков; при превышении кэш очищается (≈240 МБ при 1024 измерениях)
_vector_cache: dict[str, "np.ndarray"] = {}
_cache_lock = threading.Lock()


def _normalize(values: list[float]) -> "np.ndarray | None":
    arr = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or norm == 0.0:
        return None
    return arr / norm


async def top_chunks_sqlite(
    db: AsyncSession,
    query_vector: list[float],
    engine: str,
    *,
    limit: int,
    document_id: str | None = None,
    doc_type: str | None = None,
    status: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    organization: str | None = None,
) -> list[ChunkHit]:
    """Косинус по нормализованным векторам из кэша. Только вектора того же движка."""
    query = np.asarray(query_vector, dtype=np.float32)
    qnorm = float(np.linalg.norm(query))
    if qnorm == 0.0:
        return []
    query = query / qnorm

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.page_number,
            DocumentChunk.chunk_text,
            Document.filename,
            Document.doc_type,
            Document.status,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.embedding_model == engine)
    )
    if document_id:
        stmt = stmt.where(DocumentChunk.document_id == document_id)
    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    if status:
        stmt = stmt.where(Document.status == status)
    if date_from:
        stmt = stmt.where(Document.created_at >= date_from)
    if date_to:
        stmt = stmt.where(Document.created_at <= date_to)
    if organization:
        stmt = stmt.where(DocumentChunk.chunk_text.ilike(f"%{organization}%"))
    rows = (await db.execute(stmt)).all()
    if not rows:
        return []

    # догружаем только вектора, которых нет в кэше
    missing = [row.id for row in rows if row.id not in _vector_cache]
    if missing:
        for start in range(0, len(missing), 500):
            batch = missing[start:start + 500]
            vec_rows = await db.execute(
                select(DocumentChunk.id, DocumentChunk.embedding).where(DocumentChunk.id.in_(batch))
            )
            loaded = {chunk_id: _normalize(embedding or []) for chunk_id, embedding in vec_rows.all()}
            with _cache_lock:
                if len(_vector_cache) + len(loaded) > VECTOR_CACHE_LIMIT:
                    _vector_cache.clear()
                for chunk_id, vec in loaded.items():
                    if vec is not None:
                        _vector_cache[chunk_id] = vec

    kept = [row for row in rows if row.id in _vector_cache]
    if not kept:
        return []
    matrix = np.stack([_vector_cache[row.id] for row in kept])
    scores = matrix @ query
    order = np.argsort(-scores)[:limit]
    return [
        ChunkHit(
            document_id=kept[i].document_id,
            filename=kept[i].filename,
            doc_type=kept[i].doc_type,
            status=kept[i].status,
            page_number=kept[i].page_number,
            chunk_text=kept[i].chunk_text,
            similarity=float(scores[i]),
        )
        for i in order
    ]


def clear_vector_cache() -> None:
    with _cache_lock:
        _vector_cache.clear()
