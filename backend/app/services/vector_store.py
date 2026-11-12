"""Векторное хранилище чанков: PostgreSQL + pgvector (HNSW) или JSON-вектора (SQLite).

Все функции принимают ``engine`` — название движка эмбеддингов ("ollama" | "fallback"),
которым получен запрос. Сравниваются только векторы того же движка.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_pipeline.text_utils import cosine
from app.models.document import DocumentChunk

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
