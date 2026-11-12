"""Векторное хранилище чанков: PostgreSQL + pgvector (HNSW) или JSON-вектора (SQLite).

Все функции принимают ``engine`` — название движка эмбеддингов ("ollama" | "fallback"),
которым получен запрос. Сравниваются только векторы того же движка.
"""
from __future__ import annotations

import heapq
import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import func, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.ml_pipeline.text_utils import cosine
from app.models.document import Document, DocumentChunk
from app.services.mmap_index import MmapVectorIndex

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
    # HNSW обходит ef_search кандидатов и только потом применяет WHERE: при фильтрах кандидатов
    # нужно больше, иначе выдача станет короче limit. Значение действует до конца транзакции.
    filtered = bool(document_id or extra_sql)
    ef_search = max(settings.HNSW_EF_SEARCH, limit * (4 if filtered else 1))
    await db.execute(sql_text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
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
# Нормализованные вектора (float32) лежат в памяти процесса. Кэш — LRU на OrderedDict:
# при переполнении вытесняется самый давно использованный чанк, а не весь кэш (раньше кэш
# обнулялся целиком, и при большой базе каждый запрос заново парсил JSON всех векторов).
# Чтение идёт батчами по ключу id, поэтому память на запрос ограничена размером батча и top-K.
VECTOR_CACHE_LIMIT = 60_000  # чанков (≈240 МБ при 1024 измерениях)
SQLITE_SCAN_BATCH = 2_000  # строк за один проход сканирования
_vector_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_cache_lock = threading.Lock()


def _normalize(values: list[float]) -> "np.ndarray | None":
    arr = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or norm == 0.0:
        return None
    return arr / norm


def _cache_get(chunk_id: str) -> "np.ndarray | None":
    with _cache_lock:
        vec = _vector_cache.get(chunk_id)
        if vec is not None:
            _vector_cache.move_to_end(chunk_id)
        return vec


def _cache_put(items: dict[str, "np.ndarray"]) -> None:
    with _cache_lock:
        for chunk_id, vec in items.items():
            _vector_cache[chunk_id] = vec
            _vector_cache.move_to_end(chunk_id)
        while len(_vector_cache) > VECTOR_CACHE_LIMIT:
            _vector_cache.popitem(last=False)  # LRU: вытесняем старейший


async def _load_vectors(db: AsyncSession, chunk_ids: list[str]) -> None:
    """Догружает в кэш вектора, которых там нет (JSON → float32), батчами по 500."""
    for start in range(0, len(chunk_ids), 500):
        batch = chunk_ids[start:start + 500]
        result = await db.execute(
            select(DocumentChunk.id, DocumentChunk.embedding).where(DocumentChunk.id.in_(batch))
        )
        loaded = {}
        for chunk_id, embedding in result.all():
            vec = _normalize(embedding or [])
            if vec is not None:
                loaded[chunk_id] = vec
        _cache_put(loaded)


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
    """Косинус по нормализованным векторам: батчевый проход по id, top-K в куче.

    Только вектора того же движка. Сложность — O(N) по числу чанков (SQLite не индексирует
    векторы), поэтому для миллионов чанков нужен PostgreSQL + pgvector (см. README).
    """
    if limit <= 0:
        return []
    if await use_mmap_index(db, engine):
        return await top_chunks_mmap(
            db, query_vector, engine, limit=limit, document_id=document_id, doc_type=doc_type,
            status=status, date_from=date_from, date_to=date_to, organization=organization,
        )
    query = np.asarray(query_vector, dtype=np.float32)
    qnorm = float(np.linalg.norm(query))
    if qnorm == 0.0:
        return []
    query = query / qnorm

    base = (
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
        base = base.where(DocumentChunk.document_id == document_id)
    if doc_type:
        base = base.where(Document.doc_type == doc_type)
    if status:
        base = base.where(Document.status == status)
    if date_from:
        base = base.where(Document.created_at >= date_from)
    if date_to:
        base = base.where(Document.created_at <= date_to)
    if organization:
        base = base.where(DocumentChunk.chunk_text.ilike(f"%{organization}%"))

    heap: list[tuple[float, int, ChunkHit]] = []  # min-heap из top-K
    counter = 0
    last_id: str | None = None
    while True:
        stmt = base
        if last_id is not None:
            stmt = stmt.where(DocumentChunk.id > last_id)
        stmt = stmt.order_by(DocumentChunk.id).limit(SQLITE_SCAN_BATCH)
        rows = (await db.execute(stmt)).all()
        if not rows:
            break
        last_id = rows[-1].id

        missing = [row.id for row in rows if _cache_get(row.id) is None]
        if missing:
            await _load_vectors(db, missing)
        kept = [(row, _cache_get(row.id)) for row in rows]
        kept = [(row, vec) for row, vec in kept if vec is not None]
        if not kept:
            continue
        scores = np.stack([vec for _, vec in kept]) @ query
        best = np.argsort(-scores)[:limit]
        for index in best:
            row = kept[int(index)][0]
            score = float(scores[index])
            if len(heap) < limit:
                heapq.heappush(heap, (score, counter, _hit(row, score)))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, _hit(row, score)))
            counter += 1
    return [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]


def _hit(row, score: float) -> ChunkHit:
    return ChunkHit(
        document_id=row.document_id,
        filename=row.filename,
        doc_type=row.doc_type,
        status=row.status,
        page_number=row.page_number,
        chunk_text=row.chunk_text,
        similarity=score,
    )


def clear_vector_cache() -> None:
    with _cache_lock:
        _vector_cache.clear()


def vector_cache_size() -> int:
    with _cache_lock:
        return len(_vector_cache)


# ---------- mmap-индекс 1-bit + int8 для слабых машин (см. services/mmap_index.py) ----------
_INDEX_SYNC_BATCH = 2_000
_count_cache: dict[str, tuple[float, int]] = {}
_COUNT_TTL_SECONDS = 30.0
_indexes: dict[str, MmapVectorIndex] = {}


def _index_dir(engine: str) -> Path:
    return settings.DATA_DIR / "vector_index" / f"{engine}_{settings.EMBEDDING_DIM}"


def _get_index(engine: str) -> MmapVectorIndex:
    key = str(_index_dir(engine))
    index = _indexes.get(key)
    if index is None:
        index = MmapVectorIndex(_index_dir(engine), settings.EMBEDDING_DIM, rerank_bits=settings.VECTOR_RERANK_BITS)
        _indexes[key] = index
    return index


async def _engine_chunk_count(db: AsyncSession, engine: str) -> int:
    now = time.monotonic()
    cached = _count_cache.get(engine)
    if cached and now - cached[0] < _COUNT_TTL_SECONDS:
        return cached[1]
    count = int(await db.scalar(select(func.count()).select_from(DocumentChunk).where(
        DocumentChunk.embedding_model == engine)) or 0)
    _count_cache[engine] = (now, count)
    return count


async def use_mmap_index(db: AsyncSession, engine: str) -> bool:
    """auto: mmap, когда чанков больше VECTOR_CACHE_LIMIT (там float-кэш уже не помещается в RAM слабой машины)."""
    mode = settings.VECTOR_INDEX_MODE
    if mode == "float" or is_postgres(db):
        return False
    if mode == "mmap_binary":
        return True
    return await _engine_chunk_count(db, engine) > VECTOR_CACHE_LIMIT


async def sync_mmap_index(db: AsyncSession, engine: str) -> MmapVectorIndex:
    """Дозаписывает в индекс новые чанки (rowid больше курсора). Удалённые остаются и отсеиваются при выборке."""
    index = _get_index(engine)
    index.check_consistency()  # до чтения курсора: иначе курсор удалённого каталога пропустит строки
    top = int(await db.scalar(sql_text("SELECT COALESCE(MAX(rowid), 0) FROM document_chunks")) or 0)
    if top < index.cursor:
        # строки удалены с конца либо VACUUM перенумеровал rowid — курсор недостоверен, строим заново
        index.rebuild()
    while index.cursor < top:
        rows = (await db.execute(
            sql_text(
                "SELECT rowid AS rid, id, embedding FROM document_chunks "
                "WHERE rowid > :cursor AND rowid <= :top AND embedding_model = :engine "
                "ORDER BY rowid LIMIT :batch"
            ),
            {"cursor": index.cursor, "top": top, "engine": engine, "batch": _INDEX_SYNC_BATCH},
        )).all()
        if not rows:
            index.append([], None, cursor=top)
            break
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            embedding = row.embedding
            values = json.loads(embedding) if isinstance(embedding, str) else (embedding or [])
            if len(values) == settings.EMBEDDING_DIM:
                vec = np.asarray(values, dtype=np.float32)
                if float(np.linalg.norm(vec)) > 0.0:
                    ids.append(row.id)
                    vectors.append(vec)
        last = int(rows[-1].rid)
        if ids:
            index.append(ids, np.stack(vectors), cursor=last)
        else:
            index.append([], None, cursor=last)
    if index.ivf_needs_build(settings.IVF_MIN_VECTORS, settings.IVF_REBUILD_RATIO):
        # перестройка тяжёлая (K-Means + переукладка файлов), поэтому не блокирует event loop
        await asyncio.to_thread(index.build_ivf, settings.IVF_CLUSTERS)
    return index


async def top_chunks_mmap(
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
    """Top-k через mmap-индекс: POPCNT-скан 1-битных сигнатур → int8-доранжирование → фильтры по таблице.

    Фильтры применяются после выбора кандидатов, поэтому их берём с запасом (×10 от limit).
    """
    index = await sync_mmap_index(db, engine)
    query = np.asarray(query_vector, dtype=np.float32)
    if index.count == 0 or float(np.linalg.norm(query)) == 0.0:
        return []
    filtered = any([document_id, doc_type, status, date_from, date_to, organization])
    want = limit * (10 if filtered else 1)
    candidates = index.search(query, want, settings.VECTOR_INDEX_RERANK_FACTOR, nprobe=settings.IVF_NPROBE,
                              early_words=settings.VECTOR_EARLY_WORDS, early_z=settings.VECTOR_EARLY_Z)
    if not candidates:
        return []
    score_by_id = {c.chunk_id: c.score for c in candidates}
    base = (
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
        .where(DocumentChunk.id.in_(list(score_by_id)))
    )
    if document_id:
        base = base.where(DocumentChunk.document_id == document_id)
    if doc_type:
        base = base.where(Document.doc_type == doc_type)
    if status:
        base = base.where(Document.status == status)
    if date_from:
        base = base.where(Document.created_at >= date_from)
    if date_to:
        base = base.where(Document.created_at <= date_to)
    if organization:
        base = base.where(DocumentChunk.chunk_text.ilike(f"%{organization}%"))
    rows = (await db.execute(base)).all()  # удалённые чанки сюда не попадают
    hits = [_hit(row, score_by_id[row.id]) for row in rows]
    hits.sort(key=lambda h: h.similarity, reverse=True)
    return hits[:limit]
