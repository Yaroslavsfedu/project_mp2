"""Быстрый SQLite-косинус (кэш нормализованных векторов) должен совпадать с эталонной реализацией."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.ml_pipeline.text_utils import hashing_embedding
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.services.vector_store import clear_vector_cache, top_chunks_python, top_chunks_sqlite

TEXTS = [
    "договор поставки компьютерной техники и оплата",
    "акт выполненных работ по ремонту здания",
    "счёт на оплату поставки мониторов",
    "приказ о назначении ответственного лица",
]


async def _make_document(session, chunks_spec: list[tuple[str, str]]) -> str:
    """chunks_spec: [(текст, движок эмбеддингов)]. Возвращает id документа."""
    user = User(email=f"vec-{uuid.uuid4().hex[:8]}@example.ru", password_hash="x", full_name="t", role="operator")
    session.add(user)
    await session.flush()
    document = Document(
        user_id=user.id,
        filename=f"vec-{uuid.uuid4().hex[:8]}.txt",
        s3_key_original="k",
        status=DocumentStatus.COMPLETED,
        doc_type="Договор",
    )
    session.add(document)
    await session.flush()
    session.add_all(
        DocumentChunk(
            document_id=document.id,
            page_number=1,
            chunk_index=i,
            chunk_text=text,
            embedding=hashing_embedding(text, settings.EMBEDDING_DIM),
            embedding_model=engine,
        )
        for i, (text, engine) in enumerate(chunks_spec)
    )
    await session.commit()
    return document.id


async def test_numpy_search_matches_reference_ranking():
    clear_vector_cache()
    async with SessionLocal() as session:
        doc_id = await _make_document(session, [(text, "fallback") for text in TEXTS])
        query = hashing_embedding("поставка компьютерной техники", settings.EMBEDDING_DIM)
        fast = await top_chunks_sqlite(session, query, "fallback", limit=4, document_id=doc_id)
        loaded = (await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        )).scalars().all()
        reference = top_chunks_python(list(loaded), query, "fallback", 4)

    # Сравниваем оценки по позициям и совпадение ненулевых результатов.
    # Чанки с нулевой оценкой (нет общих слов с запросом) идут в произвольном порядке при равенстве.
    for hit, (_, score) in zip(fast, reference, strict=True):
        assert abs(hit.similarity - score) < 1e-4
    positive_fast = {h.chunk_text for h in fast if h.similarity > 0}
    positive_ref = {c.chunk_text for c, s in reference if s > 0}
    assert positive_fast == positive_ref


async def test_other_engine_vectors_are_not_mixed():
    """Чанк, посчитанный другим движком, в выдачу не попадает — даже при полном совпадении текста."""
    clear_vector_cache()
    text = "уникальная фраза для проверки движка эмбеддингов"
    async with SessionLocal() as session:
        doc_id = await _make_document(session, [(text, "ollama"), (TEXTS[0], "fallback")])
        query = hashing_embedding(text, settings.EMBEDDING_DIM)
        fallback_hits = await top_chunks_sqlite(session, query, "fallback", limit=5, document_id=doc_id)
        ollama_hits = await top_chunks_sqlite(session, query, "ollama", limit=5, document_id=doc_id)

    assert all(h.chunk_text != text for h in fallback_hits)
    assert [h.chunk_text for h in ollama_hits] == [text]


async def test_stale_dimension_vectors_do_not_break_search():
    """Старые векторы другой размерности (после смены модели) не должны ронять поиск (раньше — 500)."""
    clear_vector_cache()
    text = "договор поставки для проверки смешанных размерностей"
    async with SessionLocal() as session:
        doc_id = await _make_document(session, [(text, "fallback"), (TEXTS[1], "fallback")])
        # чанк со старым 512-мерным вектором в той же базе
        stale = DocumentChunk(
            document_id=doc_id,
            page_number=1,
            chunk_index=99,
            chunk_text="старый чанк 512 измерений",
            embedding=hashing_embedding("старый чанк", 512),
            embedding_model="fallback",
        )
        session.add(stale)
        await session.commit()
        query = hashing_embedding(text, settings.EMBEDDING_DIM)
        hits = await top_chunks_sqlite(session, query, "fallback", limit=5, document_id=doc_id)

    assert hits, "поиск должен вернуть корректные чанки"
    assert hits[0].chunk_text == text
    assert all(h.chunk_text != "старый чанк 512 измерений" for h in hits)
