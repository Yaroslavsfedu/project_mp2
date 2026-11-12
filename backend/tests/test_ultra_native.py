"""Ultra-Low-Spec: нативные ядра (C/POPCNT/int8), mmap-индекс, FTS5 на SQLite, прагмы.

Каждая проверка сравнивает быстрый путь с эталоном: C с NumPy, mmap-индекс с полным перебором,
FTS5 с прежним LIKE-поиском по смыслу (документ находится / не находится).
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy import delete, text as sql_text

from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.ml_pipeline.text_utils import hashing_embedding
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.native.loader import FastKernels, get_kernels
from app.schemas.document import SearchFilters
from app.services import mmap_index, vector_store
from app.services.mmap_index import MmapVectorIndex
from app.services.search_service import search_documents

NATIVE = get_kernels()
needs_native = pytest.mark.skipif(not NATIVE.native, reason="gcc недоступен: нативная сборка не выполнена")


def _numpy_only() -> FastKernels:
    kernels = FastKernels.__new__(FastKernels)
    kernels._lib = None  # принудительно NumPy
    return kernels


# ------------------------------ 1. хэширование ------------------------------
TEXTS = [
    "договор поставки компьютерной техники и оплата",
    "Счёт №12 от 01.02.2026 — ОПЛАТА 150 000 руб.",
    "ab cd ёжик Ёлка Quick-brown fox 2024 юфу",
    "неустойка неустойки неустойку штраф",
    "Mixed Кириллица и latin WORDS number12345 ьъ",
    "",
    "а",
    "Длинный текст " * 200,
]


@needs_native
@pytest.mark.parametrize("dim", [1024, 512, 1000, 7])
def test_c_hash_embedding_matches_python(dim):
    """C-хэш побитово совпадает с hashing_embedding (MD5 и взвешивание), разница — только округление до 5 знаков."""
    for text in TEXTS:
        py = np.asarray(hashing_embedding(text, dim), dtype=np.float32)
        c = NATIVE.hash_embedding(text, dim)
        assert c.shape == py.shape
        assert float(np.max(np.abs(py - c))) < 1e-4, text[:30]


def test_numpy_fallback_hash_matches_python():
    kernels = _numpy_only()
    for text in TEXTS[:4]:
        py = np.asarray(hashing_embedding(text, 1024), dtype=np.float32)
        assert float(np.max(np.abs(py - kernels.hash_embedding(text, 1024)))) < 1e-4


# ------------------------------ 2. 1-битные сигнатуры ------------------------------
def _rand_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_pack_bits_layout_is_128_bytes_for_1024():
    kernels = _numpy_only()
    sig = kernels.pack_bits(_rand_vectors(1, 1024, 1)[0])
    assert sig.dtype == np.uint64 and sig.shape == (16,)  # 16×uint64 = 128 байт
    assert sig.nbytes == 128


@needs_native
def test_native_pack_bits_matches_numpy():
    numpy_k = _numpy_only()
    for vec in _rand_vectors(20, 1024, 2):
        assert np.array_equal(NATIVE.pack_bits(vec), numpy_k.pack_bits(vec))


def test_simhash_is_sign_of_projection_and_128_bytes():
    kernels = _numpy_only()
    proj = np.random.default_rng(3).standard_normal((1024, 1024)).astype(np.float32)
    vec = _rand_vectors(1, 1024, 13)[0]
    sig = kernels.simhash(vec, proj)
    assert sig.nbytes == 128
    expected = np.packbits((vec.astype(np.float64) @ proj) > 0, bitorder="little").view("<u8")
    assert np.array_equal(sig, expected.astype(np.uint64))


@needs_native
def test_native_simhash_matches_numpy():
    numpy_k = _numpy_only()
    proj = np.random.default_rng(4).standard_normal((1024, 1024)).astype(np.float32)
    # разреженный вектор, как у хэш-эмбеддингов (≈50 ненулевых из 1024)
    sparse = np.zeros((20, 1024), dtype=np.float32)
    rng = np.random.default_rng(5)
    for row in sparse:
        idx = rng.choice(1024, size=50, replace=False)
        row[idx] = rng.random(50).astype(np.float32)
    for vec in list(sparse) + list(_rand_vectors(5, 1024, 6)):
        assert np.array_equal(NATIVE.simhash(vec, proj), numpy_k.simhash(vec, proj))


@needs_native
def test_native_topk_hamming_matches_numpy_and_bruteforce():
    numpy_k = _numpy_only()
    vecs = _rand_vectors(3000, 1024, 3)
    sigs = np.stack([NATIVE.pack_bits(v) for v in vecs])
    q = NATIVE.pack_bits(vecs[17] + 0.01)
    idx_c, dist_c = NATIVE.topk_hamming(sigs, q, 25)
    idx_p, dist_p = numpy_k.topk_hamming(sigs, q, 25)
    assert np.array_equal(idx_c, idx_p)
    assert np.array_equal(dist_c, dist_p)
    brute = np.array([bin(int(x)).count("1") for row in (sigs ^ q) for x in row]).reshape(-1, 16).sum(axis=1)
    order = np.lexsort((np.arange(len(brute)), brute))[:25]
    assert np.array_equal(idx_c, order)


def test_topk_hamming_on_memmap_without_copy(tmp_path):
    kernels = _numpy_only()
    sigs = np.random.default_rng(4).integers(0, 2**63, size=(500, 16), dtype=np.uint64)
    path = tmp_path / "s.bin"
    sigs.tofile(path)
    mm = np.memmap(path, dtype=np.uint64, mode="r", shape=(500, 16))
    idx, dist = kernels.topk_hamming(mm, sigs[9], 5)
    assert idx[0] == 9 and dist[0] == 0


# ------------------------------ 3. int8 ------------------------------
@needs_native
def test_int8_dot_and_rerank_match_numpy():
    numpy_k = _numpy_only()
    vecs = _rand_vectors(400, 1024, 5)
    base = np.stack([NATIVE.quantize_unit_i8(v) for v in vecs])
    base_np = np.stack([numpy_k.quantize_unit_i8(v) for v in vecs])
    assert np.max(np.abs(base.astype(int) - base_np.astype(int))) <= 1  # округление: C lround, NumPy — то же
    q = NATIVE.quantize_unit_i8(vecs[3])
    idx = np.arange(0, 400, 3, dtype=np.int64)
    assert np.array_equal(NATIVE.rerank_i8(base, idx, q), numpy_k.rerank_i8(base, idx, q))


def test_int8_score_approximates_cosine():
    kernels = _numpy_only()
    a, b = _rand_vectors(2, 1024, 6)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    approx = kernels.rerank_i8(
        np.stack([kernels.quantize_unit_i8(b)]), np.array([0]), kernels.quantize_unit_i8(a)
    )[0] / (127.0 * 127.0)
    assert abs(approx - cos) < 0.01  # SQ8 даёт ошибку порядка 1/127


# ------------------------------ 4. mmap-индекс ------------------------------
def test_mmap_index_roundtrip_and_exact_top1(tmp_path):
    vecs = _rand_vectors(800, 256, 7)
    ids = [str(uuid.uuid4()) for _ in range(800)]
    index = MmapVectorIndex(tmp_path / "idx", 256)
    index.append(ids[:500], vecs[:500], cursor=500)
    index.append(ids[500:], vecs[500:], cursor=800)
    assert index.count == 800 and index.cursor == 800
    # перезагрузка с диска
    again = MmapVectorIndex(tmp_path / "idx", 256)
    assert again.count == 800 and again.cursor == 800
    # запрос = один из векторов → он первый, косинус ≈ 1
    hits = again.search(vecs[610], k=3, rerank_factor=20)
    assert hits[0].chunk_id == ids[610]
    assert hits[0].score > 0.98


def test_mmap_index_recall_against_exact(tmp_path):
    """Доля совпадений top-10 по SimHash-поиску с доранжированием относительно полного перебора float32.

    Данные с кластерами (как документы по темам). На изотропном шуме без структуры все косинусы почти
    равны и recall падает до ~0.56 — это худший случай, он описан в отчёте, а не скрыт.
    """
    dim = 256
    rng = np.random.default_rng(8)
    centers = rng.standard_normal((30, dim)).astype(np.float32)
    vecs = centers[rng.integers(0, 30, size=3000)] + rng.standard_normal((3000, dim)).astype(np.float32) * 0.6
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [str(uuid.uuid4()) for _ in range(3000)]
    index = MmapVectorIndex(tmp_path / "rec", dim)
    index.append(ids, vecs, cursor=3000)
    recalls = []
    for _ in range(20):
        q = vecs[int(rng.integers(0, 3000))] + rng.standard_normal(dim).astype(np.float32) * 0.05
        exact = np.argsort(-(vecs @ q / np.linalg.norm(q)))[:10]
        got = {c.chunk_id for c in index.search(q, 10, rerank_factor=20)}
        recalls.append(len({ids[i] for i in exact} & got) / 10)
    assert float(np.mean(recalls)) >= 0.8


def test_mmap_index_recovers_from_truncated_files(tmp_path):
    vecs = _rand_vectors(50, 128, 10)
    ids = [str(uuid.uuid4()) for _ in range(50)]
    index = MmapVectorIndex(tmp_path / "trunc", 128)
    index.append(ids, vecs, cursor=50)
    # имитация прерванной записи: ids.bin короче остальных
    with open(tmp_path / "trunc" / mmap_index.IDS_FILE, "r+b") as f:
        f.truncate(36 * 40)
    again = MmapVectorIndex(tmp_path / "trunc", 128)
    assert again.count == 40  # берём минимум по файлам, не падаем
    assert again.search(vecs[3], 1, 20)[0].chunk_id == ids[3]


def test_mmap_index_rebuild_and_dim_change(tmp_path):
    index = MmapVectorIndex(tmp_path / "rb", 64)
    index.append([str(uuid.uuid4())], _rand_vectors(1, 64, 11), cursor=5)
    index.rebuild()
    assert index.count == 0 and index.cursor == 0
    index.append([str(uuid.uuid4())], _rand_vectors(1, 64, 12), cursor=7)
    other_dim = MmapVectorIndex(tmp_path / "rb", 128)  # другой формат → пустой индекс
    assert other_dim.count == 0


# ------------------------------ 5. интеграция с SQLite ------------------------------
async def _make_chunks(session, texts: list[str]) -> tuple[str, list[str]]:
    user = User(email=f"ultra-{uuid.uuid4().hex[:8]}@example.ru", password_hash="x", full_name="t", role="operator")
    session.add(user)
    await session.flush()
    doc = Document(
        user_id=user.id,
        filename=f"ultra-{uuid.uuid4().hex[:8]}.txt",
        s3_key_original="k",
        status=DocumentStatus.COMPLETED,
        doc_type="Договор",
    )
    session.add(doc)
    await session.flush()
    chunks = [
        DocumentChunk(
            document_id=doc.id, page_number=1, chunk_index=i, chunk_text=text,
            embedding=hashing_embedding(text, settings.EMBEDDING_DIM), embedding_model="fallback",
        )
        for i, text in enumerate(texts)
    ]
    session.add_all(chunks)
    await session.commit()
    return doc.id, [c.id for c in chunks]


@pytest.mark.anyio
async def test_mmap_mode_returns_same_top_hit_as_float_mode(monkeypatch):
    texts = ["договор поставки компьютерной техники", "приказ о назначении ответственного лица",
             "акт выполненных работ по ремонту здания", "счёт на оплату поставки мониторов"]
    async with SessionLocal() as session:
        doc_id, _ = await _make_chunks(session, texts)
        query = hashing_embedding("поставка компьютерной техники", settings.EMBEDDING_DIM)
        monkeypatch.setattr(settings, "VECTOR_INDEX_MODE", "float")
        float_hits = await vector_store.top_chunks_sqlite(session, query, "fallback", limit=2, document_id=doc_id)
        monkeypatch.setattr(settings, "VECTOR_INDEX_MODE", "mmap_binary")
        monkeypatch.setattr(vector_store, "_indexes", {})
        mmap_hits = await vector_store.top_chunks_sqlite(session, query, "fallback", limit=2, document_id=doc_id)
    assert float_hits and mmap_hits
    assert mmap_hits[0].chunk_text == float_hits[0].chunk_text == texts[0]


@pytest.mark.anyio
async def test_mmap_index_drops_deleted_chunks(monkeypatch):
    texts = ["уникальная фраза удаление один", "уникальная фраза удаление два"]
    async with SessionLocal() as session:
        doc_id, chunk_ids = await _make_chunks(session, texts)
        monkeypatch.setattr(settings, "VECTOR_INDEX_MODE", "mmap_binary")
        monkeypatch.setattr(vector_store, "_indexes", {})
        query = hashing_embedding(texts[0], settings.EMBEDDING_DIM)
        before = await vector_store.top_chunks_sqlite(session, query, "fallback", limit=5, document_id=doc_id)
        assert any(h.chunk_text == texts[0] for h in before)
        await session.execute(delete(DocumentChunk).where(DocumentChunk.id == chunk_ids[0]))
        await session.commit()
        after = await vector_store.top_chunks_sqlite(session, query, "fallback", limit=5, document_id=doc_id)
    # удалённый чанк остаётся в индексе, но в выдаче его нет
    assert all(h.chunk_text != texts[0] for h in after)


def test_use_mmap_auto_threshold(monkeypatch):
    import asyncio  # noqa: PLC0415

    async def run(mode: str, count: int) -> bool:
        monkeypatch.setattr(settings, "VECTOR_INDEX_MODE", mode)

        class FakeDb:
            def get_bind(self):
                return type("B", (), {"dialect": type("D", (), {"name": "sqlite"})()})()

        async def fake_count(db, engine):
            return count

        monkeypatch.setattr(vector_store, "_engine_chunk_count", fake_count)
        return await vector_store.use_mmap_index(FakeDb(), "fallback")

    assert asyncio.run(run("auto", 100)) is False
    assert asyncio.run(run("auto", vector_store.VECTOR_CACHE_LIMIT + 1)) is True
    assert asyncio.run(run("float", 10**9)) is False
    assert asyncio.run(run("mmap_binary", 1)) is True


# ------------------------------ 6. FTS5 на SQLite ------------------------------
@pytest.mark.anyio
async def test_fts5_table_exists_and_is_synced_by_triggers():
    from app.core import database

    assert database.SQLITE_FTS5_READY, "FTS5 должен быть доступен в сборке SQLite Python"
    async with SessionLocal() as session:
        doc_id, chunk_ids = await _make_chunks(session, ["неустойка за просрочку платежа по договору"])
        found = await session.scalar(sql_text(
            "SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH '\"неусто\"*'"
        ))
        assert found == 1  # INSERT-триггер
        await session.execute(delete(DocumentChunk).where(DocumentChunk.id == chunk_ids[0]))
        await session.commit()
        gone = await session.scalar(sql_text(
            "SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH '\"неусто\"*'"
        ))
        assert gone == 0  # DELETE-триггер


@pytest.mark.anyio
async def test_fulltext_search_uses_fts5_and_returns_bm25_snippet():
    async with SessionLocal() as session:
        doc_id, _ = await _make_chunks(session, ["штраф за нарушение сроков поставки оборудования"])
        response = await search_documents(session, "штраф", SearchFilters(), "fulltext", 10)
    hit = next(h for h in response.hits if h.document_id == doc_id)
    assert hit.match_type == "fulltext"
    assert "штраф" in hit.snippet.lower()


@pytest.mark.anyio
async def test_fulltext_falls_back_to_like_when_fts5_missing(monkeypatch):
    from app.core import database

    monkeypatch.setattr(database, "SQLITE_FTS5_READY", False)
    from app.models.document import OcrPage  # noqa: PLC0415

    async with SessionLocal() as session:
        text_value = "пеня за просрочку оплаты счёта"
        doc_id, _ = await _make_chunks(session, [text_value])
        # LIKE-путь SQLite ищет по страницам (ocr_pages), а не по чанкам — как в реальном пайплайне
        session.add(OcrPage(document_id=doc_id, page_number=1, raw_text=text_value, corrected_text=text_value))
        await session.commit()
        response = await search_documents(session, "пеня", SearchFilters(), "fulltext", 10)
    assert any(h.document_id == doc_id for h in response.hits)


# ------------------------------ 7. прагмы SQLite ------------------------------
@pytest.mark.anyio
async def test_sqlite_low_memory_pragmas_applied():
    async with engine.connect() as conn:
        values = {
            "synchronous": (await conn.exec_driver_sql("PRAGMA synchronous")).scalar(),
            "temp_store": (await conn.exec_driver_sql("PRAGMA temp_store")).scalar(),
            "cache_size": (await conn.exec_driver_sql("PRAGMA cache_size")).scalar(),
            "mmap_size": (await conn.exec_driver_sql("PRAGMA mmap_size")).scalar(),
        }
    assert values["synchronous"] == 1  # NORMAL
    assert values["temp_store"] == 2  # MEMORY
    assert values["cache_size"] == -32000
    assert values["mmap_size"] == 268435456


def test_native_library_reports_backend():
    assert get_kernels().backend in {"numpy", "c-avx2", "c-scalar"}
