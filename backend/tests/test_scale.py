"""Масштабирование (2+ млн документов): шардирование файлов, keyset-пагинация, LRU-кэш векторов."""
from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services import vector_store
from app.services.storage_service import original_key, page_image_key, shard_dirs, storage
from app.ml_pipeline.text_utils import hashing_embedding

KEY_RE = re.compile(r"^uploads/[0-9a-f]{2}/[0-9a-f]{2}/(?P<id>[0-9a-f-]{36})_(?P<name>.+)$")


# ------------------------------- шардирование -------------------------------
def test_original_key_is_two_level_sharded():
    doc_id = str(uuid.uuid4())
    key = original_key(doc_id, "dogovor.pdf")
    match = KEY_RE.match(key)
    assert match, key
    assert match["id"] == doc_id
    assert key.split("/")[1] == doc_id[:2] and key.split("/")[2] == doc_id[2:4]


def test_page_image_key_is_sharded_per_document():
    doc_id = str(uuid.uuid4())
    key = page_image_key(doc_id, 3)
    assert key == f"pages/{doc_id[:2]}/{doc_id[2:4]}/{doc_id}/page-3.png"


@pytest.mark.parametrize("bad", ["", "../../etc/passwd", "abc", "a" * 36, "../" + "a" * 33])
def test_document_id_must_be_canonical_uuid(bad):
    """ID попадает в путь на диске — принимаем только канонический UUID (защита от Path Traversal)."""
    with pytest.raises(ValueError):
        shard_dirs(bad)


def test_shard_distribution_is_flat_enough():
    """При 200 тыс. документов в каталоге не больше ~нескольких файлов: 2 млн → ~30 на лист (65 536 листов)."""
    ids = [str(uuid.uuid4()) for _ in range(200_000)]
    leaves = Counter(shard_dirs(i) for i in ids)
    assert len(leaves) > 60_000  # почти все из 65 536 листов задействованы
    # среднее ~3 на каталог (распределение Пуассона): максимум ~12, а не 200 000 в одном каталоге
    assert max(leaves.values()) < 25


def test_storage_key_for_legacy_prefixes_still_safe():
    """Файлы прежней схемы и новые пути не должны выходить за корень хранилища."""
    with pytest.raises(ValueError):
        storage._local_path("../outside.txt")


# ------------------------------- API: файлы -------------------------------
async def _current_user_id(client: AsyncClient, headers: dict) -> str:
    return (await client.get("/api/v1/auth/me", headers=headers)).json()["id"]


async def test_upload_writes_sharded_original(client: AsyncClient, auth_headers):
    files = {"files": ("shard-check.txt", "Договор поставки № 77".encode(), "text/plain")}
    response = await client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
    assert response.status_code == 201, response.text
    document_id = response.json()["documents"][0]["id"]
    async with SessionLocal() as session:
        document = await session.get(Document, document_id)
    assert document.s3_key_original.startswith("uploads/")
    assert (storage.root / document.s3_key_original).exists()


async def test_legacy_flat_file_is_readable_and_deleted(client: AsyncClient, auth_headers):
    """Документ, загруженный до шардирования (originals/{id}/...), открывается и удаляется как раньше."""
    user_id = await _current_user_id(client, auth_headers)
    doc_id = str(uuid.uuid4())
    legacy_key = f"originals/{doc_id}/legacy.txt"
    await storage.put(legacy_key, "Старый документ".encode())
    async with SessionLocal() as session:
        session.add(
            Document(
                id=doc_id, user_id=user_id, filename="legacy.txt", s3_key_original=legacy_key,
                mime_type="text/plain", file_size=14, status=DocumentStatus.FAILED,
            )
        )
        await session.commit()

    response = await client.get(f"/api/v1/documents/{doc_id}/file", headers=auth_headers)
    assert response.status_code == 200
    assert response.content == "Старый документ".encode()

    response = await client.get(f"/api/v1/files/{legacy_key}", headers=auth_headers)
    assert response.status_code == 200

    response = await client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert response.status_code == 204
    assert not (storage.root / legacy_key).exists()
    assert not (storage.root / f"originals/{doc_id}").exists()


async def test_files_endpoint_serves_sharded_uploads(client: AsyncClient, auth_headers):
    files = {"files": ("serve.txt", b"serve me", "text/plain")}
    response = await client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
    key = None
    async with SessionLocal() as session:
        document = await session.get(Document, response.json()["documents"][0]["id"])
        key = document.s3_key_original
    response = await client.get(f"/api/v1/files/{key}", headers=auth_headers)
    assert response.status_code == 200
    assert response.content == b"serve me"
    # посторонние каталоги по-прежнему закрыты
    response = await client.get("/api/v1/files/secrets/key.txt", headers=auth_headers)
    assert response.status_code == 404


# ------------------------------- список: keyset -------------------------------
async def _seed_documents(user_id: str, tag: str, count: int) -> list[str]:
    base = datetime.now(UTC) - timedelta(days=1)
    ids: list[str] = []
    async with SessionLocal() as session:
        for i in range(count):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            session.add(
                Document(
                    id=doc_id, user_id=user_id, filename=f"{tag}-{i:03}.txt",
                    s3_key_original=f"uploads/x/{doc_id}.txt", mime_type="text/plain",
                    file_size=1, status=DocumentStatus.COMPLETED, progress=100, doc_type=tag,
                    created_at=base - timedelta(minutes=i),
                )
            )
        await session.commit()
    return ids


async def test_keyset_pagination_walks_all_rows_once(client: AsyncClient, auth_headers):
    user_id = await _current_user_id(client, auth_headers)
    tag = f"keyset-{uuid.uuid4().hex[:6]}"
    expected = await _seed_documents(user_id, tag, 25)

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        params = {"doc_type": tag, "limit": 7}
        if cursor:
            params["cursor"] = cursor
        response = await client.get("/api/v1/documents", params=params, headers=auth_headers)
        assert response.status_code == 200, response.text
        body = response.json()
        seen.extend(item["id"] for item in body["items"])
        pages += 1
        if cursor:
            assert body["total"] is None  # в курсорном режиме счётчик не считается
        else:
            assert body["total"] == 25  # первая страница без курсора — offset-режим с точным счётчиком
        if not body["has_more"]:
            break
        cursor = body["next_cursor"]
        assert cursor
    assert pages == 4  # 7+7+7+4
    assert len(seen) == len(set(seen)) == 25
    assert seen == [i for i in expected]  # порядок: новые сверху (created_at DESC)


async def test_keyset_with_explicit_after_params(client: AsyncClient, auth_headers):
    user_id = await _current_user_id(client, auth_headers)
    tag = f"explicit-{uuid.uuid4().hex[:6]}"
    expected = await _seed_documents(user_id, tag, 5)
    first = (await client.get(
        "/api/v1/documents", params={"doc_type": tag, "limit": 2}, headers=auth_headers
    )).json()["items"]
    last = first[-1]
    response = await client.get(
        "/api/v1/documents",
        params={"doc_type": tag, "limit": 10, "after_created_at": last["created_at"], "after_id": last["id"]},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    ids = [item["id"] for item in response.json()["items"]]
    assert ids == expected[2:]


async def test_offset_pagination_keeps_total(client: AsyncClient, auth_headers):
    user_id = await _current_user_id(client, auth_headers)
    tag = f"offset-{uuid.uuid4().hex[:6]}"
    await _seed_documents(user_id, tag, 12)
    body = (await client.get(
        "/api/v1/documents", params={"doc_type": tag, "page": 2, "limit": 5}, headers=auth_headers
    )).json()
    assert body["total"] == 12
    assert body["total_estimated"] is False
    assert len(body["items"]) == 5
    assert body["has_more"] is True
    assert body["next_cursor"]  # переход на курсор возможен и из offset-страницы
    via_cursor = (await client.get(
        "/api/v1/documents", params={"doc_type": tag, "limit": 5, "cursor": body["next_cursor"]},
        headers=auth_headers,
    )).json()
    page3 = (await client.get(
        "/api/v1/documents", params={"doc_type": tag, "page": 3, "limit": 5}, headers=auth_headers
    )).json()
    # body — страница 2, её next_cursor должен вернуть ровно страницу 3 offset-режима
    assert [i["id"] for i in via_cursor["items"]] == [i["id"] for i in page3["items"]]


async def test_deep_offset_is_rejected_with_hint(client: AsyncClient, auth_headers):
    response = await client.get("/api/v1/documents", params={"page": 10_000, "limit": 100}, headers=auth_headers)
    assert response.status_code == 400
    assert "cursor" in response.json()["detail"]


async def test_bad_cursor_and_unsupported_sort(client: AsyncClient, auth_headers):
    response = await client.get("/api/v1/documents", params={"cursor": "%%%"}, headers=auth_headers)
    assert response.status_code == 400
    response = await client.get(
        "/api/v1/documents", params={"cursor": "MjAyNi0wMS0wMVQwMDowMDowMHwx", "sort": "filename"},
        headers=auth_headers,
    )
    assert response.status_code == 400


async def test_count_cache_returns_same_total_within_ttl(client: AsyncClient, auth_headers):
    from app.services import document_list_service as dls

    user_id = await _current_user_id(client, auth_headers)
    tag = f"cache-{uuid.uuid4().hex[:6]}"
    await _seed_documents(user_id, tag, 3)
    dls.clear_count_cache()
    first = (await client.get("/api/v1/documents", params={"doc_type": tag}, headers=auth_headers)).json()["total"]
    await _seed_documents(user_id, tag, 2)
    second = (await client.get("/api/v1/documents", params={"doc_type": tag}, headers=auth_headers)).json()["total"]
    assert first == second == 3  # второй запрос взят из кэша счётчика (TTL 30 с)


# ------------------------------- векторы: LRU и батчи -------------------------------
async def _vector_fixture(count: int) -> str:
    """Создаёт документ с count чанками (fallback-эмбеддинги). Возвращает id пользователя."""
    async with SessionLocal() as session:
        user_id = str(uuid.uuid4())
        from app.models.user import User

        session.add(User(id=user_id, email=f"lru-{user_id[:8]}@example.ru", password_hash="x", full_name="t"))
        await session.flush()
        for i in range(count):
            doc = Document(
                user_id=user_id, filename=f"lru-{i}.txt", s3_key_original="k", status=DocumentStatus.COMPLETED,
                doc_type="LRU",
            )
            session.add(doc)
            await session.flush()
            text_value = f"договор поставки номер {i} оплата {'мониторы' if i % 2 else 'бумага'}"
            session.add(
                DocumentChunk(
                    document_id=doc.id, page_number=1, chunk_index=0, chunk_text=text_value,
                    embedding=hashing_embedding(text_value), embedding_model="fallback",
                )
            )
        await session.commit()
    return user_id


async def test_sqlite_vector_search_lru_and_batches_match_reference(monkeypatch):
    """Маленький кэш и батч по 3 строки: результат должен совпасть с полным расчётом, а кэш не переполняться."""

    await _vector_fixture(17)
    vector_store.clear_vector_cache()
    monkeypatch.setattr(vector_store, "VECTOR_CACHE_LIMIT", 5)
    monkeypatch.setattr(vector_store, "SQLITE_SCAN_BATCH", 3)
    query = hashing_embedding("договор оплата мониторы")

    async with SessionLocal() as session:
        hits = await vector_store.top_chunks_sqlite(session, query, "fallback", limit=6, doc_type="LRU")
        chunks = list((await session.execute(
            select(DocumentChunk).join(Document).where(Document.doc_type == "LRU")
        )).scalars())
    reference = vector_store.top_chunks_python(chunks, query, "fallback", 6)

    assert len(hits) == 6
    # хеш-эмбеддинги одинаковы для «номер 1» и «номер 5» (цифры не учитываются): это ничьи,
    # поэтому сравниваем не тексты, а значения косинуса — они обязаны совпасть по позициям
    assert [round(h.similarity, 5) for h in hits] == [round(score, 5) for _, score in reference]
    assert vector_store.vector_cache_size() <= 5  # LRU удерживает лимит, не обнуляя кэш
    vector_store.clear_vector_cache()


async def test_lru_evicts_oldest_not_everything(monkeypatch):
    vector_store.clear_vector_cache()
    monkeypatch.setattr(vector_store, "VECTOR_CACHE_LIMIT", 3)
    vecs = {f"c{i}": np.ones(4, dtype=np.float32) for i in range(3)}
    vector_store._cache_put(vecs)
    assert vector_store._cache_get("c0") is not None  # c0 стал самым свежим
    vector_store._cache_put({"c3": np.ones(4, dtype=np.float32)})
    assert vector_store.vector_cache_size() == 3
    assert vector_store._cache_get("c1") is None  # вытеснен самый давний (c1)
    assert vector_store._cache_get("c0") is not None  # свежий сохранён
    vector_store.clear_vector_cache()


def test_hnsw_and_fts_settings_defaults():
    from app.core.config import settings

    assert settings.HNSW_EF_SEARCH >= 40
    assert settings.FTS_CANDIDATE_LIMIT == 2000
    assert settings.LIST_OFFSET_MAX == 50_000
    assert settings.COUNT_ESTIMATE_THRESHOLD == 100_000
