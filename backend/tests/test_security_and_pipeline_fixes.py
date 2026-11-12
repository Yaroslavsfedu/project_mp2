"""Регрессионные тесты на найденные и исправленные проблемы.

Каждый тест воспроизводит конкретную ошибку, которая была в коде до исправления.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient
from starlette.testclient import TestClient

from app.core.config import settings
from app.core.ratelimit import login_limiter
from app.ml_pipeline import llm_service
from app.ml_pipeline.text_utils import hashing_embedding
from app.core.http import content_disposition
from app.models.document import DocumentChunk
from app.services.search_service import top_chunks_python
from app.services.storage_service import safe_filename


# ---------------- регистрация и роли ----------------
async def test_register_ignores_requested_role(client: AsyncClient, auth_headers):
    """Раньше любой аноним регистрировался админом, передав role=admin."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "escalation@example.ru", "password": "secret123", "role": "admin"},
    )
    assert response.status_code == 201
    assert response.json()["user"]["role"] == "operator"

    token = response.json()["access_token"]
    forbidden = await client.get("/api/v1/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert forbidden.status_code == 403


# ---------------- rate limit ----------------
async def test_rate_limit_ignores_spoofed_forwarded_for(client: AsyncClient, monkeypatch):
    """Раньше подмена X-Forwarded-For обходила лимит входа (14 попыток без 429)."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    login_limiter.reset()
    statuses = []
    for i in range(12):
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.ru", "password": "wrong-pass"},
            headers={"X-Forwarded-For": f"10.1.1.{i}"},
        )
        statuses.append(response.status_code)
    login_limiter.reset()
    assert 429 in statuses


# ---------------- WebSocket ----------------
def test_websocket_requires_valid_token():
    """Раньше соединение открывалось без токена и отдавало статусы всех документов."""
    from app.main import app

    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/ws/documents") as ws:
                ws.receive_text()


# ---------------- файлы ----------------
async def test_upload_keeps_cyrillic_filename(client: AsyncClient, auth_headers, sample_pdf):
    """Раньше русские имена превращались в «______.pdf»."""
    data = sample_pdf.read_bytes()
    response = await client.post(
        "/api/v1/documents/upload",
        files={"files": ("Договор поставки.pdf", data, "application/pdf")},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    document = response.json()["documents"][0]
    assert document["filename"] == "Договор поставки.pdf"


async def test_text_upload_with_binary_content_rejected(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/documents/upload",
        files={"files": ("notes.txt", b"hello\x00\x01\x02binary", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 400


async def test_original_served_with_extension_based_type(client: AsyncClient, auth_headers):
    """Раньше HTML, загруженный как .txt с Content-Type text/html, отдавался как text/html."""
    payload = b"<script>alert(1)</script>"
    uploaded = await client.post(
        "/api/v1/documents/upload",
        files={"files": ("page.txt", payload, "text/html")},
        headers=auth_headers,
    )
    assert uploaded.status_code == 201, uploaded.text
    document_id = uploaded.json()["documents"][0]["id"]
    response = await client.get(f"/api/v1/documents/{document_id}/file", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["x-content-type-options"] == "nosniff"
    await client.delete(f"/api/v1/documents/{document_id}", headers=auth_headers)


def test_safe_filename_blocks_traversal_and_keeps_letters():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("Счёт №208.pdf") == "Счёт _208.pdf"
    assert "/" not in safe_filename("a/b\\c.pdf")


def test_content_disposition_is_latin1_safe():
    header = content_disposition("Акт сверки №15.pdf")
    header.encode("latin-1")  # не должно падать (иначе Starlette вернёт 500)
    assert "filename*=UTF-8''" in header


# ---------------- ИИ / пайплайн ----------------
async def test_page_correction_falls_back_without_llm(monkeypatch):
    async def offline() -> bool:
        return False

    monkeypatch.setattr(llm_service.ollama, "is_available", offline)
    text, engine = await llm_service.correct_page_text("Дого-\nвор  поставки ,  № 1")
    assert engine == "fallback"
    assert "Договор" in text


async def test_page_correction_rejects_truncated_llm_output(monkeypatch):
    """Если LLM «пересказала» или обрезала текст, берём детерминированный вариант."""
    async def online() -> bool:
        return True

    async def short_answer(prompt, system, timeout=None):
        return "Кратко."

    monkeypatch.setattr(llm_service.ollama, "is_available", online)
    monkeypatch.setattr(llm_service.ollama, "generate_text", short_answer)
    original = "Договор поставки компьютерной техники заключён между сторонами " * 5
    text, engine = await llm_service.correct_page_text(original)
    assert engine == "fallback"
    assert text.startswith("Договор поставки")


def test_vectors_from_other_engine_are_not_compared():
    """Раньше запрос (hashing) сравнивался с векторами bge-m3 — результат был случайным."""
    query = hashing_embedding("поставка техники", 1024)
    same = DocumentChunk(document_id="d1", page_number=1, chunk_text="a", embedding=query, embedding_model="fallback")
    other = DocumentChunk(document_id="d2", page_number=1, chunk_text="b", embedding=query, embedding_model="ollama")
    ranked = top_chunks_python([same, other], query, "fallback", 10)
    assert [chunk.document_id for chunk, _ in ranked] == ["d1"]


def test_embedding_dimension_matches_bge_m3():
    assert settings.EMBEDDING_DIM == 1024
