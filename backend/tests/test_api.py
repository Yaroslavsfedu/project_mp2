import asyncio

from httpx import AsyncClient


async def test_health(client: AsyncClient):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_protected_endpoint_requires_token(client: AsyncClient):
    response = await client.get("/api/v1/documents")
    assert response.status_code == 401


async def test_login_and_profile(client: AsyncClient, auth_headers):
    response = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.ru"


async def test_upload_rejects_fake_pdf(client: AsyncClient, auth_headers):
    files = {"files": ("fake.pdf", b"this is not a pdf", "application/pdf")}
    response = await client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
    assert response.status_code == 400
    assert "Magic Bytes" in response.json()["detail"]


async def test_upload_rejects_unknown_extension(client: AsyncClient, auth_headers):
    files = {"files": ("script.exe", b"MZ\x90\x00", "application/octet-stream")}
    response = await client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
    assert response.status_code == 415


async def test_full_pipeline_search_and_export(client: AsyncClient, auth_headers, sample_pdf):
    content = sample_pdf.read_bytes()
    files = {"files": (sample_pdf.name, content, "application/pdf")}
    response = await client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
    assert response.status_code == 201, response.text
    document_id = response.json()["documents"][0]["id"]

    detail = None
    for _ in range(60):
        await asyncio.sleep(0.5)
        detail_response = await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)
        assert detail_response.status_code == 200
        detail = detail_response.json()
        if detail["status"] in ("completed", "failed"):
            break

    assert detail is not None
    assert detail["status"] == "completed", detail.get("error_message")
    assert detail["doc_type"] == "Договор"
    assert detail["pages"], "должна быть хотя бы одна страница"
    page = detail["pages"][0]
    assert page["bounding_boxes"], "Bounding Boxes должны быть заполнены"
    assert page["image_url"]
    assert detail["entities"]["amounts"], "суммы должны быть извлечены"
    assert detail["summary"]

    # изображение страницы отдаётся с авторизацией
    image = await client.get(f"/api/v1/documents/{document_id}/pages/1/image", headers=auth_headers)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"

    # поиск находит документ
    search = await client.post(
        "/api/v1/search",
        json={"query": "неустойка за просрочку поставки", "mode": "hybrid", "limit": 10},
        headers=auth_headers,
    )
    assert search.status_code == 200
    hits = search.json()["hits"]
    assert any(hit["document_id"] == document_id for hit in hits)

    # экспорт в markdown
    export = await client.get(
        f"/api/v1/documents/{document_id}/export?format=md", headers=auth_headers
    )
    assert export.status_code == 200
    assert "Договор" in export.text

    # верификация оператором
    verify = await client.put(
        f"/api/v1/documents/{document_id}/verify",
        json={"summary": "Проверено в тесте", "is_verified": True},
        headers=auth_headers,
    )
    assert verify.status_code == 200
    assert verify.json()["status"] == "verified"
    assert verify.json()["is_verified"] is True

    # чат по документу
    chat = await client.post(
        f"/api/v1/documents/{document_id}/chat",
        json={"question": "Какая стоимость договора?"},
        headers=auth_headers,
    )
    assert chat.status_code == 200
    assert chat.json()["answer"]

    # удаление
    delete = await client.delete(f"/api/v1/documents/{document_id}", headers=auth_headers)
    assert delete.status_code == 204
    missing = await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)
    assert missing.status_code == 404
