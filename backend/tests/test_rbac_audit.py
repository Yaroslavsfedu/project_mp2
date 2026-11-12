"""Регрессии: роли на записи, владение при экспорте/перезапуске, журнал входа и экспорта."""
import uuid

import pytest
from httpx import AsyncClient

from app.core.database import SessionLocal
from app.models.audit import AuditLog
from sqlalchemy import select


async def _register_and_login(client: AsyncClient, role: str | None = None) -> dict[str, str]:
    email = f"rbac-{uuid.uuid4().hex[:8]}@example.ru"
    password = "rbac12345"
    reg = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert reg.status_code in (200, 201), reg.text
    user_id = reg.json()["user"]["id"]
    if role:
        # назначаем роль через админа (первый зарегистрированный в тестовой БД может быть не админом)
        admin = await client.post("/api/v1/auth/login", json={"email": "test@example.ru", "password": "test12345"})
        if admin.status_code != 200:
            pytest.skip("нет админа для назначения роли")
        headers = {"Authorization": f"Bearer {admin.json()['access_token']}"}
        upd = await client.patch(f"/api/v1/admin/users/{user_id}", json={"role": role}, headers=headers)
        assert upd.status_code == 200, upd.text
    login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _upload_txt(client: AsyncClient, headers: dict[str, str], name: str = "rbac.txt") -> str:
    resp = await client.post(
        "/api/v1/documents/upload",
        files=[("files", (name, "Договор поставки № 9 от 01.02.2024".encode(), "text/plain"))],
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["documents"][0]["id"]


async def test_viewer_cannot_upload(client: AsyncClient, auth_headers):
    viewer = await _register_and_login(client, role="viewer")
    resp = await client.post(
        "/api/v1/documents/upload",
        files=[("files", ("v.txt", b"hello", "text/plain"))],
        headers=viewer,
    )
    assert resp.status_code == 403


async def test_viewer_cannot_reprocess_or_delete(client: AsyncClient, auth_headers):
    owner = await _register_and_login(client, role="operator")
    doc_id = await _upload_txt(client, owner)
    viewer = await _register_and_login(client, role="viewer")
    assert (await client.post(f"/api/v1/documents/{doc_id}/reprocess", headers=viewer)).status_code == 403
    assert (await client.delete(f"/api/v1/documents/{doc_id}", headers=viewer)).status_code == 403
    assert (await client.get(f"/api/v1/documents/{doc_id}", headers=viewer)).status_code == 200


async def test_documents_are_shared_for_reading_and_export(client: AsyncClient, auth_headers):
    """Документы общие: чтение и экспорт доступны любой роли; перезапуск — operator/admin."""
    owner = await _register_and_login(client, role="operator")
    doc_id = await _upload_txt(client, owner)
    other = await _register_and_login(client, role="operator")
    viewer = await _register_and_login(client, role="viewer")
    assert (await client.get(f"/api/v1/documents/{doc_id}/export", params={"format": "json"}, headers=viewer)).status_code == 200
    assert (await client.get(f"/api/v1/documents/{doc_id}", headers=other)).status_code == 200
    assert (await client.post(f"/api/v1/documents/{doc_id}/reprocess", headers=other)).status_code == 200
    assert (await client.post(f"/api/v1/documents/{doc_id}/reprocess", headers=viewer)).status_code == 403


async def test_only_owner_or_admin_can_delete(client: AsyncClient, auth_headers):
    owner = await _register_and_login(client, role="operator")
    doc_id = await _upload_txt(client, owner)
    other = await _register_and_login(client, role="operator")
    assert (await client.delete(f"/api/v1/documents/{doc_id}", headers=other)).status_code == 403
    assert (await client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)).status_code == 204


async def test_login_and_export_are_audited(client: AsyncClient, auth_headers):
    owner = await _register_and_login(client, role="operator")
    email = (await client.get("/api/v1/auth/me", headers=owner)).json()["email"]
    doc_id = await _upload_txt(client, owner)
    assert (await client.get(f"/api/v1/documents/{doc_id}/export", params={"format": "md"}, headers=owner)).status_code == 200
    await client.post("/api/v1/auth/login", json={"email": email, "password": "неверный-пароль"})

    async with SessionLocal() as session:
        actions = set(
            (await session.execute(select(AuditLog.action).where(AuditLog.user_email == email))).scalars().all()
        )
        failed = set(
            (await session.execute(select(AuditLog.action).where(AuditLog.action == "login_failed"))).scalars().all()
        )
    assert "login" in actions
    assert "document_exported" in actions
    assert "login_failed" in failed


async def test_quality_report_is_served_from_project_reports(client: AsyncClient, auth_headers):
    resp = await client.get("/api/v1/admin/quality", headers=auth_headers)
    # отчёт может отсутствовать в свежем клоне; тогда 404 с понятным текстом, а не 500
    assert resp.status_code in (200, 404)
    if resp.status_code == 404:
        assert "не сформирован" in resp.json()["detail"]
