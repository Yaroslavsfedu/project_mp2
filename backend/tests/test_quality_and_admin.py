"""Метрики качества (CER/WER/F1), журнал аудита и RBAC администратора."""
from __future__ import annotations

import json

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.audit import AuditLog
from app.models.user import User, UserRole
from app.services import quality_service


# ---------------------------------------------------------------- метрики
def test_levenshtein_basic():
    assert quality_service.levenshtein("кот", "кот") == 0
    assert quality_service.levenshtein("кот", "кит") == 1
    assert quality_service.levenshtein("", "абв") == 3


def test_cer_normalizes_whitespace_and_case():
    reference = "ДОГОВОР  поставки № 47/2024"
    hypothesis = "договор поставки № 47/2024\n"
    assert quality_service.cer(reference, hypothesis) == 0.0


def test_cer_counts_errors():
    score = quality_service.cer("договор поставки", "договор паставки")
    assert 0 < score < 0.2


def test_wer_uses_token_level_distance():
    # слово переставлено: ошибка одного токена, а не «разъехавшаяся» строка
    assert quality_service.wer("акт выполненных работ", "акт работ выполненных") == pytest.approx(2 / 3)
    assert quality_service.wer("счёт на оплату", "счёт на оплату") == 0.0


def test_wer_empty_reference():
    assert quality_service.wer("", "") == 0.0
    assert quality_service.wer("", "лишние слова") == 1.0


def test_entity_score_exact_and_missing():
    reference = {"persons": ["Иванов И. И."], "dates": ["12.03.2024"]}
    predicted = {"persons": ["Иванов И. И."], "dates": ["13.03.2024"], "organizations": ["ООО «Тест»"]}
    score = quality_service.score_entities(reference, predicted)
    assert score.per_field["persons"]["tp"] == 1
    assert score.per_field["dates"]["fp"] == 1 and score.per_field["dates"]["fn"] == 1
    assert score.per_field["organizations"]["fp"] == 1
    assert 0 < score.f1 < 1


def test_entity_score_fuzzy_addresses():
    reference = {"addresses": ["344002, г. Ростов-на-Дону, пр. Будённовский, д. 21, офис 4"]}
    predicted = {"addresses": ["г. Ростов-на-Дону, пр. Будённовский, д. 21, офис 4"]}
    score = quality_service.score_entities(reference, predicted)
    assert score.per_field["addresses"]["tp"] == 1
    assert score.per_field["addresses"]["fp"] == 0


def test_build_report_aggregates_and_marks_worst():
    per_document = [
        {
            "filename": "a.pdf",
            "cer": 0.01,
            "wer": 0.02,
            "reference_entities": {"dates": ["12.03.2024"]},
            "predicted_entities": {"dates": ["12.03.2024"]},
            "doc_type_expected": "Договор",
            "doc_type_predicted": "Договор",
            "doc_type_correct": True,
            "duration_sec": 0.5,
            "status": "completed",
        },
        {
            "filename": "b.pdf",
            "cer": 0.2,
            "wer": 0.3,
            "reference_entities": {"dates": ["12.03.2024"]},
            "predicted_entities": {"dates": []},
            "doc_type_expected": "Счет",
            "doc_type_predicted": "Акт",
            "doc_type_correct": False,
            "duration_sec": 1.5,
            "status": "completed",
        },
    ]
    report = quality_service.build_report(per_document, {"llm_engine": "fallback"}, skipped=[{"filename": "scan.jpg"}])

    assert report["documents"] == 2
    assert report["metrics"]["doc_type_accuracy"] == 0.5
    assert report["metrics"]["avg_duration_sec"] == 1.0
    assert report["worst_documents"][0]["filename"] == "b.pdf"
    assert report["skipped"][0]["filename"] == "scan.jpg"

    markdown = quality_service.report_to_markdown(report)
    assert "CER" in markdown and "b.pdf" in markdown


def test_save_and_load_report(tmp_path):
    report = {"generated_at": "2026-01-01T00:00:00", "documents": 1, "metrics": {"cer": 0.1}}
    json_path, md_path = quality_service.save_report(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["documents"] == 1
    assert md_path.exists()
    assert quality_service.load_report(json_path)["metrics"]["cer"] == 0.1
    assert quality_service.load_report(tmp_path / "missing.json") is None


# ---------------------------------------------------------------- rate limiting
def test_sliding_window_limiter():
    from app.core.ratelimit import SlidingWindowLimiter

    limiter = SlidingWindowLimiter(limit=3, window=60.0)
    assert [limiter.allow("ip", now=moment) for moment in (0, 1, 2, 3)] == [True, True, True, False]
    assert limiter.allow("ip", now=61) is True
    assert limiter.allow("other-ip", now=61) is True


async def test_login_rate_limited(client: AsyncClient, monkeypatch):
    from app.core.config import settings
    from app.core.ratelimit import login_limiter

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(login_limiter, "limit", 1)
    login_limiter.reset()

    payload = {"email": "ratelimit@example.ru", "password": "wrong-password"}
    first = await client.post("/api/v1/auth/login", json=payload)
    second = await client.post("/api/v1/auth/login", json=payload)

    login_limiter.reset()
    assert first.status_code == 401
    assert second.status_code == 429


# ---------------------------------------------------------------- аудит
async def _make_user(email: str, role: UserRole) -> dict:
    async with SessionLocal() as session:
        user = User(email=email, password_hash="-", full_name="Тест", role=role, is_active=True)
        session.add(user)
        await session.commit()
        return {"id": user.id, "email": user.email, "role": role}


async def test_log_action_persists_entry(client: AsyncClient):
    async with SessionLocal() as session:
        user = User(email="audit-target@example.ru", password_hash="-", full_name="Аудит", role=UserRole.OPERATOR)
        session.add(user)
        await session.commit()
        await session.refresh(user)

        from app.services.audit_service import log_action

        await log_action(
            session,
            action="document_uploaded",
            user=user,
            details={"size": 1234},
            ip_address="127.0.0.1",
        )
        await session.commit()

    async with SessionLocal() as session:
        entry = await session.scalar(
            select(AuditLog).where(AuditLog.user_email == "audit-target@example.ru")
        )
        assert entry is not None
        assert entry.action == "document_uploaded"
        assert entry.details["size"] == 1234
        assert entry.ip_address == "127.0.0.1"


async def test_verify_endpoint_writes_audit(
    client: AsyncClient, auth_headers: dict[str, str], sample_pdf
):
    documents = await client.get("/api/v1/documents?limit=1", headers=auth_headers)
    items = documents.json()["items"]
    if not items:
        with sample_pdf.open("rb") as handle:
            upload = await client.post(
                "/api/v1/documents/upload",
                files={"files": (sample_pdf.name, handle, "application/pdf")},
                headers=auth_headers,
            )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["documents"][0]["id"]
    else:
        document_id = items[0]["id"]

    detail = await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)
    entities = detail.json()["extracted"]["entities"] if detail.json().get("extracted") else {}

    response = await client.put(
        f"/api/v1/documents/{document_id}/verify",
        json={"entities": entities, "is_verified": True},
        headers=auth_headers,
    )
    assert response.status_code == 200

    async with SessionLocal() as session:
        entry = await session.scalar(
            select(AuditLog)
            .where(AuditLog.document_id == document_id, AuditLog.action == "document_verified")
            .order_by(AuditLog.created_at.desc())
        )
        assert entry is not None


async def _token_for(email: str, role: UserRole) -> dict[str, str]:
    """Создаёт пользователя с нужной ролью и возвращает заголовок авторизации."""
    from app.core.security import create_access_token

    user = await _make_user(email, role)
    return {"Authorization": f"Bearer {create_access_token(subject=user['id'])}"}


async def test_admin_endpoints_require_admin_role(client: AsyncClient):
    headers = await _token_for("operator-rbac@example.ru", UserRole.OPERATOR)
    for path in ("/api/v1/admin/users", "/api/v1/admin/audit", "/api/v1/admin/overview"):
        response = await client.get(path, headers=headers)
        assert response.status_code == 403, path


async def test_admin_can_list_users_and_change_role(client: AsyncClient):
    headers = await _token_for("admin-tests@example.ru", UserRole.ADMIN)
    async with SessionLocal() as session:
        target = await session.scalar(select(User).where(User.email == "operator-rbac@example.ru"))
        target_id = target.id
        await session.commit()

    users = await client.get("/api/v1/admin/users", headers=headers)
    assert users.status_code == 200
    assert any(item["email"] == "operator-rbac@example.ru" for item in users.json())

    updated = await client.patch(
        f"/api/v1/admin/users/{target_id}", json={"role": "viewer"}, headers=headers
    )
    assert updated.status_code == 200
    assert updated.json()["role"] == "viewer"

    overview = await client.get("/api/v1/admin/overview", headers=headers)
    assert overview.status_code == 200
    assert overview.json()["users_total"] >= 2

    audit = await client.get("/api/v1/admin/audit?action=user_updated", headers=headers)
    assert audit.status_code == 200
    assert any(entry["action"] == "user_updated" for entry in audit.json())


async def test_admin_cannot_block_self(client: AsyncClient):
    headers = await _token_for("admin-self@example.ru", UserRole.ADMIN)
    me = (await client.get("/api/v1/auth/me", headers=headers)).json()
    response = await client.patch(
        f"/api/v1/admin/users/{me['id']}", json={"is_active": False}, headers=headers
    )
    assert response.status_code == 400


async def test_system_status_reports_components(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.get("/api/v1/system/status", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()

    assert set(payload) >= {"llm", "ocr", "gpu", "cpu", "hints", "storage_backend"}
    assert payload["ocr"]["engine"] == "none"  # в тестах нет ни Tesseract, ни Ollama
    assert payload["gpu"]["available"] is False
    assert any("OCR" in hint or "Ollama" in hint for hint in payload["hints"])
