"""Администрирование: пользователи, журнал аудита, отчёт о качестве модели."""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import require_roles
from app.models.audit import AuditLog
from app.models.document import Document, ExtractedData, OcrPage
from app.models.user import User, UserRole
from app.schemas.admin import AdminOverview, AuditLogOut, UserAdminOut, UserUpdate
from app.services import quality_service
from app.services.audit_service import log_action

router = APIRouter(prefix="/admin", tags=["admin"])

BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_ROOT.parent
REPORTS_DIR = Path(settings.DATA_DIR).resolve() / "reports" / "quality"
# отчёт, закоммиченный в репозиторий (reports/quality), виден админке без отдельного прогона
PROJECT_REPORTS_DIR = PROJECT_ROOT / "reports" / "quality"


def _quality_report_dir() -> Path:
    return REPORTS_DIR if (REPORTS_DIR / "quality.json").exists() else PROJECT_REPORTS_DIR


@router.get("/users", response_model=list[UserAdminOut])
async def list_users(
    admin: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    counts = dict(
        (row[0], row[1])
        for row in (
            await db.execute(select(Document.user_id, func.count(Document.id)).group_by(Document.user_id))
        ).all()
    )
    users = (await db.execute(select(User).order_by(User.created_at))).scalars().all()
    return [
        UserAdminOut(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            documents_count=counts.get(user.id, 0),
        )
        for user in users
    ]


@router.patch("/users/{user_id}", response_model=UserAdminOut)
async def update_user(
    user_id: str,
    payload: UserUpdate,
    admin: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Пользователь не найден")
    if user.id == admin.id and (payload.role or payload.is_active is False):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Нельзя изменить собственную роль или заблокировать себя"
        )

    changes: dict[str, str] = {}
    if payload.role and payload.role != user.role:
        # нельзя оставить систему без администратора
        if user.role == UserRole.ADMIN and payload.role != UserRole.ADMIN:
            admins = await db.scalar(
                select(func.count(User.id)).where(User.role == UserRole.ADMIN, User.is_active.is_(True))
            ) or 0
            if admins <= 1:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "В системе должен остаться хотя бы один администратор")
        changes["role"] = f"{user.role} → {payload.role}"
        user.role = payload.role
    if payload.is_active is not None and payload.is_active != user.is_active:
        changes["is_active"] = f"{user.is_active} → {payload.is_active}"
        user.is_active = payload.is_active
    if payload.full_name is not None:
        user.full_name = payload.full_name

    await log_action(
        db,
        action="user_updated",
        user=admin,
        details={"target": user.email, "changes": changes},
    )
    await db.commit()
    await db.refresh(user)
    documents_count = await db.scalar(
        select(func.count(Document.id)).where(Document.user_id == user.id)
    ) or 0
    return UserAdminOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        documents_count=documents_count,
    )


@router.get("/audit", response_model=list[AuditLogOut])
async def list_audit(
    admin: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(100, ge=1, le=500),
    action: str | None = None,
    document_id: str | None = None,
):
    stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if document_id:
        stmt = stmt.where(AuditLog.document_id == document_id)
    rows = (await db.execute(stmt)).scalars().all()
    return [AuditLogOut.model_validate(row) for row in rows]


@router.get("/overview", response_model=AdminOverview)
async def overview(
    admin: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    users_total = await db.scalar(select(func.count(User.id))) or 0
    documents_total = await db.scalar(select(func.count(Document.id))) or 0
    pages_total = await db.scalar(select(func.count(OcrPage.id))) or 0
    verified_total = await db.scalar(
        select(func.count(ExtractedData.id)).where(ExtractedData.is_verified.is_(True))
    ) or 0
    storage_bytes = await db.scalar(select(func.coalesce(func.sum(Document.file_size), 0))) or 0
    audit_events = await db.scalar(select(func.count(AuditLog.id))) or 0

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    action_rows = (
        await db.execute(
            select(AuditLog.action, func.count(AuditLog.id))
            .where(AuditLog.created_at >= since)
            .group_by(AuditLog.action)
        )
    ).all()

    return AdminOverview(
        users_total=users_total,
        documents_total=documents_total,
        pages_total=pages_total,
        verified_total=verified_total,
        storage_bytes=int(storage_bytes),
        audit_events=audit_events,
        actions_24h=[{"action": row[0], "count": row[1]} for row in action_rows],
    )


@router.get("/quality")
async def get_quality_report(admin: Annotated[User, Depends(require_roles(UserRole.ADMIN))]):
    """Последний отчёт о качестве (CER/WER/F1) из `data/reports/quality`."""
    report_dir = _quality_report_dir()
    report = quality_service.load_report(report_dir / "quality.json")
    if report is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Отчёт ещё не сформирован. Запустите оценку качества в админ-панели.",
        )
    markdown_path = report_dir / "quality.md"
    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else None
    return {"report": report, "markdown": markdown}


@router.post("/quality/run")
async def run_quality_evaluation(
    admin: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int | None = Query(None, ge=1, le=50),
):
    """Прогон эталонного датасета в изолированной временной БД.

    Считает CER/WER/NER F1 и сохраняет отчёт; рабочие документы не затрагиваются.
    """
    dataset = PROJECT_ROOT / "samples" / "ground_truth.json"
    if not dataset.exists():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Эталонный датасет не найден. Сгенерируйте его: python -m scripts.generate_samples",
        )

    eval_dir = Path(settings.DATA_DIR).resolve() / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONPATH": str(BACKEND_ROOT),
        "DATABASE_URL": f"sqlite+aiosqlite:///{eval_dir}/eval.db",
        "DATA_DIR": str(eval_dir / "data"),
        "SEED_DEMO": "false",
        "QUEUE_BACKEND": "inline",
    }
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "--dataset",
        str(dataset),
        "--out",
        str(REPORTS_DIR),
        "--quiet",
    ]
    if limit:
        command += ["--limit", str(limit)]

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(BACKEND_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=900)
    except asyncio.TimeoutError:
        process.kill()
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Оценка качества заняла слишком много времени") from None

    if process.returncode != 0:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Оценка не выполнена: {stderr.decode('utf-8', 'ignore')[-1200:]}",
        )

    report = quality_service.load_report(REPORTS_DIR / "quality.json")
    if report is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Отчёт не сформирован")

    await log_action(
        db,
        action="quality_evaluated",
        user=admin,
        details={"documents": report.get("documents"), "metrics": report.get("metrics", {})},
    )
    await db.commit()
    return {
        "report": report,
        "markdown": (REPORTS_DIR / "quality.md").read_text(encoding="utf-8"),
        "stdout": stdout.decode("utf-8", "ignore")[-2000:],
    }
