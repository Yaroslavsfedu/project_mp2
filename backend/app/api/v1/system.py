from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.ml_pipeline.llm_service import llm_status
from app.ml_pipeline.ocr_engine import available_engine
from app.models.document import Document, OcrPage, ExtractedData, DocumentStatus
from app.models.user import User
from app.schemas.document import DashboardStats

router = APIRouter(tags=["system"])


@router.get("/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION, "app": settings.APP_NAME}


@router.get("/system/status")
async def system_status(user: Annotated[User, Depends(get_current_user)]):
    """Состояние локального ИИ-стека — важно для On-Premise развёртывания."""
    return {
        "llm": await llm_status(),
        "ocr_engine": available_engine(),
        "storage_backend": settings.STORAGE_BACKEND,
        "queue_backend": settings.QUEUE_BACKEND,
        "preprocess_enabled": settings.PREPROCESS_ENABLED,
        "max_upload_mb": settings.MAX_UPLOAD_MB,
    }


@router.get("/dashboard/stats", response_model=DashboardStats)
async def dashboard_stats(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    total = await db.scalar(select(func.count(Document.id))) or 0
    status_rows = await db.execute(select(Document.status, func.count(Document.id)).group_by(Document.status))
    by_status = {row[0]: row[1] for row in status_rows}
    type_rows = await db.execute(
        select(Document.doc_type, func.count(Document.id)).group_by(Document.doc_type)
    )
    by_type = {row[0] or "Не определён": row[1] for row in type_rows}
    verified = await db.scalar(
        select(func.count(ExtractedData.id)).where(ExtractedData.is_verified.is_(True))
    ) or 0
    pages = await db.scalar(select(func.count(OcrPage.id))) or 0
    avg_conf = await db.scalar(select(func.avg(OcrPage.avg_confidence))) or 0.0

    day_rows = await db.execute(
        select(func.date(Document.created_at), func.count(Document.id))
        .group_by(func.date(Document.created_at))
        .order_by(func.date(Document.created_at))
    )
    last_7 = [{"date": str(row[0]), "count": row[1]} for row in day_rows][-7:]

    return DashboardStats(
        total_documents=total,
        by_status=by_status,
        by_type=by_type,
        verified_count=verified,
        pages_processed=pages,
        avg_confidence=round(float(avg_conf), 3),
        last_7_days=last_7,
    )
