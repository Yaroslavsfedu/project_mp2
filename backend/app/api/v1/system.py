from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.hardware import cpu_info, gpu_info
from app.core.security import get_current_user
from app.ml_pipeline.llm_service import llm_status
from app.ml_pipeline.ocr_engine import resolve_engine, tesseract_available
from app.models.document import Document, ExtractedData, OcrPage
from app.models.user import User
from app.schemas.document import DashboardStats

router = APIRouter(tags=["system"])


@router.get("/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION, "app": settings.APP_NAME}


@router.get("/system/status")
async def system_status(user: Annotated[User, Depends(get_current_user)]):
    """Состояние локального ИИ-стека и подсказки по запуску моделей.

    Важно для On-Premise развёртывания: интерфейс должен честно показывать,
    какие модели установлены, есть ли GPU и чего не хватает для полного пайплайна.
    """
    llm = await llm_status()
    ocr_engine = await resolve_engine()
    gpu = gpu_info()

    hints: list[str] = []
    if not llm["available"]:
        hints.append(
            "Ollama недоступна: анализ выполняется эвристическим fallback-движком. "
            "Запустите `ollama serve` (или контейнер ollama) для полноценной локальной LLM."
        )
    elif not llm["model_installed"]:
        hints.append(f"Модель {llm['model']} не загружена: выполните `ollama pull {llm['model']}`.")
    if llm["available"] and not llm["embedding_model_installed"]:
        hints.append(
            f"Модель эмбеддингов {llm['embedding_model']} не загружена: "
            f"`ollama pull {llm['embedding_model']}` (семантический поиск работает в fallback-режиме)."
        )
    if ocr_engine == "none":
        hints.append(
            "OCR-движок не найден: установите Tesseract (`apt install tesseract-ocr tesseract-ocr-rus`) "
            "или загрузите VLM (`ollama pull qwen2-vl:7b`) и включите OCR_ENGINE=auto/ollama. "
            "PDF с текстовым слоем, DOCX и TXT обрабатываются и без OCR."
        )
    if gpu.get("available") and gpu.get("recommended_llm") != llm["model"]:
        hints.append(
            f"Для GPU {gpu.get('name')} ({gpu.get('memory_mb')} МБ) рекомендуется модель "
            f"{gpu.get('recommended_llm')} — она лучше использует доступную видеопамять."
        )

    return {
        "llm": llm,
        "ocr": {
            "engine": ocr_engine,
            "configured": settings.OCR_ENGINE,
            "languages": settings.OCR_LANGUAGES,
            "tesseract_installed": tesseract_available(),
            "preprocess_enabled": settings.PREPROCESS_ENABLED,
        },
        "gpu": gpu,
        "cpu": cpu_info(),
        "storage_backend": settings.STORAGE_BACKEND,
        "queue_backend": settings.QUEUE_BACKEND,
        "max_upload_mb": settings.MAX_UPLOAD_MB,
        "hints": hints,
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
