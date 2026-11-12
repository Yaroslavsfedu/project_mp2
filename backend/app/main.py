"""Точка входа FastAPI-приложения IDP-платформы."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import api_router
from app.api.v1.documents import SERVE_MEDIA_TYPES
from app.core.config import settings
from app.core.database import init_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.events import redis_listener
from app.services.seed_service import seed_demo_data
from app.services.storage_service import FILE_PREFIXES, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("idp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.pages_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("База данных готова: %s", settings.DATABASE_URL)
    if settings.SEED_DEMO:
        asyncio.create_task(seed_demo_data())
    listener = None
    if settings.QUEUE_BACKEND == "celery":
        listener = asyncio.create_task(redis_listener())
    if settings.SECRET_KEY.startswith("change-me"):
        logger.warning(
            "SECRET_KEY не задан: используется значение по умолчанию. "
            "Сгенерируйте свой ключ (scripts/install.sh делает это автоматически)."
        )
    yield
    if listener is not None:
        listener.cancel()
    logger.info("Остановка приложения")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "IDP — Intelligent Document Processing. Полностью локальный контур: "
        "OCR + локальная LLM (Ollama/Qwen 2.5) + гибридный поиск + ручная верификация."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get(f"{settings.API_V1_PREFIX}/files/{{key:path}}", tags=["files"])
async def get_stored_file(
    key: str,
    user: Annotated[User, Depends(get_current_user)],
):
    """Выдача файлов из локального хранилища (аналог S3 presigned URL)."""
    # uploads/ — новые оригиналы (шардирование), originals/ — прежняя плоская схема (совместимость)
    allowed_prefixes = FILE_PREFIXES
    if not key.startswith(allowed_prefixes) or ".." in key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Файл не найден")
    try:
        path = await storage.local_path(key)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Файл не найден") from None
    if not Path(path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Файл не найден")
    suffix = Path(key).suffix.lower()
    return FileResponse(
        path,
        media_type=SERVE_MEDIA_TYPES.get(suffix, "application/octet-stream"),
        headers={"X-Content-Type-Options": "nosniff"},
    )


# Собранный фронтенд (production-образ Nginx/uvicorn) отдаётся с того же origin
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend_dist.exists():  # pragma: no cover
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
