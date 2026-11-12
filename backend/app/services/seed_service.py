"""Демо-данные для первого запуска (демонстрация возможностей платформы)."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from sqlalchemy import func, select

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.document import Document, DocumentStatus
from app.models.user import User, UserRole
from app.services import pipeline_service
from app.services.storage_service import original_key, storage

logger = logging.getLogger(__name__)

DEMO_EMAIL = "demo@idp-project.ru"
DEMO_PASSWORD = "demo12345"

SAMPLES_DIR = Path(__file__).resolve().parents[3] / "samples"


async def ensure_demo_user() -> User:
    async with SessionLocal() as session:
        user = await session.scalar(select(User).where(User.email == DEMO_EMAIL))
        if user:
            return user
        user = User(
            email=DEMO_EMAIL,
            password_hash=hash_password(DEMO_PASSWORD),
            full_name="Оператор верификации",
            role=UserRole.ADMIN,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        logger.info("Создан демо-пользователь %s / %s", DEMO_EMAIL, DEMO_PASSWORD)
        return user


async def seed_demo_data() -> None:
    try:
        user = await ensure_demo_user()
        async with SessionLocal() as session:
            existing = await session.scalar(select(func.count(Document.id))) or 0
        if existing:
            return

        from scripts.generate_samples import generate_samples

        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        files = [
            path
            for path in generate_samples(SAMPLES_DIR)
            # сканы (scan_*) требуют OCR-движка, а ground_truth.json — не документ
            if path.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
            and not path.name.startswith("scan_")
        ]
        if not files:
            return

        created: list[str] = []
        async with SessionLocal() as session:
            for path in files:
                data = path.read_bytes()
                document_id = str(uuid.uuid4())
                key = original_key(document_id, path.name)
                await storage.put(key, data)
                session.add(
                    Document(
                        id=document_id,
                        user_id=user.id,
                        filename=path.name,
                        s3_key_original=key,
                        mime_type="application/pdf" if path.suffix == ".pdf" else "application/octet-stream",
                        file_size=len(data),
                        status=DocumentStatus.PENDING,
                    )
                )
                created.append(document_id)
            await session.commit()

        logger.info("Загружено демонстрационных документов: %s", len(created))
        for document_id in created:
            await pipeline_service.process_document(document_id)
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось подготовить демо-данные")
