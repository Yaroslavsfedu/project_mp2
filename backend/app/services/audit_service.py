"""Запись событий аудита (без падения основного запроса при ошибке записи)."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog
from app.models.document import Document
from app.models.user import User

logger = logging.getLogger(__name__)


async def log_action(
    db: AsyncSession,
    *,
    action: str,
    user: User | None = None,
    document: Document | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    """Добавляет запись в журнал. Коммит выполняет вызывающий код."""
    try:
        db.add(
            AuditLog(
                user_id=user.id if user else None,
                user_email=user.email if user else "system",
                action=action,
                document_id=document.id if document else None,
                document_name=document.filename if document else "",
                details=details or {},
                ip_address=ip_address,
            )
        )
    except Exception:  # noqa: BLE001 - аудит не должен ломать бизнес-операцию
        logger.exception("Не удалось записать событие аудита: %s", action)
