"""Celery-воркер фоновой обработки документов (QUEUE_BACKEND=celery).

Тот же пайплайн, что и в inline-режиме: здесь только обёртка для очереди.
События статусов рассылаются через WebSocket API (в prod-контуре — Redis Pub/Sub).
"""
from __future__ import annotations

import asyncio
import logging

from celery import Celery

from app.core.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "idp",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=60 * 30,
    task_soft_time_limit=60 * 25,
    worker_max_tasks_per_child=20,
    broker_connection_retry_on_startup=True,
)


@celery_app.task(name="idp.process_document", bind=True, max_retries=2)
def process_document_task(self, document_id: str) -> str:
    """Запускает асинхронный пайплайн обработки документа в отдельном event loop."""
    from app.core.database import engine, init_db
    from app.services.pipeline_service import process_document

    async def runner() -> None:
        try:
            await init_db()
            await process_document(document_id)
        finally:
            # соединения asyncpg привязаны к event loop этой задачи — не переиспользуем их
            await engine.dispose()

    try:
        asyncio.run(runner())
        return document_id
    except Exception as exc:  # noqa: BLE001
        logger.exception("Celery task failed for %s", document_id)
        raise self.retry(exc=exc, countdown=13) from exc

