"""Шина событий статусов обработки для WebSocket-уведомлений.

* ``QUEUE_BACKEND=inline`` — события доставляются подписчикам в памяти процесса.
* ``QUEUE_BACKEND=celery`` — обработку делает отдельный воркер (другой процесс),
  поэтому каждое событие дополнительно публикуется в Redis Pub/Sub, а каждый
  процесс API слушает канал и пересылает события своим WebSocket-клиентам.
  Так работают и несколько воркеров uvicorn, и отдельный Celery-воркер.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

REDIS_CHANNEL = "idp:document-events"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # медленный клиент не должен тормозить остальных
                continue

    async def subscribe(self) -> AsyncGenerator[dict[str, Any], None]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                self._subscribers.discard(queue)


bus = EventBus()


async def publish_event(event: dict[str, Any]) -> None:
    """Публикует событие локально и (в celery-режиме) в Redis для остальных процессов."""
    await bus.publish(event)
    if settings.QUEUE_BACKEND == "celery":
        await _redis_publish(event)


async def _redis_publish(event: dict[str, Any]) -> None:
    import redis.asyncio as aioredis

    client = aioredis.from_url(settings.REDIS_URL)
    try:
        await client.publish(REDIS_CHANNEL, json.dumps(event, ensure_ascii=False, default=str))
    except Exception as exc:  # noqa: BLE001 - потеря одного уведомления не должна ломать обработку
        logger.warning("Не удалось опубликовать событие в Redis: %s", exc)
    finally:
        await client.aclose()


async def redis_listener() -> None:
    """Фоновая задача API: переносит события из Redis в локальную шину (с переподключением)."""
    import redis.asyncio as aioredis

    while True:
        client = aioredis.from_url(settings.REDIS_URL)
        try:
            pubsub = client.pubsub()
            await pubsub.subscribe(REDIS_CHANNEL)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    await bus.publish(json.loads(message["data"]))
                except (ValueError, TypeError):
                    logger.warning("Некорректное событие из Redis проигнорировано")
        except asyncio.CancelledError:
            await client.aclose()
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis Pub/Sub недоступен (%s), повтор через 3 с", exc)
            await asyncio.sleep(3)
        finally:
            await client.aclose()
