"""Шина событий для WebSocket-уведомлений о статусах обработки.

В dev-режиме — in-memory pub/sub, в prod-контуре те же события рассылает
Celery-воркер через Redis Pub/Sub (см. `ml_pipeline/celery_worker.py`).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any


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
            except asyncio.QueueFull:  # pragma: no cover
                continue

    async def subscribe(self) -> AsyncGenerator[dict[str, Any], None]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                self._subscribers.discard(queue)


bus = EventBus()
