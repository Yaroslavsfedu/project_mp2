"""Лимитер запросов для критичных эндпоинтов (вход и загрузка).

Два хранилища счётчиков:

* **Redis** (`RATE_LIMIT_BACKEND=redis|auto`) — общий для всех воркеров uvicorn и Celery;
  фиксированное окно: INCR + EXPIRE на ключ `rl:<scope>:<ip>:<номер окна>`.
* **память процесса** (`RATE_LIMIT_BACKEND=memory`, или откат, если Redis недоступен) —
  скользящее окно, работает без внешних сервисов.

Режим `auto` (по умолчанию) использует Redis, если он отвечает, и незаметно переключается
на память при сбое. Сбой фиксируется в логе; повторная попытка Redis — через 30 секунд.

IP клиента берётся из `X-Forwarded-For` только при `TRUST_PROXY_HEADERS=true`, и тогда
используется ПОСЛЕДНИЙ адрес в цепочке: его добавил доверенный прокси, а левые адреса
может подставить клиент. Без доверенного прокси учитывается адрес соединения.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.core.config import settings

logger = logging.getLogger("idp.ratelimit")

REDIS_RETRY_SECONDS = 30.0


class SlidingWindowLimiter:
    """Скользящее окно: не более `limit` запросов за `window` секунд на ключ (в памяти процесса)."""

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        moment = now if now is not None else time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and moment - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(moment)
            # не даём словарю расти бесконечно
            if len(self._hits) > 10_000:
                for stale_key in [k for k, v in self._hits.items() if not v][:5_000]:
                    self._hits.pop(stale_key, None)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class RedisWindowCounter:
    """Фиксированное окно в Redis; общий счётчик для всех процессов."""

    def __init__(self, name: str, window: int):
        self.name = name
        self.window = int(window)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                settings.REDIS_URL, socket_timeout=0.5, socket_connect_timeout=0.5
            )
        return self._client

    async def allow(self, key: str, limit: int) -> bool:
        bucket = int(time.time() // self.window)
        redis_key = f"rl:{self.name}:{key}:{bucket}"
        client = self._get_client()
        count = await client.incr(redis_key)
        if count == 1:
            await client.expire(redis_key, self.window + 1)
        return count <= limit


class HybridLimiter:
    """Лимит на сценарий: Redis (общий) с откатом на память (локальный)."""

    def __init__(self, name: str, limit: int, window: float):
        self.name = name
        self.limit = limit
        self.window = window
        self.local = SlidingWindowLimiter(limit, window)
        self._redis = RedisWindowCounter(name, int(window))
        self._redis_retry_at = 0.0

    def reset(self) -> None:
        self.local.reset()

    def _use_redis(self) -> bool:
        backend = settings.RATE_LIMIT_BACKEND
        if backend == "memory":
            return False
        return time.monotonic() >= self._redis_retry_at

    async def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        if self._use_redis():
            try:
                return await self._redis.allow(key, self.limit)
            except Exception as exc:  # noqa: BLE001 — любой сбой Redis не должен ронять API
                self._redis_retry_at = time.monotonic() + REDIS_RETRY_SECONDS
                logger.warning(
                    "Redis недоступен для лимитера %s, использую память процесса: %s",
                    self.name,
                    exc,
                )
        self.local.limit = self.limit  # limit может меняться (настройки, тесты)
        return self.local.allow(key)


login_limiter = HybridLimiter("login", settings.RATE_LIMIT_LOGIN_PER_MINUTE, 60.0)
upload_limiter = HybridLimiter("upload", settings.RATE_LIMIT_UPLOAD_PER_MINUTE, 60.0)


def client_key(request: Request, suffix: str = "") -> str:
    ip = request.client.host if request.client else "unknown"
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        parts = [part.strip() for part in forwarded.split(",") if part.strip()]
        if parts:
            ip = parts[-1]
    return f"{ip}:{suffix}"


async def limit_login(request: Request) -> None:
    if not settings.RATE_LIMIT_ENABLED:
        return
    if not await login_limiter.allow(client_key(request, "login")):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Слишком много попыток входа. Повторите через минуту.",
        )


async def limit_upload(request: Request) -> None:
    if not settings.RATE_LIMIT_ENABLED:
        return
    if not await upload_limiter.allow(client_key(request, "upload")):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Превышен лимит загрузок в минуту. Подождите и повторите.",
        )
