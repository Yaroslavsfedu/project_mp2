"""Простой лимитер запросов для критичных эндпоинтов (без внешних зависимостей).

Полноценный rate limiting для промышленной эксплуатации ставится на уровне
API-gateway/nginx, но платформа должна быть защищена и в «коробочном» варианте:
локальная сеть вуза, слабые пароли, автоматические сканеры.

Реализация — скользящее окно в памяти процесса. Для нескольких воркеров
лимиты считаются на каждый процесс (документировано в README).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.core.config import settings


class SlidingWindowLimiter:
    """Скользящее окно: не более `limit` запросов за `window` секунд на ключ."""

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


login_limiter = SlidingWindowLimiter(settings.RATE_LIMIT_LOGIN_PER_MINUTE, 60.0)
upload_limiter = SlidingWindowLimiter(settings.RATE_LIMIT_UPLOAD_PER_MINUTE, 60.0)


def client_key(request: Request, suffix: str = "") -> str:
    ip = request.client.host if request.client else "unknown"
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip = forwarded.split(",")[0].strip() or ip
    return f"{ip}:{suffix}"


async def limit_login(request: Request) -> None:
    if not settings.RATE_LIMIT_ENABLED:
        return
    if not login_limiter.allow(client_key(request, "login")):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Слишком много попыток входа. Повторите через минуту.",
        )


async def limit_upload(request: Request) -> None:
    if not settings.RATE_LIMIT_ENABLED:
        return
    if not upload_limiter.allow(client_key(request, "upload")):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Превышен лимит загрузок в минуту. Подождите и повторите.",
        )
