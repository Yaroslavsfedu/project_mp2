"""Лимитер: доверие к X-Forwarded-For, общий счётчик в Redis и откат на память."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from starlette.requests import Request

from app.core.config import settings
from app.core.ratelimit import HybridLimiter, client_key


def _request(headers: dict[str, str], client_host: str = "203.0.113.7") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (client_host, 40000),
    }
    return Request(scope)


@pytest.fixture()
def restore_settings(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_BACKEND", "auto")
    yield


def test_forwarded_for_ignored_without_trusted_proxy(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    request = _request({"X-Forwarded-For": "10.0.0.1"})
    assert client_key(request, "login") == "203.0.113.7:login"


def test_forwarded_for_uses_last_hop_when_proxy_trusted(monkeypatch):
    """Левый адрес подставляет клиент; последний добавляет доверенный прокси."""
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    request = _request({"X-Forwarded-For": "1.2.3.4, 198.51.100.9"})
    assert client_key(request, "login") == "198.51.100.9:login"


def test_redis_unavailable_falls_back_to_memory(monkeypatch, restore_settings):
    monkeypatch.setattr(settings, "REDIS_URL", "redis://127.0.0.1:1/0")
    limiter = HybridLimiter(f"fallback-{uuid.uuid4().hex[:6]}", limit=2, window=60.0)
    results = [asyncio.run(limiter.allow("ip")) for _ in range(3)]
    assert results == [True, True, False]


def test_memory_backend_never_calls_redis(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_BACKEND", "memory")
    monkeypatch.setattr(settings, "REDIS_URL", "redis://127.0.0.1:1/0")
    limiter = HybridLimiter(f"mem-{uuid.uuid4().hex[:6]}", limit=1, window=60.0)
    assert asyncio.run(limiter.allow("ip")) is True
    assert asyncio.run(limiter.allow("ip")) is False
    assert limiter._redis_retry_at == 0.0  # Redis ни разу не трогался


def _redis_available() -> bool:
    try:
        import redis

        return bool(redis.Redis.from_url(settings.REDIS_URL, socket_timeout=0.5).ping())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _redis_available(), reason="Redis недоступен на REDIS_URL")
def test_two_workers_share_redis_counter(monkeypatch, restore_settings):
    """Два экземпляра лимитера (два воркера) считают один и тот же счётчик."""
    monkeypatch.setattr(settings, "RATE_LIMIT_BACKEND", "redis")
    name = f"shared-{uuid.uuid4().hex[:8]}"
    worker_a = HybridLimiter(name, limit=3, window=60.0)
    worker_b = HybridLimiter(name, limit=3, window=60.0)

    async def scenario() -> list[bool]:  # один event loop: клиент redis привязан к нему
        return [
            await worker_a.allow("ip"),
            await worker_b.allow("ip"),
            await worker_a.allow("ip"),
            await worker_b.allow("ip"),
        ]

    results = asyncio.run(scenario())
    assert results == [True, True, True, False]
    # уборка тестовых ключей
    import redis

    client = redis.Redis.from_url(settings.REDIS_URL)
    for key in client.scan_iter(match=f"rl:{name}:*"):
        client.delete(key)
