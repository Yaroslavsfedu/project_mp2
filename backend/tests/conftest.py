"""Тестовое окружение: изолированная БД и хранилище во временной директории."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

TMP_DIR = Path(tempfile.mkdtemp(prefix="idp-tests-"))

# переменные должны быть выставлены до импорта app.core.config
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TMP_DIR}/test.db"
os.environ["DATA_DIR"] = str(TMP_DIR)
os.environ["SEED_DEMO"] = "false"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["QUEUE_BACKEND"] = "inline"
os.environ["LLM_FALLBACK_ENABLED"] = "true"
os.environ["OLLAMA_URL"] = "http://127.0.0.1:1"  # заведомо недоступна
os.environ["SECRET_KEY"] = "test-secret"
# лимиты проверяются отдельным тестом, иначе фикстура логина упирается в 429
os.environ["RATE_LIMIT_ENABLED"] = "false"
# счётчики в памяти процесса: Redis-бэкенд проверяется отдельным тестом
os.environ["RATE_LIMIT_BACKEND"] = "memory"

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.core.database import init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
async def prepared_database():
    await init_db()
    yield


@pytest.fixture()
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    from scripts.generate_samples import generate_samples

    files = generate_samples(TMP_DIR / "samples")
    return next(path for path in files if path.suffix == ".pdf" and "dogovor" in path.name)


@pytest.fixture()
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    email = "test@example.ru"
    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "test12345"}
    )
    if response.status_code == 401:
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "test12345", "full_name": "Тестировщик"},
        )
    assert response.status_code in (200, 201), response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
