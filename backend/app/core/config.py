"""Настройки приложения (Pydantic BaseSettings).

Все значения можно переопределить переменными окружения или файлом .env.
По умолчанию приложение поднимается в «локальном» режиме:
SQLite + локальный диск вместо MinIO + inline-обработка вместо Celery.
Для продуктивного стека достаточно переопределить DATABASE_URL, STORAGE_BACKEND
и QUEUE_BACKEND (см. .env.example и docker-compose.yml).
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/ и корень репозитория
BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    # .env ищем в корне проекта (там его создаёт install.sh), а не в текущей директории:
    # dev-скрипты запускают backend из папки backend/, и без этого .env игнорировался бы
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- Общие ---
    APP_NAME: str = "IDP Платформа — оцифровка документов"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # --- Безопасность ---
    SECRET_KEY: str = "change-me-in-production-3f9a1c7d2b6e4a8f"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    MAX_UPLOAD_MB: int = 50
    CORS_ORIGINS: str = "*"

    # Rate limiting (защита от перебора паролей и флуда загрузками)
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_LOGIN_PER_MINUTE: int = 10
    RATE_LIMIT_UPLOAD_PER_MINUTE: int = 30
    # Доверять X-Forwarded-For можно только когда API стоит за своим reverse-proxy (nginx),
    # иначе клиент подменяет заголовок и обходит лимит.
    TRUST_PROXY_HEADERS: bool = False

    # --- Данные ---
    DATA_DIR: Path = BACKEND_ROOT / "data"
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BACKEND_ROOT / 'data' / 'idp.db'}"

    # --- Объектное хранилище: local | minio ---
    STORAGE_BACKEND: str = "local"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "idp-documents"
    MINIO_SECURE: bool = False
    PRESIGNED_URL_TTL_MINUTES: int = 15

    # --- Очереди: inline | celery ---
    QUEUE_BACKEND: str = "inline"
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Локальный ИИ (Ollama) ---
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    OLLAMA_VISION_MODEL: str = "qwen2-vl:7b"
    EMBEDDING_MODEL: str = "bge-m3"
    # размерность bge-m3 (колонка pgvector VECTOR(1024) в спецификации)
    EMBEDDING_DIM: int = 1024
    LLM_TIMEOUT_SECONDS: int = 180
    # Если Ollama недоступна — работаем на эвристическом fallback-движке
    LLM_FALLBACK_ENABLED: bool = True

    # --- OCR ---
    # auto | tesseract | native(pdf text layer only)
    OCR_ENGINE: str = "auto"
    OCR_LANGUAGES: str = "rus+eng"
    OCR_DPI: int = 150
    PREPROCESS_ENABLED: bool = True

    # --- Демо-данные ---
    SEED_DEMO: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def storage_dir(self) -> Path:
        return self.DATA_DIR / "storage"

    @property
    def pages_dir(self) -> Path:
        return self.DATA_DIR / "pages"

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
