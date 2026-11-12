"""Асинхронная сессия SQLAlchemy 2.0."""
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


def _ensure_sqlite_dir(url: str) -> None:
    """Создаёт каталог под файл SQLite, иначе на чистой установке подключение падает."""
    if url.startswith("sqlite") and ":memory:" not in url and "///" in url:
        Path(url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(settings.DATABASE_URL)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    # SQLite не поддерживает pool-настройки PostgreSQL
    **({} if settings.DATABASE_URL.startswith("sqlite") else {"pool_size": 10, "max_overflow": 20}),
)

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """Создание схемы и идемпотентные дополнения (Docker-контур не запускает Alembic)."""
    from app import models  # noqa: F401  (регистрация метаданных)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _ensure_extras()


_extras_ready = False


async def _ensure_extras() -> None:
    """Догоняет схему для уже существующих БД: новые колонки и PostgreSQL-расширения.

    Выполняется один раз на процесс (Celery вызывает init_db на каждую задачу).
    """
    global _extras_ready
    if _extras_ready:
        return
    await _apply_extras()
    _extras_ready = True


async def _apply_extras() -> None:
    # SQL напрямую, а не inspect(): инспектор SQLAlchemy не знает тип pgvector
    async with engine.connect() as conn:
        if engine.dialect.name == "sqlite":
            rows = await conn.execute(text("PRAGMA table_info(document_chunks)"))
            columns = {row[1] for row in rows}
        else:
            rows = await conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'document_chunks'")
            )
            columns = {row[0] for row in rows}
    if "embedding_model" not in columns:
        await _execute(
            "ALTER TABLE document_chunks ADD COLUMN embedding_model VARCHAR(32) NOT NULL DEFAULT 'fallback'",
            required=True,
        )
    if engine.dialect.name != "postgresql":
        return
    # Обязательные: без pgvector векторный поиск работает через JSON-вектора (медленнее)
    await _execute("CREATE EXTENSION IF NOT EXISTS vector", required=False)
    await _execute(
        f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_vec vector({settings.EMBEDDING_DIM})",
        required=False,
    )
    await _execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding_vec vector_cosine_ops)",
        required=False,
    )
    # полнотекстовый индекс с русской морфологией (то же выражение, что в search_service)
    await _execute(
        "CREATE INDEX IF NOT EXISTS ix_ocr_pages_fts_russian ON ocr_pages USING gin ("
        "to_tsvector('russian', coalesce(corrected_text, '') || ' ' || coalesce(raw_text, '')))",
        required=False,
    )
    # pg_trgm ускоряет поиск по имени файла; в минимальных сборках PostgreSQL его может не быть
    await _execute("CREATE EXTENSION IF NOT EXISTS pg_trgm", required=False)
    await _execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_filename_trgm ON documents USING gin (filename gin_trgm_ops)",
        required=False,
    )


async def _execute(statement: str, *, required: bool) -> None:
    import logging

    try:
        async with engine.begin() as conn:
            await conn.execute(text(statement))
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        logging.getLogger(__name__).warning("Пропущено (%s): %s", statement.split(" ON ")[0][:60], exc)
