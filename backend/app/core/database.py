"""Асинхронная сессия SQLAlchemy 2.0."""
import importlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
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
    # SQLite не поддерживает pool-настройки PostgreSQL; вместо них — ожидание блокировки записи
    # (фоновые задачи и запросы API пишут в одну базу; без таймаута сразу падает "database is locked")
    **(
        {"connect_args": {"timeout": 30}}
        if settings.DATABASE_URL.startswith("sqlite")
        else {"pool_size": 10, "max_overflow": 20}
    ),
)

if engine.dialect.name == "sqlite":
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_on_connect(dbapi_connection, _record) -> None:  # noqa: ANN001
        # WAL: читатели не блокируют писателей — иначе фоновая обработка и API мешают друг другу
        # ("database is locked"). busy_timeout — ждать освобождения блокировки, а не падать сразу.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


# True, когда на PostgreSQL создан индекс pg_trgm по имени файла (ILIKE по filename идёт по индексу).
trgm_filename_index_ready = False


@asynccontextmanager
async def gin_only(db: AsyncSession):
    """Запрещает Seq Scan внутри блока (PostgreSQL). Нужно для полнотекстовых кандидатов.

    Без этого планировщик на частом слове выбирает последовательное чтение с ранним выходом по LIMIT:
    оно быстро только когда слово почти везде. Если слово в 1% строк, чтение пройдёт почти всю таблицу.
    С запретом стоимость задаёт GIN-индекс: пропорционально числу совпадений, а не размеру таблицы.
    Действует только внутри транзакции (SET LOCAL) и сразу возвращается обратно.
    """
    bind = db.get_bind() if hasattr(db, "get_bind") else None
    if getattr(getattr(bind, "dialect", None), "name", "") != "postgresql":
        yield
        return
    await db.execute(text("SET LOCAL enable_seqscan = off"))
    try:
        yield
    finally:
        try:
            await db.execute(text("SET LOCAL enable_seqscan = on"))
        except Exception:  # noqa: BLE001 - транзакция могла быть прервана ошибкой запроса
            pass


async def init_db() -> None:
    """Создание схемы и идемпотентные дополнения (Docker-контур не запускает Alembic)."""
    # регистрация метаданных всех моделей; pyflakes не учитывает noqa, поэтому importlib
    importlib.import_module("app.models")  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _ensure_extras()


_extras_ready = False

FTS_COLUMN_SQL = (
    "ALTER TABLE ocr_pages ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS "
    "(to_tsvector('russian', coalesce(corrected_text, '') || ' ' || coalesce(raw_text, ''))) STORED"
)


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
    # составные индексы списка документов (для уже созданных таблиц create_all их не добавит)
    for statement in (
        "CREATE INDEX IF NOT EXISTS ix_documents_status_doctype_created "
        "ON documents (status, doc_type, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS ix_documents_created_id ON documents (created_at DESC, id)",
    ):
        await _execute(statement, required=False)
    if engine.dialect.name != "postgresql":
        return
    # Обязательные: без pgvector векторный поиск работает через JSON-вектора (медленнее)
    await _execute("CREATE EXTENSION IF NOT EXISTS vector", required=False)
    await _execute(
        f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_vec vector({settings.EMBEDDING_DIM})",
        required=False,
    )
    await _execute(
        # HNSW с явными параметрами: m=16 связей на узел и ef_construction=64 — компромисс
        # между recall и памятью. Граф держится в shared_buffers/RAM: ~(m*2*8 + dim*4) байт на вектор.
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding_vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
        required=False,
    )
    # полнотекстовый индекс с русской морфологией (то же выражение, что в search_service)
    await _execute(
        "CREATE INDEX IF NOT EXISTS ix_ocr_pages_fts_russian ON ocr_pages USING gin ("
        "to_tsvector('russian', coalesce(corrected_text, '') || ' ' || coalesce(raw_text, '')))",
        required=False,
    )
    # хранимый tsvector: ranking читает готовый вектор, а не пересчитывает to_tsvector для каждой строки
    await _execute(FTS_COLUMN_SQL, required=False)
    await _execute(
        "CREATE INDEX IF NOT EXISTS ix_ocr_pages_search_tsv ON ocr_pages USING gin (search_tsv)",
        required=False,
    )
    # pg_trgm ускоряет поиск по имени файла; в минимальных сборках PostgreSQL его может не быть
    await _execute("CREATE EXTENSION IF NOT EXISTS pg_trgm", required=False)
    await _execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_filename_trgm ON documents USING gin (filename gin_trgm_ops)",
        required=False,
    )
    global trgm_filename_index_ready
    async with engine.connect() as conn:
        found = await conn.scalar(text("SELECT 1 FROM pg_indexes WHERE indexname = 'ix_documents_filename_trgm'"))
    trgm_filename_index_ready = bool(found)


async def _execute(statement: str, *, required: bool) -> None:
    import logging

    try:
        async with engine.begin() as conn:
            await conn.execute(text(statement))
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        logging.getLogger(__name__).warning("Пропущено (%s): %s", statement.split(" ON ")[0][:60], exc)
