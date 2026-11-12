"""Асинхронная сессия SQLAlchemy 2.0."""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

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
    """Создание схемы. В продуктивном контуре используется Alembic."""
    from app import models  # noqa: F401  (регистрация метаданных)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
