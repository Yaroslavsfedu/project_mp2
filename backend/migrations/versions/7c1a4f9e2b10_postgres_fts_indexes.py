"""PostgreSQL: GIN-индексы для полнотекстового поиска (морфология 'russian')

Revision ID: 7c1a4f9e2b10
Revises: 03d068b7ae33
Create Date: 2026-11-12

Миграция применяется только на PostgreSQL: в SQLite используется
префиксный LIKE-поиск, и индексы tsvector там не нужны.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "7c1a4f9e2b10"
down_revision: Union[str, None] = "03d068b7ae33"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_ocr_pages_fts_russian"


def _extension_available(name: str) -> bool:
    """Есть ли расширение в установке PostgreSQL (пакет postgresql-contrib и т. п.)."""
    bind = op.get_bind()
    row = bind.execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = :name"), {"name": name}
    ).first()
    return row is not None


def _optional(sql: str, label: str) -> bool:
    """Выполняет необязательный шаг в SAVEPOINT: при ошибке транзакция миграции не отменяется."""
    bind = op.get_bind()
    try:
        with bind.begin_nested():
            bind.execute(sa.text(sql))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[миграция] пропущено ({label}): {str(exc).splitlines()[0]}")
        return False


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    # to_tsvector('russian', ...) с константной конфигурацией иммутабельна,
    # поэтому индекс по выражению допустим.
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {INDEX_NAME}
        ON ocr_pages
        USING gin (
            to_tsvector(
                'russian',
                coalesce(corrected_text, '') || ' ' || coalesce(raw_text, '')
            )
        )
        """
    )
    # pg_trgm ускоряет фильтры по имени файла; нужен пакет postgresql-contrib.
    # Без него миграция не падает: индекс по имени просто не создаётся.
    if _extension_available("pg_trgm"):
        _optional("CREATE EXTENSION IF NOT EXISTS pg_trgm", "pg_trgm")
        _optional(
            "CREATE INDEX IF NOT EXISTS ix_documents_filename_trgm "
            "ON documents USING gin (filename gin_trgm_ops)",
            "trgm-индекс",
        )
    else:
        print("[миграция] pg_trgm недоступен на сервере: индекс по имени файла не создан")

def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    op.execute("DROP INDEX IF EXISTS ix_documents_filename_trgm")
