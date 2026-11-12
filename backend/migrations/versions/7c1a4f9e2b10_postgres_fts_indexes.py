"""PostgreSQL: GIN-индексы для полнотекстового поиска (морфология 'russian')

Revision ID: 7c1a4f9e2b10
Revises: 03d068b7ae33
Create Date: 2026-11-12

Миграция применяется только на PostgreSQL: в SQLite используется
префиксный LIKE-поиск, и индексы tsvector там не нужны.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "7c1a4f9e2b10"
down_revision: Union[str, None] = "03d068b7ae33"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_ocr_pages_fts_russian"


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
    # pg_trgm заметно ускоряет фильтры по организациям и подсветку фрагментов
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_documents_filename_trgm
        ON documents USING gin (filename gin_trgm_ops)
        """
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    op.execute("DROP INDEX IF EXISTS ix_documents_filename_trgm")
