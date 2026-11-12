"""PostgreSQL: хранимый tsvector для полнотекстового поиска (ускорение ранжирования)

Revision ID: c3e8a1b5d7f9
Revises: b7d2e9f1a3c4
Create Date: 2026-11-12

Колонка search_tsv вычисляется из текста страницы и хранится на диске. Ранжирование
ts_rank() читает её напрямую, поэтому запрос не пересчитывает to_tsvector для каждой
найденной страницы (раньше это давало ~400 мс на 1000 документов).
Миграция идемпотентна и не падает на сервере без нужных возможностей.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3e8a1b5d7f9"
down_revision: Union[str, None] = "b7d2e9f1a3c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FTS_COLUMN_SQL = (
    "ALTER TABLE ocr_pages ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS "
    "(to_tsvector('russian', coalesce(corrected_text, '') || ' ' || coalesce(raw_text, ''))) STORED"
)


def _optional(sql: str, label: str) -> bool:
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
    if _optional(FTS_COLUMN_SQL, "search_tsv"):
        _optional(
            "CREATE INDEX IF NOT EXISTS ix_ocr_pages_search_tsv ON ocr_pages USING gin (search_tsv)",
            "GIN по search_tsv",
        )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_ocr_pages_search_tsv")
    op.execute("ALTER TABLE ocr_pages DROP COLUMN IF EXISTS search_tsv")
