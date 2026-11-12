"""Масштабирование списка документов: составные B-Tree индексы (2+ млн строк)

Revision ID: d5a1c9e3f7b2
Revises: c3e8a1b5d7f9
Create Date: 2026-11-12

* (status, doc_type, created_at DESC) — список с фильтрами и сортировкой по дате;
* (created_at DESC, id) — keyset-пагинация (курсор) без OFFSET.

Индексы создаются через IF NOT EXISTS, поэтому миграция идемпотентна и работает и на
PostgreSQL, и на SQLite (оба поддерживают DESC в индексе).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "d5a1c9e3f7b2"
down_revision: Union[str, None] = "c3e8a1b5d7f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_documents_status_doctype_created "
    "ON documents (status, doc_type, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_documents_created_id ON documents (created_at DESC, id)",
)


def upgrade() -> None:
    for statement in INDEXES:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_created_id")
    op.execute("DROP INDEX IF EXISTS ix_documents_status_doctype_created")
