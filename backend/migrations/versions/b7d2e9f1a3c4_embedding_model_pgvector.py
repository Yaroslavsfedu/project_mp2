"""Векторный поиск: embedding_model у чанков и колонка pgvector (HNSW)

Revision ID: b7d2e9f1a3c4
Revises: 7c1a4f9e2b10
Create Date: 2026-11-12

* embedding_model — каким движком получен вектор (ollama / fallback); сравниваем
  только векторы одного движка;
* PostgreSQL: колонка embedding_vec VECTOR(1024) + HNSW-индекс для косинусного поиска.
Миграция идемпотентна: init_db() в приложении делает то же самое при старте.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7d2e9f1a3c4"
down_revision: Union[str, None] = "7c1a4f9e2b10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VECTOR_DIM = 1024


def upgrade() -> None:
    bind = op.get_bind()
    columns = {c["name"] for c in sa.inspect(bind).get_columns("document_chunks")}
    if "embedding_model" not in columns:
        op.add_column(
            "document_chunks",
            sa.Column("embedding_model", sa.String(length=32), nullable=False, server_default="fallback"),
        )
    if bind.dialect.name != "postgresql":
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_vec vector({VECTOR_DIM})")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding_vec vector_cosine_ops)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
        op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding_vec")
    op.drop_column("document_chunks", "embedding_model")
