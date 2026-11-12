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


def _extension_available(name: str) -> bool:
    row = op.get_bind().execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = :name"), {"name": name}
    ).first()
    return row is not None


def _optional(sql: str, label: str) -> bool:
    """Необязательный шаг в SAVEPOINT: ошибка не отменяет всю миграцию."""
    bind = op.get_bind()
    try:
        with bind.begin_nested():
            bind.execute(sa.text(sql))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[миграция] пропущено ({label}): {str(exc).splitlines()[0]}")
        return False


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
    # pgvector — расширение из образа pgvector. Без него вектора хранятся в JSON,
    # а поиск работает медленнее: миграция не падает, а предупреждает.
    if not _extension_available("vector"):
        print("[миграция] расширение vector недоступно: колонка embedding_vec не создана")
        return
    if not _optional("CREATE EXTENSION IF NOT EXISTS vector", "vector"):
        return
    _optional(f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_vec vector({VECTOR_DIM})", "embedding_vec")
    _optional(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding_vec vector_cosine_ops)",
        "HNSW-индекс",
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
        op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding_vec")
    op.drop_column("document_chunks", "embedding_model")
