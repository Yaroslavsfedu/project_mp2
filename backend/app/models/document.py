import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.user import new_uuid, utcnow


class DocumentStatus(StrEnum):
    PENDING = "pending"
    OCR_PROCESSING = "ocr_processing"
    LLM_PROCESSING = "llm_processing"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    s3_key_original: Mapped[str] = mapped_column(String(1024))
    s3_key_processed: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default=DocumentStatus.PENDING, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    doc_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    pages: Mapped[list["OcrPage"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="OcrPage.page_number"
    )
    extracted: Mapped["ExtractedData | None"] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",  # summary/entities нужны почти в каждом ответе API
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class OcrPage(Base):
    __tablename__ = "ocr_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, default=1)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    corrected_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    bounding_boxes: Mapped[list] = mapped_column(JSON, default=list)
    image_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    image_width: Mapped[int] = mapped_column(Integer, default=0)
    image_height: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="ocr")  # native | ocr

    document: Mapped[Document] = relationship(back_populates="pages")


class ExtractedData(Base):
    __tablename__ = "extracted_data"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, index=True
    )
    summary: Mapped[str] = mapped_column(Text, default="")
    entities: Mapped[dict] = mapped_column(JSON, default=dict)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    engine: Mapped[str] = mapped_column(String(32), default="fallback")  # ollama | fallback
    is_verified: Mapped[bool] = mapped_column(default=False)
    verified_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    document: Mapped[Document] = relationship(back_populates="extracted")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, default=1)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    chunk_text: Mapped[str] = mapped_column(Text, default="")
    # В PostgreSQL-контуре используется TSVECTOR + pgvector,
    # в локальном (SQLite) — нормализованный текст и JSON-вектор.
    embedding: Mapped[list] = mapped_column(JSON, default=list)

    document: Mapped[Document] = relationship(back_populates="chunks")
