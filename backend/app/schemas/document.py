from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.schemas.analysis import ExtractedEntities


# ---------------- Auth ----------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    full_name: str = ""


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    role: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------------- OCR ----------------
class BoundingBox(BaseModel):
    id: str
    text: str
    confidence: float = 1.0
    bbox: list[float]  # [x_min, y_min, x_max, y_max] в пикселях страницы


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int
    raw_text: str
    corrected_text: str | None = None
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    avg_confidence: float = 0.0
    source: str = "ocr"
    image_url: str | None = None


# ---------------- Documents ----------------
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    mime_type: str
    file_size: int
    page_count: int
    status: str
    progress: int = 0
    doc_type: str | None = None
    summary: str | None = None
    is_verified: bool = False
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentOut):
    pages: list[PageOut] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    analysis_engine: str | None = None
    document_url: str | None = None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    # None в курсорном режиме: точный счётчик там не нужен и на миллионах строк стоит дорого
    total: int | None
    # true — total взят из статистики PostgreSQL (pg_class.reltuples), а не посчитан точно
    total_estimated: bool = False
    page: int
    limit: int
    next_cursor: str | None = None
    has_more: bool = False


class UploadResponse(BaseModel):
    documents: list[DocumentOut]
    task_id: str


class VerifyPayload(BaseModel):
    document_type: str | None = None
    summary: str | None = None
    entities: ExtractedEntities | None = None
    keywords: list[str] | None = None
    pages: list[dict[str, Any]] | None = None  # [{"page_number": 1, "corrected_text": "..."}]
    is_verified: bool = True


# ---------------- Search ----------------
class SearchFilters(BaseModel):
    doc_type: str | None = None
    status: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    organization: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: Literal["hybrid", "fulltext", "semantic"] = "hybrid"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=20, ge=1, le=100)


class SearchHit(BaseModel):
    document_id: str
    filename: str
    doc_type: str | None = None
    status: str
    page_number: int | None = None
    snippet: str
    score: float
    match_type: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    took_ms: float
    total: int
    hits: list[SearchHit]


# ---------------- Chat / RAG ----------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatSource(BaseModel):
    page_number: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    engine: str = "fallback"


# ---------------- Dashboard ----------------
class DashboardStats(BaseModel):
    total_documents: int
    by_status: dict[str, int]
    by_type: dict[str, int]
    verified_count: int
    pages_processed: int
    avg_confidence: float
    last_7_days: list[dict[str, Any]]
