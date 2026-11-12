from __future__ import annotations

import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, inspect as sa_inspect, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.ratelimit import limit_upload
from app.core.security import get_current_user
from app.ml_pipeline.llm_service import answer_question, embed_texts
from app.ml_pipeline.text_utils import cosine
from app.models.document import Document, DocumentChunk, DocumentStatus, ExtractedData, OcrPage
from app.models.user import User, UserRole
from app.schemas.document import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    DocumentDetail,
    DocumentList,
    DocumentOut,
    PageOut,
    UploadResponse,
    VerifyPayload,
)
from app.services import pipeline_service
from app.services.audit_service import log_action
from app.services.storage_service import safe_filename, storage

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".docx", ".txt", ".md"}
# проверка реального типа файла по Magic Bytes (а не по расширению)
MAGIC_PREFIXES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".docx": (b"PK\x03\x04",),
    ".txt": (),
    ".md": (),
}


def validate_upload(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Неподдерживаемый формат «{suffix or 'без расширения'}». "
            f"Допустимо: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Файл пуст")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Размер файла превышает {settings.MAX_UPLOAD_MB} МБ",
        )
    prefixes = MAGIC_PREFIXES[suffix]
    if prefixes and not any(data.startswith(prefix) for prefix in prefixes):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Содержимое файла не соответствует его расширению (проверка Magic Bytes)",
        )
    return suffix


def to_out(document: Document) -> DocumentOut:
    # защита от ленивой загрузки связи в синхронном контексте (MissingGreenlet)
    extracted = None if "extracted" in sa_inspect(document).unloaded else document.extracted
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        mime_type=document.mime_type,
        file_size=document.file_size,
        page_count=document.page_count,
        status=document.status,
        progress=document.progress,
        doc_type=document.doc_type,
        summary=extracted.summary if extracted else None,
        is_verified=bool(extracted.is_verified) if extracted else False,
        error_message=document.error_message,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


async def get_document_or_404(db: AsyncSession, document_id: str) -> Document:
    document = await db.scalar(
        select(Document)
        .options(selectinload(Document.pages), selectinload(Document.extracted))
        .where(Document.id == document_id)
    )
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Документ не найден")
    return document


# --------------------------------- upload ---------------------------------
@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    files: Annotated[list[UploadFile], File(description="Один или несколько документов")],
    _: Annotated[None, Depends(limit_upload)] = None,
):
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Файлы не переданы")

    created: list[Document] = []
    for upload in files:
        data = await upload.read()
        suffix = validate_upload(upload.filename or "document", data)
        document_id = str(uuid.uuid4())
        filename = safe_filename(Path(upload.filename or "document").stem) + suffix
        key = f"originals/{document_id}/{filename}"
        await storage.put(key, data)
        document = Document(
            id=document_id,
            user_id=user.id,
            filename=filename,
            s3_key_original=key,
            mime_type=upload.content_type
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream",
            file_size=len(data),
            status=DocumentStatus.PENDING,
            progress=0,
        )
        db.add(document)
        created.append(document)
        await log_action(
            db,
            action="document_uploaded",
            user=user,
            document=document,
            details={"size": len(data), "mime": document.mime_type},
            ip_address=request.client.host if request.client else None,
        )

    await db.commit()
    for document in created:
        await db.refresh(document)
        pipeline_service.enqueue(document.id)

    return UploadResponse(
        documents=[to_out(document) for document in created],
        task_id=str(uuid.uuid4()),
    )


# ---------------------------------- list ----------------------------------
@router.get("", response_model=DocumentList)
async def list_documents(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status_filter: str | None = Query(None, alias="status"),
    doc_type: str | None = None,
    query: str | None = Query(None, alias="q"),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    sort: str = Query("created_at_desc"),
):
    stmt = select(Document).options(selectinload(Document.extracted))
    count_stmt = select(func.count(Document.id))

    conditions = []
    if status_filter:
        conditions.append(Document.status == status_filter)
    if doc_type:
        conditions.append(Document.doc_type == doc_type)
    if date_from:
        conditions.append(Document.created_at >= date_from)
    if date_to:
        conditions.append(Document.created_at <= date_to)
    if query:
        pattern = f"%{query.lower()}%"
        subquery = select(OcrPage.document_id).where(
            or_(func.lower(OcrPage.raw_text).like(pattern), func.lower(OcrPage.corrected_text).like(pattern))
        )
        conditions.append(or_(func.lower(Document.filename).like(pattern), Document.id.in_(subquery)))

    for condition in conditions:
        stmt = stmt.where(condition)
        count_stmt = count_stmt.where(condition)

    ordering = {
        "created_at_asc": Document.created_at.asc(),
        "created_at_desc": Document.created_at.desc(),
        "filename": Document.filename.asc(),
        "status": Document.status.asc(),
    }.get(sort, Document.created_at.desc())

    total = await db.scalar(count_stmt) or 0
    rows = await db.execute(stmt.order_by(ordering).offset((page - 1) * limit).limit(limit))
    items = [to_out(document) for document in rows.scalars().unique()]

    return DocumentList(items=items, total=total, page=page, limit=limit)


# --------------------------------- detail ---------------------------------
@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    document = await get_document_or_404(db, document_id)
    extracted = document.extracted
    pages = [
        PageOut(
            page_number=p.page_number,
            raw_text=p.raw_text,
            corrected_text=p.corrected_text,
            bounding_boxes=p.bounding_boxes or [],
            image_width=p.image_width,
            image_height=p.image_height,
            avg_confidence=p.avg_confidence,
            source=p.source,
            image_url=f"{settings.API_V1_PREFIX}/documents/{document.id}/pages/{p.page_number}/image",
        )
        for p in sorted(document.pages, key=lambda x: x.page_number)
    ]
    return DocumentDetail(
        **to_out(document).model_dump(),
        pages=pages,
        entities=extracted.entities if extracted else {},
        keywords=extracted.keywords if extracted else [],
        analysis_engine=extracted.engine if extracted else None,
        document_url=f"{settings.API_V1_PREFIX}/documents/{document.id}/file",
    )


@router.get("/{document_id}/pages/{page_number}/image")
async def get_page_image(
    document_id: str,
    page_number: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    page = await db.scalar(
        select(OcrPage).where(OcrPage.document_id == document_id, OcrPage.page_number == page_number)
    )
    if page is None or not page.image_key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Изображение страницы не найдено")
    data = await storage.get(page.image_key)
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


@router.get("/{document_id}/file")
async def get_original_file(
    document_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    document = await get_document_or_404(db, document_id)
    path = await storage.local_path(document.s3_key_original)
    if not Path(path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Файл недоступен в хранилище")
    return FileResponse(path, filename=document.filename, media_type=document.mime_type)


# --------------------------------- verify ---------------------------------
@router.put("/{document_id}/verify", response_model=DocumentDetail)
async def verify_document(
    document_id: str,
    payload: VerifyPayload,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    if user.role == UserRole.VIEWER:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Роль «viewer» не может изменять документы")
    document = await get_document_or_404(db, document_id)
    extracted = document.extracted
    if extracted is None:
        extracted = ExtractedData(document_id=document.id)
        db.add(extracted)

    if payload.document_type:
        document.doc_type = payload.document_type
    if payload.summary is not None:
        extracted.summary = payload.summary
    if payload.entities is not None:
        extracted.entities = payload.entities.model_dump()
    if payload.keywords is not None:
        extracted.keywords = payload.keywords

    if payload.pages:
        by_number = {p.page_number: p for p in document.pages}
        for page_payload in payload.pages:
            page = by_number.get(page_payload.get("page_number"))
            if page and page_payload.get("corrected_text") is not None:
                page.corrected_text = page_payload["corrected_text"]

    extracted.is_verified = payload.is_verified
    extracted.verified_by = user.id
    extracted.verified_at = datetime.utcnow() if payload.is_verified else None
    if payload.is_verified:
        document.status = DocumentStatus.VERIFIED

    await log_action(
        db,
        action="document_verified" if payload.is_verified else "document_edited",
        user=user,
        document=document,
        details={
            "doc_type": document.doc_type,
            "entities_total": sum(len(v) for v in (extracted.entities or {}).values() if isinstance(v, list)),
        },
        ip_address=request.client.host if request.client else None,
    )
    await db.commit()
    await pipeline_service.publish_status(document, document.status, document.progress, "Документ проверен оператором")
    return await get_document(document_id, user, db)


@router.post("/{document_id}/reprocess", response_model=DocumentOut)
async def reprocess_document(
    document_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    document = await get_document_or_404(db, document_id)
    document.status = DocumentStatus.PENDING
    document.progress = 0
    document.error_message = None
    if document.extracted is not None:
        # после повторной обработки результат требует новой проверки оператором
        document.extracted.is_verified = False
        document.extracted.verified_at = None

    await log_action(
        db,
        action="document_reprocessed",
        user=user,
        document=document,
        ip_address=request.client.host if request.client else None,
    )
    await db.commit()
    pipeline_service.enqueue(document.id)
    return to_out(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    document = await get_document_or_404(db, document_id)
    if user.role != UserRole.ADMIN and document.user_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Удалять чужие документы может только администратор")
    await log_action(
        db,
        action="document_deleted",
        user=user,
        document=document,
        ip_address=request.client.host if request.client else None,
    )
    await storage.delete_prefix(f"originals/{document_id}")
    await storage.delete_prefix(f"pages/{document_id}")
    await db.delete(document)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------- chat ----------------------------------
@router.post("/{document_id}/chat", response_model=ChatResponse)
async def chat_with_document(
    document_id: str,
    payload: ChatRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    document = await get_document_or_404(db, document_id)
    rows = await db.execute(select(DocumentChunk).where(DocumentChunk.document_id == document_id))
    chunks = list(rows.scalars())
    if not chunks:
        # fallback: используем текст страниц
        pages = sorted(document.pages, key=lambda p: p.page_number)[:5]
        chunks_text = [(p.page_number, (p.corrected_text or p.raw_text)) for p in pages if p.raw_text]
        if not chunks_text:
            raise HTTPException(status.HTTP_409_CONFLICT, "Документ ещё не обработан")
    else:
        vectors, _ = await embed_texts([payload.question])
        query_vector = vectors[0]
        ranked = sorted(chunks, key=lambda c: cosine(query_vector, c.embedding or []), reverse=True)[:4]
        chunks_text = [(c.page_number, c.chunk_text) for c in ranked]

    answer, engine, best = await answer_question(payload.question, [text for _, text in chunks_text])
    sources = [
        ChatSource(page_number=page, snippet=text[:400]) for page, text in chunks_text[:3]
    ]
    return ChatResponse(answer=answer, sources=sources, engine=engine)
