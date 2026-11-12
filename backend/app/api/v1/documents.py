from __future__ import annotations

import mimetypes
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import inspect as sa_inspect, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.ratelimit import limit_upload
from app.core.security import get_current_user
from app.ml_pipeline.llm_service import answer_question, embed_texts
from app.services.document_list_service import ListQuery, decode_cursor, list_documents_page
from app.services.search_service import rank_document_chunks
from app.models.document import Document, DocumentStatus, ExtractedData, OcrPage
from app.models.user import User, UserRole, utcnow
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
from app.services.storage_service import original_key, safe_filename, storage

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".docx", ".txt", ".md"}
# проверка реального типа файла по Magic Bytes (а не по расширению)
TEXT_SUFFIXES = {".txt", ".md"}
# Тип для отдачи оригинала: определяется расширением, а не заголовком клиента.
SERVE_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
}
INLINE_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
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
    if suffix in TEXT_SUFFIXES and b"\x00" in data[:8192]:
        # у текстовых файлов нет нулевых байтов; бинарник, переименованный в .txt, отсекаем
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Файл .txt/.md содержит бинарные данные")
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


def require_writer(user: User) -> None:
    """Загрузка, верификация, повторная обработка и удаление — только operator и admin."""
    if user.role == UserRole.VIEWER:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Роль «viewer» не может изменять документы")


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
    require_writer(user)
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Файлы не переданы")

    # сначала валидируем весь пакет: при ошибке в одном файле ничего не записываем
    prepared: list[tuple[str, bytes, str]] = []
    for upload in files:
        data = await upload.read()
        suffix = validate_upload(upload.filename or "document", data)
        prepared.append((upload.filename or "document", data, suffix))

    created: list[Document] = []
    for (original_name, data, suffix), upload in zip(prepared, files, strict=True):
        document_id = str(uuid.uuid4())
        filename = safe_filename(Path(original_name).stem) + suffix
        key = original_key(document_id, filename)
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
        # документ нужно записать ДО журнала аудита: строка аудита ссылается на него по FK,
        # а без связи в ORM порядок вставки не гарантирован (PostgreSQL проверяет FK)
        await db.flush()
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
    page: int = Query(1, ge=1, description="Номер страницы (offset-режим)"),
    limit: int = Query(20, ge=1, le=100),
    status_filter: str | None = Query(None, alias="status"),
    doc_type: str | None = None,
    query: str | None = Query(None, alias="q"),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    sort: str = Query("created_at_desc"),
    cursor: str | None = Query(None, description="Курсор из next_cursor предыдущего ответа (keyset)"),
    after_created_at: datetime | None = Query(None, description="Альтернатива cursor: дата последней строки"),
    after_id: str | None = Query(None, max_length=36, description="Альтернатива cursor: id последней строки"),
):
    """Список документов.

    * Без ``cursor`` — классическая пагинация ``page``/``limit``; глубина ограничена ``LIST_OFFSET_MAX``.
    * С ``cursor`` — keyset-пагинация: стабильная и быстрая на миллионах строк. Первая страница —
      без ``cursor``, далее передавайте ``next_cursor`` из ответа, пока ``has_more`` = true.
    """
    offset = (page - 1) * limit
    keyset = cursor is not None or after_created_at is not None or after_id is not None
    if not keyset and offset > settings.LIST_OFFSET_MAX:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Глубокая страница (смещение больше {settings.LIST_OFFSET_MAX}). "
            "Используйте курсорную пагинацию: параметр cursor из поля next_cursor",
        )
    list_query = ListQuery(
        status=status_filter,
        doc_type=doc_type,
        date_from=date_from,
        date_to=date_to,
        query=query,
        sort=sort,
    )
    decoded = None
    if cursor:
        decoded = decode_cursor(cursor)
    elif after_created_at is not None or after_id is not None:
        if after_created_at is None or not after_id:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Нужны оба параметра: after_created_at и after_id"
            )
        if after_created_at.tzinfo is None:
            after_created_at = after_created_at.replace(tzinfo=UTC)
        decoded = (after_created_at, after_id)
    result = await list_documents_page(db, list_query, page=page, limit=limit, cursor=decoded)
    return DocumentList(
        items=[to_out(document) for document in result.items],
        total=result.total,
        total_estimated=result.total_estimated,
        page=page,
        limit=limit,
        next_cursor=result.next_cursor,
        has_more=result.has_more,
    )


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
    suffix = Path(document.filename).suffix.lower()
    # тип задаём по расширению и запрещаем браузеру «угадывать» тип (nosniff),
    # иначе HTML, загруженный как .txt, исполнился бы в контексте приложения
    return FileResponse(
        path,
        filename=document.filename,
        media_type=SERVE_MEDIA_TYPES.get(suffix, "application/octet-stream"),
        content_disposition_type="inline" if suffix in INLINE_SUFFIXES else "attachment",
        headers={"X-Content-Type-Options": "nosniff"},
    )


# --------------------------------- verify ---------------------------------
@router.put("/{document_id}/verify", response_model=DocumentDetail)
async def verify_document(
    document_id: str,
    payload: VerifyPayload,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    require_writer(user)
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
    extracted.verified_at = utcnow() if payload.is_verified else None
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
    require_writer(user)
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
    require_writer(user)
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
    await storage.delete_document_files(document_id, document.s3_key_original)
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
    vectors, engine = await embed_texts([payload.question])
    chunks_text = await rank_document_chunks(db, document_id, vectors[0], engine, 4)
    if not chunks_text:
        # индекса нет (например, документ обработан другим движком эмбеддингов) —
        # берём текст страниц целиком
        pages = sorted(document.pages, key=lambda p: p.page_number)[:5]
        chunks_text = [(p.page_number, (p.corrected_text or p.raw_text)) for p in pages if p.raw_text]
        if not chunks_text:
            raise HTTPException(status.HTTP_409_CONFLICT, "Документ ещё не обработан")

    answer, engine, best = await answer_question(payload.question, [text for _, text in chunks_text])
    sources = [
        ChatSource(page_number=page, snippet=text[:400]) for page, text in chunks_text[:3]
    ]
    return ChatResponse(answer=answer, sources=sources, engine=engine)
