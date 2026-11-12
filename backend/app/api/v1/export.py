from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.http import content_disposition
from app.core.security import get_current_user
from app.models.document import Document
from app.models.user import User
from app.schemas.document import SearchFilters
from app.services import export_service
from app.services.search_service import search_documents

router = APIRouter(tags=["export"])


@router.get("/documents/{document_id}/export")
async def export_document(
    document_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = Query("json", pattern="^(json|csv|docx|md|txt)$"),
):
    document = await db.scalar(
        select(Document)
        .options(selectinload(Document.pages), selectinload(Document.extracted))
        .where(Document.id == document_id)
    )
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Документ не найден")

    exporter = export_service.EXPORTERS[format]
    content, media_type, extension = exporter(document)
    filename = f"{document.filename.rsplit('.', 1)[0]}-{export_service.timestamp()}{extension}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": content_disposition(filename)},
    )


@router.post("/documents/export")
async def export_batch(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = Query("xlsx", pattern="^(xlsx|csv)$"),
    ids: Annotated[list[str] | None, Query()] = None,
    query: str | None = None,
    doc_type: str | None = None,
):
    """Пакетный экспорт сводной таблицы по выбранным документам."""
    document_ids = list(dict.fromkeys(ids or []))
    if not document_ids and query:
        result = await search_documents(db, query, SearchFilters(doc_type=doc_type), "hybrid", 100)
        document_ids = list(dict.fromkeys(hit.document_id for hit in result.hits))

    stmt = select(Document).options(selectinload(Document.extracted))
    if document_ids:
        stmt = stmt.where(Document.id.in_(document_ids))
    elif doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    documents = list((await db.execute(stmt.order_by(Document.created_at.desc()))).scalars().unique())
    if not documents:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Нет документов для экспорта")

    content, media_type, extension = (
        export_service.export_batch_xlsx(documents)
        if format == "xlsx"
        else export_service.export_batch_csv(documents)
    )
    filename = f"documents-export-{export_service.timestamp()}{extension}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": content_disposition(filename)},
    )
