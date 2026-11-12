from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.document import SearchRequest, SearchResponse
from app.services.search_service import search_documents

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    return await search_documents(db, payload.query, payload.filters, payload.mode, payload.limit)
