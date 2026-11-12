from fastapi import APIRouter

from app.api.v1 import admin, auth, documents, export, search, system, ws

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(admin.router)
api_router.include_router(system.router)
api_router.include_router(documents.router)
api_router.include_router(export.router)
api_router.include_router(search.router)
api_router.include_router(ws.router)
