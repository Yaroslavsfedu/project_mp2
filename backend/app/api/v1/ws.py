"""WebSocket-канал статусов обработки документов."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.security import decode_token
from app.services.events import bus

router = APIRouter(tags=["ws"])


@router.websocket("/ws/documents")
async def documents_ws(websocket: WebSocket, token: str | None = None):
    if token:
        try:
            decode_token(token)
        except Exception:  # noqa: BLE001
            await websocket.close(code=4401)
            return
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "connected"}, ensure_ascii=False))

    async def reader() -> None:
        # держим соединение открытым, отвечаем на ping
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    reader_task = asyncio.create_task(reader())
    try:
        async for event in bus.subscribe():
            await websocket.send_text(json.dumps({"type": "document_status", **event}, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        reader_task.cancel()
