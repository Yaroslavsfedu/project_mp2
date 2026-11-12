from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class UserAdminOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    documents_count: int = 0


class UserUpdate(BaseModel):
    role: Literal["admin", "operator", "viewer"] | None = None
    is_active: bool | None = None
    full_name: str | None = Field(default=None, max_length=255)


class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_email: str
    action: str
    document_id: str | None = None
    document_name: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AdminOverview(BaseModel):
    users_total: int
    documents_total: int
    pages_total: int
    verified_total: int
    storage_bytes: int
    audit_events: int
    actions_24h: list[dict[str, Any]]
