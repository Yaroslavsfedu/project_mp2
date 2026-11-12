from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.ratelimit import limit_login
from app.core.security import create_access_token, get_current_user, hash_password, verify_password
from app.models.user import User, UserRole
from app.services.audit_service import log_action
from app.schemas.document import TokenOut, UserCreate, UserLogin, UserOut

def _ip(request: Request) -> str | None:
    return request.client.host if request.client else None


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(limit_login)],
)
async def register(payload: UserCreate, request: Request, db: Annotated[AsyncSession, Depends(get_db)]):
    exists = await db.scalar(select(User).where(User.email == payload.email.lower()))
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Пользователь с таким email уже существует")
    # Роль из запроса НЕ принимается: иначе любой аноним зарегистрируется админом.
    # Первый пользователь системы становится администратором, остальные — operator
    # (повышение — только через админ-панель: PATCH /admin/users/{id}).
    users_count = await db.scalar(select(User.id).limit(1))
    role = UserRole.ADMIN if users_count is None else UserRole.OPERATOR
    user = User(
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=role,
    )
    db.add(user)
    await db.flush()
    await log_action(db, action="user_registered", user=user, details={"role": role}, ip_address=_ip(request))
    await db.commit()
    await db.refresh(user)
    token = create_access_token(user.id, {"role": user.role})
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenOut, dependencies=[Depends(limit_login)])
async def login(payload: UserLogin, request: Request, db: Annotated[AsyncSession, Depends(get_db)]):
    user = await db.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        await log_action(db, action="login_failed", details={"email": payload.email.lower()}, ip_address=_ip(request))
        await db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный email или пароль")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Учётная запись заблокирована")
    await log_action(db, action="login", user=user, ip_address=_ip(request))
    await db.commit()
    token = create_access_token(user.id, {"role": user.role})
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
async def me(user: Annotated[User, Depends(get_current_user)]):
    return UserOut.model_validate(user)
