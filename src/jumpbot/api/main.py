from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from jumpbot import __version__
from jumpbot.api.schemas import HealthRead, JumpRead, UserCreate, UserRead
from jumpbot.api.security import require_api_key
from jumpbot.config import get_settings
from jumpbot.db.models import JumpHistory, User
from jumpbot.db.session import get_session
from jumpbot.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    configure_logging()
    get_settings().storage_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="JumpBot API",
    version=__version__,
    description="Training-oriented vertical jump video analysis",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthRead, tags=["system"])
async def health() -> HealthRead:
    return HealthRead(status="ok", version=__version__)


@app.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_or_update_user(
    payload: UserCreate,
    _: None = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> User:
    user = await session.scalar(select(User).where(User.telegram_user_id == payload.telegram_user_id))
    if user is None:
        user = User(**payload.model_dump())
        session.add(user)
    else:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(user, field, value)
    await session.commit()
    await session.refresh(user)
    return user


@app.get("/users/{telegram_user_id}/jumps", response_model=list[JumpRead])
async def list_jumps(
    telegram_user_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    _: None = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> list[JumpHistory]:
    user = await session.scalar(select(User).where(User.telegram_user_id == telegram_user_id))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    result = await session.scalars(
        select(JumpHistory)
        .where(JumpHistory.user_id == user.id)
        .order_by(JumpHistory.created_at.desc())
        .limit(limit)
    )
    return list(result)
