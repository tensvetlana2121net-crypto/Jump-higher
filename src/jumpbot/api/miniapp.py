import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from jumpbot.api.schemas import JumpRead
from jumpbot.api.telegram_auth import TelegramIdentity, require_telegram_user
from jumpbot.config import get_settings
from jumpbot.db.models import AnalysisStatus, JumpHistory, User
from jumpbot.db.session import get_session
from jumpbot.services.storage import LocalStorage, sha256_file
from jumpbot.services.video_validation import validate_video_file
from jumpbot.worker import analyze_video_task

router = APIRouter(prefix="/miniapp/api", tags=["miniapp"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
TelegramDep = Annotated[TelegramIdentity, Depends(require_telegram_user)]

ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}
JUMP_NAMES = {"axel", "loop", "salchow", "flip", "lutz", "toe_loop"}
ROTATION_COUNTS = {1, 2, 3, 4}
ANALYSIS_MODES = {"single", "cascade"}


def make_jump_type(jump_name: str, rotation_count: int) -> str:
    if jump_name not in JUMP_NAMES or rotation_count not in ROTATION_COUNTS:
        raise ValueError("Unsupported jump classification")
    return f"{rotation_count}_{jump_name}"


def make_analysis_type(
    analysis_mode: str,
    jump_name: str,
    rotation_count: int,
    cascade_element_count: int | None,
) -> str:
    if analysis_mode not in ANALYSIS_MODES:
        raise ValueError("Unsupported analysis mode")
    if analysis_mode == "single":
        return make_jump_type(jump_name, rotation_count)
    if cascade_element_count not in {2, 3}:
        raise ValueError("Cascade must contain 2 or 3 elements")
    return f"cascade_{cascade_element_count}"


async def _ensure_user(session: AsyncSession, identity: TelegramIdentity) -> User:
    user = await session.scalar(select(User).where(User.telegram_user_id == identity.id))
    if user is None:
        user = User(
            telegram_user_id=identity.id,
            username=identity.username,
            first_name=identity.first_name,
            language_code=identity.language_code,
        )
        session.add(user)
    else:
        user.username = identity.username
        user.first_name = identity.first_name
        user.language_code = identity.language_code
    await session.commit()
    await session.refresh(user)
    return user


@router.get("/me")
async def get_me(identity: TelegramDep, session: SessionDep) -> dict[str, object]:
    user = await _ensure_user(session, identity)
    return {"telegram_user_id": user.telegram_user_id, "first_name": user.first_name}


@router.get("/analyses", response_model=list[JumpRead])
async def list_analyses(
    identity: TelegramDep,
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=50),
) -> list[JumpHistory]:
    user = await _ensure_user(session, identity)
    result = await session.scalars(
        select(JumpHistory)
        .where(JumpHistory.user_id == user.id)
        .order_by(JumpHistory.created_at.desc())
        .limit(limit)
    )
    return list(result)


@router.get("/analyses/{jump_id}", response_model=JumpRead)
async def get_analysis(
    jump_id: uuid.UUID, identity: TelegramDep, session: SessionDep
) -> JumpHistory:
    user = await _ensure_user(session, identity)
    jump = await session.scalar(
        select(JumpHistory).where(
            JumpHistory.id == jump_id, JumpHistory.user_id == user.id
        )
    )
    if jump is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return jump


@router.post("/analyses", response_model=JumpRead, status_code=status.HTTP_202_ACCEPTED)
async def create_analysis(
    identity: TelegramDep,
    session: SessionDep,
    video: Annotated[UploadFile, File()],
    athlete_height_cm: Annotated[float, Form(ge=100, le=250)],
    jump_name: Annotated[str, Form()],
    rotation_count: Annotated[int, Form()],
    analysis_mode: Annotated[str, Form()] = "single",
    cascade_element_count: Annotated[int | None, Form()] = None,
) -> JumpHistory:
    settings = get_settings()
    if video.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported video format")
    try:
        jump_type = make_analysis_type(
            analysis_mode, jump_name, rotation_count, cascade_element_count
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    user = await _ensure_user(session, identity)
    jump_id = uuid.uuid4()
    filename = Path(video.filename or "jump.mp4").name
    try:
        path = LocalStorage(settings.storage_dir).video_path(user.id, jump_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    max_bytes = settings.max_video_mb * 1024 * 1024
    written = 0
    try:
        with path.open("xb") as destination:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"Максимальный размер — {settings.max_video_mb} МБ")
                destination.write(chunk)
        metadata = validate_video_file(path, max_bytes, settings.max_video_seconds)
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Video upload failed") from exc
    finally:
        await video.close()

    jump = JumpHistory(
        id=jump_id,
        user_id=user.id,
        status=AnalysisStatus.QUEUED,
        source_file_key=str(path),
        source_file_sha256=sha256_file(path),
        duration_ms=round(metadata.duration_s * 1000),
        jump_type=jump_type,
        calibration_method="athlete_height",
        metric_data={
            "athlete_height_cm": athlete_height_cm,
            "analysis_mode": analysis_mode,
            "cascade_element_count": cascade_element_count,
        },
    )
    session.add(jump)
    try:
        await session.commit()
        await session.refresh(jump)
        analyze_video_task.delay(str(jump.id))
    except Exception:
        await session.rollback()
        path.unlink(missing_ok=True)
        raise
    return jump
