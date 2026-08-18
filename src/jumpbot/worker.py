import asyncio
import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from aiogram import Bot
from celery import Celery

from jumpbot.config import get_settings
from jumpbot.cv.pipeline import analyze_jump
from jumpbot.db.models import AnalysisStatus, JumpHistory, User
from jumpbot.db.session import SessionLocal

settings = get_settings()
celery_app = Celery("jumpbot", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_track_started=True, task_time_limit=300, task_soft_time_limit=270)
logger = logging.getLogger(__name__)


@celery_app.task(name="jumpbot.analyze_video", bind=True, max_retries=1)
def analyze_video_task(self: object, jump_id: str) -> dict[str, object]:
    return asyncio.run(_analyze_video(uuid.UUID(jump_id)))


async def _analyze_video(jump_id: uuid.UUID) -> dict[str, object]:
    async with SessionLocal() as session:
        jump = await session.get(JumpHistory, jump_id)
        if jump is None:
            raise ValueError("Analysis job not found")
        if jump.status == AnalysisStatus.COMPLETED:
            return jump.metric_data or {}
        user = await session.get(User, jump.user_id)
        jump.status = AnalysisStatus.PROCESSING
        await session.commit()

        source_path = Path(jump.source_file_key or "")
        try:
            height_m = float(user.height_cm) / 100 if user and user.height_cm else None
            result = analyze_jump(source_path, height_m)
            payload = result.as_dict()
            jump.status = AnalysisStatus.COMPLETED
            jump.source_fps = Decimal(str(round(result.fps, 3)))
            jump.frame_count = result.frame_count
            jump.takeoff_frame = result.phases.takeoff
            jump.apex_frame = result.phases.apex
            jump.landing_frame = result.phases.landing
            jump.flight_time_ms = round(result.flight_time_s * 1000)
            jump.height_flight_cm = Decimal(str(round(result.height_flight_m * 100, 2)))
            if result.height_displacement_m is not None:
                jump.height_displacement_cm = Decimal(
                    str(round(result.height_displacement_m * 100, 2))
                )
            if result.takeoff_velocity_mps is not None:
                jump.takeoff_velocity_mps = Decimal(str(round(result.takeoff_velocity_mps, 3)))
            if result.max_propulsion_velocity_mps is not None:
                jump.max_propulsion_mps = Decimal(
                    str(round(result.max_propulsion_velocity_mps, 3))
                )
            jump.max_angular_velocity_dps = Decimal(
                str(round(result.max_angular_velocity_dps or 0, 2))
            )
            jump.confidence_score = Decimal(str(round(result.confidence_score, 3)))
            jump.quality_flags = result.quality_flags
            jump.phase_data = payload["phases"]  # type: ignore[assignment]
            jump.metric_data = payload
            jump.completed_at = datetime.now(UTC)
            await session.commit()
            if user and settings.telegram_bot_token:
                await _notify_user(
                    user.telegram_user_id,
                    "Анализ завершён. Высота по времени полёта: "
                    f"{result.height_flight_m * 100:.1f} см. "
                    f"Достоверность: {result.confidence_score:.0%}.",
                )
            return payload
        except ValueError as exc:
            jump.status = AnalysisStatus.REJECTED
            jump.error_code = "video_quality"
            jump.error_message = str(exc)
            jump.completed_at = datetime.now(UTC)
            await session.commit()
            if user and settings.telegram_bot_token:
                await _notify_user(
                    user.telegram_user_id,
                    f"Видео не удалось надёжно проанализировать: {exc}. "
                    "Попробуйте переснять сбоку.",
                )
            return {"status": "rejected", "reason": str(exc)}
        except Exception as exc:
            logger.exception("Jump analysis failed", extra={"jump_id": str(jump_id)})
            jump.status = AnalysisStatus.FAILED
            jump.error_code = "processing_error"
            jump.error_message = str(exc)[:1000]
            jump.completed_at = datetime.now(UTC)
            await session.commit()
            raise
        finally:
            if settings.keep_source_video_days == 0 and source_path.is_file():
                source_path.unlink(missing_ok=True)


async def _notify_user(telegram_user_id: int, text: str) -> None:
    bot = Bot(settings.telegram_bot_token)
    try:
        await bot.send_message(telegram_user_id, text)
    except Exception:
        logger.exception(
            "Telegram notification failed",
            extra={"telegram_user_id": telegram_user_id},
        )
    finally:
        await bot.session.close()
