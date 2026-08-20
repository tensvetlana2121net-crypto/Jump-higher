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
from jumpbot.cv.types import AnalysisResult
from jumpbot.db.models import AnalysisStatus, JumpHistory, User
from jumpbot.db.session import SessionLocal

settings = get_settings()
celery_app = Celery("jumpbot", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_track_started=True, task_time_limit=300, task_soft_time_limit=270)
logger = logging.getLogger(__name__)


def _format_analysis_message(result: AnalysisResult) -> str:
    lines = [
        "✅ Анализ прыжка завершён",
        f"Максимальная высота прыжка: {result.jump_height_m * 100:.1f} см",
        f"Время полёта: {result.flight_time_s:.3f} с",
        f"Наклон корпуса при отрыве: {result.takeoff_inclination_deg:+.1f}°",
        f"Максимальный наклон до отрыва: {result.max_inclination_deg:.1f}°",
    ]
    lines.append(f"Высота по времени полёта: {result.height_flight_m * 100:.1f} см")
    if result.height_trajectory_m is not None:
        lines.append(
            f"Высота по траектории центра масс: {result.height_trajectory_m * 100:.1f} см"
        )
    if result.height_ballistic_m is not None:
        lines.append(f"Высота по параболе полёта: {result.height_ballistic_m * 100:.1f} см")
    if result.height_displacement_m is not None:
        lines.append(f"Подъём центра масс: {result.height_displacement_m * 100:.1f} см")
    if result.takeoff_velocity_mps is not None:
        lines.append(
            "Расчётная вертикальная скорость взлёта: "
            f"{result.takeoff_velocity_mps:.2f} м/с"
        )
    if result.max_propulsion_velocity_mps is not None:
        lines.append(
            "Максимальная вертикальная скорость при отталкивании: "
            f"{result.max_propulsion_velocity_mps:.2f} м/с"
        )
    if result.max_angular_velocity_dps is not None:
        lines.append(
            f"Средняя угловая скорость вращения: {result.max_angular_velocity_dps:.1f} °/с"
        )
        lines.append(
            f"Частота вращения: {result.max_angular_velocity_dps / 6.0:.0f} об/мин"
        )
    if result.takeoff_foot_angle_deg is not None:
        lines.append(
            "Ориентация конька при отрыве относительно горизонта кадра: "
            f"{result.takeoff_foot_angle_deg:+.1f}°"
        )
    if result.landing_foot_angle_deg is not None:
        lines.append(
            "Ориентация конька при приземлении относительно горизонта кадра: "
            f"{result.landing_foot_angle_deg:+.1f}°"
        )
    if result.rotation_degrees is not None and result.rotation_turns is not None:
        direction = {
            "clockwise": "по часовой стрелке",
            "counterclockwise": "против часовой стрелки",
        }.get(result.rotation_direction, "направление не определено")
        lines.append(
            f"Осевое вращение корпуса: {result.rotation_degrees:.1f}° "
            f"({result.rotation_turns:.2f} оборота), {direction}"
        )
    else:
        lines.append("Выраженное вращение: не обнаружено")
    lines.append(f"Достоверность: {result.confidence_score:.0%}")
    if result.quality_flags:
        flag_labels = {
            "low_fps": "низкая частота кадров",
            "low_landmark_visibility": "часть тела распознана неуверенно",
            "interpolated_pose_gap": "краткое перекрытие спортсмена восстановлено по траектории",
            "partial_com_fallback": "центр масс частично восстановлен по центру бёдер",
            "ankle_based_ground_contact": "контакт с полом оценён по лодыжкам",
            "unstable_trunk_orientation": "вращение нельзя определить надёжно",
            "implausible_rotation_speed": "скачок позы исказил скорость вращения",
            "inconsistent_height_estimates": "способы расчёта высоты расходятся",
        }
        labels = [flag_labels.get(flag, flag) for flag in result.quality_flags]
        lines.append("Предупреждения качества: " + "; ".join(labels))
    lines.append("Оценка выполнена по 2D-видео и не является медицинским 3D-измерением.")
    return "\n".join(lines)


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
        delete_source = settings.keep_source_video_days == 0
        try:
            requested_height_cm = (jump.metric_data or {}).get("athlete_height_cm")
            height_m = (
                float(requested_height_cm) / 100
                if requested_height_cm is not None
                else None
            )
            result = analyze_jump(source_path, height_m, settings.pose_backend)
            payload = result.as_dict()
            jump.status = AnalysisStatus.COMPLETED
            jump.source_fps = Decimal(str(round(result.fps, 3)))
            jump.frame_count = result.frame_count
            jump.takeoff_frame = result.phases.takeoff
            jump.apex_frame = result.phases.apex
            jump.landing_frame = result.phases.landing
            jump.flight_time_ms = round(result.flight_time_s * 1000)
            jump.height_flight_cm = Decimal(str(round(result.jump_height_m * 100, 2)))
            if result.height_displacement_m is not None:
                jump.height_displacement_cm = Decimal(
                    str(round(result.height_displacement_m * 100, 2))
                )
            if result.takeoff_velocity_mps is not None:
                jump.takeoff_velocity_mps = Decimal(str(round(result.takeoff_velocity_mps, 3)))
            if result.max_propulsion_velocity_mps is not None:
                jump.max_propulsion_mps = Decimal(str(round(result.max_propulsion_velocity_mps, 3)))
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
                await _notify_user(user.telegram_user_id, _format_analysis_message(result))
            return payload
        except ValueError as exc:
            jump.status = AnalysisStatus.REJECTED
            jump.error_code = "video_quality"
            jump.error_message = str(exc)
            jump.completed_at = datetime.now(UTC)
            await session.commit()
            if user and settings.telegram_bot_token:
                advice = {
                    "Pose tracking contains a long gap": (
                        "Основной спортсмен был перекрыт или пропал из кадра более чем "
                        "на 0,75 с. Держите его в кадре и по возможности без полного перекрытия."
                    ),
                    "Landing was not detected": (
                        "В ролике нужно оставить не менее 1 секунды после касания льда, "
                        "стопы и коньки должны быть видны."
                    ),
                    "Take-off was not detected": (
                        "Начните ролик за 1 секунду до отрыва и не обрезайте коньки."
                    ),
                    "Athlete height is required": (
                        "Сначала укажите рост спортсмена командой /height 170, "
                        "затем отправьте видео повторно."
                    ),
                    "Implausibly long flight interval": (
                        "Не удалось надёжно отделить полёт от скольжения или других движений. "
                        "Оставьте один прыжок и снимайте неподвижной камерой."
                    ),
                    "Trajectory contains a gap that is too long": (
                        "Одна из ключевых точек тела долго не была видна. "
                        "Нужно, чтобы бёдра, плечи и "
                        "хотя бы одна стопа не исчезали из кадра надолго."
                    ),
                }.get(str(exc), str(exc))
                await _notify_user(
                    user.telegram_user_id,
                    f"Видео не удалось надёжно проанализировать. {advice}",
                )
            return {"status": "rejected", "reason": str(exc)}
        except Exception as exc:
            delete_source = False
            logger.exception("Jump analysis failed", extra={"jump_id": str(jump_id)})
            jump.status = AnalysisStatus.FAILED
            jump.error_code = "processing_error"
            jump.error_message = str(exc)[:1000]
            jump.completed_at = datetime.now(UTC)
            await session.commit()
            if user and settings.telegram_bot_token:
                await _notify_user(
                    user.telegram_user_id,
                    "Анализ завершился технической ошибкой. "
                    "Видео сохранено в истории; попробуйте повторить позже.",
                )
            raise
        finally:
            if delete_source and source_path.is_file():
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
