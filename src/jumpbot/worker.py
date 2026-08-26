import asyncio
import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from time import perf_counter

from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError
from aiogram.types import BufferedInputFile
from celery import Celery
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import select

from jumpbot.config import get_settings
from jumpbot.cv.assessment import assess_declared_rotation, compare_with_personal_baseline
from jumpbot.cv.pipeline import analyze_jump
from jumpbot.cv.types import AnalysisResult
from jumpbot.db.models import AnalysisStatus, JumpHistory, User
from jumpbot.db.session import SessionLocal

settings = get_settings()
celery_app = Celery("jumpbot", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_track_started=True, task_time_limit=300, task_soft_time_limit=270)
logger = logging.getLogger(__name__)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _result_card_png(result: AnalysisResult) -> bytes:
    image = Image.new("RGB", (1080, 1450), "#F5F3EF")
    draw = ImageDraw.Draw(image)
    title_font = _font(48, bold=True)
    label_font = _font(27)
    value_font = _font(43, bold=True)
    footer_font = _font(34, bold=True)

    if result.fps < 50 and result.rotation_degrees is not None:
        rotation_degrees = round(result.rotation_degrees / 10.0) * 10.0
        rotation_value = f"≈{rotation_degrees:.0f}°"
        turns_value = f"≈{rotation_degrees / 360.0:.2f}"
    else:
        rotation_value = (
            f"{result.rotation_degrees:.1f}°" if result.rotation_degrees is not None else "—"
        )
        turns_value = f"{result.rotation_turns:.2f}" if result.rotation_turns is not None else "—"

    height = result.height_ballistic_m or result.jump_height_m
    metrics = (
        ("Высота по параболе полёта", f"{height * 100:.1f} см"),
        (
            "Подъём центра масс",
            f"{result.height_displacement_m * 100:.1f} см"
            if result.height_displacement_m is not None
            else "—",
        ),
        (
            "Вертикальная скорость взлёта",
            f"{result.takeoff_velocity_mps:.2f} м/с"
            if result.takeoff_velocity_mps is not None
            else "—",
        ),
        (
            "Скорость при отталкивании",
            f"{result.max_propulsion_velocity_mps:.2f} м/с"
            if result.max_propulsion_velocity_mps is not None
            else "—",
        ),
        ("Наклон до отрыва", f"{result.max_inclination_deg:.1f}°"),
        ("Осевое вращение корпуса", rotation_value),
        ("Количество оборотов", turns_value),
        (
            "Скорость вращения",
            f"{result.max_angular_velocity_dps:.0f}°/с"
            if result.max_angular_velocity_dps is not None
            else "—",
        ),
        (
            "Частота вращения",
            f"{result.max_angular_velocity_dps / 6.0:.0f} об/мин"
            if result.max_angular_velocity_dps is not None
            else "—",
        ),
        ("Наклон при отрыве", f"{result.takeoff_inclination_deg:+.1f}°"),
    )
    colors = ("#DDEBE6", "#E8E2F0", "#E9E5D2", "#DCE7F2", "#F0E0DC")

    draw.text((60, 55), "Анализ прыжка", fill="#35443F", font=title_font)
    for index, (label, value) in enumerate(metrics):
        column, row = index % 2, index // 2
        left = 60 + column * 500
        top = 155 + row * 220
        draw.rounded_rectangle(
            (left, top, left + 460, top + 185),
            radius=28,
            fill=colors[index % len(colors)],
        )
        draw.multiline_text(
            (left + 28, top + 24),
            label,
            fill="#52615C",
            font=label_font,
            spacing=5,
        )
        draw.text((left + 28, top + 112), value, fill="#263833", font=value_font)

    draw.rounded_rectangle((60, 1285, 1020, 1385), radius=30, fill="#D7E7DF")
    footer = "Отличный результат! Попробуй ещё раз!"
    footer_box = draw.textbbox((0, 0), footer, font=footer_font)
    footer_width = footer_box[2] - footer_box[0]
    draw.text(((1080 - footer_width) / 2, 1313), footer, fill="#35443F", font=footer_font)
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


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
        lines.append(f"Высота по траектории центра масс: {result.height_trajectory_m * 100:.1f} см")
    if result.height_ballistic_m is not None:
        lines.append(f"Высота по параболе полёта: {result.height_ballistic_m * 100:.1f} см")
    if result.height_displacement_m is not None:
        lines.append(f"Подъём центра масс: {result.height_displacement_m * 100:.1f} см")
    if result.takeoff_velocity_mps is not None:
        lines.append(
            f"Расчётная вертикальная скорость взлёта: {result.takeoff_velocity_mps:.2f} м/с"
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
        lines.append(f"Частота вращения: {result.max_angular_velocity_dps / 6.0:.0f} об/мин")
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
        if result.fps < 50:
            displayed_degrees = round(result.rotation_degrees / 10.0) * 10.0
            uncertainty = round((result.max_angular_velocity_dps or 0.0) / result.fps / 5.0) * 5.0
            lines.append(
                f"Осевое вращение корпуса: ≈{displayed_degrees:.0f}° "
                f"({displayed_degrees / 360.0:.2f} оборота), "
                f"погрешность около ±{max(5.0, uncertainty):.0f}°"
            )
        else:
            lines.append(
                f"Осевое вращение корпуса: {result.rotation_degrees:.1f}° "
                f"({result.rotation_turns:.2f} оборота)"
            )
        lines.append("Направление вращения по одному 2D-видео надёжно не определяется.")
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
            analysis_mode = (jump.metric_data or {}).get("analysis_mode", "single")
            if analysis_mode == "cascade":
                jump.status = AnalysisStatus.REJECTED
                jump.error_code = "cascade_segmentation_pending"
                jump.error_message = (
                    "Каскад распознан как отдельный тип видео. Поэлементное выделение "
                    "2–3 прыжков ещё калибруется; видео не анализировалось как один прыжок."
                )
                jump.completed_at = datetime.now(UTC)
                await session.commit()
                if user and settings.telegram_bot_token:
                    await _notify_user(
                        user.telegram_user_id,
                        "Каскад принят правильно, но поэлементный анализ пока калибруется. "
                        "Я не стал выдавать весь каскад за один прыжок. Сохраните видео — "
                        "оно подходит для следующей версии анализатора каскадов.",
                    )
                return {"status": "rejected", "reason": "cascade segmentation pending"}
            height_m = float(requested_height_cm) / 100 if requested_height_cm is not None else None
            analysis_started = perf_counter()
            result = analyze_jump(
                source_path,
                height_m,
                settings.pose_backend,
                settings.pose_tracking_roi_enabled,
                settings.pose_camera_stabilization_enabled,
            )
            logger.info(
                "Jump analysis computation finished",
                extra={
                    "jump_id": str(jump_id),
                    "analysis_seconds": round(perf_counter() - analysis_started, 3),
                },
            )
            payload = result.as_dict()
            payload["technique_assessment"] = assess_declared_rotation(payload, jump.jump_type)
            previous_metrics = await session.scalars(
                select(JumpHistory.metric_data)
                .where(
                    JumpHistory.user_id == jump.user_id,
                    JumpHistory.jump_type == jump.jump_type,
                    JumpHistory.status == AnalysisStatus.COMPLETED,
                    JumpHistory.id != jump.id,
                )
                .order_by(JumpHistory.created_at.desc())
                .limit(5)
            )
            personal_comparison = compare_with_personal_baseline(
                payload,
                [metrics for metrics in previous_metrics if metrics],
            )
            if personal_comparison is not None:
                payload["personal_comparison"] = personal_comparison
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
                await _notify_result_card(user.telegram_user_id, result)
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
                    "Take-off occurs too close to video start": (
                        "Начало полёта оказалось у самой границы видео. Оставьте не менее "
                        "1 секунды перед первым прыжком; для каскада выберите режим каскада."
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
                    "Analysis confidence is too low": (
                        "Модель не смогла уверенно удержать главного спортсмена. "
                        "Для каскада нужен отдельный режим; для одиночного прыжка "
                        "обрежьте видео до 1–2 секунд до отрыва и после выезда."
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


async def _notify_result_card(telegram_user_id: int, result: AnalysisResult) -> None:
    bot = Bot(settings.telegram_bot_token)
    try:
        image_bytes = _result_card_png(result)
        for attempt in range(3):
            try:
                image = BufferedInputFile(image_bytes, filename="jump-result.png")
                await bot.send_photo(telegram_user_id, image, request_timeout=90)
                return
            except TelegramNetworkError:
                if attempt == 2:
                    raise
                await asyncio.sleep(3 * (attempt + 1))
    except Exception:
        logger.exception(
            "Telegram result card delivery failed",
            extra={"telegram_user_id": telegram_user_id},
        )
    finally:
        await bot.session.close()
