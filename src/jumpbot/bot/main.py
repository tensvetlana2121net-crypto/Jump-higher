import asyncio
import logging
import uuid
from decimal import Decimal

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message
from sqlalchemy import select

from jumpbot.config import get_settings
from jumpbot.db.models import AnalysisStatus, JumpHistory, User
from jumpbot.db.session import SessionLocal
from jumpbot.logging import configure_logging
from jumpbot.services.quota import consume_analysis
from jumpbot.services.storage import LocalStorage, sha256_file
from jumpbot.services.video_validation import validate_video_file
from jumpbot.worker import analyze_video_task

router = Router()
settings = get_settings()


class HeightSetup(StatesGroup):
    waiting_for_height = State()


async def ensure_user(message: Message) -> User:
    assert message.from_user is not None
    async with SessionLocal() as session:
        user = await session.scalar(
            select(User).where(User.telegram_user_id == message.from_user.id)
        )
        if user is None:
            user = User(
                telegram_user_id=message.from_user.id,
                username=message.from_user.username,
                first_name=message.from_user.first_name,
                language_code=message.from_user.language_code or "ru",
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user


@router.message(Command("start"))
async def start(message: Message) -> None:
    await ensure_user(message)
    await message.answer(
        "Отправьте видео прыжка. Затем бот попросит ввести рост спортсмена "
        "в сантиметрах и автоматически начнёт анализ.\n\n"
        "Снимайте сбоку, держите камеру неподвижно и оставьте стопы в кадре.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("height"))
async def set_height(message: Message) -> None:
    await message.answer(
        "Сначала отправьте видео прыжка. Затем бот попросит ввести рост спортсмена "
        "в сантиметрах специально для этого анализа. Рост в аккаунте не сохраняется."
    )


@router.message(Command("history"))
async def history(message: Message) -> None:
    user = await ensure_user(message)
    async with SessionLocal() as session:
        rows = list(
            await session.scalars(
                select(JumpHistory)
                .where(JumpHistory.user_id == user.id)
                .order_by(JumpHistory.created_at.desc())
                .limit(5)
            )
        )
    if not rows:
        await message.answer("История пока пуста.")
        return
    lines = [
        f"{row.created_at:%d.%m}: {row.height_flight_cm or '—'} см ({row.status.value})"
        for row in rows
    ]
    await message.answer("Последние анализы:\n" + "\n".join(lines))


@router.message(F.video | F.document)
async def receive_video(message: Message, bot: Bot, state: FSMContext) -> None:
    video = message.video or message.document
    if video is None:
        return
    await state.set_state(HeightSetup.waiting_for_height)
    await state.update_data(
        pending_file_id=video.file_id,
        pending_file_size=video.file_size or 0,
        pending_mime_type=getattr(video, "mime_type", None) or "video/mp4",
        pending_file_name=getattr(video, "file_name", None) or "jump.mp4",
    )
    await message.answer(
        "Введите рост спортсмена в сантиметрах одним числом, например: <b>160</b>. "
        "После этого я автоматически начну анализ уже отправленного видео. "
        "Рост в аккаунте не сохраняется.",
        parse_mode=ParseMode.HTML,
    )
    return


@router.message(HeightSetup.waiting_for_height, F.text)
async def receive_height_for_pending_video(
    message: Message, bot: Bot, state: FSMContext
) -> None:
    try:
        height = Decimal((message.text or "").strip().replace(",", "."))
    except Exception:
        await message.answer("Введите рост одним числом в сантиметрах, например: 160.")
        return
    if not Decimal("100") <= height <= Decimal("250"):
        await message.answer("Введите рост от 100 до 250 сантиметров.")
        return
    user = await ensure_user(message)
    data = await state.get_data()
    await state.clear()
    await message.answer(f"Рост принят для этого анализа: {height} см. Начинаю анализ видео.")
    await _process_video(
        message,
        bot,
        user,
        str(data["pending_file_id"]),
        int(data["pending_file_size"]),
        str(data["pending_mime_type"]),
        str(data["pending_file_name"]),
        height,
    )


async def _process_video(
    message: Message,
    bot: Bot,
    user: User,
    file_id: str,
    file_size: int,
    mime_type: str,
    name: str,
    athlete_height_cm: Decimal,
) -> None:
    max_bytes = settings.max_video_mb * 1024 * 1024
    if file_size <= 0 or file_size > max_bytes:
        await message.answer(f"Максимальный размер — {settings.max_video_mb} МБ.")
        return
    allowed_mime_types = {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
        "video/webm",
    }
    if mime_type not in allowed_mime_types:
        await message.answer("Поддерживаются только MP4, MOV, AVI, MKV и WebM.")
        return
    jump_id = uuid.uuid4()
    try:
        path = LocalStorage(settings.storage_dir).video_path(user.id, jump_id, name)
    except ValueError as exc:
        await message.answer(str(exc))
        return

    try:
        await bot.download(file_id, destination=path)
        metadata = validate_video_file(path, max_bytes, settings.max_video_seconds)
    except ValueError as exc:
        path.unlink(missing_ok=True)
        await message.answer(f"Видео отклонено: {exc}")
        return
    except Exception:
        logging.exception("Video download or validation failed")
        path.unlink(missing_ok=True)
        await message.answer("Не удалось безопасно проверить видео. Попробуйте другой файл.")
        return

    async with SessionLocal() as session:
        if not await consume_analysis(session, user.id):
            path.unlink(missing_ok=True)
            await message.answer("Бесплатный недельный лимит исчерпан.")
            return
        jump = JumpHistory(
            id=jump_id,
            user_id=user.id,
            status=AnalysisStatus.QUEUED,
            source_file_key=str(path),
            source_file_sha256=sha256_file(path),
            duration_ms=round(metadata.duration_s * 1000),
            calibration_method="athlete_height",
            metric_data={"athlete_height_cm": float(athlete_height_cm)},
        )
        session.add(jump)
        await session.commit()
    analyze_video_task.delay(str(jump_id))
    await message.answer("Видео принято. Анализирую!")


async def main() -> None:
    configure_logging()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
    bot = Bot(settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
