import asyncio
import logging
import uuid
from decimal import Decimal

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import select

from jumpbot.config import get_settings
from jumpbot.db.models import AnalysisStatus, JumpHistory, User
from jumpbot.db.session import SessionLocal
from jumpbot.logging import configure_logging
from jumpbot.services.quota import consume_analysis
from jumpbot.services.storage import LocalStorage, sha256_file
from jumpbot.worker import analyze_video_task

router = Router()
settings = get_settings()


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
        "Пришлите рост командой <code>/height 182</code>, затем отправьте видео прыжка.\n\n"
        "Снимайте сбоку, держите камеру неподвижно и оставьте стопы в кадре.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("height"))
async def set_height(message: Message) -> None:
    user = await ensure_user(message)
    parts = (message.text or "").split()
    if len(parts) != 2:
        await message.answer("Пример: /height 182")
        return
    try:
        height = Decimal(parts[1].replace(",", "."))
    except Exception:
        await message.answer("Укажите рост числом в сантиметрах.")
        return
    if not Decimal("100") <= height <= Decimal("250"):
        await message.answer("Допустимый диапазон роста: 100–250 см.")
        return
    async with SessionLocal() as session:
        stored = await session.get(User, user.id)
        assert stored is not None
        stored.height_cm = height
        await session.commit()
    await message.answer(f"Рост сохранён: {height} см.")


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
async def receive_video(message: Message, bot: Bot) -> None:
    user = await ensure_user(message)
    video = message.video or message.document
    if video is None:
        return
    file_size = video.file_size or 0
    if file_size > settings.max_video_mb * 1024 * 1024:
        await message.answer(f"Максимальный размер — {settings.max_video_mb} МБ.")
        return
    name = getattr(video, "file_name", None) or "jump.mp4"
    jump_id = uuid.uuid4()
    try:
        path = LocalStorage(settings.storage_dir).video_path(user.id, jump_id, name)
    except ValueError as exc:
        await message.answer(str(exc))
        return

    async with SessionLocal() as session:
        if not await consume_analysis(session, user.id):
            await message.answer("Бесплатный недельный лимит исчерпан.")
            return
        jump = JumpHistory(
            id=jump_id,
            user_id=user.id,
            status=AnalysisStatus.QUEUED,
            source_file_key=str(path),
            calibration_method="athlete_height" if user.height_cm else "flight_time",
        )
        session.add(jump)
        await session.commit()

    await bot.download(video, destination=path)
    async with SessionLocal() as session:
        stored = await session.get(JumpHistory, jump_id)
        assert stored is not None
        stored.source_file_sha256 = sha256_file(path)
        await session.commit()
    analyze_video_task.delay(str(jump_id))
    await message.answer(f"Видео принято. Номер анализа: <code>{jump_id}</code>")


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
