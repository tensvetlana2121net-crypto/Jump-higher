import asyncio
import logging
import uuid
from decimal import Decimal

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from sqlalchemy import func, select

from jumpbot.bot.welcome_card import welcome_card_png
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
    waiting_for_mode = State()
    waiting_for_height = State()


class PublicationSetup(StatesGroup):
    waiting_for_rotation = State()
    waiting_for_name = State()
    waiting_for_cascade_details = State()


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
    caption = (
        "Научился падать, научись взлетать!\n\n"
        "Загрузи видео до 10 сек.\n"
        "«Предварительное вращение» бот не считает!\n"
        "Используй приложения для анализа статистики."
    )
    image_bytes = welcome_card_png()
    for attempt in range(3):
        try:
            await message.answer_photo(
                BufferedInputFile(image_bytes, filename="jump-higher-welcome.png"),
                request_timeout=90,
            )
            return
        except TelegramNetworkError:
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
    logging.getLogger(__name__).error("Welcome card delivery failed after retries")
    await message.answer(caption, request_timeout=30)


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


async def _publish_jump(
    telegram_user_id: int,
    jump_id: uuid.UUID,
    *,
    jump_type: str | None = None,
    declared_jump_label: str | None = None,
) -> bool:
    async with SessionLocal() as session:
        jump = await session.scalar(
            select(JumpHistory)
            .join(User, User.id == JumpHistory.user_id)
            .where(
                JumpHistory.id == jump_id,
                User.telegram_user_id == telegram_user_id,
                JumpHistory.status == AnalysisStatus.COMPLETED,
            )
        )
        if jump is None:
            return False
        if jump_type is not None:
            jump.jump_type = jump_type
        metrics = dict(jump.metric_data or {})
        if declared_jump_label:
            metrics["declared_jump_label"] = declared_jump_label
        metrics["published_to_app"] = True
        jump.metric_data = metrics
        jump.published_to_app = True
        await session.commit()
        return True


@router.callback_query(F.data.startswith("app:publish:"))
async def choose_publish(callback: CallbackQuery, state: FSMContext) -> None:
    if callback.from_user is None:
        return
    try:
        jump_id = uuid.UUID((callback.data or "").rsplit(":", 1)[1])
    except (ValueError, IndexError):
        await callback.answer("Некорректный результат", show_alert=True)
        return
    async with SessionLocal() as session:
        jump = await session.scalar(
            select(JumpHistory)
            .join(User, User.id == JumpHistory.user_id)
            .where(JumpHistory.id == jump_id, User.telegram_user_id == callback.from_user.id)
        )
    if jump is None or jump.status != AnalysisStatus.COMPLETED:
        await callback.answer("Результат не найден", show_alert=True)
        return
    metrics = jump.metric_data or {}
    mode = str(metrics.get("analysis_mode", "single"))
    await state.update_data(publication_jump_id=str(jump_id), publication_mode=mode)
    await callback.answer()
    if callback.message is None:
        return
    await callback.message.edit_reply_markup(reply_markup=None)
    if mode == "cascade":
        await state.set_state(PublicationSetup.waiting_for_cascade_details)
        await callback.message.answer(
            "Заполните данные прыжка: напишите состав каскада, например:\n"
            "<b>Двойной аксель + ойлер + тройной тулуп</b>",
            parse_mode=ParseMode.HTML,
        )
        return
    await state.set_state(PublicationSetup.waiting_for_rotation)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Одинарный", callback_data="app:rotation:1"),
                InlineKeyboardButton(text="Двойной", callback_data="app:rotation:2"),
            ],
            [
                InlineKeyboardButton(text="Тройной", callback_data="app:rotation:3"),
                InlineKeyboardButton(text="Четверной", callback_data="app:rotation:4"),
            ],
        ]
    )
    await callback.message.answer("Укажите количество оборотов:", reply_markup=keyboard)


@router.callback_query(F.data.startswith("app:keep:"))
async def keep_best_later(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer("Результат оставлен только в боте")
    if callback.message is not None:
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer(
            "Хорошо, этот результат не переношу. Можно отправить новое видео и дождаться "
            "лучшего результата."
        )


@router.callback_query(PublicationSetup.waiting_for_rotation, F.data.startswith("app:rotation:"))
async def choose_publication_rotation(callback: CallbackQuery, state: FSMContext) -> None:
    try:
        rotation = int((callback.data or "").rsplit(":", 1)[1])
    except (ValueError, IndexError):
        rotation = 0
    if rotation not in {1, 2, 3, 4}:
        await callback.answer("Выберите количество оборотов", show_alert=True)
        return
    data = await state.get_data()
    await state.update_data(publication_rotation=rotation)
    await callback.answer()
    if callback.message is None:
        return
    if data.get("publication_mode") == "floor_tour":
        jump_id = uuid.UUID(str(data["publication_jump_id"]))
        labels = {
            1: "Одинарный тур",
            2: "Двойной тур",
            3: "Тройной тур",
            4: "Четверной тур",
        }
        published = await _publish_jump(
            callback.from_user.id,
            jump_id,
            jump_type=f"{rotation}_floor_tour",
            declared_jump_label=labels[rotation],
        )
        await state.clear()
        await callback.message.answer(
            "Результат добавлен в приложение." if published else "Не удалось найти результат."
        )
        return
    await state.set_state(PublicationSetup.waiting_for_name)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Аксель", callback_data="app:name:axel"),
                InlineKeyboardButton(text="Риттбергер", callback_data="app:name:loop"),
            ],
            [
                InlineKeyboardButton(text="Сальхов", callback_data="app:name:salchow"),
                InlineKeyboardButton(text="Флип", callback_data="app:name:flip"),
            ],
            [
                InlineKeyboardButton(text="Луц", callback_data="app:name:lutz"),
                InlineKeyboardButton(text="Тулуп", callback_data="app:name:toe_loop"),
            ],
        ]
    )
    await callback.message.answer("Выберите название прыжка:", reply_markup=keyboard)


@router.callback_query(PublicationSetup.waiting_for_name, F.data.startswith("app:name:"))
async def choose_publication_name(callback: CallbackQuery, state: FSMContext) -> None:
    jump_name = (callback.data or "").rsplit(":", 1)[-1]
    names = {
        "axel": "Аксель",
        "loop": "Риттбергер",
        "salchow": "Сальхов",
        "flip": "Флип",
        "lutz": "Луц",
        "toe_loop": "Тулуп",
    }
    if jump_name not in names:
        await callback.answer("Выберите название прыжка", show_alert=True)
        return
    data = await state.get_data()
    rotation = int(data["publication_rotation"])
    jump_id = uuid.UUID(str(data["publication_jump_id"]))
    rotation_labels = {1: "Одинарный", 2: "Двойной", 3: "Тройной", 4: "Четверной"}
    published = await _publish_jump(
        callback.from_user.id,
        jump_id,
        jump_type=f"{rotation}_{jump_name}",
        declared_jump_label=f"{rotation_labels[rotation]} {names[jump_name]}",
    )
    await state.clear()
    await callback.answer()
    if callback.message is not None:
        await callback.message.answer(
            "Результат добавлен в приложение." if published else "Не удалось найти результат."
        )


@router.message(PublicationSetup.waiting_for_cascade_details, F.text)
async def receive_cascade_details(message: Message, state: FSMContext) -> None:
    label = " ".join((message.text or "").split())
    if not 5 <= len(label) <= 120:
        await message.answer("Введите состав каскада текстом длиной от 5 до 120 символов.")
        return
    if message.from_user is None:
        return
    data = await state.get_data()
    jump_id = uuid.UUID(str(data["publication_jump_id"]))
    published = await _publish_jump(
        message.from_user.id,
        jump_id,
        declared_jump_label=label,
    )
    await state.clear()
    await message.answer(
        "Результат добавлен в приложение." if published else "Не удалось найти результат."
    )


@router.message(Command("myid"))
async def my_id(message: Message) -> None:
    if message.from_user is None:
        return
    await message.answer(
        f"Ваш Telegram ID: <code>{message.from_user.id}</code>",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("stats"))
async def stats(message: Message) -> None:
    if message.from_user is None or not settings.is_admin(message.from_user.id):
        await message.answer(
            "Статистика доступна только владельцу. Отправьте /myid, чтобы узнать свой "
            "Telegram ID для безопасного добавления в настройку сервера."
        )
        return
    async with SessionLocal() as session:
        total_users = int(await session.scalar(select(func.count(User.id))) or 0)
        rows = list(await session.scalars(select(JumpHistory)))
    completed = sum(row.status == AnalysisStatus.COMPLETED for row in rows)
    rejected = sum(row.status == AnalysisStatus.REJECTED for row in rows)
    failed = sum(row.status == AnalysisStatus.FAILED for row in rows)
    floor_tours = sum(
        row.jump_type == "floor_tour" or row.jump_type.endswith("_floor_tour") for row in rows
    )
    cascades = sum(
        row.jump_type == "cascade" or row.jump_type.startswith("cascade_") for row in rows
    )
    ice_singles = len(rows) - floor_tours - cascades
    await message.answer(
        "Статистика Jump Higher\n\n"
        f"Пользователей: {total_users}\n"
        f"Видео всего: {len(rows)}\n"
        f"Успешных анализов: {completed}\n"
        f"Отклонено по качеству: {rejected}\n"
        f"Технических ошибок: {failed}\n\n"
        f"Одиночные на льду: {ice_singles}\n"
        f"Каскады: {cascades}\n"
        f"Туры в зале: {floor_tours}\n\n"
        "Доступ остаётся бесплатным; оплаты и подписки отключены."
    )


@router.message(F.video | F.document)
async def receive_video(message: Message, bot: Bot, state: FSMContext) -> None:
    video = message.video or message.document
    if video is None:
        return
    await state.set_state(HeightSetup.waiting_for_mode)
    await state.update_data(
        pending_file_id=video.file_id,
        pending_file_size=video.file_size or 0,
        pending_mime_type=getattr(video, "mime_type", None) or "video/mp4",
        pending_file_name=getattr(video, "file_name", None) or "jump.mp4",
    )
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Одиночный прыжок", callback_data="mode:single")],
            [InlineKeyboardButton(text="Каскад (2–3 элемента)", callback_data="mode:cascade")],
            [InlineKeyboardButton(text="Туры в зале (на полу)", callback_data="mode:floor_tour")],
        ]
    )
    await message.answer("Что снято на видео?", reply_markup=keyboard)
    return


@router.callback_query(HeightSetup.waiting_for_mode, F.data.startswith("mode:"))
async def receive_analysis_mode(callback: CallbackQuery, state: FSMContext) -> None:
    mode = (callback.data or "").partition(":")[2]
    if mode not in {"single", "cascade", "floor_tour"}:
        await callback.answer("Неизвестный режим", show_alert=True)
        return
    await state.update_data(analysis_mode=mode)
    await state.set_state(HeightSetup.waiting_for_height)
    await callback.answer()
    if callback.message is not None:
        labels = {
            "single": "одиночного прыжка на льду",
            "cascade": "каскада на льду",
            "floor_tour": "тура в зале на полу",
        }
        label = labels[mode]
        await callback.message.answer(
            f"Выбран анализ {label}. Введите рост спортсмена одним числом, например: "
            "<b>160</b>. Рост в аккаунте не сохраняется.",
            parse_mode=ParseMode.HTML,
        )


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
        str(data.get("analysis_mode", "single")),
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
    analysis_mode: str,
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
            jump_type={
                "cascade": "cascade",
                "floor_tour": "floor_tour",
            }.get(analysis_mode, "countermovement"),
            published_to_app=False,
            calibration_method="athlete_height",
            metric_data={
                "athlete_height_cm": float(athlete_height_cm),
                "analysis_mode": analysis_mode,
                "training_surface": "floor" if analysis_mode == "floor_tour" else "ice",
                "submission_source": "bot",
            },
        )
        session.add(jump)
        await session.commit()
    analyze_video_task.delay(str(jump_id))
    if analysis_mode == "cascade":
        await message.answer("Каскад принят. Выделяю элементы отдельно, не как один прыжок.")
    elif analysis_mode == "floor_tour":
        await message.answer("Тур в зале принят. Анализирую прыжок на полу!")
    else:
        await message.answer("Одиночный прыжок принят. Анализирую!")


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
