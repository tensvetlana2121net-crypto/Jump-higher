# JumpBot

Telegram-бот и HTTP API для оценки вертикального прыжка по видео со смартфона.

> Проект предназначен для тренировочной аналитики. Результаты не являются медицинским заключением или лабораторным измерением.

## Возможности

- приём видео через Telegram;
- очередь обработки Celery + Redis;
- извлечение позы MediaPipe (33 landmarks);
- высота по времени полёта и перемещению центра бёдер;
- скорость отталкивания, фазы прыжка и оценка качества;
- Free/Pro-квоты;
- история анализов через FastAPI;
- экспорт JSON/CSV;
- PostgreSQL, Docker Compose, Alembic и pytest.

## Быстрый старт

1. Скопируйте `.env.example` в `.env` и укажите `TELEGRAM_BOT_TOKEN`.
2. Запустите инфраструктуру:

```bash
docker compose up --build
```

3. API будет доступно на `http://localhost:8000`, документация — на `/docs`.

Для локальной разработки без Docker:

```bash
python -m venv .venv
pip install -e ".[dev,cv]"
pytest
uvicorn jumpbot.api.main:app --reload
```

## Съёмка

- камера неподвижна и расположена сбоку;
- спортсмен целиком виден в кадре, включая стопы;
- перед прыжком нужно постоять неподвижно 1–2 секунды;
- предпочтительно 60 или 120 FPS;
- один человек в кадре, ровный видимый пол;
- приземление примерно в точке отрыва.

## Архитектура

```text
Telegram/Aiogram ─┐
                  ├─ FastAPI ─ PostgreSQL
                  └─ Redis/Celery ─ OpenCV/MediaPipe ─ storage/
```

Основные каталоги:

```text
src/jumpbot/api       HTTP API
src/jumpbot/bot       Telegram handlers
src/jumpbot/cv        pose, фильтрация, фазы и метрики
src/jumpbot/db        SQLAlchemy-модели и сессии
src/jumpbot/services  квоты, файлы и экспорты
tests                 модульные тесты
```

## Команды

```bash
ruff check .
mypy src
pytest
alembic upgrade head
celery -A jumpbot.worker.celery_app worker -l INFO
python -m jumpbot.bot.main
```

## Ограничения MVP

- масштаб по росту является приближением и чувствителен к перспективе;
- середина бёдер — прокси центра масс, а не сегментная биомеханическая модель;
- автоматические кадры отрыва/приземления нужно валидировать на размеченном датасете;
- при плохом качестве бот отклоняет анализ вместо выдачи псевдоточного результата;
- исходные видео по умолчанию хранятся локально; production-развёртывание должно использовать S3 и lifecycle policy.

## Публикация на GitHub

```bash
git add .
git commit -m "feat: initial JumpBot MVP"
git branch -M main
git remote add origin https://github.com/USERNAME/jumpbot.git
git push -u origin main
```

## Лицензия

MIT. См. [LICENSE](LICENSE).
