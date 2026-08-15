FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[cv]"

COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --create-home --uid 10001 app && mkdir -p /data/storage && chown -R app:app /data
USER app

CMD ["uvicorn", "jumpbot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
