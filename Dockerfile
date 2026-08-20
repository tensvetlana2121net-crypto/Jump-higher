FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_HOME=/opt/rtmlib
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[cv]"
RUN mkdir -p "$TORCH_HOME"

# MediaPipe ships the lite/full pose models, but downloads the heavy model lazily.
# Fetch it while building so the runtime container can remain read-only.
RUN python -c "import mediapipe as mp; mp.solutions.pose.Pose(model_complexity=2).close()"

# Cache RTMPose weights in the image; production inference needs no internet.
RUN python -c "from rtmlib import BodyWithFeet; BodyWithFeet(mode='balanced', backend='onnxruntime', device='cpu')"

COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --create-home --uid 10001 app && mkdir -p /data/storage && \
    chown -R app:app /data "$TORCH_HOME"
USER app

CMD ["uvicorn", "jumpbot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
