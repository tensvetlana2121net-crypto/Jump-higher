FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_HOME=/opt/rtmlib
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-dejavu-core libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# Install dependencies in a layer that does not change with application code.
# Hatchling needs a package directory to build the dependency-only placeholder.
RUN mkdir -p src/jumpbot && touch src/jumpbot/__init__.py && \
    pip install --no-cache-dir ".[cv]" && rm -rf src
RUN mkdir -p "$TORCH_HOME"

# MediaPipe ships the lite/full pose models, but downloads the heavy model lazily.
# Fetch it while building so the runtime container can remain read-only.
RUN python -c "import mediapipe as mp; mp.solutions.pose.Pose(model_complexity=2).close()"

# Cache RTMPose weights in the image; production inference needs no internet.
RUN python -c "from rtmlib import BodyWithFeet; BodyWithFeet(mode='balanced', backend='onnxruntime', device='cpu')"

COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --create-home --uid 10001 app && mkdir -p /data/storage && \
    chown -R app:app /data "$TORCH_HOME"
USER app

CMD ["uvicorn", "jumpbot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
