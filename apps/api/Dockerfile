FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BEATFORGE_PROJECT_ROOT=/app \
    BEATFORGE_STORAGE_DIR=/app/storage \
    TORCH_HOME=/app/storage/models/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY apps/api/pyproject.toml apps/api/README.md /app/apps/api/
COPY apps/api/beatforge_api /app/apps/api/beatforge_api
RUN python -m pip install /app/apps/api

COPY scripts /app/scripts
COPY storage/demo /app/storage/demo
COPY storage/covers /app/storage/covers

EXPOSE 8000
CMD ["sh", "-c", "python scripts/generate_demo_audio.py && python scripts/seed_demo_projects.py && python -m uvicorn beatforge_api.main:app --app-dir apps/api --host 0.0.0.0 --port 8000"]
