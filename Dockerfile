# Vision AI Platform — FastAPI + UI
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    VLM_PROVIDER=mock \
    FRAMES_ROOT=/tmp/frames \
    MAX_FRAMES_PER_JOB=30

WORKDIR /app

# ffmpeg = robust video decoding; libglib2.0-0 = OpenCV runtime dep
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
