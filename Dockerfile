# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    DEVICE=cuda

WORKDIR /app

# Minimal runtime libs for OpenCV and TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install deps (Torch cu121 wheels keep size moderate and run on RunPod GPUs)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python -m pip uninstall -y opencv-python || true \
 && python -m pip install --no-cache-dir --upgrade --force-reinstall \
      opencv-python-headless

# App code
COPY pipeline_server.py handler.py /app/

# Models are baked into the image
COPY models /app/models

CMD ["python", "-u", "handler.py"]