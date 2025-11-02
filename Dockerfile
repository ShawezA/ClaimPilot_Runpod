FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Minimal libs for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Base image already has torch/torchvision (CUDA). Do not reinstall them.
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y opencv-python || true \
 && pip install --no-cache-dir --upgrade --force-reinstall opencv-python-headless

# Copy code and models into the image
COPY pipeline_server.py handler.py /app/
COPY models /app/models

# Prefer GPU unless overridden
ENV DEVICE=cuda

CMD ["python", "-u", "handler.py"]