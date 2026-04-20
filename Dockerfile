# syntax=docker/dockerfile:1.6
# Minimal CPU image sufficient to run `scripts/smoke_test.py`, the regression
# test suite, and the ruff check. GPU training is intentionally out of scope:
# anyone with a GPU should use the host Python env or a CUDA-enabled base.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 git \
 && rm -rf /var/lib/apt/lists/*

# Install CPU-only Python deps from the lock file. We intentionally install
# the CPU torch wheels to keep the image small; GPU users should override.
COPY requirements.lock.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 torchvision==0.20.1 \
 && pip install \
      numpy==2.3.4 pandas==2.3.3 pyyaml==6.0.3 omegaconf==2.3.0 \
      loguru==0.7.3 pillow==12.0.0 \
      fastapi==0.121.1 uvicorn==0.39.0 \
      pytest==9.0.2 ruff==0.14.4 scipy==1.16.0 \
      ultralytics==8.3.214

# Copy the project and install it in editable mode so `src/agridrone/` is
# importable just like on the host.
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY evaluate ./evaluate
COPY configs ./configs
COPY models ./models
RUN pip install -e .

# Expose port (Render uses 10000 by default, configurable via $PORT)
EXPOSE 10000

# Default: run the API server. Render sets $PORT automatically.
CMD ["sh", "-c", "python -m uvicorn agridrone.api.app:get_app --factory --host 0.0.0.0 --port ${PORT:-10000}"]
