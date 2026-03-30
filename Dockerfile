# ============================================================================
# Amazon ML Challenge — Production Dockerfile
# Multi-stage build optimised for AWS EC2/ECS GPU instances
# ============================================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps required at build-time only
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------- Stage 2: Runtime ---------------
FROM python:3.10-slim AS runtime

LABEL maintainer="amazon-ml-challenge"
LABEL description="Multimodal Price Prediction — DeBERTa + ResNet50 + GBM Ensemble"

# Runtime system dependencies
# libgomp1  — LightGBM / XGBoost OpenMP support
# libglib2  — PIL / image processing
# libgl1    — OpenCV headless backend (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p data/raw data/processed models submissions models/ocr

# Volume mount points (data & models injected at runtime)
VOLUME ["/app/data", "/app/models"]

# Default ports — Streamlit (8501) / FastAPI (8000)
EXPOSE 8501 8000

# Health-check (simple HTTP probe for container orchestrators)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stinfo/health || exit 1

# Default entrypoint — runs the full prediction pipeline
ENTRYPOINT ["python", "-u"]
CMD ["src/predict_model.py"]
