# MusicGen + Demucs API Dockerfile
# Multi-stage build for smaller final image

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Brussels

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    tzdata \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    curl \
    make \
    build-essential \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster, better dependency resolution (pinned version)
# Install system-wide using pip so it's accessible by all users
RUN pip install uv==0.5.30

# Use Python 3.11 (compatible with torch 2.1.0 wheels)
ENV UV_PYTHON=3.11

# Create non-root user
RUN useradd -m -u 1001 -s /bin/bash appuser

# Create data directories and set permissions upfront
# This ensures all directories exist and have correct ownership before installing dependencies
RUN mkdir -p /data/output /home/appuser/.cache /app && \
    chown -R appuser:appuser /data /home/appuser/.cache /app

# Set working directory
WORKDIR /app

# Copy project files and set ownership
COPY --chown=appuser:appuser pyproject.toml uv.lock* Makefile README.md ./

# Switch to non-root user for dependency installation
# This ensures the uv cache is created in appuser's home directory
USER appuser

# Set cache directories to appuser's home before installing
ENV HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch \
    UV_CACHE_DIR=/home/appuser/.cache/uv

# Install Python dependencies as appuser
RUN make install

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Optional: Pre-download models at build time (as appuser)
# Uncomment the following to include models in the image (increases image size significantly)
# This reduces startup time but increases image size by ~5GB
# RUN uv run python -c "from audiocraft.models import MusicGen; MusicGen.get_pretrained('facebook/musicgen-small')"
# RUN uv run python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run application
CMD ["make", "run"]
