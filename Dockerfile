# Whisper ASR API — Docker Image
# =============================================================================
# NOTE: AutoDL and similar GPU cloud platforms typically do NOT support Docker.
# For those environments, use deploy.sh instead.
# This Dockerfile is provided for standard GPU server deployments.
# =============================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r whisper && useradd -r -g whisper -d /app whisper

# App directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create directories
RUN mkdir -p /app/models /tmp/whisper_api && \
    chown -R whisper:whisper /app /tmp/whisper_api

# Switch to non-root user
USER whisper

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health/ready || exit 1

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
