# ============================================
# Bank Products Recommender API Dockerfile
# ============================================
# Multi-stage build for optimized image size

# ---------- Stage 1: Build ---------- #
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt


# ---------- Stage 2: Runtime ---------- #
FROM python:3.10-slim as runtime

# Metadata
LABEL maintainer="MLE Team"
LABEL description="Bank Products Recommender API with Prometheus monitoring"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Runtime configuration
ENV RECOMMENDER_HOST=0.0.0.0
ENV RECOMMENDER_PORT=8080
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV MODEL_NAME=ovr_grouped_santander

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start FastAPI application
CMD ["python", "-m", "uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8080"]
