# Multi-stage build for DALRN Gateway Service with Enhanced Features
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and necessary directories
RUN useradd -m -u 1000 dalrn && \
    mkdir -p /app/logs /app/data /app/cache && \
    chown -R dalrn:dalrn /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/dalrn/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=dalrn:dalrn . .

# Switch to non-root user
USER dalrn

# Add local pip packages to PATH
ENV PATH=/home/dalrn/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check with enhanced endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the enhanced application with production settings
CMD ["python", "-m", "uvicorn", "app_enhanced:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--log-level", "info"]