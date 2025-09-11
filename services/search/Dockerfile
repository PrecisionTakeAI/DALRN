# Multi-stage build for DALRN Search Service with FAISS
FROM python:3.11-slim as builder

# Install build dependencies for FAISS and scientific packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libblas3 \
    liblapack3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and directories
RUN useradd -m -u 1000 dalrn && \
    mkdir -p /app/logs /app/indexes && \
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
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8100

# Health check with PoDP validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Run the application with PoDP instrumentation
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100", "--log-level", "info"]