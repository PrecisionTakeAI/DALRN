# Multi-stage build for DALRN FHE Service with TenSEAL
FROM python:3.11-slim as builder

# Install build dependencies for TenSEAL and cryptographic libraries
# TenSEAL requires additional dependencies for SEAL library compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    libssl-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libboost-all-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with REQUIRED TenSEAL
# CRITICAL: TenSEAL is mandatory - build must fail if installation fails
# Note: TenSEAL compilation may take significant time
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir tenseal==0.3.16 || \
    (echo "FATAL: TenSEAL installation failed. This service REQUIRES real homomorphic encryption." && exit 1) && \
    pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
# TenSEAL requires these at runtime for SEAL operations
RUN apt-get update && apt-get install -y \
    curl \
    libssl3 \
    libprotobuf32 \
    libgomp1 \
    libboost-thread1.74.0 \
    libboost-system1.74.0 \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and directories
RUN useradd -m -u 1000 dalrn && \
    mkdir -p /app/logs /app/keys && \
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
EXPOSE 8200

# Health check with PoDP validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8200/health || exit 1

# Run the application with PoDP instrumentation and epsilon budget enforcement
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8200", "--log-level", "info"]