# Multi-stage build for DALRN FL/Epsilon-Ledger Service
FROM python:3.11-slim as builder

# Install build dependencies for federated learning and cryptographic libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libssl-dev \
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
    libgfortran5 \
    libgomp1 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and directories
RUN useradd -m -u 1000 dalrn && \
    mkdir -p /app/logs /app/models /app/ledger && \
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
EXPOSE 8400

# Health check with PoDP validation and epsilon budget verification
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8400/health || exit 1

# Run the application with PoDP instrumentation and epsilon ledger enforcement
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8400", "--log-level", "info"]