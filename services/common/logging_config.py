"""
Centralized logging configuration for all DALRN services
Includes structured logging, correlation IDs, and metrics integration
"""
import logging
import logging.handlers
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import uuid
from pathlib import Path

# Try to import JSON logger for structured logging
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGING = True
except ImportError:
    print("Warning: python-json-logger not installed. Using standard logging.")
    JSON_LOGGING = False

# Try to import Prometheus client for metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
    METRICS_ENABLED = True
except ImportError:
    print("Warning: prometheus-client not installed. Metrics disabled.")
    METRICS_ENABLED = False

class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to all log records for tracing"""

    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = str(uuid.uuid4())
        return True

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""

    def format(self, record):
        log_obj = {
            '@timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'N/A'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'correlation_id', 'getMessage']:
                log_obj[key] = value

        return json.dumps(log_obj)

class StructuredLogger:
    """Production-ready structured logger with correlation tracking"""

    @staticmethod
    def setup_logging(
        service_name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_json: bool = True
    ) -> logging.Logger:
        """Setup structured logging for a service"""

        # Create logger
        logger = logging.getLogger(service_name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        logger.handlers.clear()

        # Choose formatter
        if enable_json and JSON_LOGGING:
            # Use JSON formatter
            formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(correlation_id)s %(message)s',
                rename_fields={
                    'timestamp': '@timestamp',
                    'level': 'level',
                    'name': 'logger'
                }
            )
        else:
            # Use custom structured formatter
            formatter = StructuredFormatter()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            # Create log directory if needed
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=100_000_000,  # 100MB
                backupCount=10
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add correlation ID filter
        logger.addFilter(CorrelationIdFilter())

        return logger

class MetricsCollector:
    """Collect and export metrics to Prometheus"""

    def __init__(self, service_name: str, registry: Optional[CollectorRegistry] = None):
        self.service_name = service_name
        self.registry = registry or CollectorRegistry()

        if METRICS_ENABLED:
            # Request metrics
            self.request_count = Counter(
                f'{service_name}_requests_total',
                'Total requests',
                ['method', 'endpoint', 'status'],
                registry=self.registry
            )

            self.request_duration = Histogram(
                f'{service_name}_request_duration_seconds',
                'Request duration',
                ['method', 'endpoint'],
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
                registry=self.registry
            )

            # System metrics
            self.active_connections = Gauge(
                f'{service_name}_active_connections',
                'Active connections',
                registry=self.registry
            )

            self.error_count = Counter(
                f'{service_name}_errors_total',
                'Total errors',
                ['error_type'],
                registry=self.registry
            )

            # Business metrics
            self.business_operations = Counter(
                f'{service_name}_operations_total',
                'Business operations',
                ['operation_type'],
                registry=self.registry
            )

    def track_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Track HTTP request metrics"""
        if METRICS_ENABLED:
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()

            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

    def track_error(self, error_type: str):
        """Track error metrics"""
        if METRICS_ENABLED:
            self.error_count.labels(error_type=error_type).inc()

    def track_operation(self, operation_type: str):
        """Track business operation metrics"""
        if METRICS_ENABLED:
            self.business_operations.labels(operation_type=operation_type).inc()

    def set_active_connections(self, count: int):
        """Update active connections gauge"""
        if METRICS_ENABLED:
            self.active_connections.set(count)

# Middleware for FastAPI services
class LoggingMiddleware:
    """FastAPI middleware for structured logging and metrics"""

    def __init__(self, app, logger: logging.Logger, metrics: MetricsCollector):
        self.app = app
        self.logger = logger
        self.metrics = metrics
        self.active_requests = 0

    async def __call__(self, request, call_next):
        from fastapi import Request, Response

        # Generate or extract correlation ID
        correlation_id = request.headers.get(
            'X-Correlation-ID',
            str(uuid.uuid4())
        )

        # Track active connections
        self.active_requests += 1
        self.metrics.set_active_connections(self.active_requests)

        # Start timer
        start_time = time.time()

        # Log request
        self.logger.info(
            "Request received",
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'query': str(request.url.query),
                'client': request.client.host if request.client else None
            }
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Track metrics
            self.metrics.track_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )

            # Log response
            self.logger.info(
                "Request completed",
                extra={
                    'correlation_id': correlation_id,
                    'status': response.status_code,
                    'duration_ms': round(duration * 1000, 2)
                }
            )

            # Add correlation ID to response headers
            response.headers['X-Correlation-ID'] = correlation_id

            return response

        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Track error
            self.metrics.track_error(type(e).__name__)

            # Log error
            self.logger.error(
                f"Request failed: {str(e)}",
                exc_info=True,
                extra={
                    'correlation_id': correlation_id,
                    'duration_ms': round(duration * 1000, 2)
                }
            )

            raise

        finally:
            # Update active connections
            self.active_requests -= 1
            self.metrics.set_active_connections(self.active_requests)

# Service integration helper
def configure_service_logging(
    service_name: str,
    app=None,  # FastAPI app instance
    config: Optional[Dict] = None
) -> Tuple[logging.Logger, MetricsCollector]:
    """Configure logging and metrics for a service"""

    config = config or {}

    # Setup logger
    logger = StructuredLogger.setup_logging(
        service_name=service_name,
        log_level=config.get('log_level', 'INFO'),
        log_file=config.get('log_file'),
        enable_json=config.get('enable_json', True)
    )

    # Setup metrics
    metrics = MetricsCollector(service_name)

    # Add middleware if FastAPI app provided
    if app:
        from fastapi.middleware.cors import CORSMiddleware

        # Add logging middleware
        app.middleware("http")(
            LoggingMiddleware(app, logger, metrics)
        )

        # Add health check endpoint with metrics
        @app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint"""
            if METRICS_ENABLED:
                from prometheus_client import generate_latest
                return Response(
                    content=generate_latest(metrics.registry),
                    media_type="text/plain"
                )
            else:
                return {"error": "Metrics not enabled"}

    logger.info(f"{service_name} logging configured", extra={
        'log_level': config.get('log_level', 'INFO'),
        'json_logging': JSON_LOGGING,
        'metrics_enabled': METRICS_ENABLED
    })

    return logger, metrics

# Logging context manager for operations
class LoggedOperation:
    """Context manager for logging operations with timing"""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.extra = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", extra=self.extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type:
            self.logger.error(
                f"{self.operation} failed after {duration:.2f}s: {exc_val}",
                exc_info=True,
                extra={**self.extra, 'duration_seconds': duration}
            )
        else:
            self.logger.info(
                f"Completed {self.operation} in {duration:.2f}s",
                extra={**self.extra, 'duration_seconds': duration}
            )

        return False  # Don't suppress exceptions

# Usage example for services
"""
# In service startup (e.g., gateway/app.py):

from common.logging_config import configure_service_logging

# Configure logging
logger, metrics = configure_service_logging(
    service_name='gateway',
    app=app,
    config={
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', '/var/log/dalrn/gateway.log'),
        'enable_json': True
    }
)

# Use in code:
with LoggedOperation(logger, 'dispute_submission', dispute_id=dispute_id):
    # Process dispute
    pass

# Track custom metrics:
metrics.track_operation('dispute_submitted')
"""