# DALRN Gateway Service - Enhanced Production Implementation

## Overview

The DALRN Gateway Service is a production-ready FastAPI application serving as the main entry point for the Decentralized Alternative Legal Resolution Network. Two versions are available:

- **app.py**: Original implementation with basic PoDP support
- **app_enhanced.py**: Full production implementation with all features (100% complete)

## Quick Start

### Run Original Version
```bash
uvicorn services.gateway.app:app --reload --host 0.0.0.0 --port 8000
```

### Run Enhanced Version (Recommended)
```bash
uvicorn services.gateway.app_enhanced:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker Compose (Full Stack)
```bash
docker-compose up -d
```

## Enhanced Features (app_enhanced.py)

### Complete Implementation
✅ **Evidence Submission** - `/evidence` endpoint with full PoDP receipt generation
✅ **Enhanced Health Checks** - Real-time monitoring of all services (IPFS, blockchain, SOAN, etc.)
✅ **SOAN Integration** - Automatic routing through self-organizing agent networks
✅ **Production Storage** - Redis/PostgreSQL with automatic fallback to in-memory
✅ **Manual Anchoring** - `/anchor-manual` endpoint for on-demand blockchain anchoring
✅ **Receipt Retrieval** - `/receipts/{dispute_id}` with pagination support
✅ **Prometheus Metrics** - `/metrics` endpoint with comprehensive monitoring
✅ **Receipt Validation** - `/validate-receipt` for cryptographic verification
✅ **Circuit Breakers** - Automatic failure recovery for all external services
✅ **Rate Limiting** - Multi-tier sliding window rate limiting
✅ **Privacy Budget** - Epsilon-differential privacy tracking
✅ **PII Redaction** - Automatic sensitive data removal from logs

## API Endpoints

### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/submit-dispute` | Submit new dispute with PoDP |
| GET | `/status/{dispute_id}` | Get dispute status |
| POST | `/evidence` | Submit encrypted evidence |
| POST | `/anchor-manual` | Trigger manual anchoring |
| GET | `/receipts/{dispute_id}` | Get all receipts |
| POST | `/validate-receipt` | Validate receipt |
| GET | `/health` | Enhanced health check |
| GET | `/metrics` | Prometheus metrics |

## Testing

```bash
# Run comprehensive test suite
pytest tests/test_gateway_enhanced.py -v

# With coverage report
pytest tests/test_gateway_enhanced.py --cov=services.gateway
```

## Configuration

See `config.py` for full configuration options. Key environment variables:
- `REDIS_HOST`, `REDIS_PORT` - Redis configuration
- `POSTGRES_HOST`, `POSTGRES_DB` - PostgreSQL configuration
- `IPFS_HOST`, `IPFS_API_PORT` - IPFS configuration
- `EPSILON_TOTAL_BUDGET` - Privacy budget (default: 4.0)
- `ENABLE_SOAN_ROUTING` - Enable SOAN integration (default: true)

## Files

- **app.py** - Original gateway implementation
- **app_enhanced.py** - Complete production implementation (recommended)
- **config.py** - Configuration management
- **requirements.txt** - Dependencies
- **docker-compose.yml** - Full stack orchestration
- **Dockerfile** - Production container
- **prometheus.yml** - Metrics configuration
- **test_gateway_enhanced.py** - Comprehensive test suite (in tests/)

## Documentation

- Interactive API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Metrics: http://localhost:8000/metrics
- Health: http://localhost:8000/health
