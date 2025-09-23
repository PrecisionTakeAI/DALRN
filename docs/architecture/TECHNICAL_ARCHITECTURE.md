# Technical Architecture

## Technology Stack

### Core Technologies

#### Programming Languages
- **Python 3.11+:** Primary language for all services
- **Solidity 0.8.24:** Smart contract development
- **JavaScript:** Frontend components (if applicable)

#### Frameworks & Libraries
| Category | Technology | Purpose | Version |
|----------|------------|---------|---------|
| Web Framework | FastAPI | REST APIs | 0.111.0 |
| ML Framework | PyTorch | Neural networks | 2.0+ |
| Graph ML | PyTorch Geometric | GNN implementation | 2.3+ |
| Homomorphic Encryption | TenSEAL | CKKS encryption | 0.3.16 |
| Vector Search | FAISS | Similarity search | 1.8.0 |
| Federated Learning | Flower | FL coordination | 1.5+ |
| Differential Privacy | Opacus | Privacy guarantees | 1.4+ |
| Game Theory | nashpy | Nash equilibrium | 0.0.40 |
| Blockchain | Web3.py | Ethereum interaction | 6.20.1 |

### Infrastructure

#### Databases
- **PostgreSQL:** Primary relational database
- **SQLite:** Automatic fallback for development
- **Redis:** Caching and session management
- **IPFS:** Distributed file storage

#### Container & Orchestration
- **Docker:** Service containerization
- **Docker Compose:** Multi-container orchestration
- **Kubernetes:** Production orchestration (optional)

#### Monitoring & Logging
- **Prometheus:** Metrics collection
- **Grafana:** Metrics visualization
- **Structured Logging:** JSON format with correlation IDs

## Service Architecture

### Design Patterns

1. **Microservices Pattern**
   - Independent deployment
   - Service-specific databases
   - API-based communication

2. **API Gateway Pattern**
   - Central authentication
   - Request routing
   - Rate limiting

3. **Chain of Responsibility**
   - PoDP receipt generation
   - Middleware pipeline
   - Error handling chain

4. **Repository Pattern**
   - Data access abstraction
   - Database agnostic
   - Query optimization

### Service Communication

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Gateway   │────▶│  Service A  │
└─────────────┘     └─────────────┘     └─────────────┘
                            │                    │
                            ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Service B  │     │  Database   │
                    └─────────────┘     └─────────────┘
```

### Standard Service Structure

Each service follows this pattern:

```python
services/[service_name]/
├── __init__.py          # Package initialization
├── service.py           # Main service entry
├── models.py            # Pydantic models
├── routes.py            # API endpoints
├── config.py            # Configuration
├── utils.py             # Utility functions
└── tests/              # Service tests
```

## Data Flow Architecture

### Request Lifecycle

1. **Client Request**
   - HTTPS/WSS connection
   - JSON payload
   - Bearer token authentication

2. **Gateway Processing**
   - JWT validation
   - Rate limiting check
   - Request routing

3. **Service Execution**
   - Business logic
   - Database operations
   - External service calls

4. **PoDP Generation**
   - Receipt creation
   - Merkle tree construction
   - Hash computation

5. **Response**
   - JSON response
   - Receipt included
   - Status codes

### Data Persistence

```
┌──────────────┐
│   Request    │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   Service    │────▶│ PostgreSQL   │
└──────┬───────┘     └──────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│    Redis     │     │   SQLite     │
│   (Cache)    │     │  (Fallback)  │
└──────────────┘     └──────────────┘
```

## Performance Metrics

### Service-Level Metrics

| Service | Metric | Target | Achieved | Method |
|---------|--------|--------|----------|--------|
| Gateway | Throughput | 1K req/s | 10K req/s | Load testing |
| Gateway | Latency P99 | <100ms | <50ms | Percentile analysis |
| Search | Query Time | <100ms | <10ms | FAISS optimization |
| Search | Recall@10 | >90% | >95% | Accuracy testing |
| FHE | Encryption Time | <1s | ~50ms | TenSEAL benchmarks |
| FHE | Accuracy | >98% | >99% | Error analysis |
| FL | Aggregation | <10s | <1s | Parallel processing |
| Nash | Computation | <1s | <100ms | Algorithm optimization |

### System-Level Metrics

- **CPU Usage:** <70% under normal load
- **Memory Usage:** <4GB per service
- **Disk I/O:** <100MB/s sustained
- **Network I/O:** <1Gbps peak
- **Database Connections:** Pool size 20-50
- **Cache Hit Rate:** >80%

## Security Architecture

### Authentication & Authorization

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client     │────▶│   Gateway    │────▶│   Auth       │
│              │◀────│   (JWT)      │◀────│   Service    │
└──────────────┘     └──────────────┘     └──────────────┘
     Bearer Token         Validate            BCrypt
```

### Encryption Layers

1. **Transport Layer**
   - TLS 1.3 for all connections
   - Certificate pinning
   - Perfect forward secrecy

2. **Application Layer**
   - JWT for authentication
   - BCrypt for passwords (cost=12)
   - HMAC for integrity

3. **Computation Layer**
   - TenSEAL CKKS encryption
   - Secure multiparty computation
   - Zero-knowledge proofs

4. **Storage Layer**
   - Encrypted at rest
   - Key rotation
   - Backup encryption

### Threat Model

- **External Threats:** DDoS, injection attacks, MitM
- **Internal Threats:** Privilege escalation, data leakage
- **Mitigations:** Rate limiting, input validation, encryption

## Scalability Design

### Horizontal Scaling

```
        Load Balancer
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
Gateway-1 Gateway-2 Gateway-3
    │        │        │
    └────────┼────────┘
             ▼
      Service Pool
```

### Caching Strategy

1. **L1 Cache:** In-memory (service-level)
2. **L2 Cache:** Redis (shared)
3. **L3 Cache:** CDN (static assets)

### Database Scaling

- **Read Replicas:** For read-heavy workloads
- **Sharding:** By tenant_id or timestamp
- **Connection Pooling:** Prevent connection exhaustion

## Deployment Architecture

### Development Environment

```yaml
# docker-compose.yml snippet
services:
  gateway:
    image: dalrn/gateway:latest
    ports:
      - "8000:8000"
    environment:
      - ENV=development
      - DATABASE_URL=sqlite:///data/dev.db
```

### Production Environment

```yaml
# kubernetes deployment snippet
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dalrn-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gateway
  template:
    spec:
      containers:
      - name: gateway
        image: dalrn/gateway:prod
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Monitoring & Observability

### Metrics Collection

```python
# Prometheus metrics example
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@request_duration.time()
def handle_request():
    request_count.inc()
    # Process request
```

### Logging Architecture

```json
{
  "timestamp": "2025-09-23T10:00:00Z",
  "level": "INFO",
  "service": "gateway",
  "correlation_id": "abc-123",
  "user_id": "user-456",
  "action": "login",
  "duration_ms": 45,
  "status": "success"
}
```

### Health Checks

Each service exposes:
- `/health` - Basic liveness
- `/ready` - Readiness probe
- `/metrics` - Prometheus metrics

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    },
    "trace_id": "xyz-789"
  }
}
```

### Retry Strategy

- **Exponential Backoff:** 2^n seconds
- **Max Retries:** 3-5 attempts
- **Circuit Breaker:** Trip after 50% failure rate

## Development Guidelines

### Code Style
- **PEP 8:** Python style guide
- **Type Hints:** Full typing coverage
- **Docstrings:** Google style
- **Testing:** >80% coverage

### Git Workflow
- **Main Branch:** Production-ready code
- **Feature Branches:** feature/[name]
- **Pull Requests:** Required reviews
- **CI/CD:** Automated testing

---

*Technical Architecture - Last Updated: 2025-09-23*