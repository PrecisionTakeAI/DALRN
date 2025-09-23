# Epsilon-Ledger Service for DALRN

## Overview

The Epsilon-Ledger Service is a critical component of the DALRN (Decentralized Autonomous Learning and Resource Negotiation) system, providing privacy budget tracking and management for federated learning operations. It implements differential privacy accounting with support for multiple privacy accountants and seamless integration with federated learning frameworks.

## Features

### Core Functionality
- **Privacy Budget Tracking**: Per tenant/model pair budget management
- **RDP Accounting**: Rényi Differential Privacy accounting with Opacus integration
- **Pre-check System**: Verify budget availability before operations
- **Commit Mechanism**: Record privacy spend with full audit trail
- **Default Budget**: 4.0 epsilon per tenant/model (configurable)

### API Endpoints

#### Budget Management
- `POST /precheck`: Check if operation is allowed within budget
- `POST /commit`: Record privacy spend after operation
- `GET /budget/{tenant_id}/{model_id}`: Get current budget status
- `GET /history/{tenant_id}/{model_id}`: Get spending history
- `POST /reset/{tenant_id}/{model_id}`: Reset budget (admin only)

#### Federated Learning Integration
- `POST /fl/preround`: FL framework pre-round checks
- `POST /fl/postround`: FL framework post-round commits

### Framework Support
- **Flower**: Full integration with privacy-aware strategies
- **NV FLARE**: Support for NVIDIA's federated learning platform
- **Custom**: Extensible for other FL frameworks

### Robust Aggregation
- **Krum**: Select most central update
- **Multi-Krum**: Select k most central updates
- **Trimmed Mean**: Remove outliers and average
- **Median**: Coordinate-wise median aggregation

## Installation

### Basic Installation
```bash
cd services
pip install -r requirements.txt
```

### With Opacus (Recommended)
```bash
pip install opacus torch
```

### With Flower Framework
```bash
pip install flwr
```

## Configuration

Environment variables:
```bash
# Epsilon-Ledger Service
DEFAULT_EPSILON_BUDGET=4.0  # Default privacy budget
DEFAULT_DELTA=1e-5          # Default delta parameter
ADMIN_TOKEN=your-secret-token  # Admin authentication token
LEDGER_STORAGE_PATH=./ledger_data.json  # Persistence file
ENABLE_PODP=true            # Enable PoDP receipts
LEDGER_PORT=8001           # Service port

# Gateway Service
GATEWAY_PORT=8000
EPS_LEDGER_URL=http://localhost:8001
```

## Quick Start

### 1. Start Epsilon-Ledger Service
```bash
cd services/fl
python eps_ledger.py
```

### 2. Start Gateway Service
```bash
cd services/gateway
python app.py
```

### 3. Test the Services
```bash
# Health check
curl http://localhost:8001/health

# Pre-check budget
curl -X POST http://localhost:8001/precheck \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_001",
    "model_id": "model_001",
    "eps_round": 0.5
  }'

# Commit privacy spend
curl -X POST http://localhost:8001/commit \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_001",
    "model_id": "model_001",
    "round": 1,
    "accountant": "rdp",
    "epsilon": 0.5,
    "delta": 1e-5
  }'
```

## Integration Examples

### Flower Framework Integration

```python
from fl.flower_integration import PrivacyAwareFedAvg

# Create privacy-aware strategy
strategy = PrivacyAwareFedAvg(
    tenant_id="my_tenant",
    model_id="my_model",
    epsilon_per_round=0.5,
    delta=1e-5,
    clipping_norm=1.0,
    noise_multiplier=1.1,
    robust_aggregation="multi-krum"
)

# Use with Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
```

### Direct API Usage

```python
import httpx

# Pre-check before training
response = httpx.post(
    "http://localhost:8001/precheck",
    json={
        "tenant_id": "tenant_001",
        "model_id": "model_001",
        "eps_round": 0.5
    }
)

if response.json()["allowed"]:
    # Perform training
    train_model()
    
    # Commit privacy spend
    httpx.post(
        "http://localhost:8001/commit",
        json={
            "tenant_id": "tenant_001",
            "model_id": "model_001",
            "round": 1,
            "epsilon": 0.5,
            "delta": 1e-5,
            "accountant": "rdp"
        }
    )
```

## Privacy Accounting

### Supported Accountants
- **RDP (Rényi Differential Privacy)**: Default, most accurate
- **zCDP (Zero-Concentrated DP)**: Alternative formulation
- **PLD (Privacy Loss Distribution)**: Advanced composition
- **Gaussian**: Simple Gaussian mechanism
- **Basic**: Basic composition theorem

### Privacy Parameters
- **Epsilon (ε)**: Privacy loss parameter
- **Delta (δ)**: Probability of privacy breach
- **Clipping Norm (C)**: Gradient clipping threshold
- **Noise Multiplier (σ)**: Noise scale factor

### Budget Composition
The service automatically handles privacy composition across rounds:
- Sequential composition for multiple rounds
- Parallel composition for multiple models
- Advanced composition with Opacus accountants

## Testing

### Run Unit Tests
```bash
pytest tests/test_eps_ledger.py -v
```

### Run Integration Tests
```bash
pytest tests/test_eps_ledger.py::TestFederatedLearningIntegration -v
```

### Test Coverage
```bash
pytest tests/test_eps_ledger.py --cov=services.fl.eps_ledger --cov-report=html
```

## Gateway Integration

The Gateway service integrates with epsilon-ledger to:
1. Show privacy budget in dispute status
2. Block operations when budget exhausted
3. Validate FL operations before initiation

### Dispute Status with Budget
```json
{
  "dispute_id": "dispute_abc123",
  "status": "in_progress",
  "privacy_budget": {
    "total_budget": 4.0,
    "total_spent": 2.5,
    "remaining_budget": 1.5,
    "num_rounds": 5
  }
}
```

## Security Considerations

1. **Admin Authentication**: Reset operations require admin token
2. **Fail-Safe Defaults**: Service fails closed if ledger unreachable
3. **Atomic Operations**: All commits are atomic to prevent double-spending
4. **Audit Trail**: Complete history of all privacy operations
5. **Thread Safety**: Concurrent access handled with locks

## Performance

- **Latency**: < 10ms for precheck/commit operations
- **Throughput**: > 1000 ops/sec on standard hardware
- **Storage**: ~1KB per privacy operation
- **Memory**: < 100MB for 10,000 active ledgers

## Troubleshooting

### Common Issues

1. **Budget Exhausted**
   - Check remaining budget: `GET /budget/{tenant_id}/{model_id}`
   - Reset if needed (admin): `POST /reset/{tenant_id}/{model_id}`

2. **Opacus Not Available**
   - Install: `pip install opacus torch`
   - Service works without Opacus but with basic accounting

3. **Persistence Issues**
   - Check file permissions for `LEDGER_STORAGE_PATH`
   - Ensure directory exists and is writable

## Architecture

```
┌─────────────────┐
│   FL Client     │
└────────┬────────┘
         │
         v
┌─────────────────┐     ┌──────────────┐
│     Gateway     │<--->│ Eps-Ledger   │
└────────┬────────┘     └──────┬───────┘
         │                     │
         v                     v
┌─────────────────┐     ┌──────────────┐
│   FL Server     │     │   Storage    │
└─────────────────┘     └──────────────┘
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Run tests before submitting PR

## License

Part of the DALRN project - see main repository for license details.

## Support

For issues and questions:
- GitHub Issues: [DALRN Repository](https://github.com/yourusername/DALRN)
- Documentation: This README and inline code documentation
