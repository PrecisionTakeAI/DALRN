# API Reference

## Base URLs

- **Development:** `http://localhost:{port}`
- **Production:** `https://api.dalrn.com`

## Authentication

All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <jwt_token>
```

### Auth Endpoints

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}

Response 200:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Refresh Token
```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ..."
}

Response 200:
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Register
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password",
  "full_name": "John Doe"
}

Response 201:
{
  "user_id": "uuid",
  "email": "user@example.com",
  "message": "Registration successful"
}
```

## Gateway Service (Port 8000)

### Submit Dispute
```http
POST /submit-dispute
Authorization: Bearer <token>
Content-Type: application/json

{
  "parties": ["party_a", "party_b"],
  "jurisdiction": "US",
  "cid": "QmHash...",
  "enc_meta": {
    "encrypted": true,
    "algorithm": "AES-256"
  }
}

Response 200:
{
  "dispute_id": "disp_123abc",
  "status": "submitted",
  "anchor_tx": "0x...",
  "receipt_cid": "QmReceipt..."
}
```

### Get Status
```http
GET /status/{dispute_id}
Authorization: Bearer <token>

Response 200:
{
  "dispute_id": "disp_123abc",
  "phase": "negotiation",
  "receipts": [...],
  "anchor_txs": ["0x..."],
  "epsilon_budget": {
    "used": 0.5,
    "remaining": 3.5
  }
}
```

### Health Check
```http
GET /health

Response 200:
{
  "status": "healthy",
  "podp_compliant": true,
  "services_health": {
    "search": "healthy",
    "fhe": "healthy",
    "negotiation": "healthy"
  }
}
```

## Search Service (Port 8100)

### Build Index
```http
POST /build
Content-Type: application/json

{
  "embeddings": [
    [0.1, 0.2, ...],  // 768-dim vector
    [0.3, 0.4, ...],
    ...
  ]
}

Response 200:
{
  "count": 100,
  "index_id": "idx_abc123",
  "dimension": 768
}
```

### Query Vectors
```http
POST /query
Content-Type: application/json

{
  "query": [0.1, 0.2, ...],  // 768-dim vector
  "k": 10,
  "reweight_iters": 0
}

Response 200:
{
  "ids": [1, 5, 3, ...],
  "scores": [0.95, 0.92, 0.89, ...],
  "latency_ms": 8.5,
  "recall_at_10": 0.96
}
```

### gRPC Interface
```protobuf
service SearchService {
  rpc Query(QueryRequest) returns (QueryResponse);
  rpc BuildIndex(BuildRequest) returns (BuildResponse);
}

message QueryRequest {
  string dispute_id = 1;
  repeated float query_vec = 2;
  int32 k = 3;
  int32 reweight_iters = 4;
}
```

## FHE Service (Port 8200)

### Create Encryption Context
```http
POST /context/create
Content-Type: application/json

{
  "tenant_id": "tenant_123",
  "params": {
    "poly_modulus_degree": 8192,
    "global_scale": 1099511627776
  }
}

Response 200:
{
  "context_id": "ctx_abc123",
  "public_key": "base64_encoded_key",
  "parameters": {...}
}
```

### Encrypted Dot Product
```http
POST /dot
Content-Type: application/json

{
  "tenant_id": "tenant_123",
  "enc_query": "base64_encrypted_vector",
  "enc_vectors": [
    "base64_encrypted_vector_1",
    "base64_encrypted_vector_2"
  ]
}

Response 200:
{
  "enc_scores": [
    "base64_encrypted_result_1",
    "base64_encrypted_result_2"
  ],
  "computation_id": "comp_xyz",
  "receipt_id": "rcpt_123"
}
```

## Negotiation Service (Port 8300)

### Compute Nash Equilibrium
```http
POST /negotiate
Content-Type: application/json

{
  "payoff_matrix_A": [
    [3, 1],
    [0, 2]
  ],
  "payoff_matrix_B": [
    [2, 1],
    [0, 3]
  ],
  "selection_rule": "nsw",
  "batna": [1.0, 1.0]
}

Response 200:
{
  "equilibrium": {
    "player_A": [0.5, 0.5],
    "player_B": [0.5, 0.5]
  },
  "payoffs": {
    "player_A": 1.5,
    "player_B": 1.5
  },
  "explanation_cid": "QmExplanation...",
  "receipt_id": "rcpt_456"
}
```

### Enhanced Negotiation
```http
POST /negotiate/enhanced
Content-Type: application/json

{
  "dispute_id": "disp_123",
  "game_matrices": {...},
  "config": {
    "max_iterations": 100,
    "convergence_threshold": 0.001
  }
}

Response 200:
{
  "solution": {...},
  "explanation": "Detailed explanation...",
  "cid": "QmCID...",
  "receipts": [...]
}
```

## Federated Learning Service (Port 8400)

### Privacy Budget Check
```http
POST /precheck
Content-Type: application/json

{
  "tenant_id": "tenant_123",
  "model_id": "model_abc",
  "requested_epsilon": 0.5
}

Response 200:
{
  "allowed": true,
  "remaining_budget": 3.5,
  "total_budget": 4.0
}
```

### Commit Epsilon Usage
```http
POST /commit
Content-Type: application/json

{
  "tenant_id": "tenant_123",
  "model_id": "model_abc",
  "round": 5,
  "epsilon": 0.5,
  "delta": 1e-6
}

Response 200:
{
  "ok": true,
  "spent": 0.5,
  "ledger_entry_id": "entry_789"
}
```

### Get Ledger
```http
GET /ledger/{tenant_id}/{model_id}

Response 200:
{
  "entries": [
    {
      "round": 1,
      "epsilon": 0.1,
      "delta": 1e-6,
      "timestamp": "2025-09-23T10:00:00Z"
    },
    ...
  ],
  "total_spent": 0.5,
  "budget_remaining": 3.5
}
```

### Cross-Silo Registration
```http
POST /fl/cross-silo/register
Content-Type: application/json

{
  "silo_id": "silo_001",
  "organization": "healthcare_org_1",
  "num_samples": 10000,
  "public_key": "hex_encoded_key"
}

Response 200:
{
  "silo_id": "silo_001",
  "registered": true,
  "current_round": 0,
  "min_silos_required": 3,
  "current_silos": 1
}
```

## Agent Service (Port 8500)

### Initialize Network
```http
POST /api/v1/soan/initialize
Content-Type: application/json

{
  "n_nodes": 100,
  "k_edges": 6,
  "p_rewire": 0.1
}

Response 200:
{
  "network_id": "net_abc123",
  "metrics": {
    "nodes": 100,
    "edges": 300,
    "avg_degree": 6,
    "clustering": 0.45
  }
}
```

### Train GNN
```http
POST /api/v1/soan/train
Content-Type: application/json

{
  "network_id": "net_abc123",
  "epochs": 100
}

Response 200:
{
  "model_metrics": {
    "loss": 0.05,
    "accuracy": 0.95
  },
  "receipt_id": "rcpt_gnn"
}
```

### Optimize Topology
```http
POST /api/v1/soan/optimize
Content-Type: application/json

{
  "network_id": "net_abc123",
  "iterations": 20
}

Response 200:
{
  "optimization_results": {
    "initial_latency": 5.2,
    "final_latency": 3.1,
    "improvement": 40.4
  },
  "new_topology": {...}
}
```

## Common Response Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid input |
| 401 | Unauthorized | Missing/invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Error | Server error |

## Rate Limiting

All endpoints are rate limited:
- **Anonymous:** 10 requests/minute
- **Authenticated:** 100 requests/minute
- **Premium:** 1000 requests/minute

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1695470400
```

## Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {
      "field": "specific_field",
      "reason": "validation_error"
    },
    "trace_id": "abc-123-def"
  }
}
```

## WebSocket Endpoints

### Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time updates
};

// Subscribe to dispute updates
ws.send(JSON.stringify({
  "action": "subscribe",
  "dispute_id": "disp_123"
}));
```

## SDK Examples

### Python
```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/auth/login",
    json={"email": "user@example.com", "password": "password"}
)
token = response.json()["access_token"]

# Make authenticated request
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    "http://localhost:8000/status/disp_123",
    headers=headers
)
```

### JavaScript
```javascript
// Using fetch API
const login = async () => {
  const response = await fetch('http://localhost:8000/auth/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      email: 'user@example.com',
      password: 'password'
    })
  });
  const data = await response.json();
  return data.access_token;
};
```

### cURL
```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'

# Authenticated request
curl -X GET http://localhost:8000/status/disp_123 \
  -H "Authorization: Bearer <token>"
```

---

*API Reference - Version 1.0 - Last Updated: 2025-09-23*