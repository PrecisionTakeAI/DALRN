# DALRN Codebase Analysis

## 1. PROJECT OVERVIEW

Based on the codebase analysis, DALRN (Decentralized Alternative Legal Resolution Network) is a distributed system that provides privacy-preserving dispute resolution services using advanced cryptographic and machine learning techniques.

**Main Problem Solved:** The system enables secure, verifiable processing of legal disputes through encrypted data processing, automated negotiation, and cryptographic proof generation, while maintaining privacy and providing audit trails.

**Tech Stack Identified:**
- **Backend Languages:** Python 3.11+, Solidity 0.8.24
- **Frameworks:** FastAPI, gRPC, PyTorch, PyTorch Geometric
- **Databases:** PostgreSQL, Redis, IPFS
- **ML Libraries:** FAISS, TenSEAL, nashpy, scikit-learn, Flower
- **Blockchain:** Ethereum/Web3, Foundry
- **Infrastructure:** Docker, Docker Compose, Prometheus, Grafana
- **Testing:** pytest, pytest-cov, pytest-benchmark

## 2. ARCHITECTURE

### Design Patterns Observed:
- **Microservices Architecture:** Separate services for different functionalities
- **API Gateway Pattern:** Central gateway service routing to specialized services
- **Chain of Responsibility:** PoDP (Proof of Deterministic Processing) receipts
- **Repository Pattern:** Data access through service layers
- **Factory Pattern:** Receipt and chain creation

### Folder Structure:
```
DALRN/
├── services/               # Microservices
│   ├── gateway/           # API gateway with PoDP middleware
│   ├── search/            # Vector search service (FAISS)
│   ├── fhe/               # Homomorphic encryption service
│   ├── negotiation/       # Game theory negotiation service
│   ├── fl/                # Federated learning & privacy ledger
│   ├── agents/            # Self-organizing agent networks
│   ├── chain/             # Blockchain integration
│   └── common/            # Shared utilities (PoDP, IPFS)
├── infra/                 # Infrastructure configuration
│   ├── docker-compose.yml
│   ├── prometheus/        # Monitoring configuration
│   └── grafana/           # Dashboard definitions
├── tests/                 # Test suites
└── reports/               # Generated reports
```

### Data Flow:
1. **Dispute Submission** → Gateway → PoDP Receipt Generation → IPFS Storage
2. **Search Request** → Gateway → Search Service → FAISS Index → Results
3. **Encrypted Operations** → FHE Service → TenSEAL Processing → Client Decryption
4. **Negotiation** → Nash Equilibrium Computation → Explanation Generation → CID Storage
5. **All Operations** → Receipt Chain → Merkle Root → Blockchain Anchoring

## 3. CORE COMPONENTS

### Gateway Service (`services/gateway/app.py`)
- **Purpose:** Main API entry point with PoDP middleware
- **Key Classes:**
  - `SubmitDisputeRequest`: Handles dispute submission
  - `StatusResponse`: Returns dispute status
- **Functions:**
  - `redact_pii()`: Removes sensitive data from logs
  - `submit_dispute()`: Creates new dispute with receipt chain
  - `get_status()`: Retrieves dispute status and receipts

### Search Service (`services/search/service.py`)
- **Purpose:** Vector similarity search using FAISS
- **Key Classes:**
  - `VectorIndex`: Thread-safe FAISS HNSW wrapper
  - `SearchMetrics`: Performance tracking
  - `SearchServicer`: gRPC service implementation
- **Constants:**
  - `VECTOR_DIM = 768`
  - `HNSW_M = 32`
  - `P95_LATENCY_TARGET_MS = 600`

### FHE Service (`services/fhe/service.py`)
- **Purpose:** Homomorphic encryption for privacy-preserving computations
- **Key Classes:**
  - `CKKSContext`: Per-tenant encryption context
  - `DotProductRequest`: Encrypted dot product operations
- **Parameters:**
  - Polynomial degree: 8192
  - Scale: 2^40
  - Security level: 128-bit

### Negotiation Service (`services/negotiation/service.py`)
- **Purpose:** Game-theoretic dispute resolution
- **Key Components:**
  - Nash equilibrium computation using nashpy
  - Multiple selection rules (NSW, Egalitarian, Utilitarian)
  - Explanation memo generation
  - Causal Influence Diagram (CID) creation

### Agent Networks (`services/agents/`)
- **Components:**
  - `topology.py`: Watts-Strogatz network generation (N=100, k=6, p=0.1)
  - `gnn_predictor.py`: 2-layer GCN for latency prediction
  - `queue_model.py`: M/M/1 queueing model
  - `rewiring.py`: ε-greedy network optimization
  - `orchestrator.py`: Coordinates all agent operations

### Blockchain Service (`services/chain/`)
- **Smart Contract:** `AnchorReceipts.sol`
  - Anchors Merkle roots on-chain
  - Stores dispute → root mappings
  - Emits events for audit trail
- **Client:** Python wrapper for contract interaction

### Common Utilities (`services/common/`)
- **PoDP Module (`podp.py`):**
  - `Receipt`: Cryptographic receipt generation
  - `ReceiptChain`: Merkle tree construction
  - `keccak()`: Ethereum-compatible hashing
- **IPFS Module (`ipfs.py`):**
  - `put_json()`: Store data in IPFS
  - `get_json()`: Retrieve data from IPFS

## 4. API ENDPOINTS

### Gateway Service (Port 8000)
```
POST /submit-dispute
  Request: {parties[], jurisdiction, cid, enc_meta}
  Response: {dispute_id, status, anchor_tx, receipt_cid}

GET /status/{dispute_id}
  Response: {dispute_id, phase, receipts[], anchor_txs[], epsilon_budget}

POST /evidence
  Request: {dispute_id, cid, type}
  Response: {evidence_id, receipt_id}

GET /health
  Response: {status, podp_compliant, services_health}
```

### Search Service (Port 8100)
```
POST /build
  Request: {embeddings[][]}
  Response: {count, index_id}

POST /query
  Request: {query[], k, reweight_iters}
  Response: {ids[], scores[], latency_ms, recall_at_10}

gRPC: SearchService.Query()
  Input: Query{dispute_id, query_vec, k, reweight_iters}
  Output: TopK{ids[], scores[]}
```

### FHE Service (Port 8200)
```
POST /dot
  Request: {tenant_id, enc_query, enc_vectors[]}
  Response: {enc_scores[], computation_id, receipt_id}

POST /context/create
  Request: {tenant_id, params}
  Response: {context_id, public_key}

GET /health
  Response: {tenseal_available, contexts_active}
```

### Negotiation Service (Port 8300)
```
POST /negotiate
  Request: {payoff_matrix_A[][], payoff_matrix_B[][], selection_rule, batna}
  Response: {equilibrium, payoffs, explanation_cid, receipt_id}

POST /negotiate/enhanced
  Request: {dispute_id, game_matrices, config}
  Response: {solution, explanation, cid, receipts}

GET /explanation/{cid}
  Response: {memo, cid_diagram, metrics}
```

### FL/Epsilon Service (Port 8400)
```
POST /precheck
  Request: {tenant_id, model_id, requested_epsilon}
  Response: {allowed, remaining_budget, total_budget}

POST /commit
  Request: {tenant_id, model_id, round, epsilon, delta}
  Response: {ok, spent, ledger_entry_id}

GET /ledger/{tenant_id}/{model_id}
  Response: {entries[], total_spent, budget_remaining}
```

### Agent Service (Port 8500)
```
POST /api/v1/soan/initialize
  Request: {n_nodes, k_edges, p_rewire}
  Response: {network_id, metrics}

POST /api/v1/soan/train
  Request: {network_id, epochs}
  Response: {model_metrics, receipt_id}

POST /api/v1/soan/optimize
  Request: {network_id, iterations}
  Response: {optimization_results, new_topology}
```

### Authentication/Authorization
- JWT tokens mentioned but not fully implemented
- Tenant isolation in FHE service
- Rate limiting in gateway (100 req/min token bucket)

## 5. DATA MODELS

### Core Models (from Pydantic schemas)

```python
# Gateway Models
class SubmitDisputeRequest:
    parties: List[str]        # min_length=2
    jurisdiction: str         # min_length=2
    cid: str                 # IPFS CID
    enc_meta: dict

class StatusResponse:
    dispute_id: str
    phase: str
    receipts: List[Receipt]
    anchor_txs: List[str]
    epsilon_budget: dict

# PoDP Models
class Receipt:
    receipt_id: str
    dispute_id: str
    step: str
    inputs: dict
    params: dict
    artifacts: dict
    hashes: dict
    signatures: list
    ts: str
    hash: Optional[str]

class ReceiptChain:
    dispute_id: str
    receipts: List[Receipt]
    merkle_root: Optional[str]
    merkle_leaves: List[str]

# Search Models
class SearchMetrics:
    query_id: str
    k: int
    recall_at_10: float
    latency_ms: float
    reweight_enabled: bool
    total_vectors: int

# FHE Models
class CKKSContext:
    tenant_id: str
    context: Any  # TenSEAL context
    created_at: datetime
    last_used: datetime
    operations_count: int

# Negotiation Models
class NegotiationRequest:
    payoff_matrix_A: List[List[float]]
    payoff_matrix_B: List[List[float]]
    selection_rule: str  # "nsw", "egalitarian", etc.
    batna: Tuple[float, float]

class NegotiationResult:
    equilibrium: dict
    payoffs: dict
    explanation_cid: str
    fairness_metrics: dict

# FL Models
class EpsilonEntry:
    tenant_id: str
    model_id: str
    round: int
    epsilon: float
    delta: float
    mechanism: str
    timestamp: datetime
```

### Blockchain Storage (Solidity)
```solidity
struct RootInfo {
    bytes32 merkleRoot;
    uint256 timestamp;
    uint256 blockNumber;
}

mapping(bytes32 => RootInfo) latestRoots;
mapping(bytes32 => mapping(uint256 => RootInfo)) rootHistory;
```

### Database Schema
Unable to determine from codebase - PostgreSQL mentioned but schema not found

## 6. KEY ALGORITHMS & BUSINESS LOGIC

### Vector Search Algorithm (FAISS HNSW)
- **Index Type:** Hierarchical Navigable Small World
- **Parameters:** M=32, efConstruction=200, efSearch=128
- **Quantum-Inspired Reweighting:** Optional Grover-style amplification
- **Performance Target:** Recall@10 ≥95%, P95 latency <600ms

### Homomorphic Encryption (CKKS)
- **Operations:** Encrypted dot products on unit-norm vectors
- **Batching:** Vector-matrix multiplication support
- **Error Rate:** <10% vs plaintext baseline
- **Context Management:** Per-tenant isolation with TTL=3600s

### Nash Equilibrium Computation
- **Algorithm:** Lemke-Howson enumeration via nashpy
- **Selection Rules:**
  1. Nash Social Welfare (default)
  2. Egalitarian (max-min)
  3. Utilitarian (sum maximization)
  4. Kalai-Smorodinsky
  5. Nash Bargaining
- **Tie Breaking:** Deterministic selection based on hash

### Self-Organizing Networks
- **Topology:** Watts-Strogatz small-world (N=100, k=6, p=0.1)
- **GNN Architecture:** 2-layer GCN, 16 hidden dimensions
- **Queue Model:** M/M/1 with stability detection (λ < μ)
- **Rewiring:** ε-greedy with ε=0.2, 20 iterations

### PoDP Receipt Generation
- **Hash Function:** Keccak256 (Ethereum compatible)
- **Merkle Tree:** Binary tree with canonical JSON serialization
- **Determinism:** Sort keys, fixed separators, ensure_ascii=False

### Privacy Budget Management
- **Total Budget:** ε=4.0 per tenant/model
- **Accountants:** RDP, ZCDP, PLD, Gaussian
- **Overflow Prevention:** Pre-check before operations
- **Composition:** Sequential and parallel composition rules

## 7. DEPENDENCIES

### Python Packages (requirements.txt)
```
fastapi==0.111.0         # Web framework
uvicorn==0.30.1          # ASGI server
pydantic==2.7.4          # Data validation
numpy==1.26.4            # Numerical computing
faiss-cpu==1.8.0         # Vector search
nashpy==0.0.40           # Game theory
eth-hash==0.6.0          # Ethereum hashing
web3==6.20.1             # Blockchain interaction
ipfshttpclient==0.8.0a2  # IPFS client
torch>=2.0.0             # Deep learning
torch-geometric>=2.3.0   # Graph neural networks
networkx>=3.0            # Network analysis
grpcio==1.60.0           # RPC framework
scikit-learn==1.4.0      # Machine learning
```

### External Services
- **IPFS:** Distributed storage for receipts and documents
- **PostgreSQL:** Metadata and state storage
- **Redis:** Caching and session management
- **Anvil:** Local Ethereum node for development
- **Prometheus/Grafana:** Monitoring and observability

## 8. CONFIGURATION

### Environment Variables (from .env.example reference)
```
# Service Ports
GATEWAY_PORT=8000
SEARCH_PORT=8100
FHE_PORT=8200
NEGOTIATION_PORT=8300
FL_PORT=8400
AGENTS_PORT=8500

# PoDP Configuration
PODP_ENABLED=true
PODP_MERKLE_ALGO=keccak256
PODP_RECEIPT_TTL=86400

# Epsilon Budget
EPSILON_BUDGET_DEFAULT=4.0
EPSILON_OVERFLOW_PREVENTION=true

# IPFS
IPFS_API=http://ipfs:5001
IPFS_GATEWAY=http://ipfs:8080

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
REDIS_HOST=redis
REDIS_PORT=6379

# Blockchain
ANCHOR_CONTRACT_ADDRESS=(not set)
WEB3_PROVIDER_URL=http://anvil:8545
```

### Configuration Files
- `infra/docker-compose.yml`: Service orchestration
- `infra/prometheus/prometheus.yml`: Metrics collection
- `infra/grafana/dashboards/*.json`: Dashboard definitions
- `devcontainer.json`: Development environment setup

### Hardcoded Values Found
- Vector dimension: 768 (search service)
- HNSW parameters: M=32, efConstruction=200
- Network size: N=100 nodes
- Privacy budget: ε=4.0
- Token bucket rate: 100 req/min
- Context TTL: 3600 seconds

## 9. CURRENT STATE

### Completed Features
- ✅ Gateway service with PoDP middleware
- ✅ FAISS vector search with gRPC/HTTP interfaces
- ✅ TenSEAL homomorphic encryption service
- ✅ Nash equilibrium negotiation with explanations
- ✅ Epsilon-ledger privacy budget tracking
- ✅ Self-organizing agent networks
- ✅ Blockchain anchoring smart contract
- ✅ Docker orchestration with health checks
- ✅ Comprehensive test suites

### TODOs and FIXMEs Found
```python
# services/fhe/service.py:36
# TODO: Install TenSEAL via: pip install tenseal

# services/gateway/app.py:49
# In-memory storage for demo (use Redis/DB in production)

# services/chain/client.py
# Mock implementation for development
```

### Incomplete Implementations
- JWT authentication partially implemented
- Production database schema not defined
- GPU acceleration for FAISS mentioned but not implemented
- Cross-silo federated learning in progress
- Production blockchain deployment configuration missing

### Potential Issues
1. **Merge Conflict:** Gateway app.py has unresolved merge conflict markers (line 1)
2. **Import Fallbacks:** Multiple services have try/except for imports
3. **Mock Implementations:** Blockchain client returns mock transaction hashes
4. **In-Memory Storage:** Gateway uses dictionaries instead of database
5. **Missing TenSEAL:** FHE service has fallback for missing TenSEAL library

## 10. ENTRY POINTS

### Main Application Entry
```bash
# Start all services via Docker Compose
make run-all

# Or individually:
make run-gateway
make run-search
make run-fhe
make run-negotiation
make run-fl
make run-agents
```

### Service-Specific Entry Points
```python
# Gateway
uvicorn services.gateway.app:app --host 0.0.0.0 --port 8000

# Search (dual mode)
python services/search/service.py  # Starts both HTTP and gRPC

# FHE
uvicorn services.fhe.service:app --host 0.0.0.0 --port 8200

# Negotiation
uvicorn services.negotiation.service:app --host 0.0.0.0 --port 8300

# FL/Epsilon
uvicorn services.fl.eps_ledger:app --host 0.0.0.0 --port 8400
```

### Test Execution
```bash
# Run all tests
pytest tests/

# Specific test suites
pytest tests/test_gateway.py
pytest tests/test_search.py
pytest tests/test_fhe.py
pytest tests/test_negotiation_enhanced.py
pytest tests/test_eps_ledger.py
pytest tests/test_soan.py

# With coverage
pytest --cov=services tests/
```

### Blockchain Deployment
```bash
# Deploy contract locally
cd services/chain
python deploy_local.py

# Or using Foundry
forge script scripts/Deploy.s.sol --broadcast
```

### Makefile Targets (Primary Interface)
The Makefile provides 40+ targets for managing the system:
- **Core:** `run-all`, `stop-all`, `status`, `health-check`
- **Development:** `build`, `rebuild`, `logs`, `shell-<service>`
- **Testing:** `test-integration`, `test-podp`, `test-epsilon`
- **Monitoring:** `monitoring-up`, `metrics`, `podp-report`
- **Maintenance:** `backup`, `restore`, `clean`

### Scripts
- `setup.sh`: Initial environment setup
- `postStart.sh`: Post-container start initialization

---

*This analysis is based solely on the code found in the repository without referencing external documentation.*