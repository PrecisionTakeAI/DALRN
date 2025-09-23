# DALRN Local Testing Guide

## Quick Start (5 minutes)

```bash
# 1. Clone the repository (if you haven't already)
git clone https://github.com/PrecisionTakeAI/DALRN.git
cd DALRN

# 2. Set up environment
cp infra/.env.example infra/.env

# 3. Start all services with Docker
make run-all

# 4. Check health
make health-check

# 5. Access the services
# Gateway API: http://localhost:8000/docs
# Grafana: http://localhost:3000 (admin/admin)
# IPFS: http://localhost:8080
```

## Prerequisites

### Required Software
- **Docker Desktop** (Windows/Mac) or Docker Engine (Linux)
- **Docker Compose** v2.0+
- **Python 3.11+** (for running tests)
- **Make** (Windows: use Git Bash or WSL)
- **Git**

### Hardware Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk**: 10GB free space
- **CPU**: 4+ cores recommended

## Method 1: Full System Test with Docker (Recommended)

### Step 1: Prepare Environment

```bash
# Navigate to project directory
cd C:\Users\luqma\OneDrive\Documents\GitHub\DALRN

# Create environment file
cp infra/.env.example infra/.env

# Edit infra/.env if needed (optional)
# Default settings work for local testing
```

### Step 2: Build and Start Services

```bash
# Build all Docker images
make build

# Start all services
make run-all

# Wait for services to be healthy (takes ~30 seconds)
# You should see green checkmarks for each service
```

### Step 3: Verify Services Are Running

```bash
# Check service status
make status

# Check health endpoints
make health-check

# View logs
make logs
```

### Step 4: Test Core Functionality

#### A. Test Dispute Submission (Gateway)

```bash
# Submit a test dispute
curl -X POST http://localhost:8000/submit-dispute \
  -H "Content-Type: application/json" \
  -d '{
    "parties": ["party_A", "party_B"],
    "jurisdiction": "US-NY",
    "cid": "QmTest12345678901234567890",
    "enc_meta": {"encrypted": true}
  }'

# Check dispute status (use dispute_id from response)
curl http://localhost:8000/status/{dispute_id}
```

#### B. Test Vector Search

```bash
# Build search index
curl -X POST http://localhost:8100/build \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
  }'

# Query the index
curl -X POST http://localhost:8100/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "k": 5
  }'
```

#### C. Test Negotiation Service

```bash
# Run Nash equilibrium computation
curl -X POST http://localhost:8300/negotiate \
  -H "Content-Type: application/json" \
  -d '{
    "payoff_matrix_A": [[3, 0], [5, 1]],
    "payoff_matrix_B": [[3, 5], [0, 1]],
    "selection_rule": "nsw",
    "batna": [0, 0]
  }'
```

#### D. Test Privacy Budget

```bash
# Check epsilon budget
curl -X POST http://localhost:8400/precheck \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test_tenant",
    "model_id": "test_model",
    "requested_epsilon": 0.5
  }'
```

### Step 5: Access Web Interfaces

Open in your browser:
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (login: admin/admin)
- **IPFS WebUI**: http://localhost:5001/webui
- **Prometheus**: http://localhost:9090

## Method 2: Run Individual Services (Development)

### Step 1: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start Infrastructure Services

```bash
# Start only infrastructure (PostgreSQL, Redis, IPFS)
docker-compose -f infra/docker-compose.yml up postgres redis ipfs anvil
```

### Step 3: Run Services Individually

```bash
# Terminal 1: Gateway
python -m uvicorn services.gateway.app:app --reload --port 8000

# Terminal 2: Search Service
python services/search/service.py

# Terminal 3: Negotiation Service
python -m uvicorn services.negotiation.service:app --reload --port 8300

# Terminal 4: FHE Service
python -m uvicorn services.fhe.service:app --reload --port 8200

# Terminal 5: FL/Epsilon Service
python -m uvicorn services.fl.eps_ledger:app --reload --port 8400
```

## Method 3: Run Automated Tests

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_gateway.py -v
pytest tests/test_search.py -v
pytest tests/test_negotiation_enhanced.py -v
pytest tests/test_fhe.py -v
pytest tests/test_eps_ledger.py -v
pytest tests/test_soan.py -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Integration Tests

```bash
# Run integration tests (requires Docker)
make test-integration

# Test PoDP compliance
make test-podp

# Test epsilon budget constraints
make test-epsilon
```

### Performance Tests

```bash
# Run performance benchmarks
make perf-test

# Check metrics
make metrics
```

## Testing Specific Features

### 1. Self-Organizing Agent Networks (SOAN)

```bash
# Initialize network
curl -X POST http://localhost:8500/api/v1/soan/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "n_nodes": 100,
    "k_edges": 6,
    "p_rewire": 0.1
  }'

# Train GNN predictor
curl -X POST http://localhost:8500/api/v1/soan/train \
  -H "Content-Type: application/json" \
  -d '{
    "network_id": "net_123",
    "epochs": 50
  }'
```

### 2. Homomorphic Encryption (FHE)

```bash
# Note: TenSEAL may not be installed by default
# Install it first:
pip install tenseal

# Test encrypted dot product
curl -X POST http://localhost:8200/dot \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test",
    "enc_query": "base64_encrypted_data",
    "enc_vectors": ["base64_encrypted_vector"]
  }'
```

### 3. Blockchain Anchoring

```bash
# Deploy smart contract (local Anvil node)
cd services/chain
python deploy_local.py

# The contract address will be displayed
# Update infra/.env with ANCHOR_CONTRACT_ADDRESS
```

## Monitoring and Debugging

### View Logs

```bash
# All services
make logs-follow

# Specific service
make logs-gateway
make logs-search
make logs-negotiation
```

### Check Metrics

```bash
# Open Grafana
# http://localhost:3000

# View raw metrics
curl http://localhost:8000/metrics
curl http://localhost:8100/metrics
```

### Debug Issues

```bash
# Check container status
docker ps

# Inspect specific container
docker logs dalrn-gateway
docker logs dalrn-search

# Enter container shell
make shell-gateway
make shell-search
```

## Common Issues and Solutions

### Issue 1: Port Already in Use
```bash
# Error: bind: address already in use

# Solution: Stop conflicting service or change port
# Check what's using the port:
netstat -an | findstr :8000

# Change port in docker-compose.yml or .env
```

### Issue 2: Docker Not Running
```bash
# Error: Cannot connect to Docker daemon

# Solution: Start Docker Desktop
# Windows: Start Docker Desktop from Start Menu
# Linux: sudo systemctl start docker
```

### Issue 3: Python Import Errors
```bash
# Error: ModuleNotFoundError

# Solution: Install missing dependencies
pip install -r requirements.txt
pip install -r services/requirements.txt
```

### Issue 4: IPFS Connection Failed
```bash
# Error: IPFS upload failed

# Solution: Ensure IPFS is running
docker-compose -f infra/docker-compose.yml up ipfs
# Access IPFS: http://localhost:5001/webui
```

### Issue 5: TenSEAL Not Available
```bash
# Warning: TenSEAL not installed, using mock mode

# Solution (optional - FHE will work in mock mode):
pip install tenseal
# Note: Requires C++ compiler on Windows
```

## Cleanup

```bash
# Stop all services
make stop-all

# Remove containers and volumes
make clean

# Remove only volumes (keep images)
make clean-volumes

# Remove all Docker resources
docker system prune -a
```

## Quick Test Script

Create a file `test_all.sh`:

```bash
#!/bin/bash
echo "Testing DALRN Services..."

# Test Gateway
echo "1. Testing Gateway..."
curl -s http://localhost:8000/healthz | jq .

# Test Search
echo "2. Testing Search..."
curl -s http://localhost:8100/health | jq .

# Test Negotiation
echo "3. Testing Negotiation..."
curl -s http://localhost:8300/health | jq .

# Test FHE
echo "4. Testing FHE..."
curl -s http://localhost:8200/health | jq .

# Test FL
echo "5. Testing FL/Epsilon..."
curl -s http://localhost:8400/health | jq .

echo "All services tested!"
```

Run it:
```bash
chmod +x test_all.sh
./test_all.sh
```

## Next Steps

After successful testing:

1. **Explore API Documentation**: http://localhost:8000/docs
2. **View Monitoring Dashboards**: http://localhost:3000
3. **Run Performance Tests**: `make perf-test`
4. **Read Architecture Docs**: See `/plan` directory
5. **Modify and Test**: Make changes and see them live with `--reload`

## Support

If you encounter issues:
1. Check logs: `make logs`
2. Review documentation: `claude.md`
3. Check service health: `make health-check`
4. Verify Docker: `docker ps`
5. Check repository: https://github.com/PrecisionTakeAI/DALRN

---

*Testing typically takes 5-10 minutes for basic functionality, 30 minutes for full integration tests.*