# DALRN MVP Implementation Notes

## System Overview
DALRN (Distributed Adaptive Learning & Resolution Network) is a verifiable multi-party dispute resolution system that combines encrypted search, game-theoretic negotiation, and federated learning with cryptographic receipts (PoDP) and privacy budget tracking (ε-ledger).

## Core Components

### 1. Gateway Service (FastAPI)
- **Endpoints**: `/submit-dispute`, `/status/{id}`, `/evidence`
- **PoDP Middleware**: Generates receipts for every step, builds Merkle trees, uploads to IPFS
- **Anchoring**: Calls smart contract to anchor Merkle roots on-chain

### 2. Search Service (FAISS)
- **HNSW Index**: d=768 vectors, M=32, efSearch=128
- **Baseline Metrics**: Recall@10, P95 latency < 600ms on 10k docs
- **Reweighting**: Feature flag (OFF by default) for quantum-inspired reranking

### 3. FHE Service (TenSEAL)
- **CKKS Encryption**: 8192 poly degree, [60,40,40,60] moduli, scale 2^40
- **Operations**: Encrypted dot-product on unit-norm vectors
- **Client-side**: Decryption happens only on client side

### 4. Negotiation Service (nashpy)
- **Nash Equilibrium**: Lemke-Howson enumeration
- **Selection Rules**: Nash Social Welfare (default) or egalitarian
- **Output**: Deterministic equilibrium selection + explanation memo

### 5. FL/Privacy Service
- **ε-Ledger**: RDP accounting with budget precheck/commit
- **FL Framework**: Flower/NV FLARE with Opacus for DP
- **Aggregation**: Robust aggregation (median/trimmed mean)

### 6. Chain Service (Solidity)
- **AnchorReceipts Contract**: Stores Merkle roots, emits events
- **Client**: Python web3.py wrapper for anchor operations

## Architecture Principles
1. **Verifiability First**: Every operation emits PoDP receipts
2. **Privacy by Design**: No plaintext evidence in logs, no PII on-chain
3. **Deterministic Flows**: Canonical JSON, sorted keys, reproducible Merkle trees
4. **Modular Services**: Clean gRPC/HTTP interfaces between components

## Implementation Phases

### Phase 0 (Setup) ✓
- Monorepo structure created
- Service skeletons in place
- CI/CD workflows configured

### Phase 1 (Core Pipeline) - CURRENT
- PR-1: Chain contract + Foundry tests
- PR-2: Gateway with PoDP middleware
- PR-3: Search service with baseline
- PR-4: Negotiation engine
- PR-5: FHE dot-product (optional if env ready)
- PR-6: ε-ledger integration

### Phase 2 (Hardening)
- Auth/rate limiting
- Observability (Prometheus/Grafana)
- A/B testing harness

## TODO Items

### Missing Infrastructure
- [ ] Reports directory for baseline metrics
- [ ] IPFS node configuration in docker-compose
- [ ] Postgres setup for metadata storage
- [ ] Prometheus/Grafana configs

### Missing Tests
- [ ] Integration tests for end-to-end flow
- [ ] FHE parity tests vs plaintext
- [ ] Merkle tree determinism tests
- [ ] Budget exhaustion tests for ε-ledger

### Environment Setup
- [ ] TenSEAL environment verification (may need special builds)
- [ ] GPU setup for FAISS acceleration
- [ ] Foundry installation for Solidity tests

### Documentation
- [ ] API documentation generation
- [ ] Deployment runbook
- [ ] PoDP flow diagrams
- [ ] Architecture decision records (ADRs)

## Key Design Decisions

1. **Merkle Tree Algorithm**: Simple pairwise hashing with keccak256, duplicate last leaf if odd
2. **Receipt Canonicalization**: JSON with sorted keys, no whitespace, UTF-8 encoding
3. **Reweighting Default**: OFF - must be explicitly enabled via feature flag
4. **Privacy Budget**: 4.0 ε default per tenant/model pair
5. **FHE Context**: Single context per tenant, persisted between operations

## Critical Constraints
- No plaintext evidence in any logs
- All pushes must have tests
- PoDP receipts mandatory for new features
- Anchor client may stub until contract deployed
- FHE limited to dot-product only (no complex operations)

## Next Steps
1. Create first feature branch for chain contract
2. Implement AnchorReceipts.sol with tests
3. Set up local IPFS and chain infrastructure
4. Begin gateway PoDP middleware implementation