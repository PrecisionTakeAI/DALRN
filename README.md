# DALRN - Decentralized Autonomous Learning and Resource Negotiation

## 🚀 Privacy-Preserving Distributed Computing Platform

![Status](https://img.shields.io/badge/Status-Dual%20Implementation-yellow)
![Original](https://img.shields.io/badge/Original-Working-green)
![Optimized](https://img.shields.io/badge/Optimized-Needs%20Infrastructure-orange)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ⚠️ Important: System Status

**DALRN has TWO parallel implementations:**
1. **Original Services** (✅ Working) - TenSEAL, FAISS, SQLite/memory fallbacks
2. **"Optimized" Services** (🔧 Code exists, infrastructure needed) - Requires Qdrant, Concrete ML, PostgreSQL, Redis

**Performance claims (38x improvement) are UNVERIFIED and require full infrastructure setup.**

## Overview

DALRN is a microservices platform that enables organizations to collaborate on machine learning and computation tasks while preserving data privacy. The system has working implementations with automatic fallbacks, plus aspirational optimization code awaiting proper infrastructure.

### 📊 Actual Status (Verified 2025-09-23)

```bash
# Run this to see what ACTUALLY works:
python comprehensive_truth_check.py
python truth_test.py
```

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - ⚠️ **READ THIS FIRST** - Contains actual system truth
- **[Platform Overview](docs/architecture/PLATFORM_OVERVIEW.md)** - What DALRN aims to do
- **[Technical Architecture](docs/architecture/TECHNICAL_ARCHITECTURE.md)** - System design
- **[API Reference](docs/api/API_REFERENCE.md)** - Endpoint documentation

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/PrecisionTakeAI/DALRN
cd DALRN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the ORIGINAL gateway (working)
python -m services.gateway.app

# 4. Verify installation
curl http://localhost:8000/health

# 5. Check what actually works
python comprehensive_truth_check.py
```

## 🏗️ Architecture - Two Parallel Codebases

### Original Services (WORKING)
| Service | Port | Technology | Status |
|---------|------|------------|--------|
| **Gateway** | 8000 | FastAPI + JWT | ✅ Working |
| **Search** | 8100 | FAISS HNSW | ✅ Working |
| **FHE** | 8200 | TenSEAL CKKS | ✅ Working |
| **Negotiation** | 8300 | nashpy | ✅ Working |
| **FL** | 8400 | Flower + Opacus | ✅ Working |
| **Agents** | 8500 | PyTorch GNN | ❌ Import errors |

### "Optimized" Services (CODE EXISTS, INFRASTRUCTURE NEEDED)
| Service | File | Claimed Tech | Reality |
|---------|------|--------------|---------|
| **Gateway** | optimized_gateway.py | Async/HTTP2 | Untested, needs backends |
| **Search** | qdrant_search_service.py | Qdrant <5ms | No server, falls to FAISS |
| **FHE** | zama_fhe_service.py | Concrete ML <50ms | Python 3.13 incompatible |

## ✨ Key Features

### 🔐 What Actually Works
- **Federated Learning:** Flower framework with real implementation
- **Homomorphic Encryption:** TenSEAL with verified CKKS scheme
- **Vector Search:** FAISS with HNSW index
- **Game Theory:** Nash equilibrium computation
- **JWT Authentication:** Working with BCrypt
- **Automatic Fallbacks:** SQLite when no PostgreSQL, memory when no Redis

### 🚧 What Needs Infrastructure
- **Qdrant Vector Database:** Client installed, server not running
- **Zama Concrete ML:** Requires Python 3.8-3.10 (incompatible with 3.13)
- **PostgreSQL:** Not installed, using SQLite
- **Redis:** Not installed, using memory cache
- **Performance Claims:** All unverified without infrastructure

## 🔬 Reality Check

Run these commands to see the truth:

```bash
# See what's actually working
python comprehensive_truth_check.py

# Check "optimization" reality
python truth_test.py

# Try to install optimizations (will partially fail)
python install_optimizations.py
```

Expected output:
```
[OK] Gateway (Original): Original gateway exists and imports
[OK] Search (Original): FAISS-based search working
[OK] FHE (Original): TenSEAL-based FHE working
[OK] Search (Qdrant): Qdrant client=True, FAISS=True
[OK] FHE (Zama): Concrete ML=False (needs Python 3.8-3.10)
```

## 📊 Performance Claims vs Reality

| Metric | Claimed | Reality | Status |
|--------|---------|---------|--------|
| Gateway Latency | <50ms | Untested | ❓ Needs all services |
| Search (Qdrant) | <5ms | Using FAISS | ❌ No Qdrant server |
| FHE (Concrete ML) | <50ms | Using sklearn | ❌ No encryption! |
| Total Improvement | 38x | Unverified | ❌ Infrastructure missing |

## 🛠️ Development

### Prerequisites for Original (Working)
- Python 3.11+
- 8GB RAM
- No external services needed (auto-fallback)

### Prerequisites for "Optimized" (Not Working)
- Python 3.8-3.10 (for Concrete ML)
- Docker Desktop (for Qdrant)
- PostgreSQL server
- Redis server
- All services running simultaneously

### Installation

```bash
# Install base dependencies (works)
pip install -r requirements.txt

# Try to install optimizations (will partially fail on Python 3.13)
python install_optimizations.py

# What you'll see:
# ✅ Qdrant client (installed)
# ❌ Concrete ML (needs Python 3.8-3.10)
# ❌ Qdrant server (needs Docker)
# ❌ PostgreSQL (needs separate install)
# ❌ Redis (needs separate install)
```

### Running Services

```bash
# ORIGINAL services (working)
python -m services.gateway.app          # Port 8000
python -m services.search.service       # Port 8100
python -m services.fhe.service          # Port 8200
python -m services.negotiation.service  # Port 8300
python -m services.fl.service           # Port 8400

# "OPTIMIZED" services (need infrastructure)
python -m services.gateway.optimized_gateway    # Needs backends
python -m services.search.qdrant_search_service # Needs Qdrant server
python -m services.fhe.zama_fhe_service        # Needs Python 3.8-3.10
```

## 🔧 To Actually Get "Optimizations" Working

1. **Install Python 3.8-3.10** (Concrete ML requirement)
2. **Start Docker Desktop** and run:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
3. **Install PostgreSQL** from postgresql.org
4. **Install Redis** from GitHub releases
5. **Start all services simultaneously**
6. **Then maybe** the performance claims might be true

## 📈 Use Cases

The system can handle these use cases with the ORIGINAL implementation:
- Privacy-preserving analytics (TenSEAL FHE)
- Federated learning (Flower framework)
- Vector similarity search (FAISS)
- Game-theoretic optimization (nashpy)

## 🤝 Contributing

Before contributing:
1. Run `python comprehensive_truth_check.py` to understand current state
2. Don't trust documentation - verify with code
3. Test with actual running services
4. Update CLAUDE.md if you change implementation status

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

### Working Technologies
- **TenSEAL** - Actually working FHE
- **FAISS** - Actually working vector search
- **Flower** - Actually working federated learning
- **nashpy** - Actually working game theory

### Aspirational Technologies
- **Qdrant** - Would be great if server was running
- **Zama Concrete ML** - Would be great on Python 3.8-3.10
- **PostgreSQL/Redis** - Would help if installed

## 📞 Support

- **Check Reality First:** Run `python truth_test.py`
- **Installation Help:** See `INSTALLATION_STEPS.md`
- **Issues:** [GitHub Issues](https://github.com/PrecisionTakeAI/DALRN/issues)

---

**DALRN** - *Working implementations with aspirational optimizations*

*Last Updated: 2025-09-23 | Status: Dual Implementation (Original Working, Optimized Needs Infrastructure)*

**⚠️ IMPORTANT: Don't trust performance claims without running `truth_test.py` first!**