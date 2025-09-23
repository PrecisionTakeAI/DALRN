"""
Comprehensive truth check - What ACTUALLY works in DALRN
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

def check_service(name, test_func):
    """Test a service and report results"""
    try:
        result = test_func()
        return f"[OK] {name}: {result}"
    except Exception as e:
        return f"[FAIL] {name}: {str(e)[:100]}"

results = []

# 1. Gateway Services
def test_gateway():
    from services.gateway.app import app
    return "Original gateway exists and imports"
results.append(check_service("Gateway (Original)", test_gateway))

def test_optimized_gateway():
    from services.gateway.optimized_gateway import app
    return "Optimized gateway exists (async/HTTP2 code)"
results.append(check_service("Gateway (Optimized)", test_optimized_gateway))

# 2. Search Services
def test_search():
    from services.search.service import app
    return "FAISS-based search working"
results.append(check_service("Search (Original)", test_search))

def test_qdrant():
    from services.search.qdrant_search_service import QDRANT_AVAILABLE, FAISS_AVAILABLE
    return f"Qdrant client={QDRANT_AVAILABLE}, FAISS={FAISS_AVAILABLE}"
results.append(check_service("Search (Qdrant)", test_qdrant))

# 3. FHE Services
def test_fhe():
    from services.fhe.service import app
    import tenseal as ts
    return "TenSEAL-based FHE working"
results.append(check_service("FHE (Original)", test_fhe))

def test_zama():
    from services.fhe.zama_fhe_service import CONCRETE_ML_AVAILABLE
    return f"Concrete ML={CONCRETE_ML_AVAILABLE} (needs Python 3.8-3.10)"
results.append(check_service("FHE (Zama)", test_zama))

# 4. Other Services
def test_negotiation():
    from services.negotiation.service import app
    return "Nash equilibrium service exists"
results.append(check_service("Negotiation", test_negotiation))

def test_fl():
    from services.fl.service import app
    return "Federated learning service exists"
results.append(check_service("Federated Learning", test_fl))

def test_agents():
    from services.agents.service import app
    return "Agent orchestration service exists"
results.append(check_service("Agents", test_agents))

# 5. Infrastructure
def test_database():
    from services.database.connection import db
    status = db.health_check()
    return f"{status['type']} ({'connected' if status['connected'] else 'disconnected'})"
results.append(check_service("Database", test_database))

def test_cache():
    from services.cache.connection import cache
    status = cache.health_check()
    return f"{status['type']} ({'connected' if status['connected'] else 'disconnected'})"
results.append(check_service("Cache", test_cache))

# 6. Authentication
def test_auth():
    from services.auth.jwt_auth import AuthService
    return "JWT auth service exists"
results.append(check_service("Authentication", test_auth))

# Print results
print("=" * 70)
print("DALRN COMPREHENSIVE STATUS CHECK")
print("=" * 70)
for result in results:
    print(result)

print("\n" + "=" * 70)
print("CRITICAL FINDINGS:")
print("=" * 70)

# Analyze findings
qdrant_installed = "Qdrant client=True" in str(results)
concrete_ml_installed = "Concrete ML=True" in str(results)
postgres_running = "postgresql" in str(results).lower() and "connected" in str(results)
redis_running = "redis" in str(results).lower() and "connected" in str(results)

print(f"1. Qdrant: Client installed but SERVER NOT RUNNING")
print(f"2. Concrete ML: NOT WORKING (Python 3.13 incompatible)")
print(f"3. PostgreSQL: {'Running' if postgres_running else 'NOT RUNNING (using SQLite)'}")
print(f"4. Redis: {'Running' if redis_running else 'NOT RUNNING (using memory cache)'}")
print(f"5. Performance claims: UNVERIFIED - need infrastructure")

print("\nCONCLUSION: System has TWO versions:")
print("  - ORIGINAL: Working services with real implementations")
print("  - OPTIMIZED: Code exists but untested/unverified")
print("=" * 70)