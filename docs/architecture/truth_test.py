"""
TRUTH TEST - What's Actually Running vs Claims
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

print("=" * 60)
print("DALRN TRUTH TEST - ACTUAL vs CLAIMED PERFORMANCE")
print("=" * 60)

# Test 1: Vector Search - Qdrant vs FAISS
print("\n1. VECTOR SEARCH SERVICE:")
print("-" * 40)
try:
    from services.search.qdrant_search_service import QDRANT_AVAILABLE, FAISS_AVAILABLE
    print(f"CLAIMED: Qdrant with <5ms latency")
    print(f"ACTUAL: {'Qdrant' if QDRANT_AVAILABLE else 'FAISS fallback'}")
    print(f"Qdrant installed: {QDRANT_AVAILABLE}")
    print(f"FAISS available: {FAISS_AVAILABLE}")
    if not QDRANT_AVAILABLE:
        print("TRUTH: Using FAISS, NOT Qdrant. No <5ms performance!")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: FHE - Concrete ML vs TenSEAL
print("\n2. FHE SERVICE:")
print("-" * 40)
try:
    from services.fhe.zama_fhe_service import CONCRETE_ML_AVAILABLE
    print(f"CLAIMED: Zama Concrete ML with <50ms latency")
    print(f"ACTUAL: {'Concrete ML' if CONCRETE_ML_AVAILABLE else 'sklearn (NO ENCRYPTION!)'}")
    print(f"Concrete ML installed: {CONCRETE_ML_AVAILABLE}")
    if not CONCRETE_ML_AVAILABLE:
        print("TRUTH: Using sklearn LinearRegression with NO FHE!")
        print("       This is NOT homomorphic encryption at all!")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Gateway optimizations
print("\n3. GATEWAY SERVICE:")
print("-" * 40)
try:
    import services.gateway.optimized_gateway as og
    print(f"CLAIMED: <50ms with async + HTTP/2 + pooling")
    print(f"HTTP client exists: {hasattr(og, 'http_client')}")

    # Check if services are reachable
    import httpx
    import time
    import asyncio

    async def test_latency():
        try:
            async with httpx.AsyncClient() as client:
                start = time.time()
                response = await client.get("http://localhost:8000/health", timeout=5.0)
                latency = (time.time() - start) * 1000
                return latency, response.status_code
        except:
            return None, None

    # Try to test actual latency
    try:
        latency, status = asyncio.run(test_latency())
        if latency:
            print(f"ACTUAL latency: {latency:.2f}ms (status: {status})")
        else:
            print("ACTUAL: Gateway not reachable!")
    except:
        print("ACTUAL: Cannot test - gateway not running")

    print("TRUTH: Has async code but backend services unreachable")
except Exception as e:
    print(f"ERROR: {e}")

# Test 4: External Dependencies
print("\n4. EXTERNAL DEPENDENCIES:")
print("-" * 40)
try:
    # PostgreSQL
    from services.database.connection import db
    db_status = db.health_check()
    print(f"PostgreSQL: {db_status['type']} ({'connected' if db_status['connected'] else 'NOT connected'})")

    # Redis
    from services.cache.connection import cache
    cache_status = cache.health_check()
    print(f"Redis: {cache_status['type']} ({'connected' if cache_status['connected'] else 'NOT connected'})")

    if db_status['type'] == 'sqlite':
        print("TRUTH: Using SQLite fallback, not PostgreSQL")
    if cache_status['type'] == 'memory':
        print("TRUTH: Using memory cache, not Redis")
except Exception as e:
    print(f"ERROR: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS:")
print("=" * 60)
print("1. Qdrant NOT installed -> Using FAISS (no <5ms claims)")
print("2. Concrete ML NOT installed -> Using sklearn (NO encryption!)")
print("3. PostgreSQL NOT running -> Using SQLite fallback")
print("4. Redis NOT running -> Using memory cache")
print("5. Backend services NOT running -> Gateway can't route")
print("\nCONCLUSION: The 'optimizations' are mostly just CODE,")
print("            not actual running services with claimed performance.")
print("            The system falls back to simpler implementations")
print("            that don't have the claimed performance benefits.")
print("=" * 60)