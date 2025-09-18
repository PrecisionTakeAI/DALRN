"""
Performance Profiling for DALRN Gateway
Identify exact bottlenecks causing 270ms response time
"""
import cProfile
import pstats
import time
import requests
import sys
import io
from typing import Dict, List

def profile_endpoint_performance():
    """Profile the gateway endpoints to find bottlenecks"""

    print("DALRN PERFORMANCE PROFILING")
    print("=" * 50)

    # Test endpoints
    endpoints = [
        "http://localhost:8002/health",
        "http://localhost:8002/agents-fast",
        "http://localhost:8002/metrics-fast"
    ]

    results = {}

    for endpoint in endpoints:
        endpoint_name = endpoint.split('/')[-1]
        print(f"\nProfiling {endpoint_name}...")

        # Measure response times
        times = []
        for i in range(10):
            try:
                start = time.perf_counter()
                response = requests.get(endpoint, timeout=5)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

                if i == 0:  # First request
                    print(f"  Status: {response.status_code}")
                    if response.status_code == 200:
                        print(f"  Response size: {len(response.content)} bytes")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            results[endpoint_name] = {
                "avg_ms": round(avg_time, 2),
                "min_ms": round(min_time, 2),
                "max_ms": round(max_time, 2),
                "samples": len(times)
            }

            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Range: {min_time:.2f}ms - {max_time:.2f}ms")

            # Analyze performance level
            if avg_time < 50:
                print(f"  Performance: EXCELLENT")
            elif avg_time < 200:
                print(f"  Performance: GOOD")
            else:
                print(f"  Performance: NEEDS OPTIMIZATION")
        else:
            results[endpoint_name] = {"error": "No successful requests"}

    return results

def analyze_system_bottlenecks():
    """Analyze potential system-level bottlenecks"""

    print(f"\nSYSTEM BOTTLENECK ANALYSIS")
    print("-" * 30)

    bottlenecks = []

    # Check if fast gateway is running
    try:
        response = requests.get("http://localhost:8002/health", timeout=1)
        if response.status_code == 200:
            print("[PASS] Fast gateway responding")
        else:
            print("[FAIL] Fast gateway not responding properly")
            bottlenecks.append("Gateway not responding")
    except:
        print("[FAIL] Fast gateway not accessible")
        bottlenecks.append("Gateway not running")

    # Check database performance
    try:
        start = time.perf_counter()
        from services.database.production_config import get_database_service
        db = get_database_service()
        health = db.health_check()
        db_time = (time.perf_counter() - start) * 1000

        print(f"[PASS] Database health check: {db_time:.2f}ms")
        if db_time > 100:
            bottlenecks.append(f"Slow database ({db_time:.2f}ms)")

    except Exception as e:
        print(f"[FAIL] Database check failed: {e}")
        bottlenecks.append("Database issues")

    # Check import times
    print(f"\nModule import analysis:")
    modules_to_test = [
        "services.gateway.fast_app",
        "services.database.production_config",
        "services.blockchain.real_client"
    ]

    for module in modules_to_test:
        try:
            start = time.perf_counter()
            __import__(module)
            import_time = (time.perf_counter() - start) * 1000
            print(f"  {module}: {import_time:.2f}ms")

            if import_time > 500:
                bottlenecks.append(f"Slow import: {module}")

        except Exception as e:
            print(f"  {module}: FAILED - {e}")
            bottlenecks.append(f"Import error: {module}")

    return bottlenecks

def recommend_optimizations(results: Dict, bottlenecks: List[str]):
    """Recommend specific optimizations based on analysis"""

    print(f"\nPERFORMANCE OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)

    # Check if we're meeting the requirement
    avg_times = [r.get("avg_ms", 1000) for r in results.values() if isinstance(r, dict) and "avg_ms" in r]

    if avg_times:
        overall_avg = sum(avg_times) / len(avg_times)
        print(f"Overall average response time: {overall_avg:.2f}ms")

        if overall_avg < 200:
            print("[PASS] PERFORMANCE REQUIREMENT MET (<200ms)")
            return True
        else:
            print("[FAIL] PERFORMANCE REQUIREMENT NOT MET (>200ms)")

    print(f"\nIdentified issues:")
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"  {i}. {bottleneck}")

    print(f"\nRecommended fixes:")

    # Specific recommendations based on findings
    recommendations = []

    if any("Gateway not" in b for b in bottlenecks):
        recommendations.append("1. Ensure fast gateway is running on port 8002")
        recommendations.append("2. Check for port conflicts")

    if any("database" in b.lower() for b in bottlenecks):
        recommendations.append("3. Optimize database queries")
        recommendations.append("4. Add connection pooling")
        recommendations.append("5. Implement query caching")

    if any("import" in b.lower() for b in bottlenecks):
        recommendations.append("6. Lazy load heavy modules")
        recommendations.append("7. Use import caching")

    # General performance recommendations
    if overall_avg > 200:
        recommendations.extend([
            "8. Implement Redis response caching",
            "9. Use async/await throughout",
            "10. Minimize database calls per request",
            "11. Pre-load data at startup",
            "12. Use connection pooling",
            "13. Optimize JSON serialization (use orjson)",
            "14. Remove unnecessary middleware"
        ])

    for rec in recommendations:
        print(f"  {rec}")

    return False

def main():
    """Run complete performance analysis"""

    # Profile endpoint performance
    results = profile_endpoint_performance()

    # Analyze system bottlenecks
    bottlenecks = analyze_system_bottlenecks()

    # Generate recommendations
    meets_requirement = recommend_optimizations(results, bottlenecks)

    print(f"\nSUMMARY")
    print("-" * 20)
    if meets_requirement:
        print("[PASS] Performance targets achieved")
        print("[PASS] Ready for production")
    else:
        print("[FAIL] Performance optimization needed")
        print("[FAIL] Implement recommended fixes")

    return meets_requirement

if __name__ == "__main__":
    success = main()
    print(f"\nPerformance analysis complete. Target met: {success}")