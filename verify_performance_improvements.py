"""
DALRN Performance Improvement Verification
Compare original vs optimized performance
"""
import time
import requests
import statistics

def test_endpoint_performance(name, url, num_tests=10):
    """Test endpoint performance and return statistics"""
    print(f"\nTesting {name}...")

    times = []
    success_count = 0

    for i in range(num_tests):
        try:
            start_time = time.perf_counter()
            response = requests.get(url, timeout=10)
            elapsed = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                times.append(elapsed)
                success_count += 1

            if i == 0:
                print(f"  Status: {response.status_code}")
                print(f"  Size: {len(response.content)} bytes")

        except Exception as e:
            print(f"  Request {i+1} failed: {e}")

    if times:
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        p95_time = sorted(times)[int(0.95 * len(times))]
        min_time = min(times)
        max_time = max(times)

        print(f"  Success rate: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Median: {median_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")
        print(f"  Range: {min_time:.2f}ms - {max_time:.2f}ms")

        return {
            "avg": avg_time,
            "median": median_time,
            "p95": p95_time,
            "min": min_time,
            "max": max_time,
            "success_rate": success_count/num_tests*100,
            "raw_times": times
        }

    return None

def main():
    """Compare original vs optimized gateway performance"""
    print("DALRN PERFORMANCE IMPROVEMENT VERIFICATION")
    print("=" * 60)

    # Test configurations
    test_configs = [
        {
            "name": "Original Gateway (Port 8000)",
            "base_url": "http://localhost:8000",
            "endpoints": ["/health"]
        },
        {
            "name": "Fast Gateway (Port 8002)",
            "base_url": "http://localhost:8002",
            "endpoints": ["/health", "/agents-fast", "/metrics-fast"]
        },
        {
            "name": "Minimal Gateway (Port 8003)",
            "base_url": "http://localhost:8003",
            "endpoints": ["/health", "/agents-fast", "/metrics-fast", "/perf-test"]
        }
    ]

    results = {}

    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"TESTING: {config['name']}")
        print(f"{'='*60}")

        gateway_results = {}

        for endpoint in config["endpoints"]:
            url = f"{config['base_url']}{endpoint}"
            result = test_endpoint_performance(f"{endpoint}", url)

            if result:
                gateway_results[endpoint] = result

        if gateway_results:
            # Calculate overall metrics
            all_times = []
            for endpoint_result in gateway_results.values():
                all_times.extend(endpoint_result["raw_times"])

            if all_times:
                overall_avg = statistics.mean(all_times)
                overall_p95 = sorted(all_times)[int(0.95 * len(all_times))]

                print(f"\n  OVERALL PERFORMANCE:")
                print(f"    Average: {overall_avg:.2f}ms")
                print(f"    P95: {overall_p95:.2f}ms")
                print(f"    Total requests: {len(all_times)}")

                # Performance assessment
                if overall_p95 < 200:
                    print(f"    Assessment: ✓ TARGET MET (P95 < 200ms)")
                elif overall_avg < 200:
                    print(f"    Assessment: • PARTIAL (avg < 200ms)")
                else:
                    print(f"    Assessment: ✗ NEEDS IMPROVEMENT (> 200ms)")

                results[config["name"]] = {
                    "overall_avg": overall_avg,
                    "overall_p95": overall_p95,
                    "endpoints": gateway_results
                }

    # Summary comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Average: {result['overall_avg']:.2f}ms")
        print(f"  P95: {result['overall_p95']:.2f}ms")

        if result['overall_p95'] < 200:
            status = "✓ MEETS TARGET"
        elif result['overall_avg'] < 200:
            status = "• PARTIALLY MEETS TARGET"
        else:
            status = "✗ DOES NOT MEET TARGET"

        print(f"  Status: {status}")

    # Calculate improvements
    if len(results) >= 2:
        baseline_name = list(results.keys())[0]
        baseline = results[baseline_name]

        print(f"\nIMPROVEMENTS vs {baseline_name}:")

        for name, result in results.items():
            if name != baseline_name:
                avg_improvement = ((baseline['overall_avg'] - result['overall_avg']) / baseline['overall_avg']) * 100
                p95_improvement = ((baseline['overall_p95'] - result['overall_p95']) / baseline['overall_p95']) * 100

                print(f"\n{name}:")
                print(f"  Average improvement: {avg_improvement:.1f}%")
                print(f"  P95 improvement: {p95_improvement:.1f}%")

    # Final assessment
    print(f"\n{'='*60}")
    print("FINAL ASSESSMENT")
    print(f"{'='*60}")

    targets_met = sum(1 for result in results.values() if result['overall_p95'] < 200)
    total_gateways = len(results)

    if targets_met == total_gateways:
        print("🎉 ALL GATEWAYS MEET P95 < 200ms TARGET!")
    elif targets_met > 0:
        print(f"✓ {targets_met}/{total_gateways} gateways meet the target")
    else:
        print("⚠ No gateways meet the P95 < 200ms target")

    # Check if we have any gateway under 200ms average
    fast_gateways = sum(1 for result in results.values() if result['overall_avg'] < 200)

    if fast_gateways > 0:
        print(f"✓ {fast_gateways} gateway(s) have average response < 200ms")

    print(f"\nRecommendation:")
    if targets_met > 0:
        best_gateway = min(results.items(), key=lambda x: x[1]['overall_p95'])
        print(f"Use {best_gateway[0]} for production")
        print(f"P95: {best_gateway[1]['overall_p95']:.2f}ms")
    else:
        print("Continue optimization efforts")

    return targets_met > 0

if __name__ == "__main__":
    success = main()
    print(f"\nVerification complete. Performance targets met: {success}")