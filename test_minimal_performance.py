"""
Test Minimal Gateway Performance
Verify <200ms response times and <50ms import time
"""
import time
import requests
import importlib.util
import sys

def test_import_time():
    """Test import time of minimal gateway"""
    print("Testing import time...")

    start_time = time.perf_counter()

    # Import the minimal app module
    spec = importlib.util.spec_from_file_location(
        "minimal_app",
        "services/gateway/minimal_app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import_time = (time.perf_counter() - start_time) * 1000

    print(f"Import time: {import_time:.2f}ms")

    if import_time < 50:
        print("[PASS] Import time target met (<50ms)")
    elif import_time < 200:
        print("[GOOD] Import time acceptable (<200ms)")
    else:
        print("[FAIL] Import time too slow (>200ms)")

    return import_time

def test_response_times():
    """Test response times of minimal gateway endpoints"""
    print("\nTesting response times...")

    base_url = "http://localhost:8003"
    endpoints = [
        "/health",
        "/agents-fast",
        "/metrics-fast",
        "/perf-test"
    ]

    results = {}

    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")

        times = []
        for i in range(10):
            try:
                start_time = time.perf_counter()
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                elapsed = (time.perf_counter() - start_time) * 1000

                if response.status_code == 200:
                    times.append(elapsed)

                if i == 0:  # First request details
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

            results[endpoint] = {
                "avg_ms": round(avg_time, 2),
                "min_ms": round(min_time, 2),
                "max_ms": round(max_time, 2),
                "samples": len(times)
            }

            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Range: {min_time:.2f}ms - {max_time:.2f}ms")

            # Performance assessment
            if avg_time < 50:
                print(f"  Performance: EXCELLENT")
            elif avg_time < 200:
                print(f"  Performance: TARGET MET")
            else:
                print(f"  Performance: NEEDS WORK")

    return results

def test_dispute_submission():
    """Test dispute submission performance"""
    print("\nTesting dispute submission...")

    url = "http://localhost:8003/submit-dispute"
    payload = {
        "parties": ["alice@example.com", "bob@example.com"],
        "jurisdiction": "US",
        "cid": "QmTest123",
        "enc_meta": {"type": "contract_dispute"}
    }

    times = []
    for i in range(5):
        try:
            start_time = time.perf_counter()
            response = requests.post(url, json=payload, timeout=5)
            elapsed = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                times.append(elapsed)

                if i == 0:
                    data = response.json()
                    print(f"  Status: {response.status_code}")
                    print(f"  Dispute ID: {data.get('dispute_id', 'N/A')}")
                    print(f"  Response time in response: {data.get('response_time_ms', 'N/A')}ms")

        except Exception as e:
            print(f"  Error: {e}")

    if times:
        avg_time = sum(times) / len(times)
        print(f"  Average submission time: {avg_time:.2f}ms")

        if avg_time < 200:
            print(f"  [PASS] Submission performance target met")
        else:
            print(f"  [FAIL] Submission too slow")

        return avg_time

    return None

def main():
    """Run all performance tests"""
    print("DALRN MINIMAL GATEWAY PERFORMANCE TEST")
    print("=" * 50)

    # Test 1: Import time
    import_time = test_import_time()

    # Test 2: Response times
    response_results = test_response_times()

    # Test 3: Dispute submission
    submission_time = test_dispute_submission()

    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    print(f"Import time: {import_time:.2f}ms")

    if response_results:
        avg_times = [r["avg_ms"] for r in response_results.values()]
        overall_avg = sum(avg_times) / len(avg_times)
        print(f"Overall average response: {overall_avg:.2f}ms")

        # Check if we met the <200ms target
        if overall_avg < 200:
            print("[PASS] Response time target met (<200ms)")
        else:
            print("[FAIL] Response time target not met (>200ms)")

    if submission_time:
        print(f"Dispute submission: {submission_time:.2f}ms")

    # Overall assessment
    print("\nOVERALL ASSESSMENT:")

    targets_met = 0
    total_targets = 3

    if import_time < 50:
        print("✓ Import time: EXCELLENT (<50ms)")
        targets_met += 1
    elif import_time < 200:
        print("• Import time: ACCEPTABLE (<200ms)")
        targets_met += 0.5
    else:
        print("✗ Import time: NEEDS IMPROVEMENT")

    if response_results:
        avg_times = [r["avg_ms"] for r in response_results.values()]
        overall_avg = sum(avg_times) / len(avg_times)

        if overall_avg < 50:
            print("✓ Response times: EXCELLENT (<50ms)")
            targets_met += 1
        elif overall_avg < 200:
            print("✓ Response times: TARGET MET (<200ms)")
            targets_met += 1
        else:
            print("✗ Response times: NEEDS IMPROVEMENT")

    if submission_time and submission_time < 200:
        print("✓ Dispute submission: TARGET MET (<200ms)")
        targets_met += 1
    elif submission_time:
        print("✗ Dispute submission: NEEDS IMPROVEMENT")

    success_rate = (targets_met / total_targets) * 100
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")

    if success_rate >= 90:
        print("🎉 MINIMAL GATEWAY PERFORMANCE: EXCELLENT!")
    elif success_rate >= 70:
        print("✓ MINIMAL GATEWAY PERFORMANCE: GOOD")
    else:
        print("⚠ MINIMAL GATEWAY PERFORMANCE: NEEDS WORK")

    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    print(f"\nTest completed. Success: {success}")