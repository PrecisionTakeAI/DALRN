"""
PHASE 2: PERFORMANCE CLAIM VERIFICATION
Test ACTUAL performance with precise measurements
"""
import time
import statistics
import requests
import subprocess
import sys
import os

print("="*60)
print("PERFORMANCE VERIFICATION - TESTING SUB-MILLISECOND CLAIM")
print("="*60)

# Check if fast_app is running
try:
    response = requests.get("http://localhost:8002/health", timeout=1)
    print("Fast gateway already running")
except:
    print("Starting fast gateway...")
    os.chdir("services/gateway")
    gateway_process = subprocess.Popen([sys.executable, "fast_app.py"])
    os.chdir("../..")
    time.sleep(3)

try:
    # Warm-up requests
    print("Warming up...")
    for _ in range(5):
        try:
            requests.get("http://localhost:8002/health", timeout=1)
        except:
            pass

    # Precision timing test
    response_times = []

    print("\nMeasuring response times with high precision...")

    # Test different endpoints
    endpoints = [
        "http://localhost:8002/health",
        "http://localhost:8002/agents-fast",
        "http://localhost:8002/metrics-fast"
    ]

    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")
        endpoint_times = []

        for i in range(20):
            try:
                start = time.perf_counter()
                response = requests.get(endpoint, timeout=2)
                elapsed = (time.perf_counter() - start) * 1000  # Convert to milliseconds

                if response.status_code == 200:
                    endpoint_times.append(elapsed)
                    response_times.append(elapsed)

                    if i % 5 == 0:
                        print(f"  Request {i}: {elapsed:.3f}ms (status: {response.status_code})")
                else:
                    print(f"  Request {i}: FAILED (status: {response.status_code})")

            except Exception as e:
                print(f"  Request {i}: ERROR - {e}")

        if endpoint_times:
            avg = statistics.mean(endpoint_times)
            print(f"  Endpoint average: {avg:.3f}ms")

    if response_times:
        # Calculate detailed statistics
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)

        print("\n" + "-"*40)
        print("ACTUAL PERFORMANCE METRICS:")
        print(f"  Minimum: {min_time:.3f}ms")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  Median: {median_time:.3f}ms")
        print(f"  Maximum: {max_time:.3f}ms")
        print(f"  Successful requests: {len(response_times)}")
        print("-"*40)

        # Verify claim of "0.02-0.05ms average"
        print("\nCLAIM VERIFICATION:")
        if avg_time < 1:  # Less than 1ms
            print(f"✅ Sub-millisecond performance VERIFIED ({avg_time:.3f}ms)")
            performance_passed = True
        elif avg_time < 200:
            print(f"⚠️  Performance acceptable but NOT sub-millisecond as claimed")
            print(f"   Claimed: 0.02-0.05ms, Actual: {avg_time:.3f}ms")
            performance_passed = False
        else:
            print(f"❌ Performance claim FALSE")
            print(f"   Claimed: sub-millisecond, Actual: {avg_time:.3f}ms")
            performance_passed = False
    else:
        print("❌ No successful requests - performance test FAILED")
        performance_passed = False

except Exception as e:
    print(f"❌ Performance test failed: {e}")
    performance_passed = False

print(f"\nPERFORMANCE TEST RESULT: {'PASS' if performance_passed else 'FAIL'}")