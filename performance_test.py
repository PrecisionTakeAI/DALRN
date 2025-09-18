#!/usr/bin/env python
"""Performance test for DALRN gateway - 100 requests to /health endpoint"""

import requests
import time
import statistics

def test_performance(url, num_requests=100):
    """Test response time for multiple requests"""
    response_times = []
    errors = 0

    print(f"Testing {num_requests} requests to {url}")
    print("-" * 60)

    for i in range(num_requests):
        try:
            start = time.perf_counter()
            response = requests.get(url)
            end = time.perf_counter()

            if response.status_code == 200:
                response_time_ms = (end - start) * 1000
                response_times.append(response_time_ms)

                if (i + 1) % 10 == 0:
                    print(f"Completed {i+1}/{num_requests} requests...")
            else:
                errors += 1
        except Exception as e:
            errors += 1
            print(f"Error on request {i+1}: {e}")

    return response_times, errors

def main():
    print("=" * 60)
    print("DALRN Gateway Performance Test")
    print("=" * 60)

    url = "http://localhost:8000/health"
    num_requests = 100

    response_times, errors = test_performance(url, num_requests)

    if response_times:
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile

        print("\n" + "=" * 60)
        print("Performance Results:")
        print("-" * 60)
        print(f"Successful requests: {len(response_times)}/{num_requests}")
        print(f"Failed requests:     {errors}")
        print("\nResponse Times (milliseconds):")
        print(f"  Average:     {avg_time:.2f} ms")
        print(f"  Median:      {median_time:.2f} ms")
        print(f"  Min:         {min_time:.2f} ms")
        print(f"  Max:         {max_time:.2f} ms")
        print(f"  P95:         {p95_time:.2f} ms")
        print(f"  P99:         {p99_time:.2f} ms")

        print("\n" + "=" * 60)
        if avg_time < 200:
            print(f"PASS: Average response time {avg_time:.2f}ms < 200ms target")
            print("System meets production performance requirements!")
        else:
            print(f"FAIL: Average response time {avg_time:.2f}ms > 200ms target")
            print("System does NOT meet production performance requirements.")

        print("=" * 60)

        # Distribution analysis
        under_100 = sum(1 for t in response_times if t < 100)
        under_200 = sum(1 for t in response_times if t < 200)
        under_500 = sum(1 for t in response_times if t < 500)

        print("\nResponse Time Distribution:")
        print(f"  < 100ms: {under_100} requests ({100*under_100/len(response_times):.1f}%)")
        print(f"  < 200ms: {under_200} requests ({100*under_200/len(response_times):.1f}%)")
        print(f"  < 500ms: {under_500} requests ({100*under_500/len(response_times):.1f}%)")

    else:
        print("No successful responses received!")

if __name__ == "__main__":
    main()