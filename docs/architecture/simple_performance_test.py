#!/usr/bin/env python3
"""
Simple Performance Test for DALRN Gateway
Quick verification of claimed vs actual performance
"""

import time
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import sys


def test_gateway_latency(url="http://127.0.0.1:8000/health", samples=10):
    """Test simple request latency"""
    print("\n1. TESTING GATEWAY LATENCY")
    print("-" * 40)

    latencies = []

    for i in range(samples):
        try:
            start = time.time()
            response = requests.get(url, timeout=5)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            print(f"  Request {i+1}: {latency:.2f}ms (Status: {response.status_code})")
        except Exception as e:
            print(f"  Request {i+1}: FAILED - {str(e)[:50]}")

    if latencies:
        print("\n  Results:")
        print(f"    Average: {statistics.mean(latencies):.2f}ms")
        print(f"    Median:  {statistics.median(latencies):.2f}ms")
        print(f"    Min:     {min(latencies):.2f}ms")
        print(f"    Max:     {max(latencies):.2f}ms")

        # Verdict
        avg_latency = statistics.mean(latencies)
        if avg_latency < 50:
            verdict = "EXCELLENT (<50ms)"
        elif avg_latency < 100:
            verdict = "GOOD (<100ms)"
        elif avg_latency < 500:
            verdict = "ACCEPTABLE (<500ms)"
        else:
            verdict = "POOR (>500ms)"
        print(f"    Verdict: {verdict}")

        return {"average_ms": avg_latency, "verdict": verdict}

    return {"error": "All requests failed"}


def test_gateway_throughput(url="http://127.0.0.1:8000/health", num_requests=100):
    """Test concurrent request throughput"""
    print("\n2. TESTING GATEWAY THROUGHPUT")
    print("-" * 40)
    print(f"  Sending {num_requests} concurrent requests...")

    def make_request(i):
        try:
            start = time.time()
            response = requests.get(url, timeout=2)
            return {
                "success": response.status_code == 200,
                "latency": time.time() - start
            }
        except:
            return {"success": False, "latency": None}

    # Start timing
    start_time = time.time()

    # Send concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        results = [f.result() for f in as_completed(futures)]

    # Calculate results
    total_time = time.time() - start_time
    successful = [r for r in results if r["success"]]
    throughput = len(successful) / total_time if total_time > 0 else 0

    print(f"\n  Results:")
    print(f"    Total Time:     {total_time:.2f}s")
    print(f"    Successful:     {len(successful)}/{num_requests}")
    print(f"    Throughput:     {throughput:.2f} req/s")
    print(f"    Success Rate:   {len(successful)/num_requests*100:.1f}%")

    # Verdict against claim of 10K req/s
    claimed_throughput = 10000
    percentage_of_claim = (throughput / claimed_throughput) * 100

    if throughput >= claimed_throughput * 0.9:
        verdict = f"MATCHES CLAIM (>90% of 10K req/s)"
    elif throughput >= claimed_throughput * 0.5:
        verdict = f"PARTIALLY MATCHES ({percentage_of_claim:.1f}% of claim)"
    elif throughput >= 1000:
        verdict = f"ACCEPTABLE (>{throughput:.0f} req/s)"
    else:
        verdict = f"BELOW EXPECTATIONS ({throughput:.0f} req/s)"

    print(f"    Verdict:        {verdict}")

    return {
        "throughput_req_s": throughput,
        "percentage_of_claim": percentage_of_claim,
        "verdict": verdict
    }


def test_service_response_times():
    """Test response times of different endpoints"""
    print("\n3. TESTING SERVICE ENDPOINTS")
    print("-" * 40)

    endpoints = [
        ("Root", "http://127.0.0.1:8000/"),
        ("Health", "http://127.0.0.1:8000/health"),
        ("Auth Login", "http://127.0.0.1:8000/auth/login"),
    ]

    results = {}

    for name, url in endpoints:
        try:
            start = time.time()
            response = requests.get(url, timeout=5)
            latency = (time.time() - start) * 1000

            print(f"  {name:15} [{response.status_code}]: {latency:.2f}ms")
            results[name] = {
                "latency_ms": latency,
                "status_code": response.status_code
            }
        except Exception as e:
            print(f"  {name:15} [ERROR]: {str(e)[:30]}")
            results[name] = {"error": str(e)[:50]}

    return results


def generate_report(test_results):
    """Generate performance report"""

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "tests": test_results,
        "verdict": ""
    }

    # Analyze results
    issues = []

    # Check latency
    if "latency" in test_results:
        avg_latency = test_results["latency"].get("average_ms", 999)
        if avg_latency > 100:
            issues.append(f"High latency: {avg_latency:.0f}ms (claim: <50ms)")

    # Check throughput
    if "throughput" in test_results:
        throughput = test_results["throughput"].get("throughput_req_s", 0)
        percentage = test_results["throughput"].get("percentage_of_claim", 0)
        if percentage < 50:
            issues.append(f"Low throughput: {throughput:.0f} req/s ({percentage:.1f}% of 10K claim)")

    # Overall verdict
    if not issues:
        report["verdict"] = "PERFORMANCE CLAIMS VERIFIED"
    elif len(issues) == 1:
        report["verdict"] = f"MINOR DISCREPANCY: {issues[0]}"
    else:
        report["verdict"] = f"MULTIPLE ISSUES FOUND ({len(issues)} discrepancies)"

    # Save report
    with open("simple_performance_test.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate markdown
    md = f"""# DALRN Performance Test Report

**Date:** {report['timestamp']}

## Summary

**Verdict:** {report['verdict']}

## Test Results

### 1. Gateway Latency
"""

    if "latency" in test_results:
        lat = test_results["latency"]
        if "average_ms" in lat:
            md += f"- Average: {lat['average_ms']:.2f}ms\n"
            md += f"- Verdict: {lat['verdict']}\n"
        else:
            md += f"- Error: {lat.get('error', 'Unknown')}\n"

    md += "\n### 2. Gateway Throughput\n"

    if "throughput" in test_results:
        thr = test_results["throughput"]
        md += f"- Throughput: {thr.get('throughput_req_s', 0):.2f} req/s\n"
        md += f"- Percentage of claim: {thr.get('percentage_of_claim', 0):.1f}%\n"
        md += f"- Verdict: {thr.get('verdict', 'Unknown')}\n"

    md += "\n### 3. Service Endpoints\n"

    if "endpoints" in test_results:
        for name, result in test_results["endpoints"].items():
            if "latency_ms" in result:
                md += f"- {name}: {result['latency_ms']:.2f}ms\n"
            else:
                md += f"- {name}: {result.get('error', 'Failed')}\n"

    if issues:
        md += "\n## Issues Found\n\n"
        for issue in issues:
            md += f"- {issue}\n"

    md += "\n## Performance Claims vs Reality\n\n"
    md += "| Metric | Claimed | Measured | Status |\n"
    md += "|--------|---------|----------|--------|\n"

    if "latency" in test_results and "average_ms" in test_results["latency"]:
        latency_status = "OK" if test_results["latency"]["average_ms"] < 50 else "ISSUE"
        md += f"| Latency | <50ms | {test_results['latency']['average_ms']:.0f}ms | {latency_status} |\n"

    if "throughput" in test_results and "throughput_req_s" in test_results["throughput"]:
        thr_status = "OK" if test_results["throughput"]["percentage_of_claim"] > 50 else "ISSUE"
        md += f"| Throughput | 10,000 req/s | {test_results['throughput']['throughput_req_s']:.0f} req/s | {thr_status} |\n"

    with open("simple_performance_test.md", "w") as f:
        f.write(md)

    return report


def main():
    """Run performance tests"""

    print("=" * 50)
    print("DALRN SIMPLE PERFORMANCE TEST")
    print("=" * 50)
    print(f"Started: {datetime.now().isoformat()}")

    test_results = {}

    # Check if gateway is running
    print("\nChecking gateway availability...")
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code != 200:
            print(f"WARNING: Gateway returned status {response.status_code}")
    except Exception as e:
        print(f"ERROR: Gateway not responding - {e}")
        print("\nPlease start the gateway service:")
        print("  python -m services.gateway.app")
        return 1

    # Run tests
    try:
        # Test 1: Latency
        test_results["latency"] = test_gateway_latency()

        # Test 2: Throughput
        test_results["throughput"] = test_gateway_throughput()

        # Test 3: Service endpoints
        test_results["endpoints"] = test_service_response_times()

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Test failed - {e}")
        return 1

    # Generate report
    print("\n" + "=" * 50)
    print("GENERATING REPORT")
    print("-" * 50)

    report = generate_report(test_results)

    print(f"\nOVERALL VERDICT: {report['verdict']}")
    print("\nReports saved:")
    print("  - simple_performance_test.json")
    print("  - simple_performance_test.md")

    print("\n" + "=" * 50)
    print("Test complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())