#!/usr/bin/env python3
"""
Quick Performance Check Script
Fast sanity check for DALRN service performance
"""

import requests
import time
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def check_service_health(service_name: str, port: int, timeout: float = 2.0) -> dict:
    """Check if a service is healthy and measure response time"""
    url = f"http://localhost:{port}/health"

    try:
        start = time.time()
        response = requests.get(url, timeout=timeout)
        latency_ms = (time.time() - start) * 1000

        return {
            "service": service_name,
            "port": port,
            "status": "OK Running" if response.status_code == 200 else f"WARNING Status {response.status_code}",
            "latency_ms": round(latency_ms, 2),
            "healthy": response.status_code == 200
        }
    except requests.exceptions.Timeout:
        return {
            "service": service_name,
            "port": port,
            "status": "TIMEOUT",
            "latency_ms": timeout * 1000,
            "healthy": False
        }
    except requests.exceptions.ConnectionError:
        return {
            "service": service_name,
            "port": port,
            "status": "ERROR Not running",
            "latency_ms": None,
            "healthy": False
        }
    except Exception as e:
        return {
            "service": service_name,
            "port": port,
            "status": f"ERROR: {str(e)[:30]}",
            "latency_ms": None,
            "healthy": False
        }


def quick_throughput_test(url: str, num_requests: int = 100) -> dict:
    """Quick throughput test with concurrent requests"""

    def make_request(i):
        try:
            start = time.time()
            response = requests.get(url, timeout=1)
            return {
                "success": response.status_code == 200,
                "latency": time.time() - start
            }
        except:
            return {"success": False, "latency": None}

    print(f"\n  Sending {num_requests} concurrent requests...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        results = [f.result() for f in as_completed(futures)]

    total_time = time.time() - start_time

    successful = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in successful if r["latency"]]

    if latencies:
        return {
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "success_rate": round(len(successful) / num_requests * 100, 1),
            "total_time_s": round(total_time, 2),
            "throughput_req_s": round(len(successful) / total_time, 2),
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
            "min_latency_ms": round(min(latencies) * 1000, 2),
            "max_latency_ms": round(max(latencies) * 1000, 2)
        }
    else:
        return {
            "total_requests": num_requests,
            "successful_requests": 0,
            "success_rate": 0,
            "error": "All requests failed"
        }


def main():
    """Run quick performance checks"""

    print("DALRN Quick Performance Check")
    print("="*50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*50)

    results = {
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "gateway_throughput": None,
        "summary": {}
    }

    # Define services to check
    services = [
        ("Gateway", 8000),
        ("Search", 8100),
        ("FHE", 8200),
        ("Negotiation", 8300),
        ("FL", 8400),
        ("Agents", 8500)
    ]

    # 1. Check service health
    print("\n1. SERVICE HEALTH CHECK")
    print("-"*30)

    healthy_count = 0
    total_latency = 0
    latency_count = 0

    for service_name, port in services:
        result = check_service_health(service_name, port)
        results["services"][service_name] = result

        status = result["status"]
        latency = result.get("latency_ms")

        if latency:
            lat_str = f" ({latency:.0f}ms)"
            total_latency += latency
            latency_count += 1
        else:
            lat_str = ""

        print(f"  {service_name:12} [{port}]: {status}{lat_str}")

        if result["healthy"]:
            healthy_count += 1

    # 2. Gateway throughput test (if gateway is healthy)
    if results["services"]["Gateway"]["healthy"]:
        print("\n2. GATEWAY QUICK THROUGHPUT TEST")
        print("-"*30)

        throughput_result = quick_throughput_test("http://localhost:8000/health", num_requests=100)
        results["gateway_throughput"] = throughput_result

        if "throughput_req_s" in throughput_result:
            print(f"  Throughput: {throughput_result['throughput_req_s']:.2f} req/s")
            print(f"  Success Rate: {throughput_result['success_rate']}%")
            print(f"  Avg Latency: {throughput_result['avg_latency_ms']:.2f}ms")
            print(f"  Min/Max: {throughput_result['min_latency_ms']:.0f}ms / {throughput_result['max_latency_ms']:.0f}ms")

            # Performance assessment
            if throughput_result['throughput_req_s'] > 1000:
                print(f"  Assessment: EXCELLENT (>1K req/s)")
            elif throughput_result['throughput_req_s'] > 500:
                print(f"  Assessment: GOOD (>500 req/s)")
            elif throughput_result['throughput_req_s'] > 100:
                print(f"  Assessment: FAIR (>100 req/s)")
            else:
                print(f"  Assessment: POOR (<100 req/s)")
        else:
            print(f"  ERROR: Throughput test failed")
    else:
        print("\n2. GATEWAY THROUGHPUT TEST")
        print("-"*30)
        print("  SKIPPED (Gateway not healthy)")

    # 3. Summary
    print("\n3. SUMMARY")
    print("-"*30)

    results["summary"]["services_healthy"] = f"{healthy_count}/{len(services)}"
    results["summary"]["services_running_pct"] = round(healthy_count / len(services) * 100, 1)

    if latency_count > 0:
        avg_latency = total_latency / latency_count
        results["summary"]["avg_health_check_latency_ms"] = round(avg_latency, 2)
        print(f"  Services Running: {healthy_count}/{len(services)} ({results['summary']['services_running_pct']}%)")
        print(f"  Avg Health Latency: {avg_latency:.2f}ms")
    else:
        print(f"  Services Running: {healthy_count}/{len(services)}")

    if results["gateway_throughput"] and "throughput_req_s" in results["gateway_throughput"]:
        print(f"  Gateway Throughput: {results['gateway_throughput']['throughput_req_s']:.2f} req/s")

    # Overall status
    print("\n  Overall Status: ", end="")
    if healthy_count == len(services):
        print("ALL SERVICES HEALTHY")
        results["summary"]["overall_status"] = "HEALTHY"
    elif healthy_count >= len(services) * 0.5:
        print("PARTIAL AVAILABILITY")
        results["summary"]["overall_status"] = "DEGRADED"
    else:
        print("SYSTEM UNAVAILABLE")
        results["summary"]["overall_status"] = "UNAVAILABLE"

    # 4. Save results
    print("\n4. SAVING RESULTS")
    print("-"*30)

    output_file = "quick_check_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved to {output_file}")

    # Performance hints
    if results["summary"]["services_running_pct"] < 100:
        print("\nHINTS:")
        print("  - Start missing services with: python -m services.[name].service")
        print("  - Check logs for error messages")

    if results["gateway_throughput"] and results["gateway_throughput"].get("throughput_req_s", 0) < 500:
        print("\nPERFORMANCE NOTE:")
        print("  - Gateway throughput is below optimal levels")
        print("  - Consider checking CPU usage and network conditions")

    print("\n" + "="*50)
    print("Quick check complete!")

    return 0 if healthy_count > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR Check failed: {e}")
        sys.exit(1)