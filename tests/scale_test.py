"""
Scale Testing for DALRN - Verify 100,000 disputes/day capability
PRD REQUIREMENT: Handle 100,000 disputes per day
"""
import asyncio
import aiohttp
import time
import random
import statistics
import json
from typing import List, Dict
from datetime import datetime
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ScaleTester:
    """Load testing to verify 100k disputes/day requirement"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.target_per_day = 100000
        self.target_per_second = self.target_per_day / 86400  # ~1.16 per second
        self.target_per_minute = self.target_per_day / 1440  # ~69.4 per minute
        self.target_per_hour = self.target_per_day / 24  # ~4167 per hour

        # Test results
        self.response_times = []
        self.errors = []
        self.success_count = 0
        self.total_requests = 0

        # JWT token for authenticated requests
        self.auth_token = None

    async def authenticate(self):
        """Get JWT token for testing"""
        async with aiohttp.ClientSession() as session:
            # Register test user
            test_user = {
                "username": f"test_user_{random.randint(1000, 9999)}",
                "email": f"test_{random.randint(1000, 9999)}@dalrn.test",
                "password": "test_password_123"
            }

            async with session.post(
                f"{self.base_url}/auth/register",
                json=test_user
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.auth_token = data.get("access_token")
                    print(f"Authenticated as {test_user['username']}")
                else:
                    print("Failed to authenticate, using anonymous mode")

    def generate_dispute_data(self, index: int) -> Dict:
        """Generate realistic dispute data"""
        return {
            "parties": [
                f"party_a_{index}",
                f"party_b_{index}"
            ],
            "jurisdiction": random.choice(["US-CA", "US-NY", "US-TX", "EU-DE", "EU-FR"]),
            "cid": f"Qm{str(index).zfill(44)}",
            "enc_meta": {
                "encryption_scheme": "CKKS",
                "key_id": f"key_{index}",
                "test_mode": True
            }
        }

    async def send_single_request(self, session: aiohttp.ClientSession, index: int):
        """Send single dispute submission request"""
        start_time = time.perf_counter()
        dispute_data = self.generate_dispute_data(index)

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            async with session.post(
                f"{self.base_url}/submit-dispute-fast",
                json=dispute_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.response_times.append(elapsed_ms)

                if response.status in [200, 201]:
                    self.success_count += 1
                    return {"success": True, "time_ms": elapsed_ms}
                else:
                    error = f"Status {response.status}"
                    self.errors.append(error)
                    return {"success": False, "error": error}

        except asyncio.TimeoutError:
            self.errors.append("Timeout")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            self.errors.append(str(e))
            return {"success": False, "error": str(e)}

    async def run_burst_test(self, burst_size: int = 100, duration_seconds: int = 60):
        """Run burst load test"""
        print(f"\n{'='*60}")
        print(f"BURST LOAD TEST")
        print(f"Burst size: {burst_size} requests")
        print(f"Duration: {duration_seconds} seconds")
        print(f"{'='*60}\n")

        start_time = time.time()
        request_index = 0

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration_seconds:
                # Send burst of requests
                tasks = []
                for _ in range(burst_size):
                    tasks.append(self.send_single_request(session, request_index))
                    request_index += 1

                # Execute burst concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                self.total_requests += len(tasks)

                # Brief pause between bursts
                await asyncio.sleep(1)

                # Print progress
                elapsed = time.time() - start_time
                rate = self.success_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {self.success_count}/{self.total_requests} "
                      f"({rate:.1f} req/s)")

    async def run_sustained_test(self, duration_seconds: int = 300):
        """Run sustained load test at target rate"""
        print(f"\n{'='*60}")
        print(f"SUSTAINED LOAD TEST")
        print(f"Target rate: {self.target_per_second:.2f} requests/second")
        print(f"Duration: {duration_seconds} seconds")
        print(f"{'='*60}\n")

        start_time = time.time()
        request_index = 0
        requests_per_batch = int(self.target_per_second * 10)  # 10-second batches

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()

                # Send batch of requests
                tasks = []
                for _ in range(requests_per_batch):
                    tasks.append(self.send_single_request(session, request_index))
                    request_index += 1

                results = await asyncio.gather(*tasks, return_exceptions=True)
                self.total_requests += len(tasks)

                # Wait to maintain target rate
                batch_duration = time.time() - batch_start
                if batch_duration < 10:
                    await asyncio.sleep(10 - batch_duration)

                # Print progress
                elapsed = time.time() - start_time
                rate = self.success_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {self.success_count}/{self.total_requests} "
                      f"({rate:.1f} req/s)")

    def analyze_results(self):
        """Analyze test results"""
        if not self.response_times:
            print("No successful requests to analyze")
            return

        # Calculate statistics
        success_rate = (self.success_count / self.total_requests * 100) if self.total_requests > 0 else 0
        avg_time = statistics.mean(self.response_times)
        median_time = statistics.median(self.response_times)
        min_time = min(self.response_times)
        max_time = max(self.response_times)

        # Calculate percentiles
        sorted_times = sorted(self.response_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        # Extrapolate daily capacity
        test_duration = max_time / 1000  # Convert to seconds
        if test_duration > 0:
            rate_per_second = self.success_count / test_duration
            extrapolated_daily = rate_per_second * 86400
        else:
            extrapolated_daily = 0

        # Print results
        print(f"\n{'='*60}")
        print(f"SCALE TEST RESULTS")
        print(f"{'='*60}\n")

        print(f"Requests:")
        print(f"  Total sent: {self.total_requests}")
        print(f"  Successful: {self.success_count}")
        print(f"  Failed: {len(self.errors)}")
        print(f"  Success rate: {success_rate:.2f}%")

        print(f"\nResponse Times:")
        print(f"  Min: {min_time:.1f}ms")
        print(f"  Avg: {avg_time:.1f}ms")
        print(f"  Median: {median_time:.1f}ms")
        print(f"  Max: {max_time:.1f}ms")
        print(f"  P50: {p50:.1f}ms")
        print(f"  P95: {p95:.1f}ms")
        print(f"  P99: {p99:.1f}ms")

        print(f"\nThroughput:")
        print(f"  Achieved: {self.success_count / (test_duration if test_duration > 0 else 1):.1f} req/s")
        print(f"  Target: {self.target_per_second:.2f} req/s")
        print(f"  Extrapolated daily: {extrapolated_daily:,.0f} disputes")
        print(f"  Target daily: {self.target_per_day:,} disputes")

        print(f"\nRequirements Check:")
        requirements_met = []

        # Check success rate (>99%)
        if success_rate > 99:
            print(f"  [PASS] Success rate > 99%: {success_rate:.2f}%")
            requirements_met.append(True)
        else:
            print(f"  [FAIL] Success rate < 99%: {success_rate:.2f}%")
            requirements_met.append(False)

        # Check response time (<200ms for P95)
        if p95 < 200:
            print(f"  [PASS] P95 latency < 200ms: {p95:.1f}ms")
            requirements_met.append(True)
        else:
            print(f"  [FAIL] P95 latency > 200ms: {p95:.1f}ms")
            requirements_met.append(False)

        # Check throughput
        if extrapolated_daily >= self.target_per_day:
            print(f"  [PASS] Can handle 100k/day: {extrapolated_daily:,.0f}")
            requirements_met.append(True)
        else:
            print(f"  [FAIL] Cannot handle 100k/day: {extrapolated_daily:,.0f}")
            requirements_met.append(False)

        # Overall result
        print(f"\n{'='*60}")
        if all(requirements_met):
            print("[PASS] SCALE TEST PASSED - System meets 100k disputes/day requirement")
        else:
            print("[FAIL] SCALE TEST FAILED - System does not meet requirements")
        print(f"{'='*60}\n")

        # Save results to file
        self.save_results()

    def save_results(self):
        """Save test results to JSON file"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.success_count,
            "failed_requests": len(self.errors),
            "success_rate": (self.success_count / self.total_requests * 100) if self.total_requests > 0 else 0,
            "response_times": {
                "min": min(self.response_times) if self.response_times else 0,
                "avg": statistics.mean(self.response_times) if self.response_times else 0,
                "median": statistics.median(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "p95": sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0,
                "p99": sorted(self.response_times)[int(len(self.response_times) * 0.99)] if self.response_times else 0
            },
            "errors": self.errors[:100]  # First 100 errors
        }

        with open("scale_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to scale_test_results.json")

async def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="DALRN Scale Testing")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL for testing")
    parser.add_argument("--test", choices=["burst", "sustained", "full"], default="full", help="Test type")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--burst-size", type=int, default=100, help="Burst size for burst test")

    args = parser.parse_args()

    # Initialize tester
    tester = ScaleTester(base_url=args.url)

    # Authenticate
    await tester.authenticate()

    # Run tests
    if args.test == "burst" or args.test == "full":
        await tester.run_burst_test(burst_size=args.burst_size, duration_seconds=args.duration)

    if args.test == "sustained" or args.test == "full":
        await tester.run_sustained_test(duration_seconds=args.duration)

    # Analyze results
    tester.analyze_results()

if __name__ == "__main__":
    import hashlib
    asyncio.run(main())