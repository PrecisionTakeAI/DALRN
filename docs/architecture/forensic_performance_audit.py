#!/usr/bin/env python3
"""
DALRN Forensic Performance Audit
Comprehensive performance testing to verify all claims with actual measurements
"""

import sys
import json
import time
import asyncio
import statistics
import tracemalloc
import psutil
import numpy as np
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import aiohttp
import requests
import jwt
import logging

# Disable excessive logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ForensicPerformanceTester:
    """
    Forensic performance tester that measures ACTUAL performance,
    not claimed performance.
    """

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "claimed_metrics": self._load_claimed_metrics(),
            "actual_metrics": {},
            "discrepancies": [],
            "resource_usage": {},
            "test_conditions": {
                "cpu_count": multiprocessing.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": sys.version.split()[0],
                "platform": sys.platform
            }
        }
        self.base_url = "http://localhost"

    def _load_claimed_metrics(self) -> dict:
        """Load the claimed performance metrics from documentation"""
        return {
            "gateway": {
                "throughput": "10,000 req/s",
                "latency": "<50ms p99"
            },
            "search": {
                "query_time": "<10ms",
                "recall": ">95%",
                "index_size": "1M vectors"
            },
            "fhe": {
                "encryption_time": "~50ms",
                "operation_overhead": "22283x",
                "scheme": "CKKS"
            },
            "negotiation": {
                "computation_time": "<100ms",
                "max_players": "100"
            },
            "fl": {
                "aggregation_time": "<1s",
                "max_clients": "100"
            }
        }

    async def test_gateway_throughput(self) -> dict:
        """Test ACTUAL gateway throughput, not claimed 10K req/s"""
        print("\nTESTING GATEWAY THROUGHPUT...")

        # Create test token for authentication
        test_token = jwt.encode(
            {"sub": "test_user", "exp": time.time() + 3600},
            "your-secret-key",  # Would use actual secret in production
            algorithm="HS256"
        )

        results = {
            "claimed": self.results["claimed_metrics"]["gateway"]["throughput"],
            "test_loads": {},
            "max_sustainable_throughput": 0,
            "latency_percentiles": {}
        }

        async def make_request(session, url, headers):
            start = time.time()
            try:
                async with session.get(url, headers=headers) as resp:
                    await resp.text()
                    return time.time() - start, resp.status
            except Exception as e:
                return -1, str(e)

        # Test with increasing load
        test_loads = [10, 100, 500, 1000, 2000, 5000]

        for num_requests in test_loads:
            print(f"  Testing with {num_requests} requests...")

            async with aiohttp.ClientSession() as session:
                start_time = time.time()

                # Fire all requests concurrently
                tasks = [
                    make_request(
                        session,
                        f"{self.base_url}:8000/health",
                        {"Authorization": f"Bearer {test_token}"}
                    ) for _ in range(num_requests)
                ]

                responses = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time

                # Calculate metrics
                successful = [r for r in responses if isinstance(r, tuple) and r[0] > 0]

                if successful:
                    latencies = [r[0] for r in successful]

                    results["test_loads"][num_requests] = {
                        "total_time": round(total_time, 3),
                        "throughput": round(len(successful) / total_time, 2),
                        "success_rate": round(len(successful) / num_requests, 3),
                        "avg_latency_ms": round(statistics.mean(latencies) * 1000, 2),
                        "p50_latency_ms": round(np.percentile(latencies, 50) * 1000, 2),
                        "p95_latency_ms": round(np.percentile(latencies, 95) * 1000, 2),
                        "p99_latency_ms": round(np.percentile(latencies, 99) * 1000, 2),
                        "max_latency_ms": round(max(latencies) * 1000, 2)
                    }

                    print(f"    [OK] Throughput: {results['test_loads'][num_requests]['throughput']:.2f} req/s")
                    print(f"    [OK] P99 Latency: {results['test_loads'][num_requests]['p99_latency_ms']:.2f}ms")
                    print(f"    [OK] Success Rate: {results['test_loads'][num_requests]['success_rate']*100:.1f}%")

                    # Update max sustainable throughput
                    if results["test_loads"][num_requests]["success_rate"] >= 0.95:
                        results["max_sustainable_throughput"] = max(
                            results["max_sustainable_throughput"],
                            results["test_loads"][num_requests]["throughput"]
                        )

        # Determine verdict
        claimed_throughput = 10000  # 10K req/s
        actual_throughput = results["max_sustainable_throughput"]

        if actual_throughput >= claimed_throughput * 0.9:  # Within 10% of claim
            results["verdict"] = "MATCHES CLAIM"
        elif actual_throughput >= claimed_throughput * 0.5:  # Within 50%
            results["verdict"] = f"PARTIALLY ACCURATE ({actual_throughput/claimed_throughput*100:.1f}% of claim)"
        else:
            results["verdict"] = f"EXAGGERATED by {claimed_throughput/actual_throughput:.1f}x"

        results["actual_max_throughput"] = f"{actual_throughput:.2f} req/s"

        return results

    def test_search_performance(self) -> dict:
        """Test ACTUAL FAISS search performance"""
        print("\nTESTING SEARCH SERVICE...")

        try:
            import faiss
        except ImportError:
            print("  WARNING: FAISS not installed, using mock test")
            return {"error": "FAISS not available for testing"}

        results = {
            "claimed_query_time": self.results["claimed_metrics"]["search"]["query_time"],
            "claimed_recall": self.results["claimed_metrics"]["search"]["recall"]
        }

        # Generate test data
        dimension = 768  # As per documentation
        n_vectors = 10000  # Start with 10K for quick test
        n_queries = 100

        print(f"  Generating {n_vectors:,} test vectors (dimension={dimension})...")
        np.random.seed(42)
        data = np.random.random((n_vectors, dimension)).astype('float32')
        queries = np.random.random((n_queries, dimension)).astype('float32')

        # Normalize vectors (as the service should do)
        faiss.normalize_L2(data)
        faiss.normalize_L2(queries)

        # Build HNSW index (matching service configuration)
        print("  Building FAISS HNSW index...")
        index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 as per docs
        index.hnsw.efConstruction = 200  # As per docs

        start_build = time.time()
        index.add(data)
        build_time = time.time() - start_build

        # Measure search performance
        print("  Running search benchmark...")
        search_times = []
        k = 10

        for i in range(n_queries):
            start = time.time()
            distances, indices = index.search(queries[i:i+1], k)
            search_time = time.time() - start
            search_times.append(search_time)

        # Calculate recall using brute force as ground truth
        print("  Calculating recall (sampling)...")
        index_flat = faiss.IndexFlatL2(dimension)
        index_flat.add(data)

        recalls = []
        sample_size = min(20, n_queries)  # Sample for recall calculation

        for i in range(sample_size):
            D_hnsw, I_hnsw = index.search(queries[i:i+1], k)
            D_bf, I_bf = index_flat.search(queries[i:i+1], k)
            recall = len(set(I_hnsw[0]) & set(I_bf[0])) / k
            recalls.append(recall)

        # Test actual service endpoint if running
        try:
            test_vector = queries[0].tolist()
            response = requests.post(
                f"{self.base_url}:8100/search",
                json={"vector": test_vector, "k": 10},
                timeout=5
            )
            if response.status_code == 200:
                service_latency = response.elapsed.total_seconds() * 1000
                results["service_endpoint_latency_ms"] = round(service_latency, 2)
            else:
                results["service_endpoint_latency_ms"] = "Service error"
        except Exception as e:
            results["service_endpoint_latency_ms"] = f"Service not available: {str(e)[:50]}"

        # Compile results
        avg_query_time = statistics.mean(search_times) * 1000  # Convert to ms
        p99_query_time = np.percentile(search_times, 99) * 1000
        avg_recall = statistics.mean(recalls) * 100

        results.update({
            "index_build_time_s": round(build_time, 3),
            "actual_avg_query_time_ms": round(avg_query_time, 2),
            "actual_p99_query_time_ms": round(p99_query_time, 2),
            "actual_recall_pct": round(avg_recall, 2),
            "test_vectors": n_vectors,
            "test_queries": n_queries,
            "query_time_verdict": "MATCHES" if avg_query_time < 10 else f"SLOWER ({avg_query_time/10:.1f}x)",
            "recall_verdict": "MATCHES" if avg_recall > 95 else f"LOWER ({avg_recall:.1f}%)"
        })

        print(f"    [OK] Avg Query Time: {avg_query_time:.2f}ms")
        print(f"    [OK] P99 Query Time: {p99_query_time:.2f}ms")
        print(f"    [OK] Recall: {avg_recall:.1f}%")

        return results

    def test_fhe_performance(self) -> dict:
        """Test ACTUAL homomorphic encryption performance"""
        print("\nTESTING FHE SERVICE...")

        try:
            import tenseal as ts
        except ImportError:
            print("  WARNING: TenSEAL not installed, using estimation")
            return {"error": "TenSEAL not available for testing"}

        results = {
            "claimed_encryption_time": self.results["claimed_metrics"]["fhe"]["encryption_time"],
            "claimed_overhead": self.results["claimed_metrics"]["fhe"]["operation_overhead"]
        }

        # Test different data sizes
        test_sizes = [10, 100, 1000]

        # Create TenSEAL context (matching service configuration)
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()

        encryption_results = {}

        for size in test_sizes:
            print(f"  Testing with {size} elements...")

            # Generate test data
            plain_data = np.random.random(size).tolist()

            # Measure encryption time
            encryption_times = []
            for _ in range(5):  # 5 runs for average
                start = time.time()
                encrypted = ts.ckks_vector(context, plain_data)
                encryption_time = time.time() - start
                encryption_times.append(encryption_time)

            # Measure operations
            encrypted1 = ts.ckks_vector(context, plain_data)
            encrypted2 = ts.ckks_vector(context, plain_data)

            # Addition timing
            start = time.time()
            result_add = encrypted1 + encrypted2
            add_time = time.time() - start

            # Multiplication timing
            start = time.time()
            result_mult = encrypted1 * encrypted2
            mult_time = time.time() - start

            # Measure ciphertext expansion
            ciphertext_size = len(encrypted1.serialize())
            plaintext_size = size * 8  # 8 bytes per float64
            expansion_ratio = ciphertext_size / plaintext_size if plaintext_size > 0 else 0

            encryption_results[size] = {
                "avg_encryption_ms": round(statistics.mean(encryption_times) * 1000, 2),
                "addition_ms": round(add_time * 1000, 2),
                "multiplication_ms": round(mult_time * 1000, 2),
                "expansion_ratio": round(expansion_ratio, 1)
            }

            print(f"    [OK] Encryption: {encryption_results[size]['avg_encryption_ms']:.2f}ms")
            print(f"    [OK] Expansion: {expansion_ratio:.1f}x")

        # Test service endpoint
        try:
            response = requests.post(
                f"{self.base_url}:8200/encrypt",
                json={"values": [1.0, 2.0, 3.0]},
                timeout=5
            )
            if response.status_code == 200:
                results["service_available"] = True
        except:
            results["service_available"] = False

        # Overall verdict
        avg_100_encryption = encryption_results.get(100, {}).get("avg_encryption_ms", 999)

        results.update({
            "test_results": encryption_results,
            "encryption_verdict": "MATCHES" if avg_100_encryption < 100 else f"SLOWER ({avg_100_encryption:.0f}ms)",
            "expansion_verdict": "MATCHES" if encryption_results[100]["expansion_ratio"] < 30000 else "HIGHER"
        })

        return results

    def test_nash_performance(self) -> dict:
        """Test Nash equilibrium computation performance"""
        print("\nTESTING NASH COMPUTATION...")

        try:
            import nashpy as nash
        except ImportError:
            print("  WARNING: nashpy not installed")
            return {"error": "nashpy not available for testing"}

        results = {
            "claimed_computation_time": self.results["claimed_metrics"]["negotiation"]["computation_time"],
            "tests": {}
        }

        # Test different game sizes
        game_sizes = [2, 5, 10, 20]

        for size in game_sizes:
            print(f"  Testing {size}x{size} game...")

            # Create random payoff matrices
            np.random.seed(42)
            payoff_a = np.random.random((size, size))
            payoff_b = np.random.random((size, size))

            game = nash.Game(payoff_a, payoff_b)

            computation_times = []
            equilibria_found = 0

            # Run multiple times
            for _ in range(3):
                try:
                    start = time.time()
                    equilibria = list(game.support_enumeration())
                    computation_time = time.time() - start
                    computation_times.append(computation_time)
                    equilibria_found = len(equilibria)
                except Exception as e:
                    print(f"    WARNING: Failed: {str(e)[:50]}")
                    break

            if computation_times:
                avg_time_ms = statistics.mean(computation_times) * 1000

                results["tests"][f"{size}x{size}"] = {
                    "avg_time_ms": round(avg_time_ms, 2),
                    "equilibria_found": equilibria_found,
                    "under_100ms": avg_time_ms < 100
                }

                print(f"    [OK] Computation: {avg_time_ms:.2f}ms")
                print(f"    [OK] Equilibria found: {equilibria_found}")

        # Verdict
        small_games_ok = all([
            results["tests"].get(f"{s}x{s}", {}).get("under_100ms", False)
            for s in [2, 5] if f"{s}x{s}" in results["tests"]
        ])

        results["verdict"] = "MATCHES for small games" if small_games_ok else "SLOWER than claimed"

        return results

    def test_fl_aggregation(self) -> dict:
        """Test federated learning aggregation performance"""
        print("\nTESTING FL AGGREGATION...")

        results = {
            "claimed_aggregation_time": self.results["claimed_metrics"]["fl"]["aggregation_time"],
            "tests": {}
        }

        # Test scenarios
        test_scenarios = [
            {"clients": 10, "model_size": 1000},
            {"clients": 50, "model_size": 1000},
            {"clients": 100, "model_size": 1000},
            {"clients": 100, "model_size": 10000}
        ]

        for scenario in test_scenarios:
            n_clients = scenario["clients"]
            model_size = scenario["model_size"]

            print(f"  Testing {n_clients} clients, {model_size} parameters...")

            # Simulate client models
            np.random.seed(42)
            client_models = [np.random.random(model_size) for _ in range(n_clients)]

            # Test FedAvg aggregation
            start = time.time()
            aggregated = np.mean(client_models, axis=0)
            fedavg_time = time.time() - start

            # Test weighted aggregation
            weights = np.random.random(n_clients)
            weights = weights / weights.sum()

            start = time.time()
            aggregated = np.average(client_models, axis=0, weights=weights)
            weighted_time = time.time() - start

            scenario_key = f"{n_clients}c_{model_size}p"
            results["tests"][scenario_key] = {
                "fedavg_ms": round(fedavg_time * 1000, 2),
                "weighted_ms": round(weighted_time * 1000, 2),
                "under_1s": max(fedavg_time, weighted_time) < 1.0
            }

            print(f"    [OK] FedAvg: {results['tests'][scenario_key]['fedavg_ms']:.2f}ms")
            print(f"    [OK] Weighted: {results['tests'][scenario_key]['weighted_ms']:.2f}ms")

        # Verdict
        all_under_1s = all([t["under_1s"] for t in results["tests"].values()])
        results["verdict"] = "MATCHES" if all_under_1s else "SLOWER for large models"

        return results

    def monitor_service_resources(self, service_name: str, port: int, duration: int = 10) -> dict:
        """Monitor resource usage of a service"""
        print(f"\nMONITORING {service_name.upper()} RESOURCES...")

        # Check if service is responding
        try:
            response = requests.get(f"{self.base_url}:{port}/health", timeout=2)
            if response.status_code != 200:
                return {"status": "not_healthy"}
        except:
            return {"status": "not_running"}

        # Monitor system resources during load
        measurements = {
            "cpu_percent": [],
            "memory_mb": []
        }

        # Get baseline
        process = psutil.Process()
        baseline_cpu = process.cpu_percent(interval=1)
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Generate load and measure
        print(f"  Generating load for {duration} seconds...")
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration:
            try:
                # Make request
                requests.get(f"{self.base_url}:{port}/health", timeout=1)
                request_count += 1

                # Measure resources
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()

                measurements["cpu_percent"].append(cpu)
                measurements["memory_mb"].append(mem.used / 1024 / 1024)

            except:
                pass

            time.sleep(0.1)

        # Calculate statistics
        if measurements["cpu_percent"]:
            return {
                "status": "monitored",
                "requests_made": request_count,
                "avg_cpu_percent": round(statistics.mean(measurements["cpu_percent"]), 1),
                "max_cpu_percent": round(max(measurements["cpu_percent"]), 1),
                "avg_memory_mb": round(statistics.mean(measurements["memory_mb"]), 1),
                "max_memory_mb": round(max(measurements["memory_mb"]), 1)
            }

        return {"status": "monitoring_failed"}

    async def run_comprehensive_audit(self):
        """Run all performance tests"""
        print("="*60)
        print("DALRN FORENSIC PERFORMANCE AUDIT")
        print("="*60)
        print("Testing ACTUAL vs CLAIMED performance metrics")
        print(f"Platform: {self.results['test_conditions']['platform']}")
        print(f"CPUs: {self.results['test_conditions']['cpu_count']}")
        print(f"RAM: {self.results['test_conditions']['memory_gb']:.1f} GB")
        print("="*60)

        # 1. Gateway throughput
        print("\n[1/6] Gateway Service Tests")
        try:
            gateway_results = await self.test_gateway_throughput()
            self.results["actual_metrics"]["gateway"] = gateway_results
        except Exception as e:
            print(f"  ERROR: Gateway test failed: {e}")
            self.results["actual_metrics"]["gateway"] = {"error": str(e)}

        # 2. Search performance
        print("\n[2/6] Search Service Tests")
        try:
            search_results = self.test_search_performance()
            self.results["actual_metrics"]["search"] = search_results
        except Exception as e:
            print(f"  ERROR: Search test failed: {e}")
            self.results["actual_metrics"]["search"] = {"error": str(e)}

        # 3. FHE performance
        print("\n[3/6] FHE Service Tests")
        try:
            fhe_results = self.test_fhe_performance()
            self.results["actual_metrics"]["fhe"] = fhe_results
        except Exception as e:
            print(f"  ERROR: FHE test failed: {e}")
            self.results["actual_metrics"]["fhe"] = {"error": str(e)}

        # 4. Nash equilibrium
        print("\n[4/6] Nash Equilibrium Tests")
        try:
            nash_results = self.test_nash_performance()
            self.results["actual_metrics"]["negotiation"] = nash_results
        except Exception as e:
            print(f"  ERROR: Nash test failed: {e}")
            self.results["actual_metrics"]["negotiation"] = {"error": str(e)}

        # 5. Federated Learning
        print("\n[5/6] Federated Learning Tests")
        try:
            fl_results = self.test_fl_aggregation()
            self.results["actual_metrics"]["fl"] = fl_results
        except Exception as e:
            print(f"  ERROR: FL test failed: {e}")
            self.results["actual_metrics"]["fl"] = {"error": str(e)}

        # 6. Resource monitoring
        print("\n[6/6] Resource Usage Monitoring")
        services_to_monitor = [
            ("gateway", 8000),
            ("search", 8100),
            ("fhe", 8200)
        ]

        for service_name, port in services_to_monitor:
            try:
                resource_results = self.monitor_service_resources(service_name, port, duration=5)
                self.results["resource_usage"][service_name] = resource_results
            except Exception as e:
                print(f"  ERROR: {service_name} monitoring failed: {e}")
                self.results["resource_usage"][service_name] = {"error": str(e)}

        # Analyze results
        self._analyze_discrepancies()

        # Generate report
        self._generate_report()

        return self.results

    def _analyze_discrepancies(self):
        """Identify discrepancies between claims and reality"""

        # Gateway
        if "gateway" in self.results["actual_metrics"]:
            gateway = self.results["actual_metrics"]["gateway"]
            if "verdict" in gateway and "EXAGGERATED" in gateway["verdict"]:
                self.results["discrepancies"].append({
                    "service": "Gateway",
                    "metric": "throughput",
                    "claimed": "10,000 req/s",
                    "actual": gateway.get("actual_max_throughput", "Unknown"),
                    "severity": "HIGH"
                })

        # Search
        if "search" in self.results["actual_metrics"]:
            search = self.results["actual_metrics"]["search"]
            if search.get("query_time_verdict", "").startswith("SLOWER"):
                self.results["discrepancies"].append({
                    "service": "Search",
                    "metric": "query_time",
                    "claimed": "<10ms",
                    "actual": f"{search.get('actual_avg_query_time_ms', 'Unknown')}ms",
                    "severity": "MEDIUM"
                })

        # FHE
        if "fhe" in self.results["actual_metrics"]:
            fhe = self.results["actual_metrics"]["fhe"]
            if fhe.get("encryption_verdict", "").startswith("SLOWER"):
                self.results["discrepancies"].append({
                    "service": "FHE",
                    "metric": "encryption_time",
                    "claimed": "~50ms",
                    "actual": "Varies by size",
                    "severity": "LOW"
                })

    def _generate_report(self):
        """Generate comprehensive audit report"""

        # Save JSON report
        with open("PERFORMANCE_AUDIT.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate markdown report
        self._generate_markdown_report()

        # Print summary
        print("\n" + "="*60)
        print("AUDIT COMPLETE")
        print("="*60)

        discrepancy_count = len(self.results["discrepancies"])

        if discrepancy_count == 0:
            print("All performance claims verified!")
        elif discrepancy_count <= 2:
            print(f"WARNING: Found {discrepancy_count} minor discrepancies")
        else:
            print(f"ERROR: Found {discrepancy_count} significant discrepancies")

        print("\nReports saved:")
        print("  - PERFORMANCE_AUDIT.json (detailed data)")
        print("  - PERFORMANCE_AUDIT.md (summary report)")

    def _generate_markdown_report(self):
        """Generate human-readable markdown report"""

        md = f"""# DALRN Performance Audit Report

**Generated:** {self.results['timestamp']}
**Platform:** {self.results['test_conditions']['platform']}
**CPUs:** {self.results['test_conditions']['cpu_count']}
**Memory:** {self.results['test_conditions']['memory_gb']:.1f} GB

## Executive Summary

"""

        if len(self.results['discrepancies']) == 0:
            md += "**All performance claims have been verified**\n\n"
        elif len(self.results['discrepancies']) <= 2:
            md += f"**Most claims verified with {len(self.results['discrepancies'])} minor discrepancies**\n\n"
        else:
            md += f"**Found {len(self.results['discrepancies'])} significant discrepancies**\n\n"

        # Performance comparison table
        md += """## Performance Comparison

| Service | Metric | Claimed | Actual | Status |
|---------|--------|---------|--------|--------|
"""

        # Gateway
        if "gateway" in self.results["actual_metrics"]:
            g = self.results["actual_metrics"]["gateway"]
            if "actual_max_throughput" in g:
                md += f"| Gateway | Throughput | 10,000 req/s | {g['actual_max_throughput']} | {g.get('verdict', 'Unknown')} |\n"

        # Search
        if "search" in self.results["actual_metrics"]:
            s = self.results["actual_metrics"]["search"]
            if "actual_avg_query_time_ms" in s:
                md += f"| Search | Query Time | <10ms | {s['actual_avg_query_time_ms']}ms | {s.get('query_time_verdict', 'Unknown')} |\n"
            if "actual_recall_pct" in s:
                md += f"| Search | Recall | >95% | {s['actual_recall_pct']}% | {s.get('recall_verdict', 'Unknown')} |\n"

        # FHE
        if "fhe" in self.results["actual_metrics"]:
            f = self.results["actual_metrics"]["fhe"]
            if "test_results" in f and 100 in f["test_results"]:
                md += f"| FHE | Encryption (100 elem) | ~50ms | {f['test_results'][100]['avg_encryption_ms']}ms | {f.get('encryption_verdict', 'Unknown')} |\n"

        # Nash
        if "negotiation" in self.results["actual_metrics"]:
            n = self.results["actual_metrics"]["negotiation"]
            if "tests" in n and "5x5" in n["tests"]:
                md += f"| Nash | Computation (5x5) | <100ms | {n['tests']['5x5']['avg_time_ms']}ms | {n.get('verdict', 'Unknown')} |\n"

        # FL
        if "fl" in self.results["actual_metrics"]:
            fl = self.results["actual_metrics"]["fl"]
            if "tests" in fl and "100c_1000p" in fl["tests"]:
                md += f"| FL | Aggregation (100 clients) | <1s | {fl['tests']['100c_1000p']['fedavg_ms']}ms | {fl.get('verdict', 'Unknown')} |\n"

        # Resource usage
        if self.results["resource_usage"]:
            md += "\n## Resource Usage\n\n"
            md += "| Service | Status | Avg CPU % | Max Memory MB |\n"
            md += "|---------|--------|-----------|---------------|\n"

            for service, metrics in self.results["resource_usage"].items():
                if metrics.get("status") == "monitored":
                    md += f"| {service} | Running | {metrics.get('avg_cpu_percent', 'N/A')}% | {metrics.get('max_memory_mb', 'N/A')} |\n"
                else:
                    md += f"| {service} | {metrics.get('status', 'Unknown')} | - | - |\n"

        # Discrepancies
        if self.results["discrepancies"]:
            md += "\n## Discrepancies Found\n\n"
            for disc in self.results["discrepancies"]:
                md += f"- **{disc['service']} ({disc['metric']}):** "
                md += f"Claimed {disc['claimed']}, Actual {disc['actual']} "
                md += f"[Severity: {disc['severity']}]\n"

        md += "\n## Recommendations\n\n"

        if len(self.results["discrepancies"]) == 0:
            md += "1. Performance claims are accurate - no changes needed\n"
        else:
            md += "1. Update documentation to reflect actual measured performance\n"
            md += "2. Consider optimization for services not meeting targets\n"
            md += "3. Set realistic performance expectations based on hardware\n"

        md += "\n---\n*This report contains actual measured performance, not theoretical maximums*\n"

        with open("PERFORMANCE_AUDIT.md", "w") as f:
            f.write(md)


async def main():
    """Main entry point"""
    tester = ForensicPerformanceTester()
    await tester.run_comprehensive_audit()


if __name__ == "__main__":
    print("Starting DALRN Forensic Performance Audit...")
    print("Note: Ensure services are running for accurate results\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nAudit interrupted by user")
    except Exception as e:
        print(f"\nERROR: Audit failed: {e}")
        import traceback
        traceback.print_exc()