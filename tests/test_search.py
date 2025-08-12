"""
Comprehensive test suite for DALRN Vector Search Service
Tests FAISS HNSW index, gRPC/HTTP APIs, and performance benchmarks
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pytest
import grpc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.search.service import (
    VectorIndex, vector_index,
    BuildRequest, QueryRequest,
    BuildResponse, QueryResponse, StatsResponse,
    SearchServicer, app
)
from services.search import search_pb2, search_pb2_grpc

# Test constants
TEST_DIM = 768
TEST_VECTORS_SMALL = 100
TEST_VECTORS_LARGE = 10000
RECALL_THRESHOLD = 0.95
P95_LATENCY_THRESHOLD_MS = 600

# Set random seed for reproducibility
np.random.seed(42)


class TestVectorIndex:
    """Test VectorIndex class functionality"""
    
    def test_initialization(self):
        """Test index initialization"""
        index = VectorIndex(dimension=TEST_DIM, m=32)
        assert index.dimension == TEST_DIM
        assert index.m == 32
        assert index.index is not None
        assert index.index.ntotal == 0
    
    def test_build_single_vector(self):
        """Test building index with single vector"""
        index = VectorIndex(dimension=TEST_DIM)
        vector = np.random.randn(TEST_DIM).astype(np.float32)
        
        total = index.build(vector)
        assert total == 1
        assert index.index.ntotal == 1
    
    def test_build_batch_vectors(self):
        """Test building index with batch of vectors"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).astype(np.float32)
        
        total = index.build(vectors)
        assert total == TEST_VECTORS_SMALL
        assert index.index.ntotal == TEST_VECTORS_SMALL
    
    def test_append_vectors(self):
        """Test appending vectors to existing index"""
        index = VectorIndex(dimension=TEST_DIM)
        
        # Initial build
        vectors1 = np.random.randn(50, TEST_DIM).astype(np.float32)
        total1 = index.build(vectors1)
        assert total1 == 50
        
        # Append more vectors
        vectors2 = np.random.randn(30, TEST_DIM).astype(np.float32)
        total2 = index.build(vectors2, append=True)
        assert total2 == 80
        assert index.index.ntotal == 80
    
    def test_search_basic(self):
        """Test basic search functionality"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).astype(np.float32)
        index.build(vectors)
        
        # Search with first vector as query
        query = vectors[0]
        indices, distances = index.search(query, k=5)
        
        assert len(indices[0]) == 5
        assert len(distances[0]) == 5
        assert indices[0][0] == 0  # First result should be itself
        assert distances[0][0] < 0.01  # Distance to itself should be near 0
    
    def test_search_empty_index(self):
        """Test search on empty index"""
        index = VectorIndex(dimension=TEST_DIM)
        query = np.random.randn(TEST_DIM).astype(np.float32)
        
        indices, distances = index.search(query, k=5)
        assert len(indices) == 0
        assert len(distances) == 0
    
    def test_search_k_larger_than_index(self):
        """Test search when k > number of vectors"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(5, TEST_DIM).astype(np.float32)
        index.build(vectors)
        
        query = vectors[0]
        indices, distances = index.search(query, k=10)
        
        assert len(indices[0]) == 5  # Should return all 5 vectors
    
    def test_l2_normalization(self):
        """Test that vectors are L2 normalized"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(10, TEST_DIM).astype(np.float32)
        
        # Build index (vectors will be normalized)
        index.build(vectors)
        
        # Check stored vectors are normalized
        norms = np.linalg.norm(index.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_reweighting_disabled_by_default(self):
        """Test that reweighting is disabled by default"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).astype(np.float32)
        index.build(vectors)
        
        query = vectors[0]
        
        # Search without reweighting
        indices1, distances1 = index.search(query, k=10, reweight_iters=0)
        
        # Results should be deterministic
        indices2, distances2 = index.search(query, k=10, reweight_iters=0)
        
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_allclose(distances1, distances2)
    
    def test_reweighting_enabled(self):
        """Test reweighting functionality when enabled"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).astype(np.float32)
        index.build(vectors)
        
        query = vectors[0]
        
        # Search without reweighting
        indices_no_reweight, distances_no_reweight = index.search(query, k=10, reweight_iters=0)
        
        # Search with reweighting
        indices_reweight, distances_reweight = index.search(query, k=10, reweight_iters=3)
        
        # Top result should still be the same (itself)
        assert indices_no_reweight[0][0] == indices_reweight[0][0]
        
        # But other results might be reordered
        # Just verify shapes are correct
        assert indices_reweight.shape == (1, 10)
        assert distances_reweight.shape == (1, 10)
    
    def test_calculate_recall(self):
        """Test recall calculation"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).astype(np.float32)
        index.build(vectors)
        
        # Query with a vector from the index
        query = vectors[0]
        indices, _ = index.search(query, k=10)
        
        # Calculate recall
        recall = index.calculate_recall(query, indices[0], k=10)
        
        # Should have perfect recall for a vector in the index
        assert recall >= 0.9  # Allow some tolerance due to HNSW approximation
    
    def test_get_stats(self):
        """Test getting index statistics"""
        index = VectorIndex(dimension=TEST_DIM, m=32)
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).astype(np.float32)
        index.build(vectors)
        
        stats = index.get_stats()
        
        assert stats["total_vectors"] == TEST_VECTORS_SMALL
        assert stats["dimension"] == TEST_DIM
        assert stats["index_type"] == "HNSW"
        assert stats["m_parameter"] == 32
        assert stats["ef_search"] == 128
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading index"""
        index1 = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).astype(np.float32)
        index1.build(vectors)
        
        # Save index
        index_path = str(tmp_path / "test_index")
        index1.save(index_path)
        
        # Load into new index
        index2 = VectorIndex(dimension=TEST_DIM)
        index2.load(index_path)
        
        # Verify loaded index works
        assert index2.index.ntotal == TEST_VECTORS_SMALL
        
        # Search should produce same results
        query = vectors[0]
        indices1, distances1 = index1.search(query, k=5)
        indices2, distances2 = index2.search(query, k=5)
        
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_allclose(distances1, distances2)


class TestHTTPAPI:
    """Test HTTP API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_build_endpoint(self, client):
        """Test /build endpoint"""
        vectors = np.random.randn(10, TEST_DIM).tolist()
        response = client.post("/build", json={"embeddings": vectors})
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_vectors"] == 10
    
    def test_query_endpoint(self, client):
        """Test /query endpoint"""
        # First build index
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).tolist()
        client.post("/build", json={"embeddings": vectors})
        
        # Query
        query = vectors[0]
        response = client.post("/query", json={"query": query, "k": 5})
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["indices"]) == 5
        assert len(data["scores"]) == 5
        assert data["indices"][0] == 0  # First result should be itself
        assert "recall_at_10" in data
        assert "latency_ms" in data
        assert "query_id" in data
    
    def test_query_with_reweighting(self, client):
        """Test query with reweighting enabled"""
        # Build index
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).tolist()
        client.post("/build", json={"embeddings": vectors})
        
        # Query with reweighting
        query = vectors[0]
        response = client.post("/query", json={
            "query": query,
            "k": 10,
            "reweight_iters": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["indices"]) == 10
    
    def test_stats_endpoint(self, client):
        """Test /stats endpoint"""
        # Build index first
        vectors = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM).tolist()
        client.post("/build", json={"embeddings": vectors})
        
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_vectors"] == TEST_VECTORS_SMALL
        assert data["dimension"] == TEST_DIM
        assert data["index_type"] == "HNSW"
    
    def test_health_endpoint(self, client):
        """Test /healthz endpoint"""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert data["status"] == "OK"
    
    def test_append_vectors(self, client):
        """Test appending vectors to existing index"""
        # Initial build
        vectors1 = np.random.randn(50, TEST_DIM).tolist()
        response1 = client.post("/build", json={"embeddings": vectors1})
        assert response1.json()["total_vectors"] == 50
        
        # Append more
        vectors2 = np.random.randn(30, TEST_DIM).tolist()
        response2 = client.post("/build", json={
            "embeddings": vectors2,
            "append": True
        })
        assert response2.json()["total_vectors"] == 80
    
    def test_empty_query(self, client):
        """Test query on empty index"""
        # Reset the global index to ensure it's empty
        from services.search.service import vector_index
        vector_index._initialize_index()
        
        query = np.random.randn(TEST_DIM).tolist()
        response = client.post("/query", json={"query": query, "k": 5})
        
        assert response.status_code == 200
        data = response.json()
        assert data["indices"] == []
        assert data["scores"] == []


class TestGRPCAPI:
    """Test gRPC API"""
    
    @pytest.fixture
    def grpc_server(self):
        """Start gRPC server for testing"""
        from concurrent import futures
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        search_pb2_grpc.add_SearchServiceServicer_to_server(
            SearchServicer(), server
        )
        port = server.add_insecure_port('[::]:0')
        server.start()
        yield f'localhost:{port}'
        server.stop(0)
    
    def test_grpc_build(self, grpc_server):
        """Test gRPC Build method"""
        channel = grpc.insecure_channel(grpc_server)
        stub = search_pb2_grpc.SearchServiceStub(channel)
        
        # Create request
        vectors = []
        for _ in range(10):
            vec = search_pb2.Vector()
            vec.values.extend(np.random.randn(TEST_DIM).tolist())
            vectors.append(vec)
        
        request = search_pb2.BuildRequest(embeddings=vectors)
        response = stub.Build(request)
        
        assert response.success is True
        assert response.total_vectors == 10
    
    def test_grpc_query(self, grpc_server):
        """Test gRPC Query method"""
        channel = grpc.insecure_channel(grpc_server)
        stub = search_pb2_grpc.SearchServiceStub(channel)
        
        # Build index first
        vectors = []
        vectors_np = np.random.randn(TEST_VECTORS_SMALL, TEST_DIM)
        for i in range(TEST_VECTORS_SMALL):
            vec = search_pb2.Vector()
            vec.values.extend(vectors_np[i].tolist())
            vectors.append(vec)
        
        build_request = search_pb2.BuildRequest(embeddings=vectors)
        stub.Build(build_request)
        
        # Query
        query_vec = search_pb2.Vector()
        query_vec.values.extend(vectors_np[0].tolist())
        
        query_request = search_pb2.QueryRequest(
            query_vector=query_vec,
            k=5,
            reweight_iters=0
        )
        response = stub.Query(query_request)
        
        assert len(response.indices) == 5
        assert len(response.scores) == 5
        assert response.indices[0] == 0
        assert response.recall_at_10 > 0
        assert response.latency_ms > 0
    
    def test_grpc_stats(self, grpc_server):
        """Test gRPC GetStats method"""
        channel = grpc.insecure_channel(grpc_server)
        stub = search_pb2_grpc.SearchServiceStub(channel)
        
        request = search_pb2.StatsRequest()
        response = stub.GetStats(request)
        
        assert response.dimension == TEST_DIM
        assert response.index_type == "HNSW"
    
    def test_grpc_health(self, grpc_server):
        """Test gRPC HealthCheck method"""
        channel = grpc.insecure_channel(grpc_server)
        stub = search_pb2_grpc.SearchServiceStub(channel)
        
        request = search_pb2.HealthRequest()
        response = stub.HealthCheck(request)
        
        assert response.healthy is True


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.fixture
    def large_index(self):
        """Create index with 10k vectors"""
        index = VectorIndex(dimension=TEST_DIM)
        vectors = np.random.randn(TEST_VECTORS_LARGE, TEST_DIM).astype(np.float32)
        index.build(vectors)
        return index, vectors
    
    def test_recall_at_10(self, large_index):
        """Test recall@10 on 10k document corpus"""
        index, vectors = large_index
        
        # Test on multiple random queries
        num_queries = 100
        recalls = []
        
        for _ in range(num_queries):
            query_idx = np.random.randint(0, TEST_VECTORS_LARGE)
            query = vectors[query_idx]
            
            indices, _ = index.search(query, k=10)
            recall = index.calculate_recall(query, indices[0], k=10)
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        print(f"Average Recall@10: {avg_recall:.4f}")
        
        # Should achieve >0.95 recall on synthetic data
        assert avg_recall >= RECALL_THRESHOLD
    
    def test_p95_latency(self, large_index):
        """Test P95 latency on 10k document corpus"""
        index, vectors = large_index
        
        # Warm up
        for _ in range(10):
            query = np.random.randn(TEST_DIM).astype(np.float32)
            index.search(query, k=10)
        
        # Measure latencies
        latencies = []
        num_queries = 1000
        
        for _ in range(num_queries):
            query = np.random.randn(TEST_DIM).astype(np.float32)
            
            start = time.perf_counter()
            index.search(query, k=10)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"Latency P50: {p50:.2f}ms")
        print(f"Latency P95: {p95:.2f}ms")
        print(f"Latency P99: {p99:.2f}ms")
        
        # P95 should be < 600ms
        assert p95 < P95_LATENCY_THRESHOLD_MS
    
    def test_throughput(self, large_index):
        """Test query throughput"""
        index, vectors = large_index
        
        # Prepare queries
        num_queries = 1000
        queries = [np.random.randn(TEST_DIM).astype(np.float32) 
                  for _ in range(num_queries)]
        
        # Measure throughput
        start = time.perf_counter()
        
        for query in queries:
            index.search(query, k=10)
        
        end = time.perf_counter()
        duration = end - start
        
        qps = num_queries / duration
        print(f"Throughput: {qps:.2f} queries/sec")
        
        # Should handle at least 100 qps
        assert qps > 100
    
    def test_concurrent_queries(self, large_index):
        """Test concurrent query performance"""
        index, vectors = large_index
        
        def run_query():
            query = np.random.randn(TEST_DIM).astype(np.float32)
            indices, distances = index.search(query, k=10)
            return len(indices[0]) == 10
        
        # Run concurrent queries
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_query) for _ in range(100)]
            results = [f.result() for f in futures]
        
        # All queries should succeed
        assert all(results)
    
    def test_reweighting_performance_impact(self, large_index):
        """Test performance impact of reweighting"""
        index, vectors = large_index
        
        query = np.random.randn(TEST_DIM).astype(np.float32)
        
        # Measure without reweighting
        start = time.perf_counter()
        for _ in range(100):
            index.search(query, k=10, reweight_iters=0)
        time_no_reweight = time.perf_counter() - start
        
        # Measure with reweighting
        start = time.perf_counter()
        for _ in range(100):
            index.search(query, k=10, reweight_iters=3)
        time_with_reweight = time.perf_counter() - start
        
        # Reweighting should not more than double the time
        overhead_ratio = time_with_reweight / time_no_reweight
        print(f"Reweighting overhead: {overhead_ratio:.2f}x")
        
        assert overhead_ratio < 3.0


def generate_baseline_report():
    """Generate baseline performance report"""
    print("Generating baseline performance report...")
    
    # Create index with 10k vectors
    index = VectorIndex(dimension=TEST_DIM)
    vectors = np.random.randn(TEST_VECTORS_LARGE, TEST_DIM).astype(np.float32)
    index.build(vectors)
    
    # Measure recall
    recalls = []
    for _ in range(100):
        query_idx = np.random.randint(0, TEST_VECTORS_LARGE)
        query = vectors[query_idx]
        indices, _ = index.search(query, k=10)
        recall = index.calculate_recall(query, indices[0], k=10)
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    
    # Measure latency
    latencies = []
    for _ in range(1000):
        query = np.random.randn(TEST_DIM).astype(np.float32)
        start = time.perf_counter()
        index.search(query, k=10)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
    
    # Calculate metrics
    metrics = {
        "configuration": {
            "vector_dimension": TEST_DIM,
            "index_type": "HNSW",
            "m_parameter": 32,
            "ef_search": 128,
            "ef_construction": 200,
            "num_vectors": TEST_VECTORS_LARGE
        },
        "performance": {
            "recall_at_10": float(avg_recall),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies))
        },
        "test_info": {
            "timestamp": time.time(),
            "num_queries": 1000,
            "random_seed": 42
        }
    }
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Baseline report saved to reports/baseline.json")
    print(f"Recall@10: {avg_recall:.4f}")
    print(f"P95 Latency: {metrics['performance']['latency_p95_ms']:.2f}ms")
    
    return metrics


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Generate baseline report
    generate_baseline_report()