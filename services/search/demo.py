"""
Demo script for DALRN Vector Search Service
Shows how to use the service with both HTTP and direct Python API
"""

import numpy as np
import requests
import json
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from services.search.service import VectorIndex

def demo_direct_api():
    """Demo using direct Python API"""
    print("=" * 60)
    print("DALRN Vector Search Service - Direct API Demo")
    print("=" * 60)
    
    # Create index
    print("\n1. Creating HNSW index...")
    index = VectorIndex(dimension=768, m=32)
    
    # Generate synthetic data
    print("2. Generating 1000 synthetic vectors (768 dimensions)...")
    vectors = np.random.randn(1000, 768).astype(np.float32)
    
    # Build index
    print("3. Building index...")
    total = index.build(vectors)
    print(f"   - Indexed {total} vectors")
    
    # Perform search
    print("\n4. Performing search...")
    query = vectors[0]  # Use first vector as query
    
    # Search without reweighting
    start = time.time()
    indices, distances = index.search(query, k=10)
    latency = (time.time() - start) * 1000
    
    print(f"   - Found {len(indices[0])} nearest neighbors")
    print(f"   - Top 5 matches: {indices[0][:5].tolist()}")
    print(f"   - Latency: {latency:.2f}ms")
    
    # Calculate recall
    recall = index.calculate_recall(query, indices[0], k=10)
    print(f"   - Recall@10: {recall:.4f}")
    
    # Search with reweighting
    print("\n5. Testing reweighting feature...")
    start = time.time()
    indices_rw, distances_rw = index.search(query, k=10, reweight_iters=3)
    latency_rw = (time.time() - start) * 1000
    
    print(f"   - Reweighting latency: {latency_rw:.2f}ms")
    print(f"   - Overhead: {(latency_rw/latency - 1)*100:.1f}%")
    
    # Get stats
    print("\n6. Index statistics:")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    return index, vectors


def demo_http_api(base_url="http://localhost:8000"):
    """Demo using HTTP REST API"""
    print("\n" + "=" * 60)
    print("DALRN Vector Search Service - HTTP API Demo")
    print("=" * 60)
    
    # Check health
    print("\n1. Checking service health...")
    try:
        response = requests.get(f"{base_url}/healthz")
        health = response.json()
        print(f"   - Service status: {health['status']}")
    except requests.exceptions.ConnectionError:
        print("   - Service not running. Start with: python services/search/service.py")
        return
    
    # Build index
    print("\n2. Building index via HTTP...")
    vectors = np.random.randn(500, 768).tolist()
    response = requests.post(f"{base_url}/build", json={
        "embeddings": vectors,
        "append": False
    })
    result = response.json()
    print(f"   - {result['message']}")
    
    # Query index
    print("\n3. Querying index...")
    query = vectors[0]
    response = requests.post(f"{base_url}/query", json={
        "query": query,
        "k": 5,
        "reweight_iters": 0
    })
    result = response.json()
    print(f"   - Top matches: {result['indices']}")
    print(f"   - Scores: [" + ", ".join(f"{s:.3f}" for s in result['scores']) + "]")
    print(f"   - Recall@10: {result['recall_at_10']:.4f}")
    print(f"   - Latency: {result['latency_ms']:.2f}ms")
    print(f"   - Query ID: {result['query_id']}")
    
    # Get stats
    print("\n4. Getting index statistics...")
    response = requests.get(f"{base_url}/stats")
    stats = response.json()
    print(f"   - Total vectors: {stats['total_vectors']}")
    print(f"   - Index type: {stats['index_type']}")
    print(f"   - M parameter: {stats['m_parameter']}")
    print(f"   - efSearch: {stats['ef_search']}")


def demo_performance_test():
    """Demo performance testing"""
    print("\n" + "=" * 60)
    print("DALRN Vector Search Service - Performance Test")
    print("=" * 60)
    
    # Create large index
    print("\n1. Creating index with 10,000 vectors...")
    index = VectorIndex(dimension=768, m=32)
    vectors = np.random.randn(10000, 768).astype(np.float32)
    index.build(vectors)
    
    # Benchmark queries
    print("\n2. Running performance benchmark (100 queries)...")
    latencies = []
    recalls = []
    
    for i in range(100):
        query = np.random.randn(768).astype(np.float32)
        
        start = time.perf_counter()
        indices, _ = index.search(query, k=10)
        latency = (time.perf_counter() - start) * 1000
        
        recall = index.calculate_recall(query, indices[0], k=10)
        
        latencies.append(latency)
        recalls.append(recall)
        
        if (i + 1) % 20 == 0:
            print(f"   - Processed {i + 1} queries...")
    
    # Calculate metrics
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg_recall = np.mean(recalls)
    
    print("\n3. Performance Results:")
    print(f"   - Average Recall@10: {avg_recall:.4f}")
    print(f"   - Latency P50: {p50:.2f}ms")
    print(f"   - Latency P95: {p95:.2f}ms")
    print(f"   - Latency P99: {p99:.2f}ms")
    print(f"   - Throughput: {1000/np.mean(latencies):.1f} queries/sec")
    
    # Check if meets requirements
    print("\n4. Requirement Validation:")
    print(f"   - Recall@10 > 0.95: {'PASS' if avg_recall > 0.95 else 'FAIL'} ({avg_recall:.4f})")
    print(f"   - P95 < 600ms: {'PASS' if p95 < 600 else 'FAIL'} ({p95:.2f}ms)")


def main():
    """Run all demos"""
    # Direct API demo
    index, vectors = demo_direct_api()
    
    # HTTP API demo (requires server running)
    demo_http_api()
    
    # Performance test
    demo_performance_test()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()