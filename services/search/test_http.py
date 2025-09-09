"""
Quick test script for HTTP API
"""
import requests
import numpy as np
import json

def test_http_api():
    """Test HTTP API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing DALRN Vector Search Service HTTP API")
    print("=" * 50)
    
    # Test 1: Build index
    print("\n1. Building index...")
    vectors = np.random.randn(100, 768).tolist()
    try:
        response = requests.post(f"{base_url}/build", json={
            "embeddings": vectors
        }, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("   ERROR: Service not running")
        print("   Start service with: python services/search/service.py")
        return
    
    # Test 2: Query
    print("\n2. Querying index...")
    query = vectors[0]
    response = requests.post(f"{base_url}/query", json={
        "query": query,
        "k": 5
    })
    result = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Top matches: {result['indices']}")
    print(f"   Recall@10: {result['recall_at_10']:.4f}")
    print(f"   Latency: {result['latency_ms']:.2f}ms")
    
    # Test 3: Stats
    print("\n3. Getting stats...")
    response = requests.get(f"{base_url}/stats")
    stats = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Stats: {json.dumps(stats, indent=2)}")
    
    # Test 4: Health
    print("\n4. Health check...")
    response = requests.get(f"{base_url}/healthz")
    health = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Health: {health}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    print("\nNOTE: Make sure the service is running first:")
    print("  python services/search/service.py\n")
    test_http_api()