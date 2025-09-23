# DALRN Vector Search Service

High-performance vector search service using FAISS HNSW indexing with gRPC and HTTP interfaces.

## Features

- **FAISS HNSW Index**: Efficient approximate nearest neighbor search
- **Dual Interface**: Both gRPC and HTTP REST APIs
- **L2 Normalization**: Automatic unit-norm vector normalization
- **Batch Operations**: Support for batch indexing
- **Reweighting**: Optional quantum-inspired iterative reweighting (disabled by default)
- **Thread-Safe**: Concurrent query support with thread-safe operations
- **PoDP Integration**: Emits search receipts for Proof of Data Processing
- **Performance Monitoring**: Built-in metrics and benchmarking

## Configuration

### Index Parameters
- **Dimension**: 768 (default for BERT-like embeddings)
- **M Parameter**: 32 (HNSW connectivity)
- **efSearch**: 128 (search-time accuracy/speed trade-off)
- **efConstruction**: 200 (index-time quality parameter)

### Performance Targets
- **Recall@10**: > 0.95 on synthetic data
- **P95 Latency**: < 600ms on 10k documents
- **Throughput**: > 100 queries/second

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Generate gRPC code from proto files
cd services/search
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/search.proto
```

## Usage

### Starting the Service

#### HTTP Server (default)
```bash
python services/search/service.py
# Server runs on http://localhost:8000
```

#### gRPC Server
```bash
python services/search/service.py grpc
# Server runs on localhost:50051
```

### HTTP API Examples

#### Build Index
```python
import requests
import numpy as np

# Generate sample embeddings
embeddings = np.random.randn(1000, 768).tolist()

# Build index
response = requests.post("http://localhost:8000/build", json={
    "embeddings": embeddings,
    "append": False  # Set to True to append to existing index
})
print(response.json())
# {"total_vectors": 1000, "success": true, "message": "Successfully indexed 1000 vectors"}
```

#### Query Index
```python
# Generate query vector
query_vector = np.random.randn(768).tolist()

# Search for k nearest neighbors
response = requests.post("http://localhost:8000/query", json={
    "query": query_vector,
    "k": 10,
    "reweight_iters": 0  # Set > 0 to enable reweighting
})

result = response.json()
print(f"Top matches: {result['indices']}")
print(f"Scores: {result['scores']}")
print(f"Recall@10: {result['recall_at_10']}")
print(f"Latency: {result['latency_ms']}ms")
```

#### Get Statistics
```python
response = requests.get("http://localhost:8000/stats")
stats = response.json()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Index type: {stats['index_type']}")
print(f"Dimension: {stats['dimension']}")
```

#### Health Check
```python
response = requests.get("http://localhost:8000/healthz")
health = response.json()
print(f"Service healthy: {health['healthy']}")
print(f"Status: {health['status']}")
```

### gRPC API Examples

```python
import grpc
import numpy as np
from services.search import search_pb2, search_pb2_grpc

# Connect to gRPC server
channel = grpc.insecure_channel('localhost:50051')
stub = search_pb2_grpc.SearchServiceStub(channel)

# Build index
vectors = []
for _ in range(100):
    vec = search_pb2.Vector()
    vec.values.extend(np.random.randn(768).tolist())
    vectors.append(vec)

build_request = search_pb2.BuildRequest(embeddings=vectors)
build_response = stub.Build(build_request)
print(f"Indexed {build_response.total_vectors} vectors")

# Query index
query_vec = search_pb2.Vector()
query_vec.values.extend(np.random.randn(768).tolist())

query_request = search_pb2.QueryRequest(
    query_vector=query_vec,
    k=10,
    reweight_iters=0
)
query_response = stub.Query(query_request)
print(f"Found {len(query_response.indices)} matches")
print(f"Recall@10: {query_response.recall_at_10}")
```

## Reweighting Feature

The service includes an optional quantum-inspired reweighting mechanism that can improve result relevance through iterative score adjustments.

### How It Works
1. Convert distances to similarity scores
2. Apply power transformation to amplify high similarities
3. Normalize weights and adjust scores
4. Re-rank results based on adjusted scores
5. Repeat for specified iterations

### Usage
```python
# Enable reweighting with 3 iterations
response = requests.post("http://localhost:8000/query", json={
    "query": query_vector,
    "k": 10,
    "reweight_iters": 3  # Default is 0 (disabled)
})
```

**Note**: Reweighting increases query latency. Use only when improved relevance justifies the performance cost.

## Performance Benchmarking

### Running Benchmarks
```bash
# Run all tests including performance benchmarks
pytest tests/test_search.py -v

# Run only performance benchmarks
pytest tests/test_search.py::TestPerformanceBenchmarks -v

# Generate baseline performance report
python tests/test_search.py
```

### Baseline Report
The baseline report is saved to `reports/baseline.json` and includes:
- Configuration parameters
- Recall@10 metric
- Latency percentiles (P50, P95, P99)
- Test metadata

Example report:
```json
{
  "configuration": {
    "vector_dimension": 768,
    "index_type": "HNSW",
    "m_parameter": 32,
    "ef_search": 128,
    "ef_construction": 200,
    "num_vectors": 10000
  },
  "performance": {
    "recall_at_10": 0.98,
    "latency_p50_ms": 1.2,
    "latency_p95_ms": 3.5,
    "latency_p99_ms": 5.8
  }
}
```

## A/B Testing

The service supports A/B testing for comparing different configurations:

```python
from services.search.service import VectorIndex
import numpy as np

# Configuration A (baseline)
index_a = VectorIndex(dimension=768, m=32)
index_a.index.hnsw.efSearch = 128

# Configuration B (higher accuracy)
index_b = VectorIndex(dimension=768, m=48)
index_b.index.hnsw.efSearch = 256

# Compare performance
vectors = np.random.randn(10000, 768).astype(np.float32)
index_a.build(vectors)
index_b.build(vectors)

# Run queries and compare metrics
for query in queries:
    # Measure for config A
    start_a = time.time()
    results_a = index_a.search(query, k=10)
    latency_a = time.time() - start_a
    
    # Measure for config B
    start_b = time.time()
    results_b = index_b.search(query, k=10)
    latency_b = time.time() - start_b
    
    # Compare recall and latency
```

## PoDP Integration

The service emits Proof of Data Processing receipts for each query:

```json
{
  "receipt_type": "SEARCH_QUERY",
  "query_id": "abc123",
  "timestamp": 1234567890.123,
  "metrics": {
    "query_id": "abc123",
    "k": 10,
    "recall_at_10": 0.98,
    "latency_ms": 2.5,
    "reweight_enabled": false,
    "total_vectors": 10000
  }
}
```

These receipts are logged and can be integrated with the PoDP system for verification.

## Advanced Usage

### Custom Index Parameters
```python
from services.search.service import VectorIndex

# Create custom index
index = VectorIndex(dimension=512, m=16)
index.index.hnsw.efSearch = 64
index.index.hnsw.efConstruction = 100

# Use custom index
embeddings = np.random.randn(1000, 512).astype(np.float32)
index.build(embeddings)
```

### Saving and Loading Index
```python
# Save index to disk
index.save("path/to/index")

# Load index
new_index = VectorIndex(dimension=768)
new_index.load("path/to/index")
```

### Batch Processing
```python
# Process multiple queries in batch
queries = [np.random.randn(768) for _ in range(100)]
results = []

for query in queries:
    indices, distances = index.search(query, k=10)
    results.append({
        "indices": indices[0].tolist(),
        "scores": (1.0 - distances[0]).tolist()
    })
```

## Troubleshooting

### Common Issues

1. **Import Error for gRPC**
   - Ensure proto files are compiled: `python -m grpc_tools.protoc ...`
   - Check that generated files (`search_pb2.py`, `search_pb2_grpc.py`) exist

2. **Low Recall**
   - Increase `efSearch` parameter for better accuracy
   - Ensure vectors are properly normalized
   - Check if index has sufficient vectors

3. **High Latency**
   - Reduce `efSearch` for faster queries
   - Disable reweighting if not needed
   - Consider using GPU version of FAISS for large datasets

4. **Memory Issues**
   - Use memory-mapped indices for very large datasets
   - Implement index sharding for distributed search

## Testing

```bash
# Run unit tests
pytest tests/test_search.py::TestVectorIndex -v

# Run API tests
pytest tests/test_search.py::TestHTTPAPI -v
pytest tests/test_search.py::TestGRPCAPI -v

# Run with coverage
pytest tests/test_search.py --cov=services.search --cov-report=html

# Run specific test
pytest tests/test_search.py::TestVectorIndex::test_build_batch_vectors -v
```

## License

Part of the DALRN project. See main repository for license information.
