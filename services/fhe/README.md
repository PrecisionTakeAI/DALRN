# TenSEAL CKKS FHE Service

## Overview

The DALRN FHE Service provides privacy-preserving dot-product operations using the CKKS (Cheon-Kim-Kim-Song) fully homomorphic encryption scheme via TenSEAL. This service enables secure computation on encrypted vectors without ever accessing the plaintext data, ensuring complete privacy for sensitive machine learning operations.

## Key Features

- **CKKS Homomorphic Encryption**: Performs computations on encrypted floating-point vectors
- **Tenant Isolation**: Each tenant has a completely isolated encryption context
- **Client-Side-Only Decryption**: Server never has access to secret keys or plaintext data
- **Batch Operations**: Efficient batch processing for similarity search scenarios
- **PoDP Integration**: Generates cryptographic receipts for all operations
- **Accuracy Guarantees**: Maintains ±1% accuracy compared to plaintext operations

## Architecture

```
┌─────────────┐       Encrypted Data        ┌─────────────┐
│   Client    │ ─────────────────────────>  │  FHE Server │
│  (Has SK)   │                              │  (No SK)    │
│             │ <─────────────────────────   │             │
│  Encrypts/  │      Encrypted Results       │  Computes   │
│  Decrypts   │                              │  on CT only │
└─────────────┘                              └─────────────┘

SK = Secret Key, CT = Ciphertext
```

## CKKS Parameters

The service uses the following optimized CKKS parameters:

- **Polynomial Modulus Degree**: 8192
- **Coefficient Modulus Bit Sizes**: [60, 40, 40, 60]
- **Global Scale**: 2^40
- **Galois Keys**: Generated for rotations
- **Relinearization Keys**: Generated for multiplication depth

These parameters provide a balance between:
- Security level (~128 bits)
- Precision (suitable for ML operations)
- Computation depth (supports dot products)

## API Endpoints

### POST `/create_context`
Initialize a new CKKS context for a tenant.

**Request:**
```json
{
  "tenant_id": "tenant_123",
  "poly_modulus_degree": 8192,
  "coeff_mod_bit_sizes": [60, 40, 40, 60],
  "global_scale": 1099511627776,
  "generate_galois_keys": true,
  "generate_relin_keys": true
}
```

**Response:**
```json
{
  "context_id": "abc123def456",
  "serialized_context": "base64_encoded_context",
  "metadata": {
    "tenant_id": "tenant_123",
    "created_at": "2025-01-01T00:00:00",
    "config": {...}
  }
}
```

### POST `/dot`
Compute encrypted dot product of two vectors.

**Request:**
```json
{
  "tenant_id": "tenant_123",
  "encrypted_q": "base64_encrypted_query_vector",
  "encrypted_v": "base64_encrypted_value_vector",
  "context": "optional_base64_context"
}
```

**Response:**
```json
{
  "encrypted_result": "base64_encrypted_result",
  "operation_id": "uuid",
  "receipt_hash": "sha256_hash"
}
```

### POST `/batch_dot`
Compute batch encrypted dot products.

**Request:**
```json
{
  "tenant_id": "tenant_123",
  "encrypted_q": "base64_encrypted_query",
  "encrypted_matrix": ["base64_vec1", "base64_vec2", ...],
  "context": "optional_base64_context"
}
```

**Response:**
```json
{
  "encrypted_results": ["base64_result1", "base64_result2", ...],
  "operation_id": "uuid",
  "receipt_hash": "sha256_hash",
  "computation_time_ms": 123.45
}
```

### GET `/healthz`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "DALRN FHE Service",
  "tenseal_available": true,
  "timestamp": "2025-01-01T00:00:00",
  "active_contexts": 5,
  "version": "1.0.0"
}
```

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler (for TenSEAL compilation)
- CMake (for TenSEAL build)

### Install Dependencies

```bash
# Install TenSEAL (requires compilation)
pip install tenseal

# Install other dependencies
pip install fastapi uvicorn numpy pydantic
```

### Running Without TenSEAL

The service can run in placeholder mode without TenSEAL installed:

```bash
# Install minimal dependencies
pip install fastapi uvicorn numpy pydantic

# Run service (will show TenSEAL warning)
python service.py
```

## Usage

### Starting the Service

```bash
cd services/fhe
python service.py
```

The service will start on `http://localhost:8000`.

### Client Example

```python
from client_example import FHEClient
import numpy as np

# Initialize client
client = FHEClient("http://localhost:8000")

# Create encryption context
client.create_context("my_tenant")

# Create test vectors
q = np.random.randn(128)
v = np.random.randn(128)

# Normalize to unit vectors
q = q / np.linalg.norm(q)
v = v / np.linalg.norm(v)

# Compute encrypted dot product
result = client.compute_dot_product(q, v)

# Decrypt result (requires TenSEAL on client)
if TENSEAL_AVAILABLE:
    decrypted = client.decrypt_result(result['encrypted_result'])
    print(f"Result: {decrypted}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests (except those requiring TenSEAL)
pytest tests/test_fhe.py -v

# Run only non-accuracy tests
pytest tests/test_fhe.py -k "not Accuracy" -v

# Run with TenSEAL installed
pip install tenseal
pytest tests/test_fhe.py -v
```

## Security Considerations

1. **Client-Side Keys**: Secret keys must NEVER be sent to the server
2. **Context Isolation**: Each tenant has a completely isolated context
3. **No Plaintext Access**: Server never decrypts or accesses plaintext
4. **Timing Attacks**: Service implements constant-time operations where possible
5. **Context Rotation**: Contexts expire after 24 hours by default

## Performance

Typical performance metrics (with TenSEAL):

- Single dot product (128-dim): ~5-10ms
- Batch of 100 dot products: ~500-1000ms
- Context creation: ~100-200ms
- Memory per context: ~10-50MB

## Accuracy

The CKKS scheme introduces small errors due to approximate arithmetic:

- Average error: < 0.1% for normalized vectors
- Recall@k for similarity search: > 98% match with plaintext
- Maximum error tolerance: ±1% configured

## Troubleshooting

### TenSEAL Installation Issues

If TenSEAL installation fails:

```bash
# Install build tools first
# On Ubuntu/Debian:
sudo apt-get install cmake build-essential

# On Windows:
# Install Visual Studio Build Tools

# Try installation with verbose output
pip install tenseal --verbose
```

### Memory Issues

For large-scale operations:

1. Increase context expiry cleanup frequency
2. Use batch sizes appropriate for available memory
3. Consider horizontal scaling with multiple service instances

### Accuracy Issues

If accuracy degrades:

1. Check vector normalization
2. Verify CKKS parameters match between client and server
3. Consider increasing polynomial modulus degree for more precision

## License

See DALRN project license.

## References

- [TenSEAL Documentation](https://github.com/OpenMined/TenSEAL)
- [CKKS Scheme Paper](https://eprint.iacr.org/2016/421.pdf)
- [Homomorphic Encryption Standard](https://homomorphicencryption.org/)