# FHE Service Setup Guide

## Overview

The DALRN FHE (Fully Homomorphic Encryption) Service uses **TenSEAL**, a library for doing homomorphic encryption operations on tensors, with a focus on the **CKKS scheme** for privacy-preserving machine learning operations. This guide provides comprehensive setup instructions for getting the FHE service running with real encryption capabilities.

## Prerequisites

- Python 3.11 or higher
- C++ compiler (GCC 9+ or Clang 10+)
- CMake 3.16 or higher
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended for compilation

## Quick Start

### Option 1: Local Installation

1. **Install System Dependencies**

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       build-essential \
       cmake \
       libssl-dev \
       libprotobuf-dev \
       protobuf-compiler \
       libboost-all-dev \
       libomp-dev
   ```

   **macOS:**
   ```bash
   brew install cmake protobuf boost libomp
   ```

   **Windows:**
   - Install Visual Studio 2019 or later with C++ development tools
   - Install CMake from https://cmake.org/download/
   - Install vcpkg and use it to install dependencies:
     ```powershell
     vcpkg install protobuf boost-all openssl
     ```

2. **Install Python Dependencies**
   ```bash
   # Navigate to the DALRN directory
   cd DALRN

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate

   # Upgrade pip and install wheel
   pip install --upgrade pip setuptools wheel

   # Install TenSEAL (this may take 10-30 minutes)
   pip install tenseal==0.3.14

   # Install other dependencies
   pip install -r services/fhe/requirements.txt
   ```

3. **Verify Installation**
   ```python
   # Test TenSEAL installation
   python -c "import tenseal as ts; print(f'TenSEAL version: {ts.__version__}')"
   ```

4. **Run the FHE Service**
   ```bash
   cd services/fhe
   python service.py
   # Or using uvicorn:
   uvicorn service:app --host 0.0.0.0 --port 8200 --reload
   ```

### Option 2: Docker Installation (Recommended)

1. **Build the Docker Image**
   ```bash
   cd services/fhe
   docker build -t dalrn-fhe:latest .
   ```

   Note: The build process may take 15-45 minutes due to TenSEAL compilation.

2. **Run the Container**
   ```bash
   docker run -d \
     --name dalrn-fhe \
     -p 8200:8200 \
     -e PYTHONUNBUFFERED=1 \
     dalrn-fhe:latest
   ```

3. **Verify Service Health**
   ```bash
   curl http://localhost:8200/healthz
   ```

### Option 3: Docker Compose (Full Stack)

```bash
# From the DALRN root directory
docker-compose up -d fhe
```

## Configuration

### CKKS Parameters

The service uses the following default CKKS parameters optimized for dot-product operations:

```python
# In services/fhe/service.py
CKKS_CONFIG = {
    "poly_modulus_degree": 8192,      # Polynomial degree (affects security and capacity)
    "coeff_mod_bit_sizes": [60, 40, 40, 60],  # Coefficient modulus chain
    "global_scale": 2**40,             # Scaling factor for fixed-point arithmetic
    "generate_galois_keys": True,      # Enable rotations
    "generate_relin_keys": True        # Enable relinearization
}
```

These parameters provide:
- **Security Level**: 128-bit
- **Precision**: ~10-12 decimal digits
- **Max Multiplicative Depth**: 3-4 levels
- **Vector Dimension Support**: Up to 4096 elements

### Performance Tuning

For production environments, consider:

1. **Adjust Parameters Based on Use Case:**
   ```python
   # For higher precision but slower operations:
   config = CKKSConfig(
       poly_modulus_degree=16384,
       coeff_mod_bit_sizes=[60, 50, 50, 50, 60],
       global_scale=2**50
   )

   # For faster operations but lower precision:
   config = CKKSConfig(
       poly_modulus_degree=4096,
       coeff_mod_bit_sizes=[60, 40, 60],
       global_scale=2**30
   )
   ```

2. **Enable Multi-threading:**
   ```python
   import tenseal as ts
   ts.set_num_threads(8)  # Use 8 threads for operations
   ```

## Troubleshooting

### Common Installation Issues

#### 1. TenSEAL Installation Fails

**Error:** `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution (Windows):**
```powershell
# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools
# Then restart and retry pip install
```

**Error:** `CMake Error: Could not find a package configuration file`

**Solution:**
```bash
# Ensure CMake is properly installed
cmake --version  # Should be 3.16+

# On Linux, you may need to add CMake to PATH
export PATH="/usr/local/bin:$PATH"
```

#### 2. Memory Issues During Compilation

**Error:** `c++: fatal error: Killed signal terminated program cc1plus`

**Solution:**
```bash
# Increase swap space temporarily
sudo dd if=/dev/zero of=/swapfile bs=1G count=8
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Then retry installation
pip install tenseal==0.3.14

# Clean up swap after installation
sudo swapoff /swapfile
sudo rm /swapfile
```

#### 3. Import Error After Installation

**Error:** `ImportError: libSEAL.so.3.7: cannot open shared object file`

**Solution:**
```bash
# On Linux, update library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc

# On macOS
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/lib
```

### Platform-Specific Notes

#### Windows

- Use Python 3.11 (3.12+ may have compatibility issues)
- Install from Administrator command prompt
- May need to disable Windows Defender during compilation
- Consider using WSL2 for easier installation

#### macOS (Apple Silicon)

```bash
# For M1/M2 Macs, use Rosetta for better compatibility
arch -x86_64 pip install tenseal==0.3.14

# Or build from source with ARM64 support
git clone https://github.com/OpenMined/TenSEAL.git
cd TenSEAL
pip install .
```

#### Linux (Various Distributions)

**Alpine Linux:**
```bash
apk add --no-cache \
    g++ gcc cmake make \
    libressl-dev protobuf-dev \
    boost-dev openblas-dev
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 openssl-devel \
    protobuf-devel boost-devel
```

## Testing the Installation

### Basic Functionality Test

```python
"""test_tenseal_basic.py"""
import tenseal as ts
import numpy as np

# Create context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# Create and encrypt vectors
v1 = np.array([1.0, 2.0, 3.0, 4.0])
v2 = np.array([5.0, 6.0, 7.0, 8.0])

enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

# Perform encrypted dot product
enc_result = (enc_v1 * enc_v2).sum()

# Decrypt and verify
result = enc_result.decrypt()[0]
expected = np.dot(v1, v2)

print(f"Encrypted result: {result:.6f}")
print(f"Expected result: {expected:.6f}")
print(f"Error: {abs(result - expected):.10f}")

assert abs(result - expected) < 0.001, "Accuracy test failed"
print("✅ TenSEAL is working correctly!")
```

### API Test

```bash
# Start the FHE service
cd services/fhe
python service.py &

# Test the API
curl -X POST http://localhost:8200/create_context \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "test_tenant"}'

# Check health with TenSEAL status
curl http://localhost:8200/healthz | jq '.tenseal_available'
```

## Performance Benchmarks

Expected performance on typical hardware:

| Operation | Vector Size | Time (ms) | Hardware |
|-----------|------------|-----------|----------|
| Context Creation | - | 50-100 | Intel i7-9750H |
| Vector Encryption | 128 | 5-10 | Intel i7-9750H |
| Dot Product | 128 | 15-30 | Intel i7-9750H |
| Batch (100 vectors) | 128 | 1500-3000 | Intel i7-9750H |

With GPU acceleration (if available):
- 2-5x speedup for large batch operations
- Requires CUDA-enabled TenSEAL build

## Security Considerations

1. **Never expose secret keys**: The service is designed for client-side-only decryption
2. **Context isolation**: Each tenant must have a completely isolated context
3. **Parameter selection**: Use recommended parameters for 128-bit security
4. **Key rotation**: Implement regular key rotation in production
5. **Audit logging**: Enable comprehensive logging for all operations

## Integration with DALRN

The FHE service integrates with other DALRN components:

1. **Gateway Service**: Routes encrypted requests to FHE service
2. **Search Service**: Uses FHE for privacy-preserving similarity search
3. **PoDP**: Generates cryptographic receipts for all operations
4. **Chain Service**: Anchors computation proofs on blockchain

## Advanced Configuration

### Multi-GPU Support

```python
# Enable GPU acceleration if available
import tenseal as ts
if ts.is_cuda_available():
    ts.set_device("cuda:0")
    print("GPU acceleration enabled")
```

### Custom Encryption Schemes

```python
# BFV scheme for integer operations
context_bfv = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=4096,
    plain_modulus=786433
)

# BGV scheme (if supported in your TenSEAL version)
# context_bgv = ts.context(ts.SCHEME_TYPE.BGV, ...)
```

## Monitoring and Observability

The FHE service exposes metrics at `/metrics`:

```bash
# Prometheus metrics
curl http://localhost:8200/metrics

# Key metrics to monitor:
# - fhe_context_creation_total
# - fhe_encryption_duration_seconds
# - fhe_computation_duration_seconds
# - fhe_active_contexts_gauge
```

## Support and Resources

- **TenSEAL Documentation**: https://github.com/OpenMined/TenSEAL
- **SEAL Library**: https://github.com/microsoft/SEAL
- **CKKS Scheme Paper**: https://eprint.iacr.org/2016/421.pdf
- **DALRN Issues**: File issues in the DALRN repository

## Appendix: Building TenSEAL from Source

If the pip installation fails, you can build from source:

```bash
# Clone TenSEAL
git clone https://github.com/OpenMined/TenSEAL.git
cd TenSEAL

# Checkout stable version
git checkout v0.3.14

# Build and install
pip install -r requirements_dev.txt
python setup.py install

# Run tests to verify
pytest tests/
```

---

**Note**: This setup guide assumes you're setting up the FHE service as part of the larger DALRN system. For standalone deployment, adjust paths and dependencies accordingly.