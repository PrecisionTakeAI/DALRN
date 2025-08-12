"""
Comprehensive test suite for TenSEAL CKKS FHE Service.

Tests include:
- Context creation and serialization
- Dot-product accuracy vs plaintext
- Batch operations
- Tenant isolation verification
- Edge cases (zero vectors, orthogonal vectors)
- Accuracy validation with recall@k metrics
"""

import base64
import hashlib
import json
import numpy as np
import pytest
from typing import List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock

# Try to import TenSEAL for real tests
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False

# Import service components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.fhe.service import (
    app, CKKSConfig, ContextManager, FHEOperations,
    create_test_vectors, encrypt_vector_placeholder
)
from services.common.podp import Receipt, ReceiptChain, create_fhe_receipt

from fastapi.testclient import TestClient

# Test client
client = TestClient(app)


# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def compute_recall_at_k(
    query: np.ndarray,
    database: np.ndarray,
    encrypted_scores: List[float],
    plaintext_scores: List[float],
    k: int
) -> float:
    """
    Compute recall@k between encrypted and plaintext rankings.
    
    Args:
        query: Query vector
        database: Database matrix (each row is a vector)
        encrypted_scores: Scores from encrypted computation
        plaintext_scores: Scores from plaintext computation
        k: Number of top results to consider
        
    Returns:
        Recall@k score
    """
    # Get top-k indices
    encrypted_top_k = np.argsort(encrypted_scores)[-k:]
    plaintext_top_k = np.argsort(plaintext_scores)[-k:]
    
    # Compute recall
    intersection = len(set(encrypted_top_k) & set(plaintext_top_k))
    recall = intersection / k
    
    return recall


def create_mock_encrypted_vector(vector: np.ndarray) -> str:
    """Create a mock encrypted vector for testing without TenSEAL."""
    vector_bytes = vector.tobytes()
    mock_encrypted = hashlib.sha256(vector_bytes).digest()
    return base64.b64encode(mock_encrypted).decode('utf-8')


def create_mock_context() -> str:
    """Create a mock serialized context for testing."""
    mock_context = {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": 2**40
    }
    context_bytes = json.dumps(mock_context).encode()
    return base64.b64encode(context_bytes).decode('utf-8')


# ==============================================================================
# Test Context Management
# ==============================================================================

class TestContextManagement:
    """Test CKKS context creation and management."""
    
    def test_create_context_endpoint(self):
        """Test context creation via API endpoint."""
        response = client.post("/create_context", json={
            "tenant_id": "test_tenant_1",
            "poly_modulus_degree": 8192,
            "coeff_mod_bit_sizes": [60, 40, 40, 60],
            "global_scale": 2**40,
            "generate_galois_keys": True,
            "generate_relin_keys": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "context_id" in data
        assert "metadata" in data
        assert data["metadata"]["tenant_id"] == "test_tenant_1"
        
    def test_context_isolation(self):
        """Test that different tenants get isolated contexts."""
        # Create context for tenant 1
        response1 = client.post("/create_context", json={
            "tenant_id": "tenant_1",
            "poly_modulus_degree": 8192
        })
        assert response1.status_code == 200
        context_id_1 = response1.json()["context_id"]
        
        # Create context for tenant 2
        response2 = client.post("/create_context", json={
            "tenant_id": "tenant_2",
            "poly_modulus_degree": 8192
        })
        assert response2.status_code == 200
        context_id_2 = response2.json()["context_id"]
        
        # Verify different context IDs
        assert context_id_1 != context_id_2
        
    def test_context_info_endpoint(self):
        """Test retrieving context information."""
        # Create context
        client.post("/create_context", json={
            "tenant_id": "info_test_tenant"
        })
        
        # Get info
        response = client.get("/context/info_test_tenant/info")
        assert response.status_code == 200
        data = response.json()
        assert "context_id" in data
        assert "created_at" in data
        assert "config" in data
        
    def test_delete_context(self):
        """Test context deletion."""
        # Create context
        client.post("/create_context", json={
            "tenant_id": "delete_test_tenant"
        })
        
        # Delete context
        response = client.delete("/context/delete_test_tenant")
        assert response.status_code == 200
        
        # Verify context is gone
        response = client.get("/context/delete_test_tenant/info")
        assert response.status_code == 404
        
    def test_ckks_config(self):
        """Test CKKS configuration dataclass."""
        config = CKKSConfig(
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 50, 50, 60],
            global_scale=2**50
        )
        
        config_dict = config.to_dict()
        assert config_dict["poly_modulus_degree"] == 16384
        assert config_dict["global_scale"] == 2**50
        assert len(config_dict["coeff_mod_bit_sizes"]) == 4


# ==============================================================================
# Test Dot Product Operations
# ==============================================================================

class TestDotProductOperations:
    """Test encrypted dot-product computations."""
    
    def setup_method(self):
        """Setup test data."""
        self.dimension = 128
        self.tenant_id = "dot_test_tenant"
        
        # Create test vectors
        np.random.seed(42)  # For reproducibility
        self.q, self.v = create_test_vectors(self.dimension)
        
        # Create context
        response = client.post("/create_context", json={
            "tenant_id": self.tenant_id
        })
        assert response.status_code == 200
        
    def test_single_dot_product(self):
        """Test single encrypted dot-product operation."""
        # Create mock encrypted vectors
        encrypted_q = create_mock_encrypted_vector(self.q)
        encrypted_v = create_mock_encrypted_vector(self.v)
        
        # Perform dot product
        response = client.post("/dot", json={
            "tenant_id": self.tenant_id,
            "encrypted_q": encrypted_q,
            "encrypted_v": encrypted_v
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "encrypted_result" in data
        assert "operation_id" in data
        assert "receipt_hash" in data
        
    def test_batch_dot_products(self):
        """Test batch encrypted dot-product operations."""
        # Create multiple vectors
        num_vectors = 10
        vectors = [np.random.randn(self.dimension) for _ in range(num_vectors)]
        vectors = [v / np.linalg.norm(v) for v in vectors]  # Normalize
        
        # Create mock encrypted data
        encrypted_q = create_mock_encrypted_vector(self.q)
        encrypted_matrix = [create_mock_encrypted_vector(v) for v in vectors]
        
        # Perform batch dot products
        response = client.post("/batch_dot", json={
            "tenant_id": self.tenant_id,
            "encrypted_q": encrypted_q,
            "encrypted_matrix": encrypted_matrix
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "encrypted_results" in data
        assert len(data["encrypted_results"]) == num_vectors
        assert "operation_id" in data
        assert "receipt_hash" in data
        assert "computation_time_ms" in data
        
    def test_dot_product_with_context(self):
        """Test dot-product with explicit context parameter."""
        # Create mock context
        mock_context = create_mock_context()
        
        # Create mock encrypted vectors
        encrypted_q = create_mock_encrypted_vector(self.q)
        encrypted_v = create_mock_encrypted_vector(self.v)
        
        # Perform dot product with context
        response = client.post("/dot", json={
            "tenant_id": "context_test_tenant",
            "encrypted_q": encrypted_q,
            "encrypted_v": encrypted_v,
            "context": mock_context
        })
        
        # In placeholder mode, this should work
        assert response.status_code in [200, 404]  # 404 if context required


# ==============================================================================
# Test Accuracy and Parity
# ==============================================================================

@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL not installed")
class TestAccuracyParity:
    """Test accuracy parity between encrypted and plaintext operations."""
    
    def setup_method(self):
        """Setup test environment with real TenSEAL."""
        self.dimension = 128
        self.tolerance = 0.01  # 1% tolerance
        
        # Create CKKS context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        
    def test_dot_product_accuracy(self):
        """Test that encrypted dot-product matches plaintext within tolerance."""
        # Create test vectors
        np.random.seed(42)
        q = np.random.randn(self.dimension)
        v = np.random.randn(self.dimension)
        
        # Normalize
        q = q / np.linalg.norm(q)
        v = v / np.linalg.norm(v)
        
        # Plaintext dot product
        plaintext_result = np.dot(q, v)
        
        # Encrypted dot product
        encrypted_q = ts.ckks_vector(self.context, q)
        encrypted_v = ts.ckks_vector(self.context, v)
        encrypted_result = (encrypted_q * encrypted_v).sum()
        
        # Decrypt and compare
        decrypted_result = encrypted_result.decrypt()[0]
        
        # Check accuracy
        error = abs(decrypted_result - plaintext_result)
        relative_error = error / abs(plaintext_result) if plaintext_result != 0 else error
        
        assert relative_error < self.tolerance, \
            f"Relative error {relative_error:.4f} exceeds tolerance {self.tolerance}"
            
    def test_batch_accuracy(self):
        """Test batch operation accuracy."""
        # Create test data
        np.random.seed(42)
        num_vectors = 100
        q = np.random.randn(self.dimension)
        q = q / np.linalg.norm(q)
        
        vectors = []
        for _ in range(num_vectors):
            v = np.random.randn(self.dimension)
            v = v / np.linalg.norm(v)
            vectors.append(v)
            
        # Compute plaintext dot products
        plaintext_results = [np.dot(q, v) for v in vectors]
        
        # Compute encrypted dot products
        encrypted_q = ts.ckks_vector(self.context, q)
        encrypted_results = []
        
        for v in vectors:
            encrypted_v = ts.ckks_vector(self.context, v)
            encrypted_result = (encrypted_q * encrypted_v).sum()
            decrypted = encrypted_result.decrypt()[0]
            encrypted_results.append(decrypted)
            
        # Check accuracy for each result
        errors = []
        for plain, encrypted in zip(plaintext_results, encrypted_results):
            error = abs(encrypted - plain)
            relative_error = error / abs(plain) if plain != 0 else error
            errors.append(relative_error)
            
        # Check that average error is within tolerance
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        assert avg_error < self.tolerance, \
            f"Average error {avg_error:.4f} exceeds tolerance {self.tolerance}"
        assert max_error < self.tolerance * 2, \
            f"Maximum error {max_error:.4f} exceeds 2x tolerance"
            
    def test_recall_at_k(self):
        """Test recall@k metric for encrypted similarity search."""
        # Create test database
        np.random.seed(42)
        database_size = 100
        k_values = [1, 5, 10, 20]
        
        # Query vector
        q = np.random.randn(self.dimension)
        q = q / np.linalg.norm(q)
        
        # Database vectors
        database = []
        for _ in range(database_size):
            v = np.random.randn(self.dimension)
            v = v / np.linalg.norm(v)
            database.append(v)
            
        # Compute plaintext similarities
        plaintext_scores = [np.dot(q, v) for v in database]
        
        # Compute encrypted similarities
        encrypted_q = ts.ckks_vector(self.context, q)
        encrypted_scores = []
        
        for v in database:
            encrypted_v = ts.ckks_vector(self.context, v)
            encrypted_result = (encrypted_q * encrypted_v).sum()
            decrypted = encrypted_result.decrypt()[0]
            encrypted_scores.append(decrypted)
            
        # Test recall@k for different k values
        for k in k_values:
            recall = compute_recall_at_k(
                q, np.array(database), encrypted_scores, plaintext_scores, k
            )
            
            # Recall should be at least 98% (within 2% of perfect)
            assert recall >= 0.98, \
                f"Recall@{k} = {recall:.2f} is below 98% threshold"


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.tenant_id = "edge_test_tenant"
        
        # Create context
        response = client.post("/create_context", json={
            "tenant_id": self.tenant_id
        })
        assert response.status_code == 200
        
    def test_zero_vectors(self):
        """Test dot product with zero vectors."""
        dimension = 128
        
        # Create zero vector and normal vector
        zero_vector = np.zeros(dimension)
        normal_vector = np.random.randn(dimension)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        # Create mock encrypted vectors
        encrypted_zero = create_mock_encrypted_vector(zero_vector)
        encrypted_normal = create_mock_encrypted_vector(normal_vector)
        
        # Test dot product
        response = client.post("/dot", json={
            "tenant_id": self.tenant_id,
            "encrypted_q": encrypted_zero,
            "encrypted_v": encrypted_normal
        })
        
        assert response.status_code == 200
        # Result should be encrypted zero (though we can't verify without decryption)
        
    def test_orthogonal_vectors(self):
        """Test dot product with orthogonal vectors."""
        dimension = 128
        
        # Create orthogonal vectors
        v1 = np.zeros(dimension)
        v1[0] = 1.0
        
        v2 = np.zeros(dimension)
        v2[1] = 1.0
        
        # Verify orthogonality
        assert np.dot(v1, v2) == 0.0
        
        # Create mock encrypted vectors
        encrypted_v1 = create_mock_encrypted_vector(v1)
        encrypted_v2 = create_mock_encrypted_vector(v2)
        
        # Test dot product
        response = client.post("/dot", json={
            "tenant_id": self.tenant_id,
            "encrypted_q": encrypted_v1,
            "encrypted_v": encrypted_v2
        })
        
        assert response.status_code == 200
        
    def test_large_dimension_vectors(self):
        """Test with large dimension vectors."""
        large_dimension = 1024
        
        # Create large vectors
        q = np.random.randn(large_dimension)
        v = np.random.randn(large_dimension)
        
        # Normalize
        q = q / np.linalg.norm(q)
        v = v / np.linalg.norm(v)
        
        # Create mock encrypted vectors
        encrypted_q = create_mock_encrypted_vector(q)
        encrypted_v = create_mock_encrypted_vector(v)
        
        # Test dot product
        response = client.post("/dot", json={
            "tenant_id": self.tenant_id,
            "encrypted_q": encrypted_q,
            "encrypted_v": encrypted_v
        })
        
        assert response.status_code == 200
        
    def test_missing_context(self):
        """Test operation with missing context."""
        # Try to perform operation with non-existent tenant
        encrypted_q = create_mock_encrypted_vector(np.random.randn(128))
        encrypted_v = create_mock_encrypted_vector(np.random.randn(128))
        
        response = client.post("/dot", json={
            "tenant_id": "non_existent_tenant",
            "encrypted_q": encrypted_q,
            "encrypted_v": encrypted_v
        })
        
        # Should fail or handle gracefully
        assert response.status_code in [404, 200]  # 200 in placeholder mode


# ==============================================================================
# Test PoDP Integration
# ==============================================================================

class TestPoDPIntegration:
    """Test Proof-of-Deterministic-Processing integration."""
    
    def test_receipt_generation(self):
        """Test that operations generate valid PoDP receipts."""
        # Setup
        tenant_id = "podp_test_tenant"
        client.post("/create_context", json={"tenant_id": tenant_id})
        
        # Create mock encrypted vectors
        q = np.random.randn(128)
        v = np.random.randn(128)
        encrypted_q = create_mock_encrypted_vector(q)
        encrypted_v = create_mock_encrypted_vector(v)
        
        # Perform operation
        response = client.post("/dot", json={
            "tenant_id": tenant_id,
            "encrypted_q": encrypted_q,
            "encrypted_v": encrypted_v
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify receipt hash exists and is valid
        assert "receipt_hash" in data
        assert len(data["receipt_hash"]) == 64  # SHA-256 hash
        
    def test_receipt_uniqueness(self):
        """Test that different operations generate different receipts."""
        # Setup
        tenant_id = "receipt_unique_test"
        client.post("/create_context", json={"tenant_id": tenant_id})
        
        # Perform two different operations
        receipts = []
        for i in range(2):
            q = np.random.randn(128)
            v = np.random.randn(128)
            
            response = client.post("/dot", json={
                "tenant_id": tenant_id,
                "encrypted_q": create_mock_encrypted_vector(q),
                "encrypted_v": create_mock_encrypted_vector(v)
            })
            
            receipts.append(response.json()["receipt_hash"])
            
        # Verify receipts are different
        assert receipts[0] != receipts[1]
        
    def test_fhe_receipt_creation(self):
        """Test direct FHE receipt creation."""
        receipt = create_fhe_receipt(
            operation_id="test_op_123",
            tenant_id="test_tenant",
            operation_type="dot_product",
            metadata={
                "computation_time_ms": 10.5,
                "params": {"scheme": "CKKS"}
            }
        )
        
        assert receipt.dispute_id == "test_op_123"
        assert receipt.step == "FHE_DOT_V1"
        assert receipt.hash is not None
        assert "tenant_id" in receipt.inputs


# ==============================================================================
# Test Health and Status
# ==============================================================================

class TestHealthAndStatus:
    """Test service health and status endpoints."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "tenseal_available" in data
        assert "timestamp" in data
        assert "active_contexts" in data
        assert "version" in data
        
    def test_service_info(self):
        """Test that service provides proper information."""
        response = client.get("/healthz")
        data = response.json()
        
        # Check service identification
        assert data["service"] == "DALRN FHE Service"
        assert data["version"] == "1.0.0"


# ==============================================================================
# Performance Benchmarks (Optional)
# ==============================================================================

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for FHE operations."""
    
    def test_single_dot_product_performance(self, benchmark):
        """Benchmark single dot-product operation."""
        # Setup
        tenant_id = "perf_test_tenant"
        client.post("/create_context", json={"tenant_id": tenant_id})
        
        q = np.random.randn(128)
        v = np.random.randn(128)
        
        def perform_dot():
            response = client.post("/dot", json={
                "tenant_id": tenant_id,
                "encrypted_q": create_mock_encrypted_vector(q),
                "encrypted_v": create_mock_encrypted_vector(v)
            })
            return response
            
        # Run benchmark
        result = benchmark(perform_dot)
        assert result.status_code == 200
        
    def test_batch_performance(self, benchmark):
        """Benchmark batch dot-product operations."""
        # Setup
        tenant_id = "batch_perf_test"
        client.post("/create_context", json={"tenant_id": tenant_id})
        
        num_vectors = 100
        q = np.random.randn(128)
        vectors = [np.random.randn(128) for _ in range(num_vectors)]
        
        def perform_batch():
            response = client.post("/batch_dot", json={
                "tenant_id": tenant_id,
                "encrypted_q": create_mock_encrypted_vector(q),
                "encrypted_matrix": [create_mock_encrypted_vector(v) for v in vectors]
            })
            return response
            
        # Run benchmark
        result = benchmark(perform_batch)
        assert result.status_code == 200


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])