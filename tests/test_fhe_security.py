"""
Security validation tests for the FHE service.

These tests ensure that:
1. TenSEAL is required and no fake encryption is allowed
2. Tenant isolation is properly enforced
3. Encrypted data integrity is maintained
4. Accuracy requirements are met
"""

import base64
import hashlib
import json
import numpy as np
import pytest
from unittest.mock import Mock, patch

# Test that the service fails without TenSEAL
def test_service_requires_tenseal():
    """Verify the service refuses to start without TenSEAL."""
    with patch.dict('sys.modules', {'tenseal': None}):
        with pytest.raises(RuntimeError) as exc_info:
            import services.fhe.service
        assert "TenSEAL is REQUIRED" in str(exc_info.value)
        assert "homomorphic encryption" in str(exc_info.value)


# Test with TenSEAL available
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    pytest.skip("TenSEAL not installed - install it to run security tests", allow_module_level=True)


from services.fhe.service import (
    ContextManager,
    CKKSConfig,
    FHEOperations,
    app
)
from fastapi.testclient import TestClient

client = TestClient(app)


class TestSecurityEnforcement:
    """Test that security requirements are strictly enforced."""

    def test_no_placeholder_encryption(self):
        """Ensure no SHA256 or fake encryption is ever returned."""
        context_manager = ContextManager()
        config = CKKSConfig()

        # Create a real context
        context, context_id = context_manager.create_context("test_tenant", config)

        assert context is not None, "Context must be real, not None"
        assert isinstance(context_id, str)
        assert len(context_id) == 16

        # Ensure context is a real TenSEAL context
        assert hasattr(context, 'global_scale')
        assert hasattr(context, 'poly_modulus_degree')

    def test_encryption_produces_ciphertext_not_hash(self):
        """Verify that encryption produces real ciphertext, not hashes."""
        # Create a test context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        context.generate_relin_keys()

        # Create test vectors
        vector1 = np.array([1.0, 2.0, 3.0, 4.0])
        vector2 = np.array([5.0, 6.0, 7.0, 8.0])

        # Encrypt vectors
        enc_v1 = ts.ckks_vector(context, vector1.tolist())
        enc_v2 = ts.ckks_vector(context, vector2.tolist())

        # Serialize
        enc_v1_bytes = enc_v1.serialize()
        enc_v2_bytes = enc_v2.serialize()

        # Perform homomorphic dot product
        result_bytes = FHEOperations.perform_dot_product(
            context, enc_v1_bytes, enc_v2_bytes
        )

        # Verify result is not a hash
        assert len(result_bytes) > 32, "Result must be larger than SHA256 hash"

        # Try to deserialize result (would fail if it's a hash)
        enc_result = ts.ckks_vector_from(context, result_bytes)
        assert enc_result is not None

    def test_tenant_isolation(self):
        """Verify that tenants cannot access each other's contexts."""
        context_manager = ContextManager()
        config = CKKSConfig()

        # Create contexts for two tenants
        context1, id1 = context_manager.create_context("tenant1", config)
        context2, id2 = context_manager.create_context("tenant2", config)

        # Ensure contexts are different
        assert id1 != id2
        assert context1 is not context2

        # Ensure tenant1 cannot get tenant2's context
        retrieved1 = context_manager.get_context("tenant1")
        retrieved2 = context_manager.get_context("tenant2")

        assert retrieved1 is context1
        assert retrieved2 is context2
        assert retrieved1 is not retrieved2

        # Ensure no cross-contamination
        assert context_manager.get_context("tenant3") is None


class TestEncryptionAccuracy:
    """Test that encrypted operations meet accuracy requirements."""

    def test_dot_product_accuracy(self):
        """Verify encrypted dot product is within 2% of plaintext."""
        # Create context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        context.generate_relin_keys()

        # Generate secret key for decryption (client-side only)
        secret_key = context.secret_key()

        # Create unit vectors
        dimension = 128
        np.random.seed(42)
        v1 = np.random.randn(dimension)
        v2 = np.random.randn(dimension)

        # Normalize to unit vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Compute plaintext dot product
        plaintext_result = np.dot(v1, v2)

        # Encrypt vectors
        enc_v1 = ts.ckks_vector(context, v1.tolist())
        enc_v2 = ts.ckks_vector(context, v2.tolist())

        # Perform homomorphic dot product
        enc_result_bytes = FHEOperations.perform_dot_product(
            context,
            enc_v1.serialize(),
            enc_v2.serialize()
        )

        # Decrypt result (client-side only)
        enc_result = ts.ckks_vector_from(context, enc_result_bytes)
        decrypted_result = enc_result.decrypt(secret_key)[0]  # First element is the sum

        # Check accuracy (within 2%)
        relative_error = abs(decrypted_result - plaintext_result) / abs(plaintext_result)
        assert relative_error < 0.02, f"Relative error {relative_error:.4%} exceeds 2%"

    def test_cosine_similarity_accuracy(self):
        """Test that cosine similarity via dot products maintains accuracy."""
        # Create context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        context.generate_relin_keys()

        secret_key = context.secret_key()

        # Create multiple test vectors
        dimension = 256
        num_vectors = 10
        np.random.seed(123)

        vectors = []
        for _ in range(num_vectors):
            v = np.random.randn(dimension)
            v = v / np.linalg.norm(v)  # Normalize
            vectors.append(v)

        # Test query
        query = np.random.randn(dimension)
        query = query / np.linalg.norm(query)

        # Compute plaintext similarities
        plaintext_scores = [np.dot(query, v) for v in vectors]

        # Compute encrypted similarities
        enc_query = ts.ckks_vector(context, query.tolist())
        encrypted_scores = []

        for v in vectors:
            enc_v = ts.ckks_vector(context, v.tolist())
            enc_result_bytes = FHEOperations.perform_dot_product(
                context,
                enc_query.serialize(),
                enc_v.serialize()
            )
            enc_result = ts.ckks_vector_from(context, enc_result_bytes)
            decrypted = enc_result.decrypt(secret_key)[0]
            encrypted_scores.append(decrypted)

        # Check recall@k
        k = 5
        plaintext_top_k = set(np.argsort(plaintext_scores)[-k:])
        encrypted_top_k = set(np.argsort(encrypted_scores)[-k:])

        recall = len(plaintext_top_k & encrypted_top_k) / k
        assert recall >= 0.98, f"Recall@{k} = {recall:.2%} is below 98% requirement"


class TestAPIEndpointSecurity:
    """Test security of API endpoints."""

    def test_create_context_validation(self):
        """Test that context creation validates parameters."""
        # Test with invalid tenant_id
        response = client.post("/create_context", json={
            "tenant_id": "short",  # Too short
            "poly_modulus_degree": 8192
        })
        assert response.status_code == 400
        assert "at least 8 characters" in response.json()["detail"]

        # Test with insecure parameters
        response = client.post("/create_context", json={
            "tenant_id": "test_tenant_123",
            "poly_modulus_degree": 4096  # Too low for security
        })
        assert response.status_code == 400
        assert "8192 for 128-bit security" in response.json()["detail"]

    def test_dot_product_input_validation(self):
        """Test that dot product endpoint validates inputs."""
        # First create a valid context
        context_response = client.post("/create_context", json={
            "tenant_id": "test_tenant_secure",
            "poly_modulus_degree": 8192
        })
        assert context_response.status_code == 200

        # Test with missing encrypted data
        response = client.post("/dot", json={
            "tenant_id": "test_tenant_secure",
            "encrypted_q": "",
            "encrypted_v": ""
        })
        assert response.status_code == 400
        assert "Missing encrypted vectors" in response.json()["detail"]

        # Test with invalid base64
        response = client.post("/dot", json={
            "tenant_id": "test_tenant_secure",
            "encrypted_q": "not-valid-base64!@#",
            "encrypted_v": "also-invalid-base64!@#"
        })
        assert response.status_code == 400
        assert "Invalid base64" in response.json()["detail"]

        # Test with too-small encrypted data (suspicious)
        small_data = base64.b64encode(b"too small").decode()
        response = client.post("/dot", json={
            "tenant_id": "test_tenant_secure",
            "encrypted_q": small_data,
            "encrypted_v": small_data
        })
        assert response.status_code == 400
        assert "suspiciously small" in response.json()["detail"]

    def test_context_expiry(self):
        """Test that contexts expire after max age."""
        context_manager = ContextManager()
        config = CKKSConfig()

        # Create a context
        context, context_id = context_manager.create_context("expiry_test", config)

        # Initially should be retrievable
        assert context_manager.get_context("expiry_test") is not None

        # Simulate expiry
        from datetime import datetime, timedelta
        context_manager.context_expiry["expiry_test"] = datetime.utcnow() - timedelta(seconds=1)

        # Should return None and clean up
        assert context_manager.get_context("expiry_test") is None
        assert "expiry_test" not in context_manager.contexts

    def test_no_secret_key_exposure(self):
        """Ensure secret keys are never exposed by the service."""
        # Create a context
        response = client.post("/create_context", json={
            "tenant_id": "secret_test_tenant",
            "poly_modulus_degree": 8192
        })
        assert response.status_code == 200

        response_data = response.json()

        # Check serialized context doesn't contain secret key
        if response_data.get("serialized_context"):
            # Decode the context
            context_bytes = base64.b64decode(response_data["serialized_context"])

            # Try to load it and verify no secret key
            context = ts.context_from(context_bytes)

            # This should fail if no secret key
            with pytest.raises(Exception):
                test_vec = ts.ckks_vector(context, [1.0, 2.0])
                test_vec.decrypt()  # Should fail without secret key

    def test_health_check_no_sensitive_data(self):
        """Ensure health check doesn't expose sensitive information."""
        response = client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "tenseal_available" in data
        assert data["tenseal_available"] is True

        # Should not expose actual context details
        assert "contexts" not in str(data).lower()
        assert "secret" not in str(data).lower()
        assert "key" not in str(data).lower()


class TestOperationAuditability:
    """Test that operations are properly audited."""

    def test_operation_tracking(self):
        """Verify operations are tracked per tenant."""
        context_manager = ContextManager()
        config = CKKSConfig()

        # Create context
        context, _ = context_manager.create_context("audit_tenant", config)

        # Perform multiple operations
        for i in range(5):
            ctx = context_manager.get_context("audit_tenant")
            assert ctx is not None

        # Check operation count
        assert context_manager.operation_count.get("audit_tenant", 0) == 5

        # Remove context should log operation count
        context_manager.remove_context("audit_tenant")
        assert "audit_tenant" not in context_manager.operation_count


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])