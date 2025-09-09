"""
Example client for TenSEAL CKKS FHE Service.

This demonstrates how to:
1. Create encryption contexts
2. Encrypt vectors client-side (when TenSEAL is available)
3. Send encrypted data to the service
4. Receive encrypted results
5. Decrypt results client-side

Note: Full functionality requires TenSEAL installed on the client.
"""

import base64
import hashlib
import json
import numpy as np
import requests
from typing import List, Tuple

# Try to import TenSEAL for real encryption
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: TenSEAL not installed. Using placeholder encryption.")


class FHEClient:
    """Client for interacting with the FHE service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the FHE client."""
        self.base_url = base_url
        self.context = None
        self.secret_key = None
        self.tenant_id = None
        
    def create_context(self, tenant_id: str) -> dict:
        """Create a new CKKS context for encryption."""
        self.tenant_id = tenant_id
        
        if TENSEAL_AVAILABLE:
            # Create local context with secret key
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.global_scale = 2**40
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            
            # Save secret key for decryption
            self.secret_key = self.context.serialize(save_secret_key=True)
        
        # Create context on server (without secret key)
        response = requests.post(
            f"{self.base_url}/create_context",
            json={
                "tenant_id": tenant_id,
                "poly_modulus_degree": 8192,
                "coeff_mod_bit_sizes": [60, 40, 40, 60],
                "global_scale": 2**40,
                "generate_galois_keys": True,
                "generate_relin_keys": True
            }
        )
        response.raise_for_status()
        return response.json()
        
    def encrypt_vector(self, vector: np.ndarray) -> str:
        """Encrypt a vector using the CKKS scheme."""
        if TENSEAL_AVAILABLE and self.context:
            # Real encryption
            encrypted = ts.ckks_vector(self.context, vector.tolist())
            encrypted_bytes = encrypted.serialize()
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        else:
            # Placeholder encryption for testing
            vector_bytes = vector.tobytes()
            encrypted = hashlib.sha256(vector_bytes).digest()
            return base64.b64encode(encrypted).decode('utf-8')
            
    def decrypt_result(self, encrypted_result: str) -> float:
        """Decrypt an encrypted result."""
        if TENSEAL_AVAILABLE and self.secret_key:
            # Deserialize and decrypt
            encrypted_bytes = base64.b64decode(encrypted_result)
            context_with_key = ts.context_from(self.secret_key)
            result = ts.ckks_vector_from(context_with_key, encrypted_bytes)
            return result.decrypt()[0]
        else:
            # Can't decrypt without TenSEAL
            return 0.0
            
    def compute_dot_product(self, q: np.ndarray, v: np.ndarray) -> dict:
        """Compute encrypted dot product of two vectors."""
        if not self.tenant_id:
            raise ValueError("Must create context first")
            
        # Encrypt vectors
        encrypted_q = self.encrypt_vector(q)
        encrypted_v = self.encrypt_vector(v)
        
        # Send to service
        response = requests.post(
            f"{self.base_url}/dot",
            json={
                "tenant_id": self.tenant_id,
                "encrypted_q": encrypted_q,
                "encrypted_v": encrypted_v
            }
        )
        response.raise_for_status()
        return response.json()
        
    def compute_batch_dot_products(self, q: np.ndarray, matrix: List[np.ndarray]) -> dict:
        """Compute batch encrypted dot products."""
        if not self.tenant_id:
            raise ValueError("Must create context first")
            
        # Encrypt query and matrix
        encrypted_q = self.encrypt_vector(q)
        encrypted_matrix = [self.encrypt_vector(v) for v in matrix]
        
        # Send to service
        response = requests.post(
            f"{self.base_url}/batch_dot",
            json={
                "tenant_id": self.tenant_id,
                "encrypted_q": encrypted_q,
                "encrypted_matrix": encrypted_matrix
            }
        )
        response.raise_for_status()
        return response.json()


def demo_single_dot_product():
    """Demonstrate single encrypted dot product."""
    print("\n=== Single Dot Product Demo ===")
    
    # Initialize client
    client = FHEClient()
    
    # Create context
    print("Creating encryption context...")
    context_info = client.create_context("demo_tenant_1")
    print(f"Context created: {context_info['context_id']}")
    
    # Create test vectors
    dimension = 128
    q = np.random.randn(dimension)
    v = np.random.randn(dimension)
    
    # Normalize to unit vectors
    q = q / np.linalg.norm(q)
    v = v / np.linalg.norm(v)
    
    # Compute plaintext result for comparison
    plaintext_result = np.dot(q, v)
    print(f"Plaintext dot product: {plaintext_result:.6f}")
    
    # Compute encrypted dot product
    print("Computing encrypted dot product...")
    result = client.compute_dot_product(q, v)
    print(f"Operation ID: {result['operation_id']}")
    print(f"Receipt hash: {result['receipt_hash']}")
    
    # Decrypt result (if TenSEAL available)
    if TENSEAL_AVAILABLE:
        decrypted = client.decrypt_result(result['encrypted_result'])
        print(f"Decrypted result: {decrypted:.6f}")
        error = abs(decrypted - plaintext_result)
        print(f"Error: {error:.8f}")
    else:
        print("Cannot decrypt without TenSEAL")


def demo_batch_operations():
    """Demonstrate batch encrypted operations."""
    print("\n=== Batch Operations Demo ===")
    
    # Initialize client
    client = FHEClient()
    
    # Create context
    print("Creating encryption context...")
    context_info = client.create_context("demo_tenant_2")
    print(f"Context created: {context_info['context_id']}")
    
    # Create test data
    dimension = 128
    num_vectors = 10
    
    # Query vector
    q = np.random.randn(dimension)
    q = q / np.linalg.norm(q)
    
    # Database vectors
    database = []
    for i in range(num_vectors):
        v = np.random.randn(dimension)
        v = v / np.linalg.norm(v)
        database.append(v)
    
    # Compute plaintext results
    plaintext_results = [np.dot(q, v) for v in database]
    print(f"Plaintext results: {plaintext_results[:3]}... (showing first 3)")
    
    # Compute encrypted batch
    print(f"Computing {num_vectors} encrypted dot products...")
    result = client.compute_batch_dot_products(q, database)
    print(f"Operation ID: {result['operation_id']}")
    print(f"Computation time: {result['computation_time_ms']:.2f}ms")
    print(f"Number of results: {len(result['encrypted_results'])}")
    
    # Decrypt results (if TenSEAL available)
    if TENSEAL_AVAILABLE:
        decrypted_results = [
            client.decrypt_result(enc) 
            for enc in result['encrypted_results']
        ]
        print(f"Decrypted results: {decrypted_results[:3]}... (showing first 3)")
        
        # Check accuracy
        errors = [abs(d - p) for d, p in zip(decrypted_results, plaintext_results)]
        avg_error = np.mean(errors)
        print(f"Average error: {avg_error:.8f}")
    else:
        print("Cannot decrypt without TenSEAL")


def demo_cosine_similarity_search():
    """Demonstrate encrypted cosine similarity search."""
    print("\n=== Encrypted Similarity Search Demo ===")
    
    # Initialize client
    client = FHEClient()
    
    # Create context
    print("Creating encryption context...")
    context_info = client.create_context("search_tenant")
    print(f"Context created: {context_info['context_id']}")
    
    # Create synthetic database
    dimension = 128
    database_size = 100
    
    print(f"Creating database with {database_size} vectors...")
    database = []
    for i in range(database_size):
        v = np.random.randn(dimension)
        v = v / np.linalg.norm(v)  # Normalize for cosine similarity
        database.append(v)
    
    # Create query vector
    query = np.random.randn(dimension)
    query = query / np.linalg.norm(query)
    
    # Compute plaintext similarities
    plaintext_scores = [np.dot(query, v) for v in database]
    top_k = 5
    plaintext_top_k = np.argsort(plaintext_scores)[-top_k:][::-1]
    print(f"Plaintext top-{top_k} indices: {plaintext_top_k.tolist()}")
    
    # Compute encrypted similarities
    print("Computing encrypted similarities...")
    result = client.compute_batch_dot_products(query, database)
    print(f"Computation time: {result['computation_time_ms']:.2f}ms")
    
    if TENSEAL_AVAILABLE:
        # Decrypt and find top-k
        encrypted_scores = [
            client.decrypt_result(enc) 
            for enc in result['encrypted_results']
        ]
        encrypted_top_k = np.argsort(encrypted_scores)[-top_k:][::-1]
        print(f"Encrypted top-{top_k} indices: {encrypted_top_k.tolist()}")
        
        # Compute recall
        intersection = len(set(plaintext_top_k) & set(encrypted_top_k))
        recall = intersection / top_k
        print(f"Recall@{top_k}: {recall:.2%}")
    else:
        print("Cannot decrypt without TenSEAL")


if __name__ == "__main__":
    print("=" * 60)
    print("TenSEAL CKKS FHE Service Client Demo")
    print("=" * 60)
    
    # Check service availability
    try:
        response = requests.get("http://localhost:8000/healthz")
        response.raise_for_status()
        health = response.json()
        print(f"Service status: {health['status']}")
        print(f"TenSEAL available on server: {health['tenseal_available']}")
    except Exception as e:
        print(f"Warning: Service not reachable at http://localhost:8000")
        print(f"Please start the service with: python services/fhe/service.py")
        print(f"Error: {e}")
        exit(1)
    
    # Run demos
    demo_single_dot_product()
    demo_batch_operations()
    demo_cosine_similarity_search()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)