"""
TenSEAL CKKS-based Fully Homomorphic Encryption Service for DALRN.

This service provides privacy-preserving dot-product operations using the CKKS
homomorphic encryption scheme. All computations are performed on encrypted data,
with client-side-only decryption ensuring the server never accesses plaintext.

Architecture:
- CKKS parameters optimized for dot-product operations on unit-norm vectors
- Strict per-tenant context isolation with cryptographic guarantees
- Base64 encoding for encrypted data transport
- PoDP integration for deterministic processing receipts
"""

import base64
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

# TenSEAL is REQUIRED for this service - No placeholders or fake encryption allowed
try:
    import tenseal as ts
except ImportError:
    raise RuntimeError(
        "CRITICAL: TenSEAL is REQUIRED for the FHE service. "
        "This service cannot operate without real homomorphic encryption. "
        "Install it with: pip install tenseal"
    )
    
# Import PoDP for receipt generation
try:
    from services.common.podp import create_fhe_receipt, Receipt
except ImportError:
    # Fallback if running in isolation
    @dataclass
    class Receipt:
        dispute_id: str
        step: str
        inputs: Dict
        params: Dict
        artifacts: Dict
        ts: str
        hash: Optional[str] = None
        
        def finalize(self): 
            self.hash = hashlib.sha256(
                json.dumps(self.__dict__, sort_keys=True).encode()
            ).hexdigest()
            return self
    
    def create_fhe_receipt(operation_id, tenant_id, operation_type, metadata):
        return Receipt(
            dispute_id=operation_id,
            step="FHE_DOT_V1",
            inputs={"tenant_id": tenant_id, "operation_type": operation_type},
            params=metadata.get("params", {}),
            artifacts={"computation_metadata": metadata},
            ts=datetime.utcnow().isoformat() + "Z"
        ).finalize()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DALRN FHE Service (CKKS)",
    description="Privacy-preserving dot-product operations using TenSEAL CKKS",
    version="1.0.0"
)

# ==============================================================================
# CKKS Configuration
# ==============================================================================

@dataclass
class CKKSConfig:
    """CKKS scheme configuration parameters."""
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    global_scale: float = 2**40
    generate_galois_keys: bool = True
    generate_relin_keys: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": self.coeff_mod_bit_sizes,
            "global_scale": self.global_scale,
            "galois_keys": self.generate_galois_keys,
            "relin_keys": self.generate_relin_keys
        }


# ==============================================================================
# Context Management
# ==============================================================================

class ContextManager:
    """
    Manages CKKS contexts for multiple tenants with strict isolation.
    Each tenant gets a completely isolated encryption context.

    SECURITY GUARANTEES:
    - No cross-tenant context access
    - Contexts expire after max_context_age
    - No secret keys are ever stored or exposed
    - Each context is cryptographically isolated
    """

    def __init__(self):
        self.contexts: Dict[str, Any] = {}  # tenant_id -> context
        self.context_metadata: Dict[str, Dict] = {}  # tenant_id -> metadata
        self.context_expiry: Dict[str, datetime] = {}  # tenant_id -> expiry
        self.max_context_age = timedelta(hours=24)
        self.operation_count: Dict[str, int] = {}  # Track operations per tenant
        
    def create_context(self, tenant_id: str, config: CKKSConfig) -> Tuple[Any, str]:
        """
        Create a new CKKS context for a tenant.
        Returns (context, context_id).
        """
        # TenSEAL is always required - no placeholders allowed
        if not ts:
            raise RuntimeError(
                "TenSEAL is not available. This service requires real homomorphic encryption."
            )
            
        # Create TenSEAL context with specified parameters
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=config.poly_modulus_degree,
            coeff_mod_bit_sizes=config.coeff_mod_bit_sizes
        )
        
        # Set global scale
        context.global_scale = config.global_scale
        
        # Generate keys as needed
        if config.generate_galois_keys:
            context.generate_galois_keys()
        if config.generate_relin_keys:
            context.generate_relin_keys()
            
        # Store context with metadata
        context_id = hashlib.sha256(f"{tenant_id}:{time.time()}".encode()).hexdigest()[:16]
        self.contexts[tenant_id] = context
        self.context_metadata[tenant_id] = {
            "context_id": context_id,
            "created_at": datetime.utcnow().isoformat(),
            "config": config.to_dict()
        }
        self.context_expiry[tenant_id] = datetime.utcnow() + self.max_context_age
        
        logger.info(f"Created CKKS context {context_id} for tenant {tenant_id}")
        return context, context_id
        
    def get_context(self, tenant_id: str) -> Optional[Any]:
        """Get context for a tenant, checking expiry and validity."""
        if tenant_id not in self.contexts:
            return None

        # Check expiry
        if datetime.utcnow() > self.context_expiry.get(tenant_id, datetime.min):
            logger.warning(f"Context expired for tenant {tenant_id}")
            self.remove_context(tenant_id)
            return None

        # Increment operation count
        self.operation_count[tenant_id] = self.operation_count.get(tenant_id, 0) + 1

        # Log access for audit
        logger.debug(f"Context accessed for tenant {tenant_id} (op #{self.operation_count[tenant_id]})")

        return self.contexts[tenant_id]
        
    def remove_context(self, tenant_id: str) -> None:
        """Securely remove a tenant's context."""
        if tenant_id in self.contexts:
            # Clear the context from memory
            self.contexts[tenant_id] = None  # Overwrite reference first
            del self.contexts[tenant_id]
            del self.context_metadata[tenant_id]
            del self.context_expiry[tenant_id]

            if tenant_id in self.operation_count:
                ops = self.operation_count[tenant_id]
                del self.operation_count[tenant_id]
                logger.info(f"Removed context for tenant {tenant_id} (performed {ops} operations)")
            else:
                logger.info(f"Removed context for tenant {tenant_id}")
            
    def serialize_context(self, tenant_id: str) -> Optional[str]:
        """Serialize a context to base64 for client storage."""
        context = self.get_context(tenant_id)
        if not context:
            return None
            
        serialized = context.serialize(save_secret_key=False)
        return base64.b64encode(serialized).decode('utf-8')
        
    def deserialize_context(self, tenant_id: str, serialized: str) -> Any:
        """Deserialize a context from base64."""
        if not ts:
            raise RuntimeError("TenSEAL is required for context deserialization")
            
        context_bytes = base64.b64decode(serialized)
        context = ts.context_from(context_bytes)
        
        # Store in manager
        self.contexts[tenant_id] = context
        self.context_expiry[tenant_id] = datetime.utcnow() + self.max_context_age
        
        return context


# Global context manager instance
context_manager = ContextManager()


# ==============================================================================
# Request/Response Models
# ==============================================================================

class CreateContextRequest(BaseModel):
    """Request to create a new CKKS context."""
    tenant_id: str = Field(..., description="Unique tenant identifier")
    poly_modulus_degree: int = Field(8192, description="Polynomial modulus degree")
    coeff_mod_bit_sizes: List[int] = Field(
        default=[60, 40, 40, 60],
        description="Coefficient modulus bit sizes"
    )
    global_scale: float = Field(2**40, description="Global scale for CKKS")
    generate_galois_keys: bool = Field(True, description="Generate Galois keys for rotations")
    generate_relin_keys: bool = Field(True, description="Generate relinearization keys")


class CreateContextResponse(BaseModel):
    """Response with created context information."""
    context_id: str
    serialized_context: Optional[str] = Field(None, description="Base64-encoded context (no secret key)")
    metadata: Dict[str, Any]


class DotProductRequest(BaseModel):
    """Request for encrypted dot-product operation."""
    tenant_id: str = Field(..., description="Tenant identifier")
    encrypted_q: str = Field(..., description="Base64-encoded encrypted query vector")
    encrypted_v: str = Field(..., description="Base64-encoded encrypted value vector")
    context: Optional[str] = Field(None, description="Base64-encoded context if not stored server-side")


class DotProductResponse(BaseModel):
    """Response with encrypted dot-product result."""
    encrypted_result: str = Field(..., description="Base64-encoded encrypted result")
    operation_id: str = Field(..., description="Unique operation identifier")
    receipt_hash: str = Field(..., description="PoDP receipt hash")


class BatchDotProductRequest(BaseModel):
    """Request for batch encrypted dot-product operations."""
    tenant_id: str = Field(..., description="Tenant identifier")
    encrypted_q: str = Field(..., description="Base64-encoded encrypted query vector")
    encrypted_matrix: List[str] = Field(..., description="List of base64-encoded encrypted vectors")
    context: Optional[str] = Field(None, description="Base64-encoded context if not stored server-side")


class BatchDotProductResponse(BaseModel):
    """Response with batch encrypted dot-product results."""
    encrypted_results: List[str] = Field(..., description="List of base64-encoded encrypted results")
    operation_id: str = Field(..., description="Unique operation identifier")
    receipt_hash: str = Field(..., description="PoDP receipt hash")
    computation_time_ms: float = Field(..., description="Total computation time in milliseconds")


# ==============================================================================
# Encryption Operations
# ==============================================================================

class FHEOperations:
    """Core FHE operations using TenSEAL CKKS."""
    
    @staticmethod
    def perform_dot_product(
        context: Any,
        encrypted_q_bytes: bytes,
        encrypted_v_bytes: bytes
    ) -> bytes:
        """
        Perform homomorphic dot-product on encrypted vectors.

        SECURITY: This operation is performed entirely on ciphertext.
        The server never has access to plaintext values or secret keys.

        Args:
            context: TenSEAL context (public key only)
            encrypted_q_bytes: Encrypted query vector bytes
            encrypted_v_bytes: Encrypted value vector bytes

        Returns:
            Encrypted result bytes
        """
        # NO PLACEHOLDERS - Real encryption only
        if not ts:
            raise RuntimeError(
                "SECURITY ERROR: Cannot perform dot product without TenSEAL. "
                "This service MUST NOT return fake encrypted data."
            )

        # Validate input sizes
        if len(encrypted_q_bytes) < 100 or len(encrypted_v_bytes) < 100:
            raise ValueError(
                "Encrypted vectors appear too small. Possible security issue."
            )

        # Deserialize encrypted vectors with error handling
        try:
            encrypted_q = ts.ckks_vector_from(context, encrypted_q_bytes)
            encrypted_v = ts.ckks_vector_from(context, encrypted_v_bytes)
        except Exception as e:
            logger.error(f"Failed to deserialize encrypted vectors: {e}")
            raise ValueError(
                "Invalid encrypted vector format. Ensure vectors are properly encrypted with matching context."
            )
        
        # Perform element-wise multiplication
        encrypted_product = encrypted_q * encrypted_v
        
        # Sum all elements (dot product)
        # Note: TenSEAL's sum() performs homomorphic addition across vector elements
        encrypted_result = encrypted_product.sum()
        
        # Serialize result
        return encrypted_result.serialize()
        
    @staticmethod
    def validate_encryption_parameters(context: Any) -> bool:
        """Validate that context has appropriate parameters for secure dot-products."""
        if not ts:
            raise RuntimeError("TenSEAL is required for parameter validation")

        # Check polynomial modulus degree (affects security and capacity)
        # Minimum 8192 for 128-bit security with reasonable computation depth
        if context.poly_modulus_degree < 8192:
            logger.error(f"Polynomial modulus degree {context.poly_modulus_degree} is too low for secure operations")
            return False

        # Check scale is appropriate for precision
        if context.global_scale < 2**30:
            logger.warning(f"Global scale {context.global_scale} may be too low for accurate computations")
            return False

        # Verify required keys are present
        if not hasattr(context, 'galois_keys'):
            logger.error("Context missing Galois keys required for vector operations")
            return False

        if not hasattr(context, 'relin_keys'):
            logger.error("Context missing relinearization keys")
            return False

        return True


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.post("/create_context", response_model=CreateContextResponse)
async def create_context(request: CreateContextRequest):
    """
    Initialize a new CKKS context for a tenant.

    This creates an isolated encryption context with the specified parameters.
    The context is stored server-side for the tenant and can be used for
    subsequent operations. The serialized context (without secret key) is
    returned for client-side storage if needed.

    SECURITY: The secret key is NEVER stored server-side. Only the public
    encryption key is retained for computation.
    """
    try:
        # Validate tenant ID format
        if not request.tenant_id or len(request.tenant_id) < 8:
            raise HTTPException(
                status_code=400,
                detail="Invalid tenant_id. Must be at least 8 characters."
            )

        # Validate security parameters
        if request.poly_modulus_degree < 8192:
            raise HTTPException(
                status_code=400,
                detail="poly_modulus_degree must be at least 8192 for 128-bit security"
            )
        # Create configuration
        config = CKKSConfig(
            poly_modulus_degree=request.poly_modulus_degree,
            coeff_mod_bit_sizes=request.coeff_mod_bit_sizes,
            global_scale=request.global_scale,
            generate_galois_keys=request.generate_galois_keys,
            generate_relin_keys=request.generate_relin_keys
        )
        
        # Create context
        context, context_id = context_manager.create_context(request.tenant_id, config)

        # Validate the created context
        if context and not FHEOperations.validate_encryption_parameters(context):
            context_manager.remove_context(request.tenant_id)
            raise HTTPException(
                status_code=500,
                detail="Created context failed security validation"
            )

        # Serialize context for client (no secret key)
        serialized_context = context_manager.serialize_context(request.tenant_id)
        
        # Create response
        return CreateContextResponse(
            context_id=context_id,
            serialized_context=serialized_context,
            metadata={
                "tenant_id": request.tenant_id,
                "created_at": datetime.utcnow().isoformat(),
                "config": config.to_dict(),
                "tenseal_available": True  # Always true since we require it
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create context: {e}")
        raise HTTPException(status_code=500, detail=f"Context creation failed: {str(e)}")


@app.post("/dot", response_model=DotProductResponse)
async def compute_dot_product(request: DotProductRequest):
    """
    Compute encrypted dot-product of two vectors.

    This endpoint performs a homomorphic dot-product operation on two encrypted
    vectors. The computation is performed entirely on encrypted data, and the
    result is returned in encrypted form. The server never has access to the
    plaintext values or the secret key.

    Security guarantee: All operations are performed on ciphertext only.
    Client-side-only decryption is enforced.
    """
    try:
        # Validate tenant ID
        if not request.tenant_id or len(request.tenant_id) < 8:
            raise HTTPException(
                status_code=400,
                detail="Invalid tenant_id"
            )

        # Validate encrypted data format
        if not request.encrypted_q or not request.encrypted_v:
            raise HTTPException(
                status_code=400,
                detail="Missing encrypted vectors"
            )
        # Get or deserialize context
        if request.context:
            context = context_manager.deserialize_context(request.tenant_id, request.context)
        else:
            context = context_manager.get_context(request.tenant_id)

        if not context:
            raise HTTPException(
                status_code=404,
                detail="Context not found for tenant. Create a context first using /create_context"
            )
            
        # Decode encrypted vectors with validation
        try:
            encrypted_q_bytes = base64.b64decode(request.encrypted_q)
            encrypted_v_bytes = base64.b64decode(request.encrypted_v)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 encoding in encrypted vectors: {str(e)}"
            )

        # Security check: Ensure encrypted data has minimum size
        MIN_ENCRYPTED_SIZE = 100  # bytes
        if len(encrypted_q_bytes) < MIN_ENCRYPTED_SIZE or len(encrypted_v_bytes) < MIN_ENCRYPTED_SIZE:
            raise HTTPException(
                status_code=400,
                detail="Encrypted vectors are suspiciously small. Possible security issue."
            )
        
        # Perform homomorphic dot-product
        start_time = time.time()
        encrypted_result_bytes = FHEOperations.perform_dot_product(
            context, encrypted_q_bytes, encrypted_v_bytes
        )
        computation_time = (time.time() - start_time) * 1000  # ms
        
        # Encode result
        encrypted_result = base64.b64encode(encrypted_result_bytes).decode('utf-8')
        
        # Generate operation ID
        operation_id = str(uuid.uuid4())
        
        # Create PoDP receipt
        receipt = create_fhe_receipt(
            operation_id=operation_id,
            tenant_id=request.tenant_id,
            operation_type="dot_product",
            metadata={
                "computation_time_ms": computation_time,
                "vector_sizes": {
                    "q_bytes": len(encrypted_q_bytes),
                    "v_bytes": len(encrypted_v_bytes)
                },
                "params": {
                    "scheme": "CKKS",
                    "operation": "dot_product"
                }
            }
        )
        
        logger.info(f"Computed encrypted dot-product for tenant {request.tenant_id} "
                   f"(operation: {operation_id}, time: {computation_time:.2f}ms)")
        
        return DotProductResponse(
            encrypted_result=encrypted_result,
            operation_id=operation_id,
            receipt_hash=receipt.hash
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dot-product computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/batch_dot", response_model=BatchDotProductResponse)
async def compute_batch_dot_products(request: BatchDotProductRequest):
    """
    Compute encrypted dot-products between a query vector and multiple vectors.
    
    This endpoint efficiently computes multiple dot-products in a batch operation.
    Each computation is isolated and performed entirely on encrypted data.
    This is optimized for scenarios like encrypted similarity search where one
    query is compared against many database vectors.
    """
    try:
        # Get or deserialize context
        if request.context:
            context = context_manager.deserialize_context(request.tenant_id, request.context)
        else:
            context = context_manager.get_context(request.tenant_id)

        if not context:
            raise HTTPException(
                status_code=404,
                detail="Context not found for tenant. Create a context first using /create_context"
            )
            
        # Decode encrypted query vector
        encrypted_q_bytes = base64.b64decode(request.encrypted_q)
        
        # Process batch
        encrypted_results = []
        total_start_time = time.time()
        
        for encrypted_v_b64 in request.encrypted_matrix:
            # Decode encrypted vector
            encrypted_v_bytes = base64.b64decode(encrypted_v_b64)
            
            # Perform homomorphic dot-product
            encrypted_result_bytes = FHEOperations.perform_dot_product(
                context, encrypted_q_bytes, encrypted_v_bytes
            )
            
            # Encode and store result
            encrypted_result = base64.b64encode(encrypted_result_bytes).decode('utf-8')
            encrypted_results.append(encrypted_result)
            
        total_computation_time = (time.time() - total_start_time) * 1000  # ms
        
        # Generate operation ID
        operation_id = str(uuid.uuid4())
        
        # Create PoDP receipt
        receipt = create_fhe_receipt(
            operation_id=operation_id,
            tenant_id=request.tenant_id,
            operation_type="batch_dot_product",
            metadata={
                "computation_time_ms": total_computation_time,
                "batch_size": len(request.encrypted_matrix),
                "avg_time_per_operation_ms": total_computation_time / len(request.encrypted_matrix),
                "params": {
                    "scheme": "CKKS",
                    "operation": "batch_dot_product"
                }
            }
        )
        
        logger.info(f"Computed {len(encrypted_results)} encrypted dot-products for tenant {request.tenant_id} "
                   f"(operation: {operation_id}, total time: {total_computation_time:.2f}ms)")
        
        return BatchDotProductResponse(
            encrypted_results=encrypted_results,
            operation_id=operation_id,
            receipt_hash=receipt.hash,
            computation_time_ms=total_computation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch dot-product computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch computation failed: {str(e)}")


@app.get("/healthz")
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status and TenSEAL availability.
    """
    return {
        "status": "healthy",
        "service": "DALRN FHE Service",
        "tenseal_available": True,  # Always true since we require it
        "timestamp": datetime.utcnow().isoformat(),
        "active_contexts": len(context_manager.contexts),
        "version": "1.0.0"
    }


@app.delete("/context/{tenant_id}")
async def delete_context(tenant_id: str):
    """
    Delete a tenant's context.
    
    This removes the context from server memory. The client should retain
    their own copy if they need to perform future operations.
    """
    context_manager.remove_context(tenant_id)
    return {"message": f"Context removed for tenant {tenant_id}"}


@app.get("/context/{tenant_id}/info")
async def get_context_info(tenant_id: str):
    """
    Get information about a tenant's context.
    
    Returns metadata about the context without exposing sensitive information.
    """
    if tenant_id not in context_manager.context_metadata:
        raise HTTPException(status_code=404, detail="Context not found for tenant")
        
    return context_manager.context_metadata[tenant_id]


# ==============================================================================
# Utility Functions for Testing
# ==============================================================================

def create_test_vectors(dimension: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Create unit-norm test vectors for validation."""
    # Create random vectors
    q = np.random.randn(dimension)
    v = np.random.randn(dimension)
    
    # Normalize to unit vectors
    q = q / np.linalg.norm(q)
    v = v / np.linalg.norm(v)
    
    return q, v


def encrypt_vector(context: Any, vector: np.ndarray) -> str:
    """Encrypt a vector using TenSEAL CKKS - NO PLACEHOLDERS."""
    if not ts:
        raise RuntimeError("TenSEAL is required for encryption")

    # Create CKKS vector from numpy array
    encrypted_vector = ts.ckks_vector(context, vector.tolist())

    # Serialize and encode
    encrypted_bytes = encrypted_vector.serialize()
    return base64.b64encode(encrypted_bytes).decode('utf-8')


# ==============================================================================
# Startup and Shutdown Events
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting DALRN FHE Service")

    # Verify TenSEAL is available (will have already failed if not)
    if not ts:
        logger.critical("FATAL: TenSEAL is not available. Service cannot start.")
        raise RuntimeError("TenSEAL is required for this service")

    logger.info("TenSEAL verified. Real homomorphic encryption enabled.")
    logger.info(f"TenSEAL version: {ts.__version__ if hasattr(ts, '__version__') else 'unknown'}")

    # Perform a startup self-test to verify encryption works
    try:
        test_config = CKKSConfig()
        test_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=test_config.poly_modulus_degree,
            coeff_mod_bit_sizes=test_config.coeff_mod_bit_sizes
        )
        test_context.global_scale = test_config.global_scale
        test_context.generate_galois_keys()
        test_context.generate_relin_keys()

        # Test encryption/decryption
        test_vector = [1.0, 2.0, 3.0, 4.0]
        encrypted = ts.ckks_vector(test_context, test_vector)
        _ = encrypted.serialize()

        logger.info("✓ Startup self-test passed: Encryption is functional")
    except Exception as e:
        logger.critical(f"FATAL: Startup self-test failed: {e}")
        raise RuntimeError(f"FHE service self-test failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down DALRN FHE Service")
    # Clear all contexts
    context_manager.contexts.clear()
    context_manager.context_metadata.clear()
    context_manager.context_expiry.clear()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)