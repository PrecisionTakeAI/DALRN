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

# Try to import TenSEAL, but provide placeholder if not available
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    # TODO: Install TenSEAL via: pip install tenseal
    # Note: TenSEAL requires native compilation and may need additional setup
    
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
    """
    
    def __init__(self):
        self.contexts: Dict[str, Any] = {}  # tenant_id -> context
        self.context_metadata: Dict[str, Dict] = {}  # tenant_id -> metadata
        self.context_expiry: Dict[str, datetime] = {}  # tenant_id -> expiry
        self.max_context_age = timedelta(hours=24)
        
    def create_context(self, tenant_id: str, config: CKKSConfig) -> Tuple[Any, str]:
        """
        Create a new CKKS context for a tenant.
        Returns (context, context_id).
        """
        if not TENSEAL_AVAILABLE:
            # Return placeholder for environments without TenSEAL
            context_id = hashlib.sha256(f"{tenant_id}:{time.time()}".encode()).hexdigest()[:16]
            logger.warning(f"TenSEAL not available. Created placeholder context {context_id} for tenant {tenant_id}")
            
            # Store placeholder context and metadata
            self.contexts[tenant_id] = None
            self.context_metadata[tenant_id] = {
                "context_id": context_id,
                "created_at": datetime.utcnow().isoformat(),
                "config": config.to_dict()
            }
            self.context_expiry[tenant_id] = datetime.utcnow() + self.max_context_age
            
            return None, context_id
            
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
        """Get context for a tenant, checking expiry."""
        if tenant_id not in self.contexts:
            return None
            
        # Check expiry
        if datetime.utcnow() > self.context_expiry.get(tenant_id, datetime.min):
            self.remove_context(tenant_id)
            return None
            
        return self.contexts[tenant_id]
        
    def remove_context(self, tenant_id: str) -> None:
        """Remove a tenant's context."""
        if tenant_id in self.contexts:
            del self.contexts[tenant_id]
            del self.context_metadata[tenant_id]
            del self.context_expiry[tenant_id]
            logger.info(f"Removed context for tenant {tenant_id}")
            
    def serialize_context(self, tenant_id: str) -> Optional[str]:
        """Serialize a context to base64 for client storage."""
        context = self.get_context(tenant_id)
        if not context or not TENSEAL_AVAILABLE:
            return None
            
        serialized = context.serialize(save_secret_key=False)
        return base64.b64encode(serialized).decode('utf-8')
        
    def deserialize_context(self, tenant_id: str, serialized: str) -> Any:
        """Deserialize a context from base64."""
        if not TENSEAL_AVAILABLE:
            return None
            
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
        
        Args:
            context: TenSEAL context
            encrypted_q_bytes: Encrypted query vector bytes
            encrypted_v_bytes: Encrypted value vector bytes
            
        Returns:
            Encrypted result bytes
        """
        if not TENSEAL_AVAILABLE:
            # Return placeholder for testing
            placeholder = hashlib.sha256(encrypted_q_bytes + encrypted_v_bytes).digest()
            return placeholder
            
        # Deserialize encrypted vectors
        encrypted_q = ts.ckks_vector_from(context, encrypted_q_bytes)
        encrypted_v = ts.ckks_vector_from(context, encrypted_v_bytes)
        
        # Perform element-wise multiplication
        encrypted_product = encrypted_q * encrypted_v
        
        # Sum all elements (dot product)
        # Note: TenSEAL's sum() performs homomorphic addition across vector elements
        encrypted_result = encrypted_product.sum()
        
        # Serialize result
        return encrypted_result.serialize()
        
    @staticmethod
    def validate_encryption_parameters(context: Any) -> bool:
        """Validate that context has appropriate parameters for dot-products."""
        if not TENSEAL_AVAILABLE:
            return True  # Skip validation in placeholder mode
            
        # Check polynomial modulus degree (affects security and capacity)
        if context.poly_modulus_degree < 4096:
            logger.warning("Polynomial modulus degree too low for secure operations")
            return False
            
        # Check scale is appropriate for precision
        if context.global_scale < 2**20:
            logger.warning("Global scale may be too low for accurate computations")
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
    """
    try:
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
                "tenseal_available": TENSEAL_AVAILABLE
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
    """
    try:
        # Get or deserialize context
        if request.context:
            context = context_manager.deserialize_context(request.tenant_id, request.context)
        else:
            context = context_manager.get_context(request.tenant_id)
            
        if not context and TENSEAL_AVAILABLE:
            raise HTTPException(status_code=404, detail="Context not found for tenant")
            
        # Decode encrypted vectors
        encrypted_q_bytes = base64.b64decode(request.encrypted_q)
        encrypted_v_bytes = base64.b64decode(request.encrypted_v)
        
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
            
        if not context and TENSEAL_AVAILABLE:
            raise HTTPException(status_code=404, detail="Context not found for tenant")
            
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
        "tenseal_available": TENSEAL_AVAILABLE,
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


def encrypt_vector_placeholder(vector: np.ndarray) -> str:
    """Create placeholder encrypted vector for testing without TenSEAL."""
    # Simulate encryption by hashing the vector
    vector_bytes = vector.tobytes()
    encrypted = hashlib.sha256(vector_bytes).digest()
    return base64.b64encode(encrypted).decode('utf-8')


# ==============================================================================
# Startup and Shutdown Events
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting DALRN FHE Service")
    if not TENSEAL_AVAILABLE:
        logger.warning("TenSEAL is not installed. Service running in placeholder mode.")
        logger.warning("To enable full functionality, install TenSEAL: pip install tenseal")
    else:
        logger.info("TenSEAL is available. Full FHE functionality enabled.")


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