<<<<<<< HEAD
"""DALRN Gateway API with PoDP Middleware"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from services.common.podp import Receipt, ReceiptChain
from services.common.ipfs import put_json, get_json
from services.chain.client import AnchorClient

# Configure logging with PII redaction
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redaction helper for logging
def redact_pii(data: Any) -> Any:
    """Redact PII from data before logging"""
    if isinstance(data, dict):
        redacted = {}
        sensitive_keys = {'parties', 'email', 'name', 'address', 'phone', 'ssn', 'ein'}
        for k, v in data.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                redacted[k] = "[REDACTED]"
            else:
                redacted[k] = redact_pii(v)
        return redacted
    elif isinstance(data, list):
        return [redact_pii(item) for item in data]
    elif isinstance(data, str) and len(data) > 20:
        # Redact long strings that might contain sensitive data
        return f"{data[:8]}...[REDACTED]"
    return data

# In-memory storage for demo (use Redis/DB in production)
dispute_storage: Dict[str, dict] = {}
receipt_chains: Dict[str, ReceiptChain] = {}

# Initialize clients
anchor_client = AnchorClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting DALRN Gateway")
    # Initialize any async resources here
    yield
    # Cleanup
    logger.info("Shutting down DALRN Gateway")

app = FastAPI(
    title="DALRN Gateway",
    description="Decentralized Alternative Legal Resolution Network Gateway with PoDP",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SubmitDisputeRequest(BaseModel):
    """Request model for dispute submission"""
    parties: List[str] = Field(..., min_length=2, description="List of party identifiers")
    jurisdiction: str = Field(..., min_length=2, description="Jurisdiction code")
    cid: str = Field(..., description="IPFS CID of encrypted document bundle")
    enc_meta: dict = Field(default_factory=dict, description="Encrypted metadata")
    
    @field_validator('cid')
    @classmethod
    def validate_cid(cls, v):
        """Validate IPFS CID format"""
        if not v or len(v) < 10:
            raise ValueError("Invalid CID format")
        return v
    
    @field_validator('parties')
    @classmethod
    def validate_parties(cls, v):
        """Ensure at least 2 parties"""
        if len(v) < 2:
            raise ValueError("At least 2 parties required")
        return v

class SubmitDisputeResponse(BaseModel):
    """Response model for dispute submission"""
    dispute_id: str
    receipt_id: str
    anchor_uri: Optional[str] = None
    anchor_tx: Optional[str] = None
    status: str = "submitted"

class DisputeStatusResponse(BaseModel):
    """Response model for dispute status"""
    dispute_id: str
    phase: str
    receipts: List[dict]
    anchor_tx: Optional[str] = None
    eps_budget: Optional[float] = None
    last_updated: str
    receipt_chain_uri: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    ok: bool
    timestamp: str
    version: str = "1.0.0"
    services: dict = Field(default_factory=dict)

# PoDP Middleware
@app.middleware("http")
async def podp_middleware(request: Request, call_next):
    """Middleware for Proof of Data Possession tracking"""
    start_time = datetime.now(timezone.utc)
    
    # Generate request ID for tracking
    request_id = f"req_{uuid4().hex[:8]}"
    request.state.request_id = request_id
    
    # Log request (redacted)
    logger.info(
        f"Request received",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response time
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        f"Request completed",
        extra={
            "request_id": request_id,
            "duration_seconds": duration,
            "status_code": response.status_code
        }
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response

# Dependency for rate limiting (simple in-memory implementation)
request_counts: Dict[str, List[float]] = {}

async def rate_limit(request: Request):
    """Simple rate limiting dependency"""
    client_id = request.client.host if request.client else "unknown"
    current_time = datetime.now(timezone.utc).timestamp()
    
    # Clean old entries (older than 1 minute)
    if client_id in request_counts:
        request_counts[client_id] = [
            t for t in request_counts[client_id] 
            if current_time - t < 60
        ]
    else:
        request_counts[client_id] = []
    
    # Check rate limit (30 requests per minute)
    if len(request_counts[client_id]) >= 30:
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    request_counts[client_id].append(current_time)

# Endpoints
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check service connectivity
        ipfs_healthy = True  # Would check IPFS connection
        chain_healthy = True  # Would check chain connection
        
        return HealthResponse(
            ok=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            services={
                "ipfs": "healthy" if ipfs_healthy else "unhealthy",
                "chain": "healthy" if chain_healthy else "unhealthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            ok=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            services={}
        )

@app.post("/submit-dispute", 
          response_model=SubmitDisputeResponse,
          status_code=status.HTTP_201_CREATED,
          dependencies=[Depends(rate_limit)])
async def submit_dispute(
    body: SubmitDisputeRequest,
    request: Request
) -> SubmitDisputeResponse:
    """Submit a new dispute with PoDP tracking"""
    try:
        # Generate deterministic dispute ID
        dispute_id = Receipt.new_id(prefix="disp_")
        
        # Log submission (with redacted PII)
        logger.info(
            "Processing dispute submission",
            extra={
                "dispute_id": dispute_id,
                "request_id": request.state.request_id,
                "jurisdiction": body.jurisdiction,
                "party_count": len(body.parties)
            }
        )
        
        # Create initial receipt for intake
        receipt = Receipt(
            receipt_id=Receipt.new_id(prefix="rcpt_"),
            dispute_id=dispute_id,
            step="INTAKE_V1",
            inputs={
                "cid_bundle": body.cid,
                "party_count": len(body.parties),
                "submission_time": datetime.now(timezone.utc).isoformat()
            },
            params={
                "jurisdiction": body.jurisdiction,
                "version": "1.0.0"
            },
            artifacts={
                "request_id": request.state.request_id
            },
            ts=datetime.now(timezone.utc).isoformat()
        ).finalize()
        
        # Build receipt chain
        chain = ReceiptChain(
            dispute_id=dispute_id,
            receipts=[receipt]
        ).finalize()
        
        # Store receipt chain
        receipt_chains[dispute_id] = chain
        
        # Upload to IPFS
        chain_dict = chain.model_dump(exclude_none=True)
        try:
            uri = await asyncio.to_thread(put_json, chain_dict)
            logger.info(f"Receipt chain uploaded to IPFS: {uri[:30]}...")
        except Exception as e:
            logger.error(f"IPFS upload failed: {str(e)}")
            uri = None
        
        # Anchor to chain
        anchor_tx = None
        if uri:
            try:
                anchor_tx = await asyncio.to_thread(
                    anchor_client.anchor_root,
                    dispute_id,
                    chain.merkle_root,
                    b"\x00" * 32,  # Model hash placeholder
                    0,  # Round 0 for initial submission
                    uri,
                    [b"PoDP", b"INTAKE"]
                )
                logger.info(f"Anchored to chain: {anchor_tx}")
            except Exception as e:
                logger.error(f"Chain anchoring failed: {str(e)}")
        
        # Store dispute metadata
        dispute_storage[dispute_id] = {
            "dispute_id": dispute_id,
            "phase": "INTAKE",
            "parties_hash": Receipt.new_id(prefix="ph_"),  # Hash parties for privacy
            "jurisdiction": body.jurisdiction,
            "cid": body.cid,
            "enc_meta": body.enc_meta,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "anchor_tx": anchor_tx,
            "receipt_chain_uri": uri,
            "receipts": [receipt.receipt_id]
        }
        
        return SubmitDisputeResponse(
            dispute_id=dispute_id,
            receipt_id=receipt.receipt_id,
            anchor_uri=uri,
            anchor_tx=anchor_tx,
            status="submitted"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in submit_dispute: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/status/{dispute_id}", 
         response_model=DisputeStatusResponse,
         dependencies=[Depends(rate_limit)])
async def get_dispute_status(
    dispute_id: str,
    request: Request
) -> DisputeStatusResponse:
    """Get status of a dispute"""
    try:
        logger.info(
            f"Status request for dispute",
            extra={
                "dispute_id": dispute_id,
                "request_id": request.state.request_id
            }
        )
        
        # Check if dispute exists
        if dispute_id not in dispute_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dispute not found"
            )
        
        dispute = dispute_storage[dispute_id]
        
        # Get receipt chain
        chain = receipt_chains.get(dispute_id)
        receipts = []
        
        if chain:
            # Convert receipts to dict format (redacted)
            for receipt in chain.receipts:
                receipt_dict = receipt.model_dump(exclude_none=True)
                # Redact sensitive fields
                if 'inputs' in receipt_dict:
                    receipt_dict['inputs'] = redact_pii(receipt_dict['inputs'])
                receipts.append(receipt_dict)
        
        # Calculate epsilon budget (placeholder)
        eps_budget = 10.0  # Would be calculated based on DP operations
        
        return DisputeStatusResponse(
            dispute_id=dispute_id,
            phase=dispute.get("phase", "UNKNOWN"),
            receipts=receipts,
            anchor_tx=dispute.get("anchor_tx"),
            eps_budget=eps_budget,
            last_updated=dispute.get("created_at", datetime.now(timezone.utc).isoformat()),
            receipt_chain_uri=dispute.get("receipt_chain_uri")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dispute status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to docs"""
    return {
        "message": "DALRN Gateway API",
        "docs": "/docs",
        "health": "/healthz"
    }

# Additional utility endpoints for development
@app.get("/disputes", include_in_schema=False)
async def list_disputes(request: Request):
    """List all disputes (development only)"""
    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not available in production"
        )
    
    logger.info(f"Listing disputes", extra={"request_id": request.state.request_id})
    
    # Return redacted list
    disputes = []
    for dispute_id, dispute in dispute_storage.items():
        disputes.append({
            "dispute_id": dispute_id,
            "phase": dispute.get("phase"),
            "created_at": dispute.get("created_at"),
            "has_anchor": dispute.get("anchor_tx") is not None
        })
    
    return {"disputes": disputes, "count": len(disputes)}

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        "services.gateway.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
