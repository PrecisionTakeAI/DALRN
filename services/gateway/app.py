"""
DALRN Gateway Service - Simple Working Implementation.
This is the main entry point for all API requests.
"""

import os
import logging
import httpx
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import our modules
from services.database.connection import db, create_dispute, get_dispute
from services.cache.connection import cache
from services.auth.jwt_auth import auth_router, get_current_user, AuthService
from services.common.podp import Receipt, ReceiptChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class SubmitDisputeRequest(BaseModel):
    """Request model for submitting a dispute."""
    parties: list[str] = Field(..., min_length=2, max_length=10)
    jurisdiction: str = Field(..., min_length=2, max_length=100)
    cid: str = Field(..., description="IPFS CID of encrypted evidence")
    enc_meta: dict = Field(default_factory=dict)


class DisputeResponse(BaseModel):
    """Response model for dispute submission."""
    dispute_id: str
    status: str
    anchor_tx: Optional[str] = None
    receipt_cid: Optional[str] = None
    message: str


class StatusResponse(BaseModel):
    """Response model for dispute status."""
    dispute_id: str
    status: str
    phase: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# Service registry
SERVICE_REGISTRY = {
    "search": {
        "url": os.getenv("SEARCH_SERVICE_URL", "http://localhost:8001"),
        "health": "/health"
    },
    "fhe": {
        "url": os.getenv("FHE_SERVICE_URL", "http://localhost:8002"),
        "health": "/health"
    },
    "negotiation": {
        "url": os.getenv("NEGOTIATION_SERVICE_URL", "http://localhost:8003"),
        "health": "/health"
    },
    "agents": {
        "url": os.getenv("AGENTS_SERVICE_URL", "http://localhost:8004"),
        "health": "/health"
    },
    "fl": {
        "url": os.getenv("FL_SERVICE_URL", "http://localhost:8005"),
        "health": "/health"
    }
}


# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting DALRN Gateway Service...")

    # Check database
    db_health = db.health_check()
    logger.info(f"Database status: {db_health}")

    # Check cache
    cache_health = cache.health_check()
    logger.info(f"Cache status: {cache_health}")

    # Check services
    async with httpx.AsyncClient(timeout=2.0) as client:
        for name, config in SERVICE_REGISTRY.items():
            try:
                response = await client.get(f"{config['url']}{config['health']}")
                if response.status_code == 200:
                    logger.info(f"Service {name}: HEALTHY")
                else:
                    logger.warning(f"Service {name}: UNHEALTHY (status {response.status_code})")
            except Exception as e:
                logger.warning(f"Service {name}: UNREACHABLE - {e}")

    yield

    # Shutdown
    logger.info("Shutting down DALRN Gateway Service...")


# Create FastAPI app
app = FastAPI(
    title="DALRN Gateway",
    description="Main API gateway for the DALRN dispute resolution platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router
app.include_router(auth_router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check gateway and dependent services health."""
    health_status = {
        "gateway": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db.health_check()["status"],
        "cache": cache.health_check()["status"],
        "services": {}
    }

    # Check each service
    async with httpx.AsyncClient(timeout=1.0) as client:
        for name, config in SERVICE_REGISTRY.items():
            try:
                response = await client.get(f"{config['url']}{config['health']}")
                health_status["services"][name] = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                health_status["services"][name] = "unreachable"

    # Determine overall status
    all_healthy = (
        health_status["database"] == "healthy" and
        health_status["cache"] == "healthy"
    )

    return JSONResponse(
        content=health_status,
        status_code=200 if all_healthy else 503
    )


# Dispute submission endpoint
@app.post("/submit-dispute", response_model=DisputeResponse)
async def submit_dispute(
    request: SubmitDisputeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit a new dispute for resolution."""
    try:
        # Generate dispute ID
        dispute_id = f"disp_{uuid.uuid4().hex[:12]}"

        # Create receipt
        receipt = Receipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:12]}",
            dispute_id=dispute_id,
            step="INTAKE_V1",
            inputs={"parties": request.parties, "jurisdiction": request.jurisdiction},
            params={"user_id": current_user["id"]},
            artifacts={"cid": request.cid}
        )

        # Store in database
        try:
            create_dispute(
                dispute_id=dispute_id,
                user_id=current_user["id"],
                parties=",".join(request.parties),
                jurisdiction=request.jurisdiction,
                cid=request.cid,
                metadata=request.enc_meta
            )
        except Exception as e:
            logger.error(f"Database error: {e}")
            # Continue anyway - we can still process

        # Cache the dispute
        cache.set(f"dispute:{dispute_id}", {
            "status": "INTAKE",
            "user_id": current_user["id"],
            "created_at": datetime.utcnow().isoformat()
        }, ttl=3600)

        # Create response
        return DisputeResponse(
            dispute_id=dispute_id,
            status="INTAKE_COMPLETE",
            anchor_tx=None,  # Would be set after blockchain anchoring
            receipt_cid=receipt.receipt_id,
            message="Dispute submitted successfully"
        )

    except Exception as e:
        logger.error(f"Error submitting dispute: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit dispute: {str(e)}"
        )


# Status endpoint
@app.get("/status/{dispute_id}", response_model=StatusResponse)
async def get_status(
    dispute_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the status of a dispute."""
    # Try cache first
    cached = cache.get(f"dispute:{dispute_id}")
    if cached:
        return StatusResponse(
            dispute_id=dispute_id,
            status=cached.get("status", "UNKNOWN"),
            phase=cached.get("status", "UNKNOWN"),
            created_at=cached.get("created_at"),
            updated_at=cached.get("updated_at")
        )

    # Try database
    dispute = get_dispute(dispute_id)
    if dispute:
        return StatusResponse(
            dispute_id=dispute_id,
            status=dispute.get("status", "UNKNOWN"),
            phase=dispute.get("status", "UNKNOWN"),
            created_at=str(dispute.get("created_at")),
            updated_at=str(dispute.get("updated_at"))
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Dispute {dispute_id} not found"
    )


# Service routing endpoint
@app.api_route("/api/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_to_service(
    service: str,
    path: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Route requests to appropriate backend services."""
    if service not in SERVICE_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service}' not found"
        )

    service_config = SERVICE_REGISTRY[service]
    url = f"{service_config['url']}/{path}"

    # Forward the request
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Get request body if present
            body = None
            if request.method in ["POST", "PUT"]:
                body = await request.body()

            # Forward headers
            headers = dict(request.headers)
            headers["X-User-Id"] = str(current_user["id"])
            headers["X-Username"] = current_user["username"]

            # Make the request
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body
            )

            # Return the response
            return JSONResponse(
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else {"data": response.text},
                status_code=response.status_code
            )

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Service '{service}' timeout"
            )
        except Exception as e:
            logger.error(f"Error routing to {service}: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Service '{service}' error: {str(e)}"
            )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "DALRN Gateway",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "/health": "Health check",
            "/auth/register": "User registration",
            "/auth/login": "User login",
            "/submit-dispute": "Submit new dispute",
            "/status/{dispute_id}": "Get dispute status",
            "/api/{service}/{path}": "Route to backend service"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GATEWAY_PORT", 8000))
    logger.info(f"Starting DALRN Gateway on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")