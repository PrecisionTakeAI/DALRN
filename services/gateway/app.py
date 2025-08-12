"""
DALRN Gateway Service

Central gateway for managing disputes, privacy budgets, and federated learning operations.
"""

import os
import json
import logging
import httpx
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service URLs
EPS_LEDGER_URL = os.getenv("EPS_LEDGER_URL", "http://localhost:8001")
NEGOTIATION_URL = os.getenv("NEGOTIATION_URL", "http://localhost:8002")
FHE_SERVICE_URL = os.getenv("FHE_SERVICE_URL", "http://localhost:8003")
FL_SERVICE_URL = os.getenv("FL_SERVICE_URL", "http://localhost:8004")

# ============================================================================
# Data Models
# ============================================================================

class DisputeStatus(str, Enum):
    """Dispute status enumeration."""
    PENDING = "pending"
    NEGOTIATING = "negotiating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class DisputeType(str, Enum):
    """Types of disputes supported."""
    PRIVACY_VIOLATION = "privacy_violation"
    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    BUDGET_DISPUTE = "budget_dispute"
    AGGREGATION_FAILURE = "aggregation_failure"


class CreateDisputeRequest(BaseModel):
    """Request model for creating a new dispute."""
    tenant_id: str = Field(..., description="Tenant identifier")
    model_id: str = Field(..., description="Model identifier")
    dispute_type: DisputeType = Field(..., description="Type of dispute")
    description: str = Field(..., description="Dispute description")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DisputeResponse(BaseModel):
    """Response model for dispute information."""
    dispute_id: str
    tenant_id: str
    model_id: str
    status: DisputeStatus
    dispute_type: DisputeType
    description: str
    created_at: datetime
    updated_at: datetime
    privacy_budget: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any]


class FederatedLearningRequest(BaseModel):
    """Request model for initiating federated learning."""
    tenant_id: str
    model_id: str
    framework: str = Field("flower", description="FL framework to use")
    num_rounds: int = Field(10, gt=0)
    num_clients: int = Field(5, gt=0)
    privacy_config: Optional[Dict[str, Any]] = None
    aggregation_method: str = Field("fedavg")
    robust_aggregation: bool = Field(False)


# ============================================================================
# Gateway Service
# ============================================================================

class GatewayService:
    """Main gateway service for DALRN."""
    
    def __init__(self):
        self.disputes: Dict[str, Dict[str, Any]] = {}
        self.http_client = None
    
    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        return self.http_client
    
    async def check_privacy_budget(self, tenant_id: str, model_id: str) -> Dict[str, Any]:
        """Check privacy budget from epsilon-ledger service."""
        try:
            client = await self.get_http_client()
            response = await client.get(f"{EPS_LEDGER_URL}/budget/{tenant_id}/{model_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get budget: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error checking privacy budget: {e}")
            return None
    
    async def precheck_privacy_operation(
        self, tenant_id: str, model_id: str, eps_round: float
    ) -> Dict[str, Any]:
        """Pre-check if privacy operation is allowed."""
        try:
            client = await self.get_http_client()
            response = await client.post(
                f"{EPS_LEDGER_URL}/precheck",
                json={
                    "tenant_id": tenant_id,
                    "model_id": model_id,
                    "eps_round": eps_round
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Precheck failed: {response.status_code}")
                return {"allowed": False, "message": "Precheck failed"}
        except Exception as e:
            logger.error(f"Error in privacy precheck: {e}")
            return {"allowed": False, "message": str(e)}
    
    async def create_dispute(self, request: CreateDisputeRequest) -> DisputeResponse:
        """Create a new dispute."""
        dispute_id = f"dispute_{uuid4().hex[:8]}"
        
        # Check privacy budget
        budget_info = await self.check_privacy_budget(request.tenant_id, request.model_id)
        
        # Determine initial status based on budget
        initial_status = DisputeStatus.PENDING
        if budget_info and budget_info.get("remaining_budget", 0) <= 0:
            initial_status = DisputeStatus.BUDGET_EXHAUSTED
        
        dispute = {
            "dispute_id": dispute_id,
            "tenant_id": request.tenant_id,
            "model_id": request.model_id,
            "status": initial_status,
            "dispute_type": request.dispute_type,
            "description": request.description,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "privacy_budget": {
                "total_budget": budget_info.get("total_budget", 0) if budget_info else 0,
                "total_spent": budget_info.get("total_spent", 0) if budget_info else 0,
                "remaining_budget": budget_info.get("remaining_budget", 0) if budget_info else 0
            } if budget_info else None,
            "metadata": request.metadata
        }
        
        self.disputes[dispute_id] = dispute
        
        return DisputeResponse(**dispute)
    
    async def get_dispute_status(self, dispute_id: str) -> DisputeResponse:
        """Get dispute status with updated privacy budget."""
        if dispute_id not in self.disputes:
            raise HTTPException(status_code=404, detail="Dispute not found")
        
        dispute = self.disputes[dispute_id]
        
        # Update privacy budget information
        budget_info = await self.check_privacy_budget(
            dispute["tenant_id"], dispute["model_id"]
        )
        
        if budget_info:
            dispute["privacy_budget"] = {
                "total_budget": budget_info.get("total_budget", 0),
                "total_spent": budget_info.get("total_spent", 0),
                "remaining_budget": budget_info.get("remaining_budget", 0),
                "num_rounds": budget_info.get("num_rounds", 0),
                "last_update": budget_info.get("last_update")
            }
            
            # Update status if budget exhausted
            if budget_info.get("remaining_budget", 0) <= 0:
                dispute["status"] = DisputeStatus.BUDGET_EXHAUSTED
        
        dispute["updated_at"] = datetime.utcnow()
        
        return DisputeResponse(**dispute)
    
    async def initiate_federated_learning(
        self, request: FederatedLearningRequest
    ) -> Dict[str, Any]:
        """Initiate federated learning with privacy budget checks."""
        # First, check privacy budget
        eps_per_round = request.privacy_config.get("epsilon_per_round", 0.5) if request.privacy_config else 0.5
        total_eps_needed = eps_per_round * request.num_rounds
        
        precheck_result = await self.precheck_privacy_operation(
            request.tenant_id, request.model_id, total_eps_needed
        )
        
        if not precheck_result.get("allowed", False):
            return {
                "success": False,
                "message": "Insufficient privacy budget for FL operation",
                "budget_info": precheck_result
            }
        
        # Create FL job (in production, this would call FL service)
        fl_job = {
            "job_id": f"fl_{uuid4().hex[:8]}",
            "tenant_id": request.tenant_id,
            "model_id": request.model_id,
            "framework": request.framework,
            "status": "initiated",
            "num_rounds": request.num_rounds,
            "num_clients": request.num_clients,
            "aggregation_method": request.aggregation_method,
            "robust_aggregation": request.robust_aggregation,
            "privacy_budget_allocated": total_eps_needed,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "fl_job": fl_job,
            "budget_info": precheck_result
        }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="DALRN Gateway Service",
    description="Central gateway for privacy-preserving federated learning operations",
    version="1.0.0"
)

# Initialize gateway service
gateway = GatewayService()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if gateway.http_client:
        await gateway.http_client.aclose()


@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check epsilon-ledger service health
    eps_ledger_healthy = False
    try:
        client = await gateway.get_http_client()
        response = await client.get(f"{EPS_LEDGER_URL}/health")
        eps_ledger_healthy = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy",
        "service": "gateway",
        "dependencies": {
            "epsilon_ledger": eps_ledger_healthy
        }
    }


@app.post("/dispute", response_model=DisputeResponse)
async def create_dispute(request: CreateDisputeRequest):
    """
    Create a new dispute.
    
    Automatically checks privacy budget and sets appropriate status.
    """
    try:
        response = await gateway.create_dispute(request)
        logger.info(f"Created dispute {response.dispute_id} for {request.tenant_id}/{request.model_id}")
        return response
    except Exception as e:
        logger.error(f"Failed to create dispute: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{dispute_id}", response_model=DisputeResponse)
async def get_dispute_status(dispute_id: str):
    """
    Get dispute status with current privacy budget information.
    
    Shows remaining budget and whether operations are blocked due to budget exhaustion.
    """
    try:
        response = await gateway.get_dispute_status(dispute_id)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dispute status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/disputes")
async def list_disputes(
    tenant_id: Optional[str] = None,
    model_id: Optional[str] = None,
    status: Optional[DisputeStatus] = None
):
    """
    List all disputes with optional filtering.
    """
    disputes = list(gateway.disputes.values())
    
    # Apply filters
    if tenant_id:
        disputes = [d for d in disputes if d["tenant_id"] == tenant_id]
    if model_id:
        disputes = [d for d in disputes if d["model_id"] == model_id]
    if status:
        disputes = [d for d in disputes if d["status"] == status]
    
    return {"disputes": disputes, "count": len(disputes)}


@app.post("/fl/initiate")
async def initiate_federated_learning(request: FederatedLearningRequest):
    """
    Initiate federated learning with privacy budget checks.
    
    Verifies sufficient privacy budget before starting FL operations.
    """
    try:
        result = await gateway.initiate_federated_learning(request)
        
        if result["success"]:
            logger.info(f"Initiated FL job for {request.tenant_id}/{request.model_id}")
        else:
            logger.warning(f"FL initiation blocked due to budget: {result['message']}")
        
        return result
    except Exception as e:
        logger.error(f"Failed to initiate FL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/{tenant_id}/{model_id}")
async def get_privacy_budget(tenant_id: str, model_id: str):
    """
    Get current privacy budget status for a tenant/model pair.
    
    Proxies to epsilon-ledger service.
    """
    try:
        budget_info = await gateway.check_privacy_budget(tenant_id, model_id)
        
        if budget_info is None:
            raise HTTPException(status_code=503, detail="Epsilon-ledger service unavailable")
        
        return budget_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get privacy budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/budget/precheck")
async def precheck_privacy_operation(
    tenant_id: str,
    model_id: str,
    eps_round: float
):
    """
    Pre-check if a privacy operation is allowed within budget.
    
    Use this before starting any privacy-sensitive operation.
    """
    try:
        result = await gateway.precheck_privacy_operation(tenant_id, model_id, eps_round)
        return result
    except Exception as e:
        logger.error(f"Precheck failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GATEWAY_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
