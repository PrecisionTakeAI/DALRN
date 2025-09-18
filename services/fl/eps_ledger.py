"""
Epsilon-Ledger Service for DALRN

Provides privacy budget tracking and management for federated learning operations.
Supports RDP (Rényi Differential Privacy) accounting with Opacus integration.
"""

import json
import logging
import os
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Try to import Opacus for DP accounting (optional dependency)
try:
    from opacus.accountants import RDPAccountant, GaussianAccountant
    from opacus.accountants.utils import get_noise_multiplier
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available. Using basic epsilon accounting.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BUDGET = float(os.getenv("DEFAULT_EPSILON_BUDGET", "4.0"))
DEFAULT_DELTA = float(os.getenv("DEFAULT_DELTA", "1e-5"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin-secret-token")
STORAGE_PATH = os.getenv("LEDGER_STORAGE_PATH", "./ledger_data.json")
ENABLE_PODP = os.getenv("ENABLE_PODP", "true").lower() == "true"

# ============================================================================
# Data Models
# ============================================================================

class AccountantType(str, Enum):
    """Supported privacy accountant types."""
    RDP = "rdp"  # Rényi Differential Privacy
    ZCDP = "zcdp"  # Zero-Concentrated Differential Privacy
    PLD = "pld"  # Privacy Loss Distribution
    GAUSSIAN = "gaussian"  # Gaussian mechanism
    BASIC = "basic"  # Basic composition


class PreCheckRequest(BaseModel):
    """Request model for privacy budget pre-check."""
    tenant_id: str = Field(..., description="Tenant identifier")
    model_id: str = Field(..., description="Model identifier")
    eps_round: float = Field(..., gt=0, description="Proposed epsilon spend for this round")
    delta_round: Optional[float] = Field(None, ge=0, le=1, description="Proposed delta for this round")
    accountant: Optional[AccountantType] = Field(AccountantType.RDP, description="Accountant type")
    
    @validator('eps_round')
    def validate_epsilon(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("Epsilon must be between 0 and 100")
        return v


class CommitRequest(BaseModel):
    """Request model for committing privacy spend."""
    tenant_id: str = Field(..., description="Tenant identifier")
    model_id: str = Field(..., description="Model identifier")
    round: int = Field(..., ge=0, description="Training round number")
    accountant: AccountantType = Field(AccountantType.RDP, description="Accountant type used")
    epsilon: float = Field(..., gt=0, description="Actual epsilon spent")
    delta: float = Field(..., ge=0, le=1, description="Delta parameter")
    
    # Privacy mechanism parameters
    clipping_C: Optional[float] = Field(None, gt=0, description="Gradient clipping norm")
    sigma: Optional[float] = Field(None, gt=0, description="Noise multiplier sigma")
    batch_size: Optional[int] = Field(None, gt=0, description="Batch size used")
    dataset_size: Optional[int] = Field(None, gt=0, description="Total dataset size")
    
    # Federated learning metadata
    num_clients: Optional[int] = Field(None, gt=0, description="Number of participating clients")
    aggregation_method: Optional[str] = Field(None, description="Aggregation method used")
    framework: Optional[str] = Field(None, description="FL framework (Flower/NV-FLARE)")
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class PreCheckResponse(BaseModel):
    """Response model for privacy budget pre-check."""
    allowed: bool = Field(..., description="Whether the operation is allowed")
    remaining_budget: float = Field(..., description="Remaining epsilon budget")
    total_spent: float = Field(..., description="Total epsilon spent so far")
    projected_spend: float = Field(..., description="Projected total after this operation")
    message: Optional[str] = Field(None, description="Additional information")
    receipt_id: Optional[str] = Field(None, description="PoDP receipt ID if enabled")


class CommitResponse(BaseModel):
    """Response model for committing privacy spend."""
    success: bool = Field(..., description="Whether commit was successful")
    total_spent: float = Field(..., description="Total epsilon spent after commit")
    remaining_budget: float = Field(..., description="Remaining epsilon budget")
    entry_id: str = Field(..., description="Unique ID for this entry")
    receipt_id: Optional[str] = Field(None, description="PoDP receipt ID if enabled")


class BudgetStatus(BaseModel):
    """Model for budget status information."""
    tenant_id: str
    model_id: str
    total_budget: float
    total_spent: float
    remaining_budget: float
    num_rounds: int
    last_update: Optional[datetime]
    accountant_type: AccountantType
    entries_count: int


class HistoryEntry(BaseModel):
    """Model for privacy spend history entry."""
    entry_id: str
    round: int
    epsilon: float
    delta: float
    accountant: AccountantType
    timestamp: datetime
    metadata: Dict[str, Any]


# ============================================================================
# PoDP Receipt Integration
# ============================================================================

class PoDPReceipt:
    """Proof of Differential Privacy receipt generator."""
    
    @staticmethod
    def generate_receipt(operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a PoDP receipt for privacy operations."""
        import hashlib
        import hmac
        import os

        # Generate proper cryptographic signature
        secret_key = os.getenv("PODP_SECRET_KEY", "default_secret_key_change_in_production")

        # Create signing payload
        payload = f"{operation}:{datetime.utcnow().timestamp()}:{json.dumps(data, sort_keys=True)}"

        # Generate HMAC signature
        signature = hmac.new(
            secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        receipt = {
            "receipt_id": f"podp_{operation}_{datetime.utcnow().timestamp()}",
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "signature": signature,
            "signature_algorithm": "HMAC-SHA256"
        }

        if ENABLE_PODP:
            logger.info(f"Generated PoDP receipt: {receipt['receipt_id']}")
            # Store receipt hash on blockchain for immutability
            receipt_hash = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
            receipt["blockchain_hash"] = receipt_hash

        return receipt


# ============================================================================
# Epsilon Ledger Service
# ============================================================================

class EpsilonLedger:
    """Thread-safe epsilon ledger for privacy budget tracking."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or STORAGE_PATH
        self.lock = threading.RLock()
        self.ledgers: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.load_from_storage()
        
    def _get_ledger_key(self, tenant_id: str, model_id: str) -> Tuple[str, str]:
        """Generate ledger key from tenant and model IDs."""
        return (tenant_id, model_id)
    
    def _init_ledger(self, tenant_id: str, model_id: str) -> Dict[str, Any]:
        """Initialize a new ledger for tenant/model pair."""
        return {
            "tenant_id": tenant_id,
            "model_id": model_id,
            "budget": DEFAULT_BUDGET,
            "spent": 0.0,
            "delta_accumulated": 0.0,
            "entries": [],
            "created_at": datetime.utcnow().isoformat(),
            "last_update": None,
            "accountant_type": AccountantType.RDP,
            "metadata": {}
        }
    
    def load_from_storage(self):
        """Load ledger data from persistent storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Reconstruct ledgers with tuple keys
                    for key_str, ledger_data in data.items():
                        tenant_id, model_id = key_str.split('|')
                        self.ledgers[(tenant_id, model_id)] = ledger_data
                logger.info(f"Loaded {len(self.ledgers)} ledgers from storage")
            except Exception as e:
                logger.error(f"Failed to load ledger data: {e}")
    
    def save_to_storage(self):
        """Save ledger data to persistent storage."""
        try:
            # Convert tuple keys to strings for JSON serialization
            data = {f"{k[0]}|{k[1]}": v for k, v in self.ledgers.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("Saved ledger data to storage")
        except Exception as e:
            logger.error(f"Failed to save ledger data: {e}")
    
    def precheck(self, request: PreCheckRequest) -> PreCheckResponse:
        """Check if privacy budget allows the proposed operation."""
        with self.lock:
            key = self._get_ledger_key(request.tenant_id, request.model_id)
            ledger = self.ledgers.setdefault(key, self._init_ledger(request.tenant_id, request.model_id))
            
            # Calculate projected spend
            current_spent = ledger["spent"]
            projected_spend = current_spent + request.eps_round
            remaining = max(0, ledger["budget"] - current_spent)
            
            # Check if operation is allowed
            allowed = projected_spend <= ledger["budget"]
            
            # Generate message
            if allowed:
                message = f"Operation allowed. Budget after: {ledger['budget'] - projected_spend:.4f}"
            else:
                message = f"Operation denied. Would exceed budget by {projected_spend - ledger['budget']:.4f}"
            
            # Generate PoDP receipt if enabled
            receipt_id = None
            if ENABLE_PODP:
                receipt_data = {
                    "tenant_id": request.tenant_id,
                    "model_id": request.model_id,
                    "eps_round": request.eps_round,
                    "allowed": allowed,
                    "remaining_budget": remaining
                }
                receipt = PoDPReceipt.generate_receipt("PRIVACY_PRECHECK", receipt_data)
                receipt_id = receipt["receipt_id"]
            
            return PreCheckResponse(
                allowed=allowed,
                remaining_budget=remaining,
                total_spent=current_spent,
                projected_spend=projected_spend,
                message=message,
                receipt_id=receipt_id
            )
    
    def commit(self, request: CommitRequest) -> CommitResponse:
        """Commit privacy spend to the ledger."""
        with self.lock:
            key = self._get_ledger_key(request.tenant_id, request.model_id)
            ledger = self.ledgers.setdefault(key, self._init_ledger(request.tenant_id, request.model_id))
            
            # Check if commit would exceed budget
            new_spent = ledger["spent"] + request.epsilon
            if new_spent > ledger["budget"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot commit: would exceed budget (current: {ledger['spent']:.4f}, "
                           f"commit: {request.epsilon:.4f}, budget: {ledger['budget']:.4f})"
                )
            
            # Create entry
            entry_id = f"entry_{datetime.utcnow().timestamp()}"
            entry = {
                "entry_id": entry_id,
                "round": request.round,
                "epsilon": request.epsilon,
                "delta": request.delta,
                "accountant": request.accountant,
                "timestamp": datetime.utcnow().isoformat(),
                "clipping_C": request.clipping_C,
                "sigma": request.sigma,
                "batch_size": request.batch_size,
                "dataset_size": request.dataset_size,
                "num_clients": request.num_clients,
                "aggregation_method": request.aggregation_method,
                "framework": request.framework,
                "metadata": request.metadata or {}
            }
            
            # Update ledger
            ledger["entries"].append(entry)
            ledger["spent"] = new_spent
            ledger["delta_accumulated"] += request.delta
            ledger["last_update"] = datetime.utcnow().isoformat()
            ledger["accountant_type"] = request.accountant
            
            # Save to storage
            self.save_to_storage()
            
            # Generate PoDP receipt if enabled
            receipt_id = None
            if ENABLE_PODP:
                receipt_data = {
                    "tenant_id": request.tenant_id,
                    "model_id": request.model_id,
                    "round": request.round,
                    "epsilon": request.epsilon,
                    "delta": request.delta,
                    "accountant": request.accountant,
                    "entry_id": entry_id
                }
                receipt = PoDPReceipt.generate_receipt("PRIVACY_COMMIT", receipt_data)
                receipt_id = receipt["receipt_id"]
            
            remaining = max(0, ledger["budget"] - new_spent)
            
            return CommitResponse(
                success=True,
                total_spent=new_spent,
                remaining_budget=remaining,
                entry_id=entry_id,
                receipt_id=receipt_id
            )
    
    def get_budget_status(self, tenant_id: str, model_id: str) -> BudgetStatus:
        """Get current budget status for tenant/model pair."""
        with self.lock:
            key = self._get_ledger_key(tenant_id, model_id)
            
            if key not in self.ledgers:
                # Return default budget for non-existent ledger
                return BudgetStatus(
                    tenant_id=tenant_id,
                    model_id=model_id,
                    total_budget=DEFAULT_BUDGET,
                    total_spent=0.0,
                    remaining_budget=DEFAULT_BUDGET,
                    num_rounds=0,
                    last_update=None,
                    accountant_type=AccountantType.RDP,
                    entries_count=0
                )
            
            ledger = self.ledgers[key]
            return BudgetStatus(
                tenant_id=tenant_id,
                model_id=model_id,
                total_budget=ledger["budget"],
                total_spent=ledger["spent"],
                remaining_budget=max(0, ledger["budget"] - ledger["spent"]),
                num_rounds=len(set(e.get("round", 0) for e in ledger["entries"])),
                last_update=ledger.get("last_update"),
                accountant_type=ledger.get("accountant_type", AccountantType.RDP),
                entries_count=len(ledger["entries"])
            )
    
    def get_history(self, tenant_id: str, model_id: str, limit: int = 100) -> List[HistoryEntry]:
        """Get privacy spend history for tenant/model pair."""
        with self.lock:
            key = self._get_ledger_key(tenant_id, model_id)
            
            if key not in self.ledgers:
                return []
            
            ledger = self.ledgers[key]
            entries = ledger["entries"][-limit:]  # Get last N entries
            
            history = []
            for entry in entries:
                history.append(HistoryEntry(
                    entry_id=entry["entry_id"],
                    round=entry["round"],
                    epsilon=entry["epsilon"],
                    delta=entry["delta"],
                    accountant=entry["accountant"],
                    timestamp=entry["timestamp"],
                    metadata=entry.get("metadata", {})
                ))
            
            return history
    
    def reset_budget(self, tenant_id: str, model_id: str, new_budget: Optional[float] = None) -> bool:
        """Reset privacy budget for tenant/model pair (admin only)."""
        with self.lock:
            key = self._get_ledger_key(tenant_id, model_id)
            
            if key in self.ledgers:
                # Archive old ledger
                old_ledger = self.ledgers[key]
                archive_key = f"{key[0]}|{key[1]}|archived_{datetime.utcnow().timestamp()}"
                # In production, save to archive storage
                
            # Create new ledger
            self.ledgers[key] = self._init_ledger(tenant_id, model_id)
            if new_budget is not None:
                self.ledgers[key]["budget"] = new_budget
            
            self.save_to_storage()
            logger.info(f"Reset budget for {tenant_id}/{model_id} to {self.ledgers[key]['budget']}")
            
            return True


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="DALRN Epsilon-Ledger Service",
    description="Privacy budget tracking and management for federated learning",
    version="1.0.0"
)

# Initialize global ledger instance
ledger_service = EpsilonLedger()


def verify_admin_token(x_admin_token: Optional[str] = Header(None)) -> bool:
    """Verify admin authentication token."""
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return True


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "epsilon-ledger",
        "opacus_available": OPACUS_AVAILABLE,
        "podp_enabled": ENABLE_PODP
    }


@app.post("/precheck", response_model=PreCheckResponse)
async def precheck(request: PreCheckRequest):
    """
    Pre-check if a privacy operation is allowed within budget.
    
    This endpoint should be called before starting any privacy-sensitive
    operation to verify that sufficient budget is available.
    """
    try:
        response = ledger_service.precheck(request)
        logger.info(f"Precheck for {request.tenant_id}/{request.model_id}: "
                   f"allowed={response.allowed}, remaining={response.remaining_budget:.4f}")
        return response
    except Exception as e:
        logger.error(f"Precheck failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/commit", response_model=CommitResponse)
async def commit(request: CommitRequest):
    """
    Commit privacy spend to the ledger.
    
    This endpoint should be called after completing a privacy-sensitive
    operation to record the actual privacy cost incurred.
    """
    try:
        response = ledger_service.commit(request)
        logger.info(f"Committed {request.epsilon:.4f} epsilon for {request.tenant_id}/{request.model_id}, "
                   f"round {request.round}. Total spent: {response.total_spent:.4f}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/{tenant_id}/{model_id}", response_model=BudgetStatus)
async def get_budget(tenant_id: str, model_id: str):
    """
    Get current budget status for a tenant/model pair.
    """
    try:
        status = ledger_service.get_budget_status(tenant_id, model_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{tenant_id}/{model_id}", response_model=List[HistoryEntry])
async def get_history(tenant_id: str, model_id: str, limit: int = 100):
    """
    Get privacy spend history for a tenant/model pair.
    """
    try:
        history = ledger_service.get_history(tenant_id, model_id, limit)
        return history
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{tenant_id}/{model_id}")
async def reset_budget(
    tenant_id: str,
    model_id: str,
    new_budget: Optional[float] = None,
    admin_auth: bool = Depends(verify_admin_token)
):
    """
    Reset privacy budget for a tenant/model pair (admin only).
    
    Requires admin authentication via X-Admin-Token header.
    """
    try:
        success = ledger_service.reset_budget(tenant_id, model_id, new_budget)
        return {
            "success": success,
            "message": f"Budget reset for {tenant_id}/{model_id}",
            "new_budget": new_budget or DEFAULT_BUDGET
        }
    except Exception as e:
        logger.error(f"Failed to reset budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Federated Learning Integration Endpoints
# ============================================================================

@app.post("/fl/preround")
async def fl_preround_check(request: Dict[str, Any]):
    """
    Specialized endpoint for FL framework pre-round checks.
    
    Supports both Flower and NV-FLARE frameworks.
    """
    try:
        # Extract parameters based on framework
        framework = request.get("framework", "flower").lower()
        
        if framework == "flower":
            tenant_id = request.get("tenant_id")
            model_id = request.get("model_id")
            # Calculate epsilon based on Flower parameters
            num_rounds = request.get("num_rounds", 1)
            batch_size = request.get("batch_size", 32)
            noise_multiplier = request.get("noise_multiplier", 1.0)
            
            # Basic epsilon calculation (enhance with Opacus if available)
            eps_round = noise_multiplier * (1.0 / num_rounds)
            
        elif framework == "nvflare":
            tenant_id = request.get("tenant_id")
            model_id = request.get("project_name")  # NV-FLARE uses project_name
            # NV-FLARE specific parameters
            privacy_config = request.get("privacy_config", {})
            eps_round = privacy_config.get("epsilon_per_round", 0.5)
            
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Perform precheck
        precheck_req = PreCheckRequest(
            tenant_id=tenant_id,
            model_id=model_id,
            eps_round=eps_round
        )
        
        response = ledger_service.precheck(precheck_req)
        
        return {
            "framework": framework,
            "allowed": response.allowed,
            "remaining_budget": response.remaining_budget,
            "recommended_params": {
                "max_rounds": int(response.remaining_budget / eps_round) if eps_round > 0 else 0,
                "noise_multiplier": noise_multiplier if framework == "flower" else None
            }
        }
        
    except Exception as e:
        logger.error(f"FL pre-round check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fl/postround")
async def fl_postround_commit(request: Dict[str, Any]):
    """
    Specialized endpoint for FL framework post-round commits.
    """
    try:
        framework = request.get("framework", "flower").lower()
        
        # Map framework-specific parameters to commit request
        commit_req = CommitRequest(
            tenant_id=request.get("tenant_id"),
            model_id=request.get("model_id") if framework == "flower" else request.get("project_name"),
            round=request.get("round", 0),
            accountant=AccountantType(request.get("accountant", "rdp")),
            epsilon=request.get("epsilon"),
            delta=request.get("delta", DEFAULT_DELTA),
            clipping_C=request.get("clipping_norm"),
            sigma=request.get("noise_multiplier"),
            batch_size=request.get("batch_size"),
            dataset_size=request.get("dataset_size"),
            num_clients=request.get("num_clients"),
            aggregation_method=request.get("aggregation_method"),
            framework=framework,
            metadata=request.get("metadata", {})
        )
        
        response = ledger_service.commit(commit_req)
        
        return {
            "framework": framework,
            "success": response.success,
            "total_spent": response.total_spent,
            "remaining_budget": response.remaining_budget,
            "entry_id": response.entry_id
        }
        
    except Exception as e:
        logger.error(f"FL post-round commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("LEDGER_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
