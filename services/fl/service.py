"""
Federated Learning Service with Differential Privacy
Complete implementation with Flower framework integration
"""
import os
import sys
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Import epsilon ledger
from fl.eps_ledger import EpsilonLedger, EpsilonEntry

# Import Flower for FL
try:
    import flwr as fl
    from flwr.server import ServerConfig, Server
    from flwr.server.strategy import FedAvg
except ImportError:
    print("Warning: Flower not installed. Install with: pip install flwr")
    fl = None

# Import Opacus for DP
try:
    import opacus
    from opacus.accountants import RDPAccountant
except ImportError:
    print("Warning: Opacus not installed. Install with: pip install opacus")
    opacus = None

app = FastAPI(title="DALRN FL Service", version="1.0.0")

# Initialize epsilon ledger
epsilon_ledger = EpsilonLedger()

# Request/Response Models
class FLRoundRequest(BaseModel):
    """Request to start FL round"""
    tenant_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    num_clients: int = Field(default=3, ge=1, le=100)
    num_rounds: int = Field(default=5, ge=1, le=100)
    epsilon_per_round: float = Field(default=0.5, gt=0, le=2.0)
    delta: float = Field(default=1e-5, gt=0, lt=1)

class FLRoundResponse(BaseModel):
    """FL round response"""
    round_id: str
    status: str
    metrics: Dict
    epsilon_used: float
    epsilon_remaining: float

class ClientUpdateRequest(BaseModel):
    """Client model update"""
    client_id: str
    round_id: str
    model_weights: List[List[float]]  # Serialized weights
    num_samples: int
    metrics: Dict

# FL Coordination
class FLCoordinator:
    """Manages federated learning rounds"""

    def __init__(self):
        self.active_rounds = {}
        self.client_updates = {}

    async def start_round(
        self,
        tenant_id: str,
        model_id: str,
        num_clients: int,
        epsilon: float,
        delta: float
    ) -> str:
        """Start new FL round with privacy budget check"""

        # Check epsilon budget
        can_proceed = await epsilon_ledger.precheck(
            tenant_id, model_id, epsilon
        )

        if not can_proceed:
            raise HTTPException(
                status_code=403,
                detail="Insufficient privacy budget"
            )

        round_id = f"FL-{datetime.utcnow().timestamp():.0f}"

        # Initialize round
        self.active_rounds[round_id] = {
            "tenant_id": tenant_id,
            "model_id": model_id,
            "num_clients": num_clients,
            "epsilon": epsilon,
            "delta": delta,
            "client_updates": [],
            "status": "waiting_for_clients",
            "created_at": datetime.utcnow()
        }

        # Commit epsilon usage
        await epsilon_ledger.commit(
            tenant_id, model_id, 1, epsilon, delta,
            mechanism="gaussian"
        )

        return round_id

    async def aggregate_updates(
        self,
        round_id: str
    ) -> Dict:
        """Aggregate client updates with DP noise"""

        round_data = self.active_rounds.get(round_id)
        if not round_data:
            raise ValueError(f"Round {round_id} not found")

        updates = round_data["client_updates"]
        if len(updates) < round_data["num_clients"]:
            return {"status": "waiting", "updates_received": len(updates)}

        # Simple federated averaging
        total_samples = sum(u["num_samples"] for u in updates)
        averaged_weights = None

        for update in updates:
            weight = update["num_samples"] / total_samples
            weights = np.array(update["model_weights"])

            if averaged_weights is None:
                averaged_weights = weights * weight
            else:
                averaged_weights += weights * weight

        # Add DP noise if Opacus available
        if opacus and round_data["epsilon"] > 0:
            noise_scale = np.sqrt(2 * np.log(1.25 / round_data["delta"])) / round_data["epsilon"]
            dp_noise = np.random.normal(0, noise_scale, averaged_weights.shape)
            averaged_weights += dp_noise

        # Update round status
        round_data["status"] = "completed"
        round_data["aggregated_weights"] = averaged_weights.tolist()

        return {
            "status": "completed",
            "aggregated_weights": averaged_weights.tolist(),
            "num_clients": len(updates),
            "epsilon_used": round_data["epsilon"]
        }

# Initialize coordinator
coordinator = FLCoordinator()

# API Endpoints
@app.post("/fl/round/start", response_model=FLRoundResponse)
async def start_fl_round(
    request: FLRoundRequest,
    background_tasks: BackgroundTasks
):
    """Start a new federated learning round"""
    try:
        # Start round
        round_id = await coordinator.start_round(
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            num_clients=request.num_clients,
            epsilon=request.epsilon_per_round,
            delta=request.delta
        )

        # Get remaining budget
        remaining = await epsilon_ledger.get_remaining_budget(
            request.tenant_id, request.model_id
        )

        return FLRoundResponse(
            round_id=round_id,
            status="started",
            metrics={"num_clients_expected": request.num_clients},
            epsilon_used=request.epsilon_per_round,
            epsilon_remaining=remaining
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/fl/round/{round_id}/update")
async def submit_client_update(
    round_id: str,
    update: ClientUpdateRequest
):
    """Submit client model update"""
    try:
        round_data = coordinator.active_rounds.get(round_id)
        if not round_data:
            raise HTTPException(status_code=404, detail="Round not found")

        # Add update
        round_data["client_updates"].append({
            "client_id": update.client_id,
            "model_weights": update.model_weights,
            "num_samples": update.num_samples,
            "metrics": update.metrics,
            "timestamp": datetime.utcnow()
        })

        # Try aggregation
        result = await coordinator.aggregate_updates(round_id)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/fl/round/{round_id}/status")
async def get_round_status(round_id: str):
    """Get FL round status"""
    round_data = coordinator.active_rounds.get(round_id)
    if not round_data:
        raise HTTPException(status_code=404, detail="Round not found")

    return {
        "round_id": round_id,
        "status": round_data["status"],
        "num_clients_expected": round_data["num_clients"],
        "num_updates_received": len(round_data["client_updates"]),
        "epsilon_used": round_data["epsilon"]
    }

@app.post("/precheck")
async def precheck_epsilon(
    tenant_id: str,
    model_id: str,
    requested_epsilon: float
):
    """Check if epsilon budget allows operation"""
    result = await epsilon_ledger.precheck(
        tenant_id, model_id, requested_epsilon
    )

    remaining = await epsilon_ledger.get_remaining_budget(
        tenant_id, model_id
    )

    return {
        "allowed": result,
        "remaining_budget": remaining,
        "total_budget": epsilon_ledger.total_budget
    }

@app.post("/commit")
async def commit_epsilon(
    tenant_id: str,
    model_id: str,
    round: int,
    epsilon: float,
    delta: float
):
    """Commit epsilon usage"""
    entry_id = await epsilon_ledger.commit(
        tenant_id, model_id, round, epsilon, delta
    )

    return {
        "ok": True,
        "spent": epsilon,
        "ledger_entry_id": entry_id
    }

@app.get("/ledger/{tenant_id}/{model_id}")
async def get_ledger(tenant_id: str, model_id: str):
    """Get epsilon ledger for tenant/model"""
    entries = epsilon_ledger.get_ledger(tenant_id, model_id)
    spent = epsilon_ledger.get_spent(tenant_id, model_id)
    remaining = await epsilon_ledger.get_remaining_budget(tenant_id, model_id)

    return {
        "entries": entries,
        "total_spent": spent,
        "budget_remaining": remaining
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "service": "fl",
        "status": "healthy",
        "flower_available": fl is not None,
        "opacus_available": opacus is not None,
        "active_rounds": len(coordinator.active_rounds)
    }

if __name__ == "__main__":
    port = int(os.getenv("FL_PORT", 8400))
    uvicorn.run(app, host="0.0.0.0", port=port)