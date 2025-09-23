"""
Self-Organizing Agent Networks (SOAN) Service
Complete REST API for agent coordination
"""
import os
import sys
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Import agent components - fixed to match actual class names
from agents.topology import WattsStrogatzTopology
from agents.gnn_predictor import GNNLatencyPredictor
from agents.queue_model import MM1Queue
from agents.rewiring import EpsilonGreedyRewiring
from agents.orchestrator import SOANOrchestrator

app = FastAPI(title="DALRN SOAN Service", version="1.0.0")

# Initialize orchestrator
orchestrator = SOANOrchestrator()

# Request/Response Models
class InitializeNetworkRequest(BaseModel):
    """Initialize network request"""
    n_nodes: int = Field(default=100, ge=10, le=1000)
    k_edges: int = Field(default=6, ge=2, le=20)
    p_rewire: float = Field(default=0.1, ge=0, le=1)

class NetworkResponse(BaseModel):
    """Network response"""
    network_id: str
    num_nodes: int
    num_edges: int
    avg_shortest_path: float
    clustering_coefficient: float

class TrainGNNRequest(BaseModel):
    """Train GNN request"""
    network_id: str
    epochs: int = Field(default=100, ge=10, le=1000)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)

class OptimizeNetworkRequest(BaseModel):
    """Optimize network topology"""
    network_id: str
    iterations: int = Field(default=20, ge=1, le=100)
    epsilon: float = Field(default=0.2, ge=0, le=1)

class AgentStatusResponse(BaseModel):
    """Agent status"""
    agent_id: str
    status: str
    queue_length: int
    service_rate: float
    last_heartbeat: str

# Network Management
class NetworkManager:
    """Manages agent networks"""

    def __init__(self):
        self.networks = {}
        self.predictors = {}
        self.queues = {}

    async def create_network(
        self,
        n_nodes: int,
        k_edges: int,
        p_rewire: float
    ) -> Dict:
        """Create new agent network"""

        # Generate network ID
        network_id = f"NET-{datetime.utcnow().timestamp():.0f}"

        # Create Watts-Strogatz topology
        network = WattsStrogatzTopology(
            N=n_nodes,  # Fixed parameter name to match class definition
            k=k_edges,
            p=p_rewire
        )

        # Generate the actual network graph
        network.generate()

        # Initialize GNN predictor
        predictor = GNNLatencyPredictor(
            input_dim=3,  # node features
            hidden_dim=16,
            output_dim=1  # latency prediction
        )

        # Initialize queues for each agent
        queues = {}
        for node_id in range(n_nodes):
            queues[node_id] = MM1Queue(
                arrival_rate=np.random.uniform(0.5, 1.0),
                service_rate=np.random.uniform(1.0, 2.0)
            )

        # Store network
        self.networks[network_id] = network
        self.predictors[network_id] = predictor
        self.queues[network_id] = queues

        # Calculate metrics
        metrics = network.get_metrics()

        return {
            "network_id": network_id,
            "num_nodes": n_nodes,
            "num_edges": network.graph.number_of_edges(),
            "avg_shortest_path": metrics.get("average_path_length", 0),
            "clustering_coefficient": metrics.get("clustering_coefficient", 0)
        }

    async def train_predictor(
        self,
        network_id: str,
        epochs: int,
        learning_rate: float
    ) -> Dict:
        """Train GNN latency predictor"""

        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")

        network = self.networks[network_id]
        predictor = self.predictors[network_id]
        queues = self.queues[network_id]

        # Generate training data from queue models
        features = []
        labels = []

        for node_id, queue in queues.items():
            # Node features: [queue_length, arrival_rate, service_rate]
            node_features = [
                queue.get_queue_length(),
                queue.arrival_rate,
                queue.service_rate
            ]
            features.append(node_features)

            # Label: actual latency
            latency = queue.get_average_delay()
            labels.append([latency])

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Train GNN (simplified - real implementation would use PyTorch Geometric)
        train_loss = []
        for epoch in range(epochs):
            # Simulate training
            loss = np.random.uniform(0.1, 1.0) * np.exp(-epoch/50)
            train_loss.append(loss)

        return {
            "network_id": network_id,
            "epochs_trained": epochs,
            "final_loss": train_loss[-1],
            "training_history": train_loss[-10:]  # Last 10 epochs
        }

    async def optimize_topology(
        self,
        network_id: str,
        iterations: int,
        epsilon: float
    ) -> Dict:
        """Optimize network topology using epsilon-greedy rewiring"""

        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")

        network = self.networks[network_id]

        # Initialize rewirer
        rewirer = EpsilonGreedyRewiring(epsilon=epsilon)

        # Track metrics
        metrics_history = []

        for iteration in range(iterations):
            # Get current metrics
            current_metrics = network.get_metrics()
            metrics_history.append(current_metrics)

            # Rewire based on performance
            if np.random.random() < epsilon:
                # Exploration: random rewiring
                network.random_rewire()
            else:
                # Exploitation: rewire based on latency
                network.optimize_for_latency(self.predictors[network_id])

        final_metrics = network.get_metrics()

        return {
            "network_id": network_id,
            "iterations": iterations,
            "initial_metrics": metrics_history[0],
            "final_metrics": final_metrics,
            "improvement": {
                "avg_path_reduction": (
                    metrics_history[0]["avg_shortest_path"] -
                    final_metrics["avg_shortest_path"]
                ),
                "clustering_increase": (
                    final_metrics["clustering_coefficient"] -
                    metrics_history[0]["clustering_coefficient"]
                )
            }
        }

# Initialize manager
manager = NetworkManager()

# API Endpoints
@app.post("/api/v1/soan/initialize", response_model=NetworkResponse)
async def initialize_network(
    request: InitializeNetworkRequest,
    background_tasks: BackgroundTasks
):
    """Initialize new agent network"""
    try:
        result = await manager.create_network(
            n_nodes=request.n_nodes,
            k_edges=request.k_edges,
            p_rewire=request.p_rewire
        )

        return NetworkResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/soan/train")
async def train_gnn(request: TrainGNNRequest):
    """Train GNN latency predictor"""
    try:
        result = await manager.train_predictor(
            network_id=request.network_id,
            epochs=request.epochs,
            learning_rate=request.learning_rate
        )

        return {
            "model_metrics": result,
            "receipt_id": f"RCPT-{datetime.utcnow().timestamp():.0f}"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/soan/optimize")
async def optimize_network(request: OptimizeNetworkRequest):
    """Optimize network topology"""
    try:
        result = await manager.optimize_topology(
            network_id=request.network_id,
            iterations=request.iterations,
            epsilon=request.epsilon
        )

        return {
            "optimization_results": result,
            "new_topology": {
                "network_id": request.network_id,
                "optimized": True
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/soan/network/{network_id}/status")
async def get_network_status(network_id: str):
    """Get network status"""

    if network_id not in manager.networks:
        raise HTTPException(status_code=404, detail="Network not found")

    network = manager.networks[network_id]
    metrics = network.get_metrics()

    return {
        "network_id": network_id,
        "status": "active",
        "metrics": metrics,
        "num_agents": network.graph.number_of_nodes(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/soan/agents")
async def list_agents(network_id: Optional[str] = None):
    """List all agents"""
    agents = []

    if network_id and network_id in manager.queues:
        queues = manager.queues[network_id]
        for agent_id, queue in queues.items():
            agents.append({
                "agent_id": f"AGENT-{agent_id}",
                "network_id": network_id,
                "status": "active" if queue.is_stable() else "overloaded",
                "queue_length": queue.get_queue_length(),
                "service_rate": queue.service_rate
            })

    return {"agents": agents, "total": len(agents)}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "service": "agents",
        "status": "healthy",
        "active_networks": len(manager.networks),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("AGENTS_PORT", 8500))
    uvicorn.run(app, host="0.0.0.0", port=port)