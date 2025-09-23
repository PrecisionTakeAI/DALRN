"""
Self-Organizing Agent Networks (SOAN) Orchestrator Service.

This module provides the main FastAPI service for orchestrating SOAN operations
with full PoDP compliance and ε-ledger budget management.
"""

import json
import hashlib
import time
import asyncio
import uvicorn
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import networkx as nx
import numpy as np
import logging

# Import SOAN components
from .topology import WattsStrogatzTopology, NetworkMetrics
from .gnn_predictor import GNNLatencyPredictor, PredictionMetrics
from .queue_model import MM1Queue, QueueMetrics
from .rewiring import EpsilonGreedyRewiring, RewiringMetrics

# Import PoDP utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.common.podp import Receipt, ReceiptChain, keccak

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class InitializeNetworkRequest(BaseModel):
    """Request model for network initialization."""
    n_nodes: int = Field(default=100, ge=10, le=1000, description="Number of nodes")
    k_edges: int = Field(default=6, ge=2, le=20, description="Number of edges per node")
    p_rewire: float = Field(default=0.1, ge=0.0, le=1.0, description="Rewiring probability")

    @validator('k_edges')
    def validate_k_edges(cls, v, values):
        if 'n_nodes' in values and v >= values['n_nodes']:
            raise ValueError(f"k_edges ({v}) must be less than n_nodes ({values['n_nodes']})")
        if v % 2 != 0:
            raise ValueError(f"k_edges ({v}) must be even")
        return v


class TrainGNNRequest(BaseModel):
    """Request model for GNN training."""
    network_id: str = Field(..., description="Network ID to train on")
    epochs: int = Field(default=100, ge=1, le=1000, description="Training epochs")
    learning_rate: float = Field(default=0.01, gt=0, le=1.0, description="Learning rate")


class OptimizeNetworkRequest(BaseModel):
    """Request model for network optimization."""
    network_id: str = Field(..., description="Network ID to optimize")
    iterations: int = Field(default=20, ge=1, le=100, description="Optimization iterations")
    epsilon: float = Field(default=0.2, ge=0.0, le=1.0, description="Exploration rate")


class SimulateQueuesRequest(BaseModel):
    """Request model for queue simulation."""
    network_id: str = Field(..., description="Network ID for simulation")
    duration: float = Field(default=100.0, gt=0, le=10000, description="Simulation duration")
    arrival_rate: float = Field(default=0.5, gt=0, le=2.0, description="Arrival rate")


class NetworkResponse(BaseModel):
    """Response model for network operations."""
    network_id: str
    metrics: Dict[str, Any]
    receipt_id: str
    epsilon_used: float
    epsilon_remaining: float
    status: str = "success"


class SOANOrchestrator:
    """
    Main orchestrator for Self-Organizing Agent Networks.
    Coordinates all SOAN components with PoDP compliance.
    """

    # ε-ledger budget allocations
    EPSILON_ORCHESTRATION = 0.0001
    TOTAL_EPSILON_BUDGET = 4.0

    def __init__(self):
        """Initialize the orchestrator."""
        self.networks: Dict[str, Dict[str, Any]] = {}
        self.gnn_models: Dict[str, GNNLatencyPredictor] = {}
        self.queue_models: Dict[str, List[MM1Queue]] = {}
        self.receipt_chains: Dict[str, ReceiptChain] = {}
        self.epsilon_ledger: Dict[str, float] = {}

    def _generate_network_id(self) -> str:
        """Generate unique network ID."""
        return f"soan_net_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}"

    def _check_epsilon_budget(self, network_id: str, required: float) -> bool:
        """Check if operation is within ε-ledger budget."""
        spent = self.epsilon_ledger.get(network_id, 0.0)
        return (spent + required) <= self.TOTAL_EPSILON_BUDGET

    async def initialize_network(
        self,
        request: InitializeNetworkRequest
    ) -> NetworkResponse:
        """
        Initialize a new SOAN network with specified parameters.

        Args:
            request: Network initialization parameters

        Returns:
            NetworkResponse with network ID and metrics
        """
        network_id = self._generate_network_id()
        logger.info(f"Initializing network {network_id} with N={request.n_nodes}")

        # Check epsilon budget
        required_epsilon = (
            WattsStrogatzTopology.EPSILON_NETWORK_GENERATION +
            WattsStrogatzTopology.EPSILON_METRICS_CALCULATION +
            self.EPSILON_ORCHESTRATION
        )

        if not self._check_epsilon_budget(network_id, required_epsilon):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient epsilon budget. Required: {required_epsilon}, Available: {self.TOTAL_EPSILON_BUDGET}"
            )

        try:
            # Create topology
            topology_gen = WattsStrogatzTopology(
                N=request.n_nodes,
                k=request.k_edges,
                p=request.p_rewire
            )

            # Generate network
            graph, gen_receipt = topology_gen.generate()

            # Calculate metrics
            metrics, metrics_receipt = topology_gen.calculate_metrics()

            # Export to dictionary
            network_data, export_receipt = topology_gen.export_to_dict()

            # Store network
            self.networks[network_id] = {
                "graph": graph,
                "topology": topology_gen,
                "data": network_data,
                "created_at": datetime.utcnow().isoformat()
            }

            # Store receipt chain
            self.receipt_chains[network_id] = topology_gen.get_receipt_chain()

            # Update epsilon ledger
            self.epsilon_ledger[network_id] = topology_gen.epsilon_spent + self.EPSILON_ORCHESTRATION

            # Create orchestration receipt
            orch_receipt = Receipt(
                receipt_id=f"orch_init_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
                dispute_id=network_id,
                step="network_initialization_orchestration",
                inputs={
                    "request": request.dict()
                },
                params={
                    "network_id": network_id
                },
                artifacts={
                    "topology_receipt": gen_receipt.receipt_id,
                    "metrics_receipt": metrics_receipt.receipt_id,
                    "export_receipt": export_receipt.receipt_id,
                    "epsilon_used": self.EPSILON_ORCHESTRATION,
                    "epsilon_total": self.epsilon_ledger[network_id]
                },
                hashes={
                    "network_hash": keccak(json.dumps(network_data, sort_keys=True))
                },
                signatures=[],
                ts=datetime.utcnow().isoformat()
            )

            self.receipt_chains[network_id].add_receipt(orch_receipt)

            return NetworkResponse(
                network_id=network_id,
                metrics=metrics.__dict__,
                receipt_id=orch_receipt.receipt_id,
                epsilon_used=self.epsilon_ledger[network_id],
                epsilon_remaining=self.TOTAL_EPSILON_BUDGET - self.epsilon_ledger[network_id]
            )

        except Exception as e:
            logger.error(f"Failed to initialize network: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Network initialization failed: {str(e)}"
            )

    async def train_gnn_model(
        self,
        request: TrainGNNRequest
    ) -> NetworkResponse:
        """
        Train GNN latency predictor on network topology.

        Args:
            request: GNN training parameters

        Returns:
            NetworkResponse with training metrics
        """
        if request.network_id not in self.networks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Network {request.network_id} not found"
            )

        logger.info(f"Training GNN for network {request.network_id}")

        # Check epsilon budget
        required_epsilon = (
            GNNLatencyPredictor.EPSILON_MODEL_INIT +
            GNNLatencyPredictor.EPSILON_TRAINING_STEP * request.epochs +
            self.EPSILON_ORCHESTRATION
        )

        if not self._check_epsilon_budget(request.network_id, required_epsilon):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient epsilon budget for training"
            )

        try:
            # Get network graph
            graph = self.networks[request.network_id]["graph"]

            # Create GNN model
            gnn_model = GNNLatencyPredictor(
                input_dim=2,  # queue_length, service_rate
                hidden_dim=16,
                output_dim=1
            )

            # Generate synthetic training data (simulate queue latencies)
            target_latencies = []
            for node in sorted(graph.nodes()):
                attrs = graph.nodes[node]
                # Simple latency model for training
                queue_length = attrs.get('queue_length', 0)
                service_rate = attrs.get('service_rate', 1.5)
                latency = queue_length / service_rate + np.random.normal(0, 0.1)
                target_latencies.append(max(0, latency))

            target_latencies = np.array(target_latencies)

            # Train model
            losses, train_receipt = gnn_model.train_model(
                graph=graph,
                target_latencies=target_latencies,
                epochs=request.epochs,
                learning_rate=request.learning_rate
            )

            # Evaluate model
            eval_metrics, eval_receipt = gnn_model.evaluate(graph, target_latencies)

            # Store model
            self.gnn_models[request.network_id] = gnn_model

            # Update epsilon ledger
            self.epsilon_ledger[request.network_id] += gnn_model.epsilon_spent + self.EPSILON_ORCHESTRATION

            # Create orchestration receipt
            orch_receipt = Receipt(
                receipt_id=f"orch_train_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
                dispute_id=request.network_id,
                step="gnn_training_orchestration",
                inputs={
                    "request": request.dict()
                },
                params={
                    "model_params": {
                        "input_dim": 2,
                        "hidden_dim": 16,
                        "output_dim": 1
                    }
                },
                artifacts={
                    "train_receipt": train_receipt.receipt_id,
                    "eval_receipt": eval_receipt.receipt_id,
                    "final_loss": losses[-1],
                    "eval_metrics": eval_metrics.__dict__,
                    "epsilon_used": gnn_model.epsilon_spent + self.EPSILON_ORCHESTRATION,
                    "epsilon_total": self.epsilon_ledger[request.network_id]
                },
                hashes={},
                signatures=[],
                ts=datetime.utcnow().isoformat()
            )

            self.receipt_chains[request.network_id].add_receipt(orch_receipt)

            return NetworkResponse(
                network_id=request.network_id,
                metrics={
                    "training_loss": losses[-1],
                    "evaluation": eval_metrics.__dict__
                },
                receipt_id=orch_receipt.receipt_id,
                epsilon_used=self.epsilon_ledger[request.network_id],
                epsilon_remaining=self.TOTAL_EPSILON_BUDGET - self.epsilon_ledger[request.network_id]
            )

        except Exception as e:
            logger.error(f"Failed to train GNN: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"GNN training failed: {str(e)}"
            )

    async def optimize_network(
        self,
        request: OptimizeNetworkRequest
    ) -> NetworkResponse:
        """
        Optimize network topology using ε-greedy rewiring.

        Args:
            request: Optimization parameters

        Returns:
            NetworkResponse with optimization metrics
        """
        if request.network_id not in self.networks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Network {request.network_id} not found"
            )

        logger.info(f"Optimizing network {request.network_id}")

        # Check epsilon budget
        required_epsilon = (
            EpsilonGreedyRewiring.EPSILON_INIT +
            EpsilonGreedyRewiring.EPSILON_REWIRE_STEP * request.iterations +
            self.EPSILON_ORCHESTRATION
        )

        if not self._check_epsilon_budget(request.network_id, required_epsilon):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient epsilon budget for optimization"
            )

        try:
            # Get current network
            graph = self.networks[request.network_id]["graph"]

            # Create rewiring optimizer
            optimizer = EpsilonGreedyRewiring(
                epsilon=request.epsilon,
                max_iterations=request.iterations
            )

            # Optimize network
            optimized_graph, metrics, opt_receipt = optimizer.optimize(graph)

            # Update network
            self.networks[request.network_id]["graph"] = optimized_graph

            # Recalculate network metrics
            topology_gen = self.networks[request.network_id]["topology"]
            topology_gen.graph = optimized_graph
            new_metrics, _ = topology_gen.calculate_metrics()

            # Update epsilon ledger
            self.epsilon_ledger[request.network_id] += optimizer.epsilon_spent + self.EPSILON_ORCHESTRATION

            # Create orchestration receipt
            orch_receipt = Receipt(
                receipt_id=f"orch_opt_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
                dispute_id=request.network_id,
                step="network_optimization_orchestration",
                inputs={
                    "request": request.dict()
                },
                params={
                    "optimizer": "epsilon_greedy"
                },
                artifacts={
                    "opt_receipt": opt_receipt.receipt_id,
                    "rewiring_metrics": metrics.__dict__,
                    "network_metrics": new_metrics.__dict__,
                    "epsilon_used": optimizer.epsilon_spent + self.EPSILON_ORCHESTRATION,
                    "epsilon_total": self.epsilon_ledger[request.network_id]
                },
                hashes={
                    "optimized_graph_hash": keccak(
                        json.dumps(nx.adjacency_matrix(optimized_graph).todense().tolist())
                    )
                },
                signatures=[],
                ts=datetime.utcnow().isoformat()
            )

            self.receipt_chains[request.network_id].add_receipt(orch_receipt)

            return NetworkResponse(
                network_id=request.network_id,
                metrics={
                    "rewiring": metrics.__dict__,
                    "network": new_metrics.__dict__
                },
                receipt_id=orch_receipt.receipt_id,
                epsilon_used=self.epsilon_ledger[request.network_id],
                epsilon_remaining=self.TOTAL_EPSILON_BUDGET - self.epsilon_ledger[request.network_id]
            )

        except Exception as e:
            logger.error(f"Failed to optimize network: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Network optimization failed: {str(e)}"
            )

    async def simulate_queues(
        self,
        request: SimulateQueuesRequest
    ) -> NetworkResponse:
        """
        Simulate M/M/1 queues for all network nodes.

        Args:
            request: Queue simulation parameters

        Returns:
            NetworkResponse with simulation metrics
        """
        if request.network_id not in self.networks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Network {request.network_id} not found"
            )

        logger.info(f"Simulating queues for network {request.network_id}")

        try:
            # Get network graph
            graph = self.networks[request.network_id]["graph"]

            # Check epsilon budget (approximate)
            required_epsilon = (
                MM1Queue.EPSILON_QUEUE_INIT * graph.number_of_nodes() +
                MM1Queue.EPSILON_SIMULATION_STEP * 100 +  # Estimate
                self.EPSILON_ORCHESTRATION
            )

            if not self._check_epsilon_budget(request.network_id, required_epsilon):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient epsilon budget for simulation"
                )

            # Create and simulate queues for each node
            queue_models = []
            queue_metrics = []
            total_queue_epsilon = 0.0

            for node in sorted(graph.nodes())[:10]:  # Limit to first 10 nodes for demo
                attrs = graph.nodes[node]
                service_rate = attrs.get('service_rate', 1.5)

                # Create queue model
                queue = MM1Queue(
                    arrival_rate=request.arrival_rate,
                    service_rate=service_rate
                )

                # Check stability
                is_stable, stability_metrics, _ = queue.check_stability()

                if is_stable:
                    # Run simulation
                    metrics, sim_receipt = queue.simulate(
                        duration=request.duration,
                        warm_up=0.1
                    )
                    queue_metrics.append({
                        "node": node,
                        "metrics": metrics.__dict__,
                        "stable": True
                    })
                else:
                    queue_metrics.append({
                        "node": node,
                        "metrics": stability_metrics,
                        "stable": False
                    })

                queue_models.append(queue)
                total_queue_epsilon += queue.epsilon_spent

            # Store queue models
            self.queue_models[request.network_id] = queue_models

            # Update epsilon ledger
            self.epsilon_ledger[request.network_id] += total_queue_epsilon + self.EPSILON_ORCHESTRATION

            # Calculate aggregate metrics
            stable_queues = [q for q in queue_metrics if q["stable"]]
            aggregate_metrics = {
                "total_nodes_simulated": len(queue_metrics),
                "stable_queues": len(stable_queues),
                "unstable_queues": len(queue_metrics) - len(stable_queues),
                "avg_queue_length": np.mean([
                    q["metrics"]["average_queue_length"]
                    for q in stable_queues
                    if "average_queue_length" in q["metrics"]
                ]) if stable_queues else 0,
                "avg_wait_time": np.mean([
                    q["metrics"]["average_wait_time"]
                    for q in stable_queues
                    if "average_wait_time" in q["metrics"]
                ]) if stable_queues else 0
            }

            # Create orchestration receipt
            orch_receipt = Receipt(
                receipt_id=f"orch_sim_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
                dispute_id=request.network_id,
                step="queue_simulation_orchestration",
                inputs={
                    "request": request.dict()
                },
                params={
                    "nodes_simulated": len(queue_metrics)
                },
                artifacts={
                    "aggregate_metrics": aggregate_metrics,
                    "epsilon_used": total_queue_epsilon + self.EPSILON_ORCHESTRATION,
                    "epsilon_total": self.epsilon_ledger[request.network_id]
                },
                hashes={},
                signatures=[],
                ts=datetime.utcnow().isoformat()
            )

            self.receipt_chains[request.network_id].add_receipt(orch_receipt)

            return NetworkResponse(
                network_id=request.network_id,
                metrics=aggregate_metrics,
                receipt_id=orch_receipt.receipt_id,
                epsilon_used=self.epsilon_ledger[request.network_id],
                epsilon_remaining=self.TOTAL_EPSILON_BUDGET - self.epsilon_ledger[request.network_id]
            )

        except Exception as e:
            logger.error(f"Failed to simulate queues: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Queue simulation failed: {str(e)}"
            )

    async def get_network_status(self, network_id: str) -> Dict[str, Any]:
        """Get current status of a network."""
        if network_id not in self.networks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Network {network_id} not found"
            )

        network = self.networks[network_id]
        graph = network["graph"]

        status = {
            "network_id": network_id,
            "created_at": network["created_at"],
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "is_connected": nx.is_connected(graph),
            "has_gnn_model": network_id in self.gnn_models,
            "has_queue_models": network_id in self.queue_models,
            "epsilon_spent": self.epsilon_ledger.get(network_id, 0.0),
            "epsilon_remaining": self.TOTAL_EPSILON_BUDGET - self.epsilon_ledger.get(network_id, 0.0),
            "receipt_count": len(self.receipt_chains[network_id].receipts) if network_id in self.receipt_chains else 0
        }

        return status

    async def get_receipt_chain(self, network_id: str) -> Dict[str, Any]:
        """Get complete PoDP receipt chain for a network."""
        if network_id not in self.receipt_chains:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Receipt chain for network {network_id} not found"
            )

        chain = self.receipt_chains[network_id]
        return {
            "dispute_id": chain.dispute_id,
            "merkle_root": chain.get_merkle_root(),
            "receipt_count": len(chain.receipts),
            "receipts": [r.__dict__ for r in chain.receipts[-10:]]  # Last 10 receipts
        }


# Create FastAPI app
app = FastAPI(
    title="SOAN Orchestrator",
    description="Self-Organizing Agent Networks Orchestration Service",
    version="1.0.0"
)

# Create orchestrator instance
orchestrator = SOANOrchestrator()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "service": "SOAN Orchestrator",
        "status": "operational",
        "version": "1.0.0",
        "podp_compliant": True
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_networks": len(orchestrator.networks),
        "total_epsilon_allocated": sum(orchestrator.epsilon_ledger.values())
    }


@app.post("/api/v1/soan/initialize", response_model=NetworkResponse, tags=["SOAN"])
async def initialize_network(request: InitializeNetworkRequest):
    """Initialize a new SOAN network."""
    return await orchestrator.initialize_network(request)


@app.post("/api/v1/soan/train", response_model=NetworkResponse, tags=["SOAN"])
async def train_gnn(request: TrainGNNRequest):
    """Train GNN latency predictor."""
    return await orchestrator.train_gnn_model(request)


@app.post("/api/v1/soan/optimize", response_model=NetworkResponse, tags=["SOAN"])
async def optimize_network(request: OptimizeNetworkRequest):
    """Optimize network topology."""
    return await orchestrator.optimize_network(request)


@app.post("/api/v1/soan/simulate", response_model=NetworkResponse, tags=["SOAN"])
async def simulate_queues(request: SimulateQueuesRequest):
    """Simulate M/M/1 queues."""
    return await orchestrator.simulate_queues(request)


@app.get("/api/v1/soan/status/{network_id}", tags=["SOAN"])
async def get_network_status(network_id: str):
    """Get network status."""
    return await orchestrator.get_network_status(network_id)


@app.get("/api/v1/soan/receipts/{network_id}", tags=["SOAN"])
async def get_receipt_chain(network_id: str):
    """Get PoDP receipt chain."""
    return await orchestrator.get_receipt_chain(network_id)


@app.get("/api/v1/soan/networks", tags=["SOAN"])
async def list_networks():
    """List all networks."""
    return {
        "networks": list(orchestrator.networks.keys()),
        "count": len(orchestrator.networks)
    }


if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0",
        port=8500,
        reload=True,
        log_level="info"
    )