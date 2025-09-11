"""
SOAN Integration for Gateway Service

Provides agent network coordination and optimization capabilities
to the DALRN Gateway with full PoDP compliance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
import json

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from services.agents import SOANOrchestrator
from services.agents.orchestrator import SOANConfig
from services.common.podp import Receipt

logger = logging.getLogger(__name__)

# Router for SOAN endpoints
router = APIRouter(prefix="/soan", tags=["Agent Network"])

# Global orchestrator instance (singleton pattern)
_orchestrator: Optional[SOANOrchestrator] = None
_orchestrator_lock = asyncio.Lock()

# Request/Response models
class NetworkInitRequest(BaseModel):
    """Request to initialize agent network"""
    n_nodes: int = Field(default=100, ge=10, le=500, description="Number of agent nodes")
    k_neighbors: int = Field(default=6, ge=2, le=20, description="Initial neighbors per node")
    rewiring_prob: float = Field(default=0.1, ge=0.0, le=1.0, description="Rewiring probability")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class NetworkInitResponse(BaseModel):
    """Response for network initialization"""
    success: bool
    nodes: int
    edges: int
    metrics: Dict[str, float]
    receipt_id: str
    epsilon_used: float

class TrainPredictorRequest(BaseModel):
    """Request to train latency predictor"""
    training_samples: int = Field(default=1000, ge=100, le=10000)
    epochs: int = Field(default=50, ge=5, le=200)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)

class TrainPredictorResponse(BaseModel):
    """Response for predictor training"""
    success: bool
    final_loss: float
    training_time: float
    receipt_id: str
    epsilon_used: float

class OptimizeTopologyRequest(BaseModel):
    """Request to optimize network topology"""
    iterations: int = Field(default=20, ge=1, le=100)
    target_slo: float = Field(default=5.0, ge=1.0, le=20.0, description="Target SLO in seconds")

class OptimizeTopologyResponse(BaseModel):
    """Response for topology optimization"""
    success: bool
    total_rewires: int
    improvement: Dict[str, float]
    slo_compliance: Dict[str, Any]
    receipt_id: str
    epsilon_used: float

class NetworkStatusResponse(BaseModel):
    """Response for network status query"""
    initialized: bool
    nodes: Optional[int] = None
    edges: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    slo_violations: Optional[Dict[str, Any]] = None
    epsilon_usage: Dict[str, Any]
    podp_compliance: str

class AdaptationCycleResponse(BaseModel):
    """Response for adaptation cycle execution"""
    success: bool
    initial_violation_rate: float
    final_violation_rate: float
    improvement_applied: bool
    cycle_time: float
    receipt_id: str

# Helper functions
async def get_orchestrator() -> SOANOrchestrator:
    """Get or create the singleton orchestrator instance"""
    global _orchestrator
    
    async with _orchestrator_lock:
        if _orchestrator is None:
            config = SOANConfig(
                n_nodes=100,
                gnn_epochs=50,
                rewiring_iterations=20
            )
            _orchestrator = SOANOrchestrator(config)
            logger.info("Created new SOAN orchestrator instance")
    
    return _orchestrator

def generate_podp_receipt(operation: str, metadata: Dict) -> str:
    """Generate PoDP receipt for SOAN operations"""
    receipt = Receipt(
        receipt_id=Receipt.new_id(prefix="soan_"),
        dispute_id="soan_network",
        step=f"SOAN_{operation.upper()}",
        inputs=metadata,
        params={
            "version": "1.0.0",
            "component": "soan"
        },
        artifacts={
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        ts=datetime.now(timezone.utc).isoformat()
    ).finalize()
    
    return receipt.receipt_id

# Endpoints
@router.post("/initialize", response_model=NetworkInitResponse)
async def initialize_network(
    request: NetworkInitRequest,
    background_tasks: BackgroundTasks
) -> NetworkInitResponse:
    """Initialize the agent network topology"""
    try:
        orchestrator = await get_orchestrator()
        
        # Update configuration if different
        if orchestrator.config.n_nodes != request.n_nodes:
            config = SOANConfig(
                n_nodes=request.n_nodes,
                k_neighbors=request.k_neighbors,
                rewiring_prob=request.rewiring_prob
            )
            orchestrator = SOANOrchestrator(config)
            
            # Update global instance
            global _orchestrator
            async with _orchestrator_lock:
                _orchestrator = orchestrator
        
        # Initialize network
        graph = await asyncio.to_thread(
            orchestrator.initialize_network,
            seed=request.seed
        )
        
        # Compute metrics
        metrics = orchestrator.topology.compute_metrics()
        
        # Generate receipt
        receipt_id = generate_podp_receipt(
            "INIT_NETWORK",
            {
                "nodes": request.n_nodes,
                "edges": graph.number_of_edges(),
                "metrics": metrics
            }
        )
        
        # Get epsilon usage
        epsilon_usage = orchestrator.get_epsilon_usage()
        
        logger.info(f"Network initialized: {request.n_nodes} nodes, {graph.number_of_edges()} edges")
        
        return NetworkInitResponse(
            success=True,
            nodes=request.n_nodes,
            edges=graph.number_of_edges(),
            metrics=metrics,
            receipt_id=receipt_id,
            epsilon_used=epsilon_usage['total_used']
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize network: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Network initialization failed: {str(e)}"
        )

@router.post("/train", response_model=TrainPredictorResponse)
async def train_predictor(
    request: TrainPredictorRequest,
    background_tasks: BackgroundTasks
) -> TrainPredictorResponse:
    """Train the GNN latency predictor"""
    try:
        orchestrator = await get_orchestrator()
        
        if orchestrator.graph is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Network not initialized. Call /soan/initialize first"
            )
        
        # Update training epochs in config
        orchestrator.config.gnn_epochs = request.epochs
        
        # Train predictor (run in background for long training)
        stats = await asyncio.to_thread(
            orchestrator.train_latency_predictor,
            training_samples=request.training_samples,
            validation_split=request.validation_split
        )
        
        # Generate receipt
        receipt_id = generate_podp_receipt(
            "TRAIN_GNN",
            {
                "samples": request.training_samples,
                "epochs": request.epochs,
                "final_loss": stats['training_losses'][-1],
                "training_time": stats['training_time']
            }
        )
        
        # Get epsilon usage
        epsilon_usage = orchestrator.get_epsilon_usage()
        
        logger.info(f"GNN training completed: loss={stats['training_losses'][-1]:.4f}")
        
        return TrainPredictorResponse(
            success=True,
            final_loss=stats['training_losses'][-1],
            training_time=stats['training_time'],
            receipt_id=receipt_id,
            epsilon_used=epsilon_usage['gnn_predictor']['epsilon_used']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to train predictor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

@router.post("/optimize", response_model=OptimizeTopologyResponse)
async def optimize_topology(
    request: OptimizeTopologyRequest,
    background_tasks: BackgroundTasks
) -> OptimizeTopologyResponse:
    """Optimize network topology through rewiring"""
    try:
        orchestrator = await get_orchestrator()
        
        if orchestrator.graph is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Network not initialized"
            )
        
        # Predict current latencies
        await asyncio.to_thread(orchestrator.predict_latencies)
        
        # Track initial SLO compliance
        orchestrator.config.slo_threshold = request.target_slo
        initial_slo = orchestrator.track_slo_compliance()
        
        # Run optimization
        optimization_stats = await asyncio.to_thread(
            orchestrator.optimize_topology,
            iterations=request.iterations
        )
        
        # Check final SLO compliance
        final_slo = orchestrator.track_slo_compliance()
        
        # Generate receipt
        receipt_id = generate_podp_receipt(
            "OPTIMIZE_TOPOLOGY",
            {
                "iterations": request.iterations,
                "total_rewires": optimization_stats['total_rewires'],
                "initial_violations": initial_slo['violation_rate'],
                "final_violations": final_slo['violation_rate']
            }
        )
        
        # Get epsilon usage
        epsilon_usage = orchestrator.get_epsilon_usage()
        
        logger.info(
            f"Topology optimization completed: {optimization_stats['total_rewires']} rewires, "
            f"violations {initial_slo['violation_rate']:.2%} -> {final_slo['violation_rate']:.2%}"
        )
        
        return OptimizeTopologyResponse(
            success=True,
            total_rewires=optimization_stats['total_rewires'],
            improvement=optimization_stats['improvement'],
            slo_compliance={
                'initial_violation_rate': initial_slo['violation_rate'],
                'final_violation_rate': final_slo['violation_rate'],
                'improvement': initial_slo['violation_rate'] - final_slo['violation_rate']
            },
            receipt_id=receipt_id,
            epsilon_used=epsilon_usage['rewiring']['epsilon_used']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to optimize topology: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )

@router.post("/adapt", response_model=AdaptationCycleResponse)
async def run_adaptation_cycle(
    background_tasks: BackgroundTasks
) -> AdaptationCycleResponse:
    """Run a complete adaptation cycle"""
    try:
        orchestrator = await get_orchestrator()
        
        if orchestrator.graph is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Network not initialized"
            )
        
        # Run adaptation cycle
        cycle_stats = await asyncio.to_thread(
            orchestrator.run_adaptation_cycle
        )
        
        # Generate receipt
        receipt_id = generate_podp_receipt(
            "ADAPTATION_CYCLE",
            cycle_stats
        )
        
        logger.info(
            f"Adaptation cycle completed: violations {cycle_stats['initial_violation_rate']:.2%} -> "
            f"{cycle_stats['final_violation_rate']:.2%}"
        )
        
        return AdaptationCycleResponse(
            success=True,
            initial_violation_rate=cycle_stats['initial_violation_rate'],
            final_violation_rate=cycle_stats['final_violation_rate'],
            improvement_applied=cycle_stats['improvement']['optimization_applied'],
            cycle_time=cycle_stats['cycle_time'],
            receipt_id=receipt_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run adaptation cycle: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adaptation failed: {str(e)}"
        )

@router.get("/status", response_model=NetworkStatusResponse)
async def get_network_status() -> NetworkStatusResponse:
    """Get current network status and metrics"""
    try:
        orchestrator = await get_orchestrator()
        
        if orchestrator.graph is None:
            return NetworkStatusResponse(
                initialized=False,
                epsilon_usage=orchestrator.get_epsilon_usage(),
                podp_compliance="COMPLIANT"
            )
        
        # Get network state
        state = orchestrator.get_network_state()
        
        # Get PoDP compliance
        compliance_report = orchestrator.get_podp_compliance_report()
        
        return NetworkStatusResponse(
            initialized=True,
            nodes=state['nodes'],
            edges=state['edges'],
            metrics=state['topology_metrics'],
            slo_violations=state['slo_violations'],
            epsilon_usage=state['epsilon_usage'],
            podp_compliance=compliance_report['compliance_status']
        )
        
    except Exception as e:
        logger.error(f"Failed to get network status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status query failed: {str(e)}"
        )

@router.post("/save")
async def save_network_state(path: str = "./soan_state"):
    """Save network state to disk"""
    try:
        orchestrator = await get_orchestrator()
        
        if orchestrator.graph is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No network state to save"
            )
        
        await asyncio.to_thread(orchestrator.save_state, path)
        
        return {
            "success": True,
            "path": path,
            "message": "Network state saved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save state: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Save failed: {str(e)}"
        )

@router.post("/load")
async def load_network_state(path: str = "./soan_state"):
    """Load network state from disk"""
    try:
        orchestrator = await get_orchestrator()
        
        await asyncio.to_thread(orchestrator.load_state, path)
        
        return {
            "success": True,
            "path": path,
            "message": "Network state loaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to load state: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Load failed: {str(e)}"
        )

@router.get("/podp-report")
async def get_podp_compliance_report():
    """Get comprehensive PoDP compliance report"""
    try:
        orchestrator = await get_orchestrator()
        
        report = orchestrator.get_podp_compliance_report()
        
        return {
            "success": True,
            "report": report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate PoDP report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )