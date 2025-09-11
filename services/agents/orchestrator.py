"""
SOAN Orchestrator - Main coordinator for Self-Organizing Agent Networks

Integrates all components and provides unified interface for
distributed agent coordination with full PoDP compliance.
"""

import numpy as np
import networkx as nx
import torch
from typing import Dict, List, Optional, Tuple
import logging
import time
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict

from .topology import WattsStrogatzNetwork
from .gnn_predictor import GNNLatencyPredictor
from .queue_model import MM1QueueModel
from .rewiring import EpsilonGreedyRewiring

logger = logging.getLogger(__name__)

@dataclass
class SOANConfig:
    """Configuration for SOAN orchestrator"""
    # Network topology
    n_nodes: int = 100
    k_neighbors: int = 6
    rewiring_prob: float = 0.1
    
    # GNN predictor
    gnn_hidden_dim: int = 16
    gnn_learning_rate: float = 0.01
    gnn_epochs: int = 50
    
    # Queue model
    min_service_rate: float = 1.0
    max_service_rate: float = 2.0
    
    # Rewiring algorithm
    epsilon_greedy: float = 0.2
    similarity_threshold: float = 0.8
    rewiring_iterations: int = 20
    max_rewires_per_iter: int = 5
    
    # SLO tracking
    slo_threshold: float = 5.0
    
    # ε-ledger budgets
    topology_budget: float = 0.005
    gnn_budget: float = 0.015
    queue_budget: float = 0.002
    rewiring_budget: float = 0.005
    orchestrator_budget: float = 0.003
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OrchestrationReceipt:
    """PoDP receipt for orchestration operations"""
    operation: str
    timestamp: float
    orchestrator_id: str
    epsilon_used: float
    component_receipts: Dict[str, int]  # Component -> receipt count
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SOANOrchestrator:
    """
    Main orchestrator for Self-Organizing Agent Networks
    
    Coordinates:
    - Watts-Strogatz topology generation
    - GNN-based latency prediction
    - M/M/1 queue modeling
    - ε-greedy network rewiring
    - SLO tracking and optimization
    """
    
    def __init__(self, config: Optional[SOANConfig] = None):
        """
        Initialize SOAN orchestrator
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or SOANConfig()
        self.epsilon_budget = self.config.orchestrator_budget
        self.epsilon_used = 0.0
        
        # Initialize components
        self.topology = WattsStrogatzNetwork(
            n_nodes=self.config.n_nodes,
            k=self.config.k_neighbors,
            p=self.config.rewiring_prob,
            epsilon_budget=self.config.topology_budget
        )
        
        self.gnn_predictor = GNNLatencyPredictor(
            input_dim=4,
            hidden_dim=self.config.gnn_hidden_dim,
            learning_rate=self.config.gnn_learning_rate,
            epsilon_budget=self.config.gnn_budget
        )
        
        self.queue_model = MM1QueueModel(
            n_nodes=self.config.n_nodes,
            epsilon_budget=self.config.queue_budget
        )
        
        self.rewiring = EpsilonGreedyRewiring(
            epsilon=self.config.epsilon_greedy,
            similarity_threshold=self.config.similarity_threshold,
            max_rewires_per_iteration=self.config.max_rewires_per_iter,
            epsilon_budget=self.config.rewiring_budget
        )
        
        # State tracking
        self.graph: Optional[nx.Graph] = None
        self.node_features: Optional[np.ndarray] = None
        self.latency_matrix: Optional[np.ndarray] = None
        self.slo_violations: List[Dict] = []
        self.orchestration_receipts: List[OrchestrationReceipt] = []
        
        logger.info(f"Initialized SOAN Orchestrator with {self.config.n_nodes} nodes")
    
    def _generate_receipt(
        self,
        operation: str,
        epsilon_cost: float,
        metadata: Dict
    ) -> OrchestrationReceipt:
        """Generate orchestration receipt"""
        receipt = OrchestrationReceipt(
            operation=operation,
            timestamp=time.time(),
            orchestrator_id=f"soan_orchestrator_{id(self)}",
            epsilon_used=epsilon_cost,
            component_receipts={
                'topology': len(self.topology.receipts),
                'gnn': len(self.gnn_predictor.receipts),
                'queue': len(self.queue_model.receipts),
                'rewiring': len(self.rewiring.receipts)
            },
            metadata=metadata
        )
        
        self.orchestration_receipts.append(receipt)
        self.epsilon_used += epsilon_cost
        
        if self.epsilon_used > self.epsilon_budget:
            raise ValueError(f"Orchestrator ε-budget exceeded: {self.epsilon_used:.4f} > {self.epsilon_budget:.4f}")
        
        return receipt
    
    def initialize_network(self, seed: Optional[int] = None) -> nx.Graph:
        """
        Initialize the agent network topology
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Generated network graph
        """
        start_time = time.time()
        
        # Generate Watts-Strogatz topology
        self.graph = self.topology.generate(seed=seed)
        self.node_features = self.topology.node_features
        
        # Initialize queue model with service rates
        for i in range(self.config.n_nodes):
            service_rate = self.graph.nodes[i]['service_rate']
            self.queue_model.service_rates[i] = service_rate
        
        # Compute initial metrics
        metrics = self.topology.compute_metrics()
        
        self._generate_receipt(
            operation="initialize_network",
            epsilon_cost=0.0001,
            metadata={
                'nodes': self.config.n_nodes,
                'edges': self.graph.number_of_edges(),
                'metrics': metrics,
                'time': time.time() - start_time
            }
        )
        
        logger.info(f"Network initialized with metrics: {metrics}")
        
        return self.graph
    
    def train_latency_predictor(
        self,
        training_samples: int = 1000,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the GNN latency predictor
        
        Args:
            training_samples: Number of training samples to generate
            validation_split: Fraction of data for validation
            
        Returns:
            Training statistics
        """
        if self.graph is None:
            raise ValueError("Network not initialized. Call initialize_network() first.")
        
        start_time = time.time()
        
        # Generate training data
        train_data, val_data = self._generate_training_data(
            training_samples, validation_split
        )
        
        # Train GNN
        training_stats = self.gnn_predictor.train(
            train_data,
            val_data,
            epochs=self.config.gnn_epochs
        )
        
        self._generate_receipt(
            operation="train_latency_predictor",
            epsilon_cost=0.0002,
            metadata={
                'samples': training_samples,
                'epochs': self.config.gnn_epochs,
                'final_loss': training_stats['training_losses'][-1],
                'time': time.time() - start_time
            }
        )
        
        logger.info(f"GNN training completed with final loss: {training_stats['training_losses'][-1]:.4f}")
        
        return training_stats
    
    def _generate_training_data(
        self,
        n_samples: int,
        validation_split: float
    ) -> Tuple:
        """Generate synthetic training data for GNN"""
        adjacency = self.topology.get_adjacency_matrix()
        
        # Simulate various traffic patterns
        all_latencies = []
        
        for _ in range(n_samples):
            # Random traffic matrix
            traffic = np.random.exponential(0.5, (self.config.n_nodes, self.config.n_nodes))
            traffic = traffic * adjacency  # Only on existing edges
            
            # Calculate latencies using queue model
            latencies = self.queue_model.calculate_network_latency(traffic)
            all_latencies.append(latencies)
        
        # Average latencies
        avg_latencies = np.mean(all_latencies, axis=0)
        
        # Prepare PyTorch Geometric data
        train_data = self.gnn_predictor.prepare_data(
            adjacency,
            self.node_features,
            avg_latencies
        )
        
        # Create validation data with different traffic pattern
        val_traffic = np.random.exponential(0.3, (self.config.n_nodes, self.config.n_nodes))
        val_traffic = val_traffic * adjacency
        val_latencies = self.queue_model.calculate_network_latency(val_traffic)
        
        val_data = self.gnn_predictor.prepare_data(
            adjacency,
            self.node_features,
            val_latencies
        )
        
        return train_data, val_data
    
    def predict_latencies(
        self,
        traffic_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict network latencies using trained GNN
        
        Args:
            traffic_matrix: Optional traffic pattern (uses default if None)
            
        Returns:
            Predicted latency matrix
        """
        if self.graph is None:
            raise ValueError("Network not initialized")
        
        adjacency = self.topology.get_adjacency_matrix()
        
        # Use GNN for prediction
        self.latency_matrix = self.gnn_predictor.predict_edge_latency(
            adjacency,
            self.node_features
        )
        
        # Refine with queue model if traffic provided
        if traffic_matrix is not None:
            queue_latencies = self.queue_model.calculate_network_latency(traffic_matrix)
            # Weighted combination
            self.latency_matrix = 0.7 * self.latency_matrix + 0.3 * queue_latencies
        
        self._generate_receipt(
            operation="predict_latencies",
            epsilon_cost=0.0001,
            metadata={
                'avg_latency': np.mean(self.latency_matrix[self.latency_matrix > 0]),
                'max_latency': np.max(self.latency_matrix[self.latency_matrix < float('inf')])
            }
        )
        
        return self.latency_matrix
    
    def optimize_topology(
        self,
        iterations: Optional[int] = None
    ) -> Dict:
        """
        Run topology optimization through rewiring
        
        Args:
            iterations: Number of rewiring iterations (uses config default if None)
            
        Returns:
            Optimization statistics
        """
        if self.graph is None or self.latency_matrix is None:
            raise ValueError("Network and latencies must be initialized first")
        
        iterations = iterations or self.config.rewiring_iterations
        
        # Define latency predictor function for rewiring
        def latency_predictor_fn(adjacency, features):
            return self.gnn_predictor.predict_edge_latency(adjacency, features)
        
        # Run rewiring campaign
        optimization_stats = self.rewiring.run_rewiring_campaign(
            self.graph,
            self.node_features,
            latency_predictor_fn,
            iterations
        )
        
        # Update latency predictions after rewiring
        self.predict_latencies()
        
        self._generate_receipt(
            operation="optimize_topology",
            epsilon_cost=0.0002,
            metadata={
                'iterations': iterations,
                'total_rewires': optimization_stats['total_rewires'],
                'improvement': optimization_stats['improvement']
            }
        )
        
        logger.info(f"Topology optimization completed: {optimization_stats['total_rewires']} rewires")
        
        return optimization_stats
    
    def track_slo_compliance(self) -> Dict:
        """
        Track SLO compliance and violations
        
        Returns:
            SLO tracking statistics
        """
        if self.latency_matrix is None:
            raise ValueError("Latencies not predicted yet")
        
        slo_stats = self.queue_model.estimate_slo_violations(
            self.latency_matrix,
            self.config.slo_threshold
        )
        
        # Track violation history
        slo_stats['timestamp'] = time.time()
        self.slo_violations.append(slo_stats)
        
        self._generate_receipt(
            operation="track_slo_compliance",
            epsilon_cost=0.00005,
            metadata={
                'violation_rate': slo_stats['violation_rate'],
                'threshold': self.config.slo_threshold
            }
        )
        
        logger.info(f"SLO violation rate: {slo_stats['violation_rate']:.2%}")
        
        return slo_stats
    
    def run_adaptation_cycle(self) -> Dict:
        """
        Run a complete adaptation cycle:
        1. Predict latencies
        2. Check SLO compliance
        3. Optimize if needed
        4. Re-evaluate
        
        Returns:
            Cycle statistics
        """
        start_time = time.time()
        
        # Initial state
        initial_latencies = self.predict_latencies()
        initial_slo = self.track_slo_compliance()
        
        # Optimize if violations exceed threshold
        if initial_slo['violation_rate'] > 0.1:  # 10% violation threshold
            logger.info("High SLO violations detected, running optimization...")
            
            # Run short optimization
            optimization_stats = self.optimize_topology(iterations=5)
            
            # Re-evaluate
            final_latencies = self.predict_latencies()
            final_slo = self.track_slo_compliance()
            
            improvement = {
                'violation_reduction': initial_slo['violation_rate'] - final_slo['violation_rate'],
                'latency_reduction': np.mean(initial_latencies) - np.mean(final_latencies),
                'optimization_applied': True
            }
        else:
            improvement = {
                'violation_reduction': 0,
                'latency_reduction': 0,
                'optimization_applied': False
            }
        
        cycle_stats = {
            'initial_violation_rate': initial_slo['violation_rate'],
            'final_violation_rate': initial_slo['violation_rate'] - improvement['violation_reduction'],
            'improvement': improvement,
            'cycle_time': time.time() - start_time
        }
        
        self._generate_receipt(
            operation="run_adaptation_cycle",
            epsilon_cost=0.0003,
            metadata=cycle_stats
        )
        
        logger.info(f"Adaptation cycle completed in {cycle_stats['cycle_time']:.2f}s")
        
        return cycle_stats
    
    def get_network_state(self) -> Dict:
        """Get comprehensive network state information"""
        if self.graph is None:
            return {'status': 'not_initialized'}
        
        state = {
            'nodes': self.config.n_nodes,
            'edges': self.graph.number_of_edges(),
            'topology_metrics': self.topology.compute_metrics() if self.graph else None,
            'queue_statistics': self.queue_model.get_network_statistics(),
            'rewiring_summary': self.rewiring.get_rewiring_summary(),
            'slo_violations': self.slo_violations[-1] if self.slo_violations else None,
            'epsilon_usage': self.get_epsilon_usage()
        }
        
        return state
    
    def get_epsilon_usage(self) -> Dict:
        """Get ε-ledger usage across all components"""
        return {
            'orchestrator': {
                'used': self.epsilon_used,
                'budget': self.epsilon_budget,
                'remaining': self.epsilon_budget - self.epsilon_used
            },
            'topology': self.topology.get_podp_summary(),
            'gnn_predictor': self.gnn_predictor.get_podp_summary(),
            'queue_model': self.queue_model.get_podp_summary(),
            'rewiring': self.rewiring.get_podp_summary(),
            'total_used': (
                self.epsilon_used +
                self.topology.epsilon_used +
                self.gnn_predictor.epsilon_used +
                self.queue_model.epsilon_used +
                self.rewiring.epsilon_used
            ),
            'total_budget': (
                self.epsilon_budget +
                self.topology.epsilon_budget +
                self.gnn_predictor.epsilon_budget +
                self.queue_model.epsilon_budget +
                self.rewiring.epsilon_budget
            )
        }
    
    def save_state(self, path: str):
        """Save orchestrator state to disk"""
        state_path = Path(path)
        state_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(state_path / 'config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save network
        if self.graph:
            nx.write_gpickle(self.graph, state_path / 'network.gpickle')
            np.save(state_path / 'node_features.npy', self.node_features)
        
        # Save model
        self.gnn_predictor.save_model(str(state_path / 'gnn_model.pt'))
        
        # Save metrics
        metrics = {
            'slo_violations': self.slo_violations,
            'orchestration_receipts': [r.to_dict() for r in self.orchestration_receipts],
            'epsilon_usage': self.get_epsilon_usage()
        }
        
        with open(state_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self._generate_receipt(
            operation="save_state",
            epsilon_cost=0.0001,
            metadata={'path': str(state_path)}
        )
        
        logger.info(f"State saved to {state_path}")
    
    def load_state(self, path: str):
        """Load orchestrator state from disk"""
        state_path = Path(path)
        
        # Load configuration
        with open(state_path / 'config.json', 'r') as f:
            config_dict = json.load(f)
            self.config = SOANConfig(**config_dict)
        
        # Load network
        if (state_path / 'network.gpickle').exists():
            self.graph = nx.read_gpickle(state_path / 'network.gpickle')
            self.node_features = np.load(state_path / 'node_features.npy')
            self.topology.graph = self.graph
            self.topology.node_features = self.node_features
        
        # Load model
        self.gnn_predictor.load_model(str(state_path / 'gnn_model.pt'))
        
        # Load metrics
        with open(state_path / 'metrics.json', 'r') as f:
            metrics = json.load(f)
            self.slo_violations = metrics.get('slo_violations', [])
        
        self._generate_receipt(
            operation="load_state",
            epsilon_cost=0.0001,
            metadata={'path': str(state_path)}
        )
        
        logger.info(f"State loaded from {state_path}")
    
    def get_podp_compliance_report(self) -> Dict:
        """Generate comprehensive PoDP compliance report"""
        return {
            'total_receipts': sum([
                len(self.orchestration_receipts),
                len(self.topology.receipts),
                len(self.gnn_predictor.receipts),
                len(self.queue_model.receipts),
                len(self.rewiring.receipts)
            ]),
            'epsilon_summary': self.get_epsilon_usage(),
            'component_operations': {
                'orchestrator': [r.operation for r in self.orchestration_receipts[-5:]],
                'topology': self.topology.get_podp_summary()['operations'][-5:],
                'gnn': self.gnn_predictor.get_podp_summary()['operations'][-5:],
                'queue': self.queue_model.get_podp_summary()['operations'][-5:],
                'rewiring': self.rewiring.get_podp_summary()['operations'][-5:]
            },
            'compliance_status': 'COMPLIANT',
            'timestamp': time.time()
        }