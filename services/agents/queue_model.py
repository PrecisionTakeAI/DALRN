"""
M/M/1 Queue Model for Agent Network

Implements queuing theory models for latency estimation
with PoDP compliance and ε-ledger budget tracking.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import hashlib
import json
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class QueuePoDPReceipt:
    """PoDP receipt for queue operations"""
    operation: str
    timestamp: float
    queue_id: str
    epsilon_used: float
    input_hash: str
    output_hash: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'timestamp': self.timestamp,
            'queue_id': self.queue_id,
            'epsilon_used': self.epsilon_used,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'metadata': self.metadata
        }


class MM1QueueModel:
    """
    M/M/1 Queue Model for distributed agent latency estimation
    
    Implements single-server queue with:
    - Poisson arrival process (rate λ)
    - Exponential service times (rate μ)
    - FIFO discipline
    - Infinite buffer capacity
    
    ε-ledger budget allocation:
    - Queue initialization: 0.0001ε per queue
    - Latency calculation: 0.00001ε per calculation
    - Batch processing: 0.0001ε per batch
    """
    
    def __init__(
        self,
        n_nodes: int = 100,
        epsilon_budget: float = 0.002
    ):
        """
        Initialize M/M/1 queue model for network nodes
        
        Args:
            n_nodes: Number of nodes in the network
            epsilon_budget: ε-ledger budget for queue operations
        """
        self.n_nodes = n_nodes
        self.epsilon_budget = epsilon_budget
        self.epsilon_used = 0.0
        
        # Queue parameters per node
        self.service_rates = np.random.uniform(1.0, 2.0, n_nodes)  # μ values
        self.arrival_rates = np.zeros(n_nodes)  # λ values (updated dynamically)
        self.queue_lengths = np.zeros(n_nodes)
        
        # Performance metrics
        self.waiting_times = np.zeros(n_nodes)
        self.system_times = np.zeros(n_nodes)
        self.utilizations = np.zeros(n_nodes)
        
        # PoDP tracking
        self.receipts: List[QueuePoDPReceipt] = []
        
        self._generate_receipt(
            operation="initialize_queues",
            input_data={'n_nodes': n_nodes},
            output_data={'service_rates': self.service_rates.mean()},
            epsilon_cost=0.0001 * n_nodes
        )
        
        logger.info(f"Initialized M/M/1 queues for {n_nodes} nodes")
    
    def _generate_receipt(
        self,
        operation: str,
        input_data: any,
        output_data: any,
        epsilon_cost: float
    ) -> QueuePoDPReceipt:
        """Generate PoDP receipt for queue operation"""
        input_hash = hashlib.sha256(
            json.dumps(str(input_data)).encode()
        ).hexdigest()
        output_hash = hashlib.sha256(
            json.dumps(str(output_data)).encode()
        ).hexdigest()
        
        receipt = QueuePoDPReceipt(
            operation=operation,
            timestamp=time.time(),
            queue_id=f"mm1_queue_{id(self)}",
            epsilon_used=epsilon_cost,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                'n_nodes': self.n_nodes,
                'model': 'M/M/1'
            }
        )
        
        self.receipts.append(receipt)
        self.epsilon_used += epsilon_cost
        
        if self.epsilon_used > self.epsilon_budget:
            raise ValueError(f"ε-ledger budget exceeded: {self.epsilon_used:.4f} > {self.epsilon_budget:.4f}")
        
        return receipt
    
    def calculate_latency(
        self,
        node_id: int,
        arrival_rate: Optional[float] = None
    ) -> float:
        """
        Calculate expected latency for a single node using M/M/1 formula
        
        Args:
            node_id: Node identifier
            arrival_rate: Arrival rate λ (uses stored value if None)
            
        Returns:
            Expected latency (waiting time + service time)
        """
        if arrival_rate is not None:
            self.arrival_rates[node_id] = arrival_rate
        
        λ = self.arrival_rates[node_id]
        μ = self.service_rates[node_id]
        
        # Check stability condition
        if λ >= μ:
            warnings.warn(f"Unstable queue at node {node_id}: λ={λ:.3f} >= μ={μ:.3f}")
            latency = float('inf')
        else:
            # Little's Law: L = λW
            # For M/M/1: W = 1/(μ - λ)
            utilization = λ / μ
            self.utilizations[node_id] = utilization
            
            # Average waiting time in queue
            self.waiting_times[node_id] = λ / (μ * (μ - λ))
            
            # Average time in system (waiting + service)
            self.system_times[node_id] = 1 / (μ - λ)
            
            # Average queue length
            self.queue_lengths[node_id] = λ * λ / (μ * (μ - λ))
            
            latency = self.system_times[node_id]
        
        self._generate_receipt(
            operation=f"calculate_latency_node_{node_id}",
            input_data={'node': node_id, 'λ': λ, 'μ': μ},
            output_data={'latency': latency if latency != float('inf') else 'inf'},
            epsilon_cost=0.00001
        )
        
        return latency
    
    def calculate_network_latency(
        self,
        traffic_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate latency matrix for entire network
        
        Args:
            traffic_matrix: N x N matrix of traffic rates between nodes
            
        Returns:
            N x N latency matrix
        """
        n = self.n_nodes
        latency_matrix = np.zeros((n, n))
        
        # Calculate arrival rates per node (sum of incoming traffic)
        for j in range(n):
            self.arrival_rates[j] = traffic_matrix[:, j].sum()
        
        # Calculate latencies
        for i in range(n):
            for j in range(n):
                if traffic_matrix[i, j] > 0:
                    # Latency = queuing delay at destination + transmission
                    queue_latency = self.calculate_latency(j)
                    transmission_latency = 0.1  # Base transmission latency
                    
                    if queue_latency == float('inf'):
                        latency_matrix[i, j] = float('inf')
                    else:
                        latency_matrix[i, j] = queue_latency + transmission_latency
        
        self._generate_receipt(
            operation="calculate_network_latency",
            input_data={'traffic_shape': traffic_matrix.shape},
            output_data={'avg_latency': np.mean(latency_matrix[latency_matrix < float('inf')])},
            epsilon_cost=0.0001
        )
        
        return latency_matrix
    
    def update_service_rate(self, node_id: int, new_rate: float) -> QueuePoDPReceipt:
        """Update service rate μ for a node"""
        old_rate = self.service_rates[node_id]
        self.service_rates[node_id] = new_rate
        
        receipt = self._generate_receipt(
            operation=f"update_service_rate_node_{node_id}",
            input_data={'node': node_id, 'old_rate': old_rate},
            output_data={'node': node_id, 'new_rate': new_rate},
            epsilon_cost=0.00001
        )
        
        return receipt
    
    def get_queue_metrics(self, node_id: int) -> Dict[str, float]:
        """Get comprehensive queue metrics for a node"""
        metrics = {
            'service_rate': self.service_rates[node_id],
            'arrival_rate': self.arrival_rates[node_id],
            'utilization': self.utilizations[node_id],
            'queue_length': self.queue_lengths[node_id],
            'waiting_time': self.waiting_times[node_id],
            'system_time': self.system_times[node_id],
            'is_stable': self.arrival_rates[node_id] < self.service_rates[node_id]
        }
        
        return metrics
    
    def get_network_statistics(self) -> Dict[str, float]:
        """Get network-wide queue statistics"""
        stable_nodes = self.arrival_rates < self.service_rates
        
        stats = {
            'avg_service_rate': np.mean(self.service_rates),
            'avg_arrival_rate': np.mean(self.arrival_rates),
            'avg_utilization': np.mean(self.utilizations[stable_nodes]) if stable_nodes.any() else 0,
            'avg_queue_length': np.mean(self.queue_lengths[stable_nodes]) if stable_nodes.any() else 0,
            'avg_waiting_time': np.mean(self.waiting_times[stable_nodes]) if stable_nodes.any() else 0,
            'avg_system_time': np.mean(self.system_times[stable_nodes]) if stable_nodes.any() else 0,
            'stable_nodes': int(stable_nodes.sum()),
            'unstable_nodes': int((~stable_nodes).sum())
        }
        
        return stats
    
    def simulate_arrivals(
        self,
        duration: float = 10.0,
        base_rate: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Simulate Poisson arrival process for testing
        
        Args:
            duration: Simulation duration
            base_rate: Base arrival rate
            
        Returns:
            Dictionary with arrival times and counts
        """
        arrivals = {}
        
        for node_id in range(self.n_nodes):
            # Generate Poisson arrivals
            rate = base_rate * np.random.uniform(0.5, 1.5)
            n_arrivals = np.random.poisson(rate * duration)
            arrival_times = np.sort(np.random.uniform(0, duration, n_arrivals))
            
            arrivals[f'node_{node_id}'] = {
                'times': arrival_times,
                'count': n_arrivals,
                'rate': n_arrivals / duration
            }
        
        self._generate_receipt(
            operation="simulate_arrivals",
            input_data={'duration': duration, 'base_rate': base_rate},
            output_data={'total_arrivals': sum(a['count'] for a in arrivals.values())},
            epsilon_cost=0.0001
        )
        
        return arrivals
    
    def estimate_slo_violations(
        self,
        latency_matrix: np.ndarray,
        slo_threshold: float = 5.0
    ) -> Dict[str, any]:
        """
        Estimate SLO violations based on latency predictions
        
        Args:
            latency_matrix: Predicted latencies
            slo_threshold: Maximum acceptable latency
            
        Returns:
            SLO violation statistics
        """
        violations = latency_matrix > slo_threshold
        infinite_latencies = np.isinf(latency_matrix)
        
        stats = {
            'violation_rate': violations.sum() / latency_matrix.size,
            'infinite_rate': infinite_latencies.sum() / latency_matrix.size,
            'violating_pairs': np.argwhere(violations).tolist(),
            'max_latency': np.max(latency_matrix[~infinite_latencies]) if (~infinite_latencies).any() else float('inf'),
            'percentiles': {
                'p50': np.percentile(latency_matrix[~infinite_latencies], 50) if (~infinite_latencies).any() else 0,
                'p90': np.percentile(latency_matrix[~infinite_latencies], 90) if (~infinite_latencies).any() else 0,
                'p99': np.percentile(latency_matrix[~infinite_latencies], 99) if (~infinite_latencies).any() else 0
            }
        }
        
        self._generate_receipt(
            operation="estimate_slo_violations",
            input_data={'threshold': slo_threshold},
            output_data={'violation_rate': stats['violation_rate']},
            epsilon_cost=0.00001
        )
        
        return stats
    
    def optimize_service_rates(
        self,
        target_utilization: float = 0.7
    ) -> np.ndarray:
        """
        Optimize service rates to achieve target utilization
        
        Args:
            target_utilization: Target utilization level (ρ = λ/μ)
            
        Returns:
            Optimized service rates
        """
        # Calculate required service rates
        optimized_rates = self.arrival_rates / target_utilization
        
        # Apply constraints
        optimized_rates = np.clip(optimized_rates, 1.0, 2.0)
        
        # Update service rates
        old_rates = self.service_rates.copy()
        self.service_rates = optimized_rates
        
        self._generate_receipt(
            operation="optimize_service_rates",
            input_data={'target_util': target_utilization},
            output_data={'avg_change': np.mean(np.abs(optimized_rates - old_rates))},
            epsilon_cost=0.0001
        )
        
        return optimized_rates
    
    def get_podp_summary(self) -> Dict:
        """Get PoDP compliance summary"""
        return {
            'total_receipts': len(self.receipts),
            'epsilon_used': self.epsilon_used,
            'epsilon_budget': self.epsilon_budget,
            'epsilon_remaining': self.epsilon_budget - self.epsilon_used,
            'operations': [r.operation for r in self.receipts[-10:]]  # Last 10 operations
        }