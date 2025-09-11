"""
Watts-Strogatz Network Topology Generator with PoDP compliance

Implements small-world network generation with high clustering
and short path lengths for distributed agent coordination.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
import time
import hashlib
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PoDPReceipt:
    """Proof of Distributed Processing receipt"""
    operation: str
    timestamp: float
    node_id: str
    epsilon_used: float
    input_hash: str
    output_hash: str
    metadata: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

    def verify(self) -> bool:
        """Verify receipt integrity"""
        return len(self.input_hash) == 64 and len(self.output_hash) == 64


class WattsStrogatzNetwork:
    """
    Watts-Strogatz small-world network generator with PoDP instrumentation
    
    ε-ledger budget allocation:
    - Network generation: 0.002ε
    - Metric computation: 0.001ε per metric
    - Feature initialization: 0.001ε
    """
    
    def __init__(
        self,
        n_nodes: int = 100,
        k: int = 6,
        p: float = 0.1,
        epsilon_budget: float = 0.005
    ):
        """
        Initialize Watts-Strogatz network parameters
        
        Args:
            n_nodes: Number of nodes in the network
            k: Each node is connected to k nearest neighbors in ring
            p: Probability of rewiring each edge
            epsilon_budget: ε-ledger budget for network operations
        """
        self.n_nodes = n_nodes
        self.k = k
        self.p = p
        self.epsilon_budget = epsilon_budget
        self.epsilon_used = 0.0
        self.graph: Optional[nx.Graph] = None
        self.node_features: Optional[np.ndarray] = None
        self.receipts: List[PoDPReceipt] = []
        
        logger.info(f"Initialized WS network: N={n_nodes}, k={k}, p={p}")
    
    def _generate_receipt(
        self,
        operation: str,
        input_data: any,
        output_data: any,
        epsilon_cost: float
    ) -> PoDPReceipt:
        """Generate PoDP receipt for operation"""
        input_hash = hashlib.sha256(
            json.dumps(str(input_data)).encode()
        ).hexdigest()
        output_hash = hashlib.sha256(
            json.dumps(str(output_data)).encode()
        ).hexdigest()
        
        receipt = PoDPReceipt(
            operation=operation,
            timestamp=time.time(),
            node_id=f"ws_topology_{id(self)}",
            epsilon_used=epsilon_cost,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                'n_nodes': self.n_nodes,
                'k': self.k,
                'p': self.p
            }
        )
        
        self.receipts.append(receipt)
        self.epsilon_used += epsilon_cost
        
        if self.epsilon_used > self.epsilon_budget:
            raise ValueError(f"ε-ledger budget exceeded: {self.epsilon_used:.4f} > {self.epsilon_budget:.4f}")
        
        return receipt
    
    def generate(self, seed: Optional[int] = None) -> nx.Graph:
        """
        Generate Watts-Strogatz small-world network
        
        Returns:
            NetworkX graph with small-world properties
        """
        start_time = time.time()
        
        if seed is not None:
            np.random.seed(seed)
            
        # Generate WS network
        self.graph = nx.watts_strogatz_graph(self.n_nodes, self.k, self.p, seed=seed)
        
        # Initialize node features
        self.node_features = self._initialize_features()
        
        # Add features to graph nodes
        for i in range(self.n_nodes):
            self.graph.nodes[i]['features'] = self.node_features[i]
            self.graph.nodes[i]['service_rate'] = np.random.uniform(1.0, 2.0)
            
        # Generate PoDP receipt
        receipt = self._generate_receipt(
            operation="generate_watts_strogatz",
            input_data={'n': self.n_nodes, 'k': self.k, 'p': self.p, 'seed': seed},
            output_data={'nodes': self.n_nodes, 'edges': self.graph.number_of_edges()},
            epsilon_cost=0.002
        )
        
        logger.info(f"Generated WS network in {time.time() - start_time:.3f}s")
        logger.debug(f"PoDP receipt: {receipt.to_dict()}")
        
        return self.graph
    
    def _initialize_features(self) -> np.ndarray:
        """
        Initialize node features [skills, queue, failure, misc]
        
        Returns:
            Array of shape (n_nodes, 4) with node features
        """
        features = np.zeros((self.n_nodes, 4))
        
        # Skills: Random specialization levels
        features[:, 0] = np.random.uniform(0.5, 1.0, self.n_nodes)
        
        # Queue length: Initially empty
        features[:, 1] = 0.0
        
        # Failure rate: Low initial failure probability
        features[:, 2] = np.random.uniform(0.0, 0.1, self.n_nodes)
        
        # Miscellaneous: Random capacity indicator
        features[:, 3] = np.random.uniform(0.3, 1.0, self.n_nodes)
        
        receipt = self._generate_receipt(
            operation="initialize_features",
            input_data={'n_nodes': self.n_nodes},
            output_data={'shape': features.shape},
            epsilon_cost=0.001
        )
        
        return features
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute network topology metrics
        
        Returns:
            Dictionary with network metrics
        """
        if self.graph is None:
            raise ValueError("Network not generated yet")
        
        start_time = time.time()
        
        metrics = {}
        
        # Average shortest path length
        if nx.is_connected(self.graph):
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(self.graph)
        else:
            # Compute for largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(subgraph)
        
        # Clustering coefficient
        metrics['clustering_coefficient'] = nx.average_clustering(self.graph)
        
        # Algebraic connectivity (Fiedler value)
        try:
            metrics['algebraic_connectivity'] = nx.algebraic_connectivity(self.graph)
        except:
            metrics['algebraic_connectivity'] = 0.0
        
        # Degree statistics
        degrees = [d for n, d in self.graph.degree()]
        metrics['degree_mean'] = np.mean(degrees)
        metrics['degree_std'] = np.std(degrees)
        
        # Network density
        metrics['density'] = nx.density(self.graph)
        
        # Generate PoDP receipt
        receipt = self._generate_receipt(
            operation="compute_metrics",
            input_data={'graph_nodes': self.n_nodes},
            output_data=metrics,
            epsilon_cost=0.001
        )
        
        logger.info(f"Computed metrics in {time.time() - start_time:.3f}s")
        
        return metrics
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        if self.graph is None:
            raise ValueError("Network not generated yet")
            
        return nx.adjacency_matrix(self.graph).todense()
    
    def get_laplacian_matrix(self) -> np.ndarray:
        """Get Laplacian matrix for spectral analysis"""
        if self.graph is None:
            raise ValueError("Network not generated yet")
            
        return nx.laplacian_matrix(self.graph).todense()
    
    def add_edge(self, u: int, v: int) -> PoDPReceipt:
        """Add edge with PoDP tracking"""
        if self.graph is None:
            raise ValueError("Network not generated yet")
            
        self.graph.add_edge(u, v)
        
        receipt = self._generate_receipt(
            operation="add_edge",
            input_data={'u': u, 'v': v},
            output_data={'edge_added': (u, v)},
            epsilon_cost=0.0001
        )
        
        return receipt
    
    def remove_edge(self, u: int, v: int) -> PoDPReceipt:
        """Remove edge with PoDP tracking"""
        if self.graph is None:
            raise ValueError("Network not generated yet")
            
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            
        receipt = self._generate_receipt(
            operation="remove_edge",
            input_data={'u': u, 'v': v},
            output_data={'edge_removed': (u, v)},
            epsilon_cost=0.0001
        )
        
        return receipt
    
    def get_node_features(self, node_id: int) -> np.ndarray:
        """Get features for specific node"""
        if self.node_features is None:
            raise ValueError("Features not initialized")
            
        return self.node_features[node_id]
    
    def update_node_features(self, node_id: int, features: np.ndarray) -> PoDPReceipt:
        """Update node features with PoDP tracking"""
        if self.node_features is None:
            raise ValueError("Features not initialized")
            
        old_features = self.node_features[node_id].copy()
        self.node_features[node_id] = features
        
        receipt = self._generate_receipt(
            operation="update_features",
            input_data={'node': node_id, 'old': old_features.tolist()},
            output_data={'node': node_id, 'new': features.tolist()},
            epsilon_cost=0.0001
        )
        
        return receipt
    
    def get_podp_summary(self) -> Dict:
        """Get PoDP compliance summary"""
        return {
            'total_receipts': len(self.receipts),
            'epsilon_used': self.epsilon_used,
            'epsilon_budget': self.epsilon_budget,
            'epsilon_remaining': self.epsilon_budget - self.epsilon_used,
            'operations': [r.operation for r in self.receipts]
        }