"""
Watts-Strogatz Network Topology Generator for SOAN.

This module implements the network topology component with PoDP receipt generation
and ε-ledger budget tracking for all operations.
"""

import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict

# Import PoDP utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.podp import Receipt, ReceiptChain, keccak


@dataclass
class NetworkMetrics:
    """Metrics for network topology analysis."""
    clustering_coefficient: float
    average_path_length: float
    diameter: int
    density: float
    average_degree: float
    connectivity: bool
    small_world_coefficient: Optional[float] = None
    epsilon_used: float = 0.0


class WattsStrogatzTopology:
    """
    Watts-Strogatz small-world network generator with PoDP compliance.

    Parameters:
    - N: Number of nodes (default 100)
    - k: Each node connected to k nearest neighbors (default 6)
    - p: Rewiring probability (default 0.1)
    """

    # ε-ledger budget allocations
    EPSILON_NETWORK_GENERATION = 0.001
    EPSILON_METRICS_CALCULATION = 0.0005
    EPSILON_EXPORT = 0.0002

    def __init__(self, N: int = 100, k: int = 6, p: float = 0.1):
        """Initialize topology generator with specified parameters."""
        if k >= N:
            raise ValueError(f"k ({k}) must be less than N ({N})")
        if k % 2 != 0:
            raise ValueError(f"k ({k}) must be even for regular ring lattice")
        if not 0 <= p <= 1:
            raise ValueError(f"p ({p}) must be in [0, 1]")

        self.N = N
        self.k = k
        self.p = p
        self.graph: Optional[nx.Graph] = None
        self.metrics: Optional[NetworkMetrics] = None
        self.receipt_chain = ReceiptChain(
            dispute_id=f"soan_topology_{int(time.time())}"
        )
        self.epsilon_spent = 0.0

    def generate(self) -> Tuple[nx.Graph, Receipt]:
        """
        Generate Watts-Strogatz network with PoDP receipt.

        Returns:
            Tuple of (NetworkX graph, PoDP receipt)
        """
        start_time = time.time()

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"topology_gen_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="network_generation_entry",
            inputs={
                "N": self.N,
                "k": self.k,
                "p": self.p
            },
            params={
                "algorithm": "watts_strogatz",
                "seed": None  # Will be set if provided
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Generate the network
        self.graph = nx.watts_strogatz_graph(self.N, self.k, self.p)

        # Add node attributes for later use
        for node in self.graph.nodes():
            self.graph.nodes[node]['queue_length'] = 0
            self.graph.nodes[node]['service_rate'] = np.random.uniform(1.0, 2.0)
            self.graph.nodes[node]['arrival_rate'] = 0.5

        # Calculate graph hash
        adj_matrix = nx.adjacency_matrix(self.graph).todense()
        graph_hash = keccak(json.dumps(adj_matrix.tolist(), sort_keys=True).encode())

        # Create exit receipt
        exit_receipt = Receipt(
            receipt_id=f"topology_gen_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="network_generation_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "nodes_created": self.N,
                "edges_created": self.graph.number_of_edges()
            },
            artifacts={
                "graph_hash": graph_hash,
                "node_count": self.N,
                "edge_count": self.graph.number_of_edges()
            },
            hashes={
                "adjacency_matrix": graph_hash
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Update epsilon budget
        self.epsilon_spent += self.EPSILON_NETWORK_GENERATION
        exit_receipt.artifacts["epsilon_used"] = self.EPSILON_NETWORK_GENERATION
        exit_receipt.artifacts["epsilon_total"] = self.epsilon_spent

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        # Log generation time
        exit_receipt.artifacts["generation_time_ms"] = (time.time() - start_time) * 1000

        return self.graph, exit_receipt

    def calculate_metrics(self) -> Tuple[NetworkMetrics, Receipt]:
        """
        Calculate comprehensive network metrics with PoDP receipt.

        Returns:
            Tuple of (NetworkMetrics, PoDP receipt)
        """
        if self.graph is None:
            raise RuntimeError("Network must be generated before calculating metrics")

        start_time = time.time()

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"metrics_calc_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="metrics_calculation_entry",
            inputs={
                "graph_nodes": self.N,
                "graph_edges": self.graph.number_of_edges()
            },
            params={
                "metrics_requested": ["clustering", "path_length", "diameter", "density", "degree", "connectivity"]
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Calculate metrics
        clustering = nx.average_clustering(self.graph)

        # Handle path length for disconnected graphs
        if nx.is_connected(self.graph):
            avg_path_length = nx.average_shortest_path_length(self.graph)
            diameter = nx.diameter(self.graph)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)

        density = nx.density(self.graph)
        avg_degree = sum(dict(self.graph.degree()).values()) / self.N
        is_connected = nx.is_connected(self.graph)

        # Calculate small-world coefficient (sigma)
        # sigma = (C/C_rand) / (L/L_rand) where C is clustering and L is path length
        # For reference, generate random graph with same N and average degree
        random_graph = nx.erdos_renyi_graph(self.N, self.k / self.N)
        C_rand = nx.average_clustering(random_graph)

        if nx.is_connected(random_graph):
            L_rand = nx.average_shortest_path_length(random_graph)
        else:
            largest_cc_rand = max(nx.connected_components(random_graph), key=len)
            subgraph_rand = random_graph.subgraph(largest_cc_rand)
            L_rand = nx.average_shortest_path_length(subgraph_rand)

        # Avoid division by zero
        if C_rand > 0 and L_rand > 0:
            sigma = (clustering / C_rand) / (avg_path_length / L_rand)
        else:
            sigma = None

        self.metrics = NetworkMetrics(
            clustering_coefficient=clustering,
            average_path_length=avg_path_length,
            diameter=diameter,
            density=density,
            average_degree=avg_degree,
            connectivity=is_connected,
            small_world_coefficient=sigma,
            epsilon_used=self.EPSILON_METRICS_CALCULATION
        )

        # Create exit receipt
        exit_receipt = Receipt(
            receipt_id=f"metrics_calc_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="metrics_calculation_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "calculation_time_ms": (time.time() - start_time) * 1000
            },
            artifacts={
                "metrics": asdict(self.metrics)
            },
            hashes={
                "metrics_hash": keccak(json.dumps(asdict(self.metrics), sort_keys=True).encode())
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Update epsilon budget
        self.epsilon_spent += self.EPSILON_METRICS_CALCULATION
        exit_receipt.artifacts["epsilon_used"] = self.EPSILON_METRICS_CALCULATION
        exit_receipt.artifacts["epsilon_total"] = self.epsilon_spent

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        return self.metrics, exit_receipt

    def export_to_dict(self) -> Tuple[Dict[str, Any], Receipt]:
        """
        Export network to dictionary format with PoDP receipt.

        Returns:
            Tuple of (network dictionary, PoDP receipt)
        """
        if self.graph is None:
            raise RuntimeError("Network must be generated before exporting")

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"export_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="network_export_entry",
            inputs={
                "format": "dict",
                "include_attributes": True
            },
            params={},
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Export network data
        network_data = {
            "parameters": {
                "N": self.N,
                "k": self.k,
                "p": self.p
            },
            "nodes": [
                {
                    "id": node,
                    "attributes": self.graph.nodes[node]
                }
                for node in self.graph.nodes()
            ],
            "edges": [
                {"source": u, "target": v}
                for u, v in self.graph.edges()
            ],
            "metrics": asdict(self.metrics) if self.metrics else None,
            "receipt_chain": {
                "dispute_id": self.receipt_chain.dispute_id,
                "merkle_root": self.receipt_chain.merkle_root,
                "receipt_count": len(self.receipt_chain.receipts)
            }
        }

        # Create exit receipt
        exit_receipt = Receipt(
            receipt_id=f"export_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="network_export_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "node_count": len(network_data["nodes"]),
                "edge_count": len(network_data["edges"])
            },
            artifacts={
                "export_hash": keccak(json.dumps(network_data, sort_keys=True).encode()),
                "epsilon_used": self.EPSILON_EXPORT,
                "epsilon_total": self.epsilon_spent + self.EPSILON_EXPORT
            },
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Update epsilon budget
        self.epsilon_spent += self.EPSILON_EXPORT

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        return network_data, exit_receipt

    def get_receipt_chain(self) -> ReceiptChain:
        """Get the complete receipt chain for all operations."""
        return self.receipt_chain

    def get_epsilon_budget_status(self) -> Dict[str, float]:
        """Get current epsilon budget status."""
        return {
            "spent": self.epsilon_spent,
            "remaining": 4.0 - self.epsilon_spent,  # Total budget of 4.0
            "breakdown": {
                "network_generation": self.EPSILON_NETWORK_GENERATION,
                "metrics_calculation": self.EPSILON_METRICS_CALCULATION if self.metrics else 0,
                "export": self.EPSILON_EXPORT if hasattr(self, '_exported') else 0
            }
        }