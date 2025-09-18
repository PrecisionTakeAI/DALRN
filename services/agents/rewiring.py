"""
ε-greedy Network Rewiring Algorithm for SOAN.

This module implements intelligent network topology optimization
with exploration-exploitation balance and full PoDP compliance.
"""

import json
import hashlib
import time
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_similarity

# Import PoDP utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.podp import Receipt, ReceiptChain, keccak


@dataclass
class RewiringMetrics:
    """Metrics for rewiring operation evaluation."""
    edges_rewired: int
    improvement_ratio: float
    exploration_count: int
    exploitation_count: int
    avg_path_length_before: float
    avg_path_length_after: float
    clustering_before: float
    clustering_after: float
    convergence_achieved: bool
    epsilon_used: float


class EpsilonGreedyRewiring:
    """
    ε-greedy rewiring algorithm for network topology optimization.

    Parameters:
    - epsilon: Exploration probability (default 0.2)
    - max_iterations: Maximum rewiring iterations (default 20)
    - convergence_threshold: Improvement threshold for convergence (default 0.001)
    """

    # ε-ledger budget allocations
    EPSILON_INIT = 0.0003
    EPSILON_REWIRE_STEP = 0.0005
    EPSILON_FEATURE_CALC = 0.0002
    EPSILON_EVALUATION = 0.0003

    def __init__(
        self,
        epsilon: float = 0.2,
        max_iterations: int = 20,
        convergence_threshold: float = 0.001
    ):
        """Initialize rewiring algorithm with specified parameters."""
        if not 0 <= epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")
        if max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive, got {max_iterations}")

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Tracking
        self.iteration_count = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.improvement_history = []

        # PoDP tracking
        self.receipt_chain = ReceiptChain(
            dispute_id=f"rewiring_{int(time.time())}"
        )
        self.epsilon_spent = 0.0

        # Create initialization receipt
        self._create_init_receipt()

    def _create_init_receipt(self):
        """Create PoDP receipt for rewiring initialization."""
        receipt = Receipt(
            receipt_id=f"rewire_init_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="rewiring_initialization",
            inputs={
                "epsilon": self.epsilon,
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold
            },
            params={
                "algorithm": "epsilon_greedy",
                "feature_similarity": "cosine"
            },
            artifacts={
                "epsilon_used": self.EPSILON_INIT
            },
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        self.receipt_chain.add_receipt(receipt)
        self.epsilon_spent += self.EPSILON_INIT

    def extract_node_features(
        self,
        graph: nx.Graph,
        node: int
    ) -> np.ndarray:
        """
        Extract feature vector for a node.

        Features:
        - Queue length
        - Service rate
        - Degree centrality
        - Clustering coefficient
        - Average neighbor degree
        """
        attrs = graph.nodes[node]

        # Basic attributes
        queue_length = attrs.get('queue_length', 0)
        service_rate = attrs.get('service_rate', 1.5)

        # Graph metrics
        degree_centrality = nx.degree_centrality(graph)[node]
        clustering = nx.clustering(graph, node)
        avg_neighbor_degree = nx.average_neighbor_degree(graph, nodes=[node])[node]

        features = np.array([
            queue_length,
            service_rate,
            degree_centrality,
            clustering,
            avg_neighbor_degree
        ])

        self.epsilon_spent += self.EPSILON_FEATURE_CALC

        return features

    def compute_similarity_matrix(
        self,
        graph: nx.Graph
    ) -> Tuple[np.ndarray, Receipt]:
        """
        Compute pairwise similarity matrix for all nodes.

        Returns:
            Tuple of (similarity matrix, PoDP receipt)
        """
        start_time = time.time()

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"sim_matrix_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="similarity_computation_entry",
            inputs={
                "num_nodes": graph.number_of_nodes()
            },
            params={
                "similarity_metric": "cosine"
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Extract features for all nodes
        features = []
        for node in sorted(graph.nodes()):
            node_features = self.extract_node_features(graph, node)
            features.append(node_features)

        features_matrix = np.array(features)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(features_matrix)

        # Create exit receipt
        computation_time = (time.time() - start_time) * 1000
        exit_receipt = Receipt(
            receipt_id=f"sim_matrix_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="similarity_computation_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "computation_time_ms": computation_time
            },
            artifacts={
                "matrix_shape": similarity_matrix.shape,
                "mean_similarity": float(np.mean(similarity_matrix)),
                "std_similarity": float(np.std(similarity_matrix)),
                "epsilon_used": self.EPSILON_FEATURE_CALC * graph.number_of_nodes(),
                "epsilon_total": self.epsilon_spent
            },
            hashes={
                "matrix_hash": keccak(json.dumps(similarity_matrix.tolist()))
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        return similarity_matrix, exit_receipt

    def rewire_exploration(
        self,
        graph: nx.Graph
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Perform random exploration rewiring.

        Returns:
            Tuple of (edges_added, edges_removed)
        """
        edges_to_remove = set()
        edges_to_add = set()

        # Select random edge to remove
        if graph.edges():
            edge_to_remove = random.choice(list(graph.edges()))
            edges_to_remove.add(edge_to_remove)

            # Select random non-edge to add (avoiding self-loops)
            non_edges = list(nx.non_edges(graph))
            valid_non_edges = [
                (u, v) for u, v in non_edges
                if u != v and u != edge_to_remove[0] and v != edge_to_remove[1]
            ]

            if valid_non_edges:
                edge_to_add = random.choice(valid_non_edges)
                edges_to_add.add(edge_to_add)

        self.exploration_count += 1
        return edges_to_add, edges_to_remove

    def rewire_exploitation(
        self,
        graph: nx.Graph,
        similarity_matrix: np.ndarray
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Perform similarity-based exploitation rewiring.

        Returns:
            Tuple of (edges_added, edges_removed)
        """
        edges_to_remove = set()
        edges_to_add = set()

        # Find least similar connected nodes
        min_similarity = float('inf')
        edge_to_remove = None

        for u, v in graph.edges():
            similarity = similarity_matrix[u][v]
            if similarity < min_similarity:
                min_similarity = similarity
                edge_to_remove = (u, v)

        if edge_to_remove:
            edges_to_remove.add(edge_to_remove)

            # Find most similar unconnected nodes
            max_similarity = -float('inf')
            edge_to_add = None

            for u in range(len(similarity_matrix)):
                for v in range(u + 1, len(similarity_matrix)):
                    if not graph.has_edge(u, v) and u != v:
                        similarity = similarity_matrix[u][v]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            edge_to_add = (u, v)

            if edge_to_add:
                edges_to_add.add(edge_to_add)

        self.exploitation_count += 1
        return edges_to_add, edges_to_remove

    def optimize(
        self,
        graph: nx.Graph,
        objective_function: Optional[callable] = None
    ) -> Tuple[nx.Graph, RewiringMetrics, Receipt]:
        """
        Optimize network topology using ε-greedy rewiring.

        Args:
            graph: NetworkX graph to optimize
            objective_function: Optional custom objective (default: minimize avg path length)

        Returns:
            Tuple of (optimized graph, metrics, PoDP receipt)
        """
        start_time = time.time()

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"optimize_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="optimization_entry",
            inputs={
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges()
            },
            params={
                "epsilon": self.epsilon,
                "max_iterations": self.max_iterations
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Default objective: minimize average path length while maintaining connectivity
        if objective_function is None:
            def objective_function(g):
                if not nx.is_connected(g):
                    return float('inf')
                return nx.average_shortest_path_length(g)

        # Store initial metrics
        initial_objective = objective_function(graph)
        initial_clustering = nx.average_clustering(graph)
        initial_path_length = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf')

        # Create working copy
        working_graph = graph.copy()
        best_graph = working_graph.copy()
        best_objective = initial_objective

        edges_rewired_total = 0
        converged = False

        # Optimization loop
        for iteration in range(self.max_iterations):
            # Compute similarity matrix for exploitation
            similarity_matrix, _ = self.compute_similarity_matrix(working_graph)

            # ε-greedy decision
            if random.random() < self.epsilon:
                # Exploration: random rewiring
                edges_to_add, edges_to_remove = self.rewire_exploration(working_graph)
            else:
                # Exploitation: similarity-based rewiring
                edges_to_add, edges_to_remove = self.rewire_exploitation(
                    working_graph, similarity_matrix
                )

            # Apply rewiring
            for edge in edges_to_remove:
                if working_graph.has_edge(*edge):
                    working_graph.remove_edge(*edge)

            for edge in edges_to_add:
                if not working_graph.has_edge(*edge):
                    working_graph.add_edge(*edge)
                    edges_rewired_total += 1

            # Evaluate new configuration
            new_objective = objective_function(working_graph)

            # Track improvement
            improvement = (initial_objective - new_objective) / initial_objective if initial_objective != 0 else 0
            self.improvement_history.append(improvement)

            # Update best if improved
            if new_objective < best_objective:
                best_objective = new_objective
                best_graph = working_graph.copy()

            # Check convergence
            if len(self.improvement_history) >= 3:
                recent_improvements = self.improvement_history[-3:]
                if all(abs(imp) < self.convergence_threshold for imp in recent_improvements):
                    converged = True
                    break

            self.iteration_count += 1
            self.epsilon_spent += self.EPSILON_REWIRE_STEP

        # Calculate final metrics
        final_clustering = nx.average_clustering(best_graph)
        final_path_length = nx.average_shortest_path_length(best_graph) if nx.is_connected(best_graph) else float('inf')

        metrics = RewiringMetrics(
            edges_rewired=edges_rewired_total,
            improvement_ratio=(initial_objective - best_objective) / initial_objective if initial_objective != 0 else 0,
            exploration_count=self.exploration_count,
            exploitation_count=self.exploitation_count,
            avg_path_length_before=initial_path_length,
            avg_path_length_after=final_path_length,
            clustering_before=initial_clustering,
            clustering_after=final_clustering,
            convergence_achieved=converged,
            epsilon_used=self.epsilon_spent - self.epsilon_spent
        )

        # Create exit receipt
        optimization_time = (time.time() - start_time) * 1000
        exit_receipt = Receipt(
            receipt_id=f"optimize_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="optimization_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "iterations_completed": self.iteration_count,
                "optimization_time_ms": optimization_time
            },
            artifacts={
                "metrics": asdict(metrics),
                "initial_objective": initial_objective,
                "final_objective": best_objective,
                "convergence": converged,
                "epsilon_used": self.EPSILON_REWIRE_STEP * self.iteration_count,
                "epsilon_total": self.epsilon_spent
            },
            hashes={
                "final_graph_hash": keccak(json.dumps(nx.adjacency_matrix(best_graph).todense().tolist()))
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Update epsilon for evaluation
        self.epsilon_spent += self.EPSILON_EVALUATION

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        return best_graph, metrics, exit_receipt

    def adaptive_epsilon_schedule(
        self,
        iteration: int,
        decay_rate: float = 0.95
    ) -> float:
        """
        Calculate adaptive epsilon value with decay schedule.

        Args:
            iteration: Current iteration number
            decay_rate: Epsilon decay rate per iteration

        Returns:
            Updated epsilon value
        """
        return self.epsilon * (decay_rate ** iteration)

    def get_rewiring_history(self) -> Dict[str, Any]:
        """Get complete rewiring history and statistics."""
        return {
            "total_iterations": self.iteration_count,
            "exploration_count": self.exploration_count,
            "exploitation_count": self.exploitation_count,
            "exploration_ratio": self.exploration_count / (self.exploration_count + self.exploitation_count)
            if (self.exploration_count + self.exploitation_count) > 0 else 0,
            "improvement_history": self.improvement_history,
            "avg_improvement": np.mean(self.improvement_history) if self.improvement_history else 0,
            "epsilon_spent": self.epsilon_spent
        }

    def reset(self):
        """Reset algorithm state for new optimization."""
        self.iteration_count = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.improvement_history.clear()

        # Keep receipt chain for audit trail
        reset_receipt = Receipt(
            receipt_id=f"reset_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="algorithm_reset",
            inputs={},
            params={},
            artifacts={
                "state_cleared": True,
                "epsilon_retained": self.epsilon_spent
            },
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        self.receipt_chain.add_receipt(reset_receipt)

    def get_receipt_chain(self) -> ReceiptChain:
        """Get the complete receipt chain for all operations."""
        return self.receipt_chain

    def get_epsilon_budget_status(self) -> Dict[str, float]:
        """Get current epsilon budget status."""
        return {
            "spent": self.epsilon_spent,
            "remaining": 4.0 - self.epsilon_spent,  # Total budget of 4.0
            "breakdown": {
                "initialization": self.EPSILON_INIT,
                "rewiring": self.EPSILON_REWIRE_STEP * self.iteration_count,
                "features": self.EPSILON_FEATURE_CALC * len([r for r in self.receipt_chain.receipts if "feature" in r.step]),
                "evaluation": self.EPSILON_EVALUATION * len([r for r in self.receipt_chain.receipts if "eval" in r.step])
            }
        }