"""
ε-greedy Network Rewiring Algorithm

Implements intelligent network topology adaptation based on
latency predictions and feature similarity with PoDP compliance.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import logging
import time
import hashlib
import json
from dataclasses import dataclass
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class RewiringPoDPReceipt:
    """PoDP receipt for rewiring operations"""
    operation: str
    timestamp: float
    rewiring_id: str
    epsilon_used: float
    input_hash: str
    output_hash: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'timestamp': self.timestamp,
            'rewiring_id': self.rewiring_id,
            'epsilon_used': self.epsilon_used,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'metadata': self.metadata
        }


class EpsilonGreedyRewiring:
    """
    ε-greedy rewiring algorithm for network topology optimization
    
    Balances exploration (random rewiring) with exploitation
    (rewiring based on feature similarity and latency optimization)
    
    ε-ledger budget allocation:
    - Rewiring decision: 0.0001ε per iteration
    - Edge modification: 0.00005ε per edge
    - Feature similarity computation: 0.00001ε per pair
    - Connectivity check: 0.00005ε
    """
    
    def __init__(
        self,
        epsilon: float = 0.2,
        similarity_threshold: float = 0.8,
        max_rewires_per_iteration: int = 5,
        epsilon_budget: float = 0.005
    ):
        """
        Initialize rewiring algorithm
        
        Args:
            epsilon: Exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
            similarity_threshold: Minimum feature similarity for exploitation
            max_rewires_per_iteration: Maximum edge rewires per iteration
            epsilon_budget: ε-ledger budget for rewiring operations
        """
        self.epsilon = epsilon
        self.similarity_threshold = similarity_threshold
        self.max_rewires_per_iteration = max_rewires_per_iteration
        self.epsilon_budget = epsilon_budget
        self.epsilon_used = 0.0
        
        # Tracking
        self.rewiring_history: List[Dict] = []
        self.receipts: List[RewiringPoDPReceipt] = []
        self.total_rewires = 0
        
        logger.info(f"Initialized ε-greedy rewiring: ε={epsilon}, similarity={similarity_threshold}")
    
    def _generate_receipt(
        self,
        operation: str,
        input_data: any,
        output_data: any,
        epsilon_cost: float
    ) -> RewiringPoDPReceipt:
        """Generate PoDP receipt for rewiring operation"""
        input_hash = hashlib.sha256(
            json.dumps(str(input_data)).encode()
        ).hexdigest()
        output_hash = hashlib.sha256(
            json.dumps(str(output_data)).encode()
        ).hexdigest()
        
        receipt = RewiringPoDPReceipt(
            operation=operation,
            timestamp=time.time(),
            rewiring_id=f"rewiring_{id(self)}",
            epsilon_used=epsilon_cost,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                'epsilon': self.epsilon,
                'similarity_threshold': self.similarity_threshold
            }
        )
        
        self.receipts.append(receipt)
        self.epsilon_used += epsilon_cost
        
        if self.epsilon_used > self.epsilon_budget:
            raise ValueError(f"ε-ledger budget exceeded: {self.epsilon_used:.4f} > {self.epsilon_budget:.4f}")
        
        return receipt
    
    def compute_feature_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        Compute similarity between node features
        
        Uses cosine similarity: 1 - cosine_distance
        """
        similarity = 1.0 - cosine(features1, features2)
        
        self._generate_receipt(
            operation="compute_similarity",
            input_data={'feat1_shape': features1.shape, 'feat2_shape': features2.shape},
            output_data={'similarity': similarity},
            epsilon_cost=0.00001
        )
        
        return similarity
    
    def find_similar_nodes(
        self,
        graph: nx.Graph,
        node_id: int,
        node_features: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find nodes with similar features
        
        Args:
            graph: Network graph
            node_id: Reference node
            node_features: Feature matrix
            top_k: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity) tuples
        """
        similarities = []
        reference_features = node_features[node_id]
        
        for other_node in graph.nodes():
            if other_node != node_id and not graph.has_edge(node_id, other_node):
                similarity = self.compute_feature_similarity(
                    reference_features,
                    node_features[other_node]
                )
                similarities.append((other_node, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_bottleneck_edges(
        self,
        graph: nx.Graph,
        latency_matrix: np.ndarray,
        percentile: float = 90
    ) -> List[Tuple[int, int, float]]:
        """
        Identify high-latency edges that are bottlenecks
        
        Args:
            graph: Network graph
            latency_matrix: Predicted latencies
            percentile: Percentile threshold for bottleneck identification
            
        Returns:
            List of (source, target, latency) tuples
        """
        bottlenecks = []
        latency_threshold = np.percentile(
            latency_matrix[latency_matrix > 0],
            percentile
        )
        
        for edge in graph.edges():
            i, j = edge
            latency = latency_matrix[i, j]
            if latency > latency_threshold:
                bottlenecks.append((i, j, latency))
        
        # Sort by latency (worst first)
        bottlenecks.sort(key=lambda x: x[2], reverse=True)
        return bottlenecks
    
    def rewire_iteration(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        latency_matrix: np.ndarray,
        iteration: int = 0
    ) -> Dict[str, any]:
        """
        Perform one iteration of ε-greedy rewiring
        
        Args:
            graph: Network graph (modified in place)
            node_features: Node feature matrix
            latency_matrix: Current latency predictions
            iteration: Iteration number
            
        Returns:
            Dictionary with rewiring statistics
        """
        start_time = time.time()
        rewires_made = []
        
        # Decide exploration vs exploitation
        if np.random.random() < self.epsilon:
            # EXPLORATION: Random rewiring
            mode = "exploration"
            rewires_made = self._random_rewiring(graph, node_features)
        else:
            # EXPLOITATION: Feature-based rewiring
            mode = "exploitation"
            rewires_made = self._similarity_based_rewiring(
                graph, node_features, latency_matrix
            )
        
        # Ensure connectivity
        if not nx.is_connected(graph):
            self._restore_connectivity(graph)
        
        stats = {
            'iteration': iteration,
            'mode': mode,
            'rewires_made': len(rewires_made),
            'edges_removed': [r['removed'] for r in rewires_made],
            'edges_added': [r['added'] for r in rewires_made],
            'time': time.time() - start_time
        }
        
        self.rewiring_history.append(stats)
        self.total_rewires += len(rewires_made)
        
        self._generate_receipt(
            operation=f"rewire_iteration_{iteration}",
            input_data={'mode': mode, 'iteration': iteration},
            output_data={'rewires': len(rewires_made)},
            epsilon_cost=0.0001
        )
        
        logger.info(f"Rewiring iteration {iteration} ({mode}): {len(rewires_made)} rewires")
        
        return stats
    
    def _random_rewiring(
        self,
        graph: nx.Graph,
        node_features: np.ndarray
    ) -> List[Dict]:
        """
        Random edge rewiring for exploration
        """
        rewires = []
        n_rewires = min(
            self.max_rewires_per_iteration,
            graph.number_of_edges() // 10
        )
        
        edges = list(graph.edges())
        np.random.shuffle(edges)
        
        for _ in range(n_rewires):
            if not edges:
                break
                
            # Remove random edge
            edge_to_remove = edges.pop()
            u, v = edge_to_remove
            
            # Find random non-connected pair
            all_nodes = list(graph.nodes())
            np.random.shuffle(all_nodes)
            
            new_edge = None
            for node1 in all_nodes[:10]:  # Try first 10 random nodes
                for node2 in all_nodes[:10]:
                    if node1 != node2 and not graph.has_edge(node1, node2):
                        new_edge = (node1, node2)
                        break
                if new_edge:
                    break
            
            if new_edge:
                # Perform rewiring
                graph.remove_edge(u, v)
                graph.add_edge(new_edge[0], new_edge[1])
                
                rewires.append({
                    'removed': (u, v),
                    'added': new_edge,
                    'reason': 'exploration'
                })
                
                self._generate_receipt(
                    operation="random_rewire",
                    input_data={'removed': (u, v)},
                    output_data={'added': new_edge},
                    epsilon_cost=0.00005
                )
        
        return rewires
    
    def _similarity_based_rewiring(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        latency_matrix: np.ndarray
    ) -> List[Dict]:
        """
        Feature similarity-based rewiring for exploitation
        """
        rewires = []
        
        # Find bottleneck edges
        bottlenecks = self.find_bottleneck_edges(graph, latency_matrix, percentile=80)
        
        for bottleneck in bottlenecks[:self.max_rewires_per_iteration]:
            u, v, latency = bottleneck
            
            # Find similar nodes to connect instead
            similar_to_u = self.find_similar_nodes(graph, u, node_features, top_k=5)
            similar_to_v = self.find_similar_nodes(graph, v, node_features, top_k=5)
            
            # Try to create better connections
            best_alternative = None
            best_similarity = 0
            
            for node, sim in similar_to_u:
                if sim > self.similarity_threshold and not graph.has_edge(u, node):
                    if sim > best_similarity:
                        best_alternative = (u, node)
                        best_similarity = sim
            
            for node, sim in similar_to_v:
                if sim > self.similarity_threshold and not graph.has_edge(v, node):
                    if sim > best_similarity:
                        best_alternative = (v, node)
                        best_similarity = sim
            
            if best_alternative:
                # Perform rewiring
                graph.remove_edge(u, v)
                graph.add_edge(best_alternative[0], best_alternative[1])
                
                rewires.append({
                    'removed': (u, v),
                    'added': best_alternative,
                    'reason': 'similarity',
                    'similarity': best_similarity,
                    'old_latency': latency
                })
                
                self._generate_receipt(
                    operation="similarity_rewire",
                    input_data={'removed': (u, v), 'latency': latency},
                    output_data={'added': best_alternative, 'similarity': best_similarity},
                    epsilon_cost=0.00005
                )
        
        return rewires
    
    def _restore_connectivity(self, graph: nx.Graph):
        """
        Restore graph connectivity if broken
        """
        components = list(nx.connected_components(graph))
        
        if len(components) > 1:
            # Connect components
            for i in range(1, len(components)):
                # Connect to largest component
                node_from_main = list(components[0])[0]
                node_from_comp = list(components[i])[0]
                
                graph.add_edge(node_from_main, node_from_comp)
                
                logger.warning(f"Restored connectivity: connected components {i}")
                
                self._generate_receipt(
                    operation="restore_connectivity",
                    input_data={'components': len(components)},
                    output_data={'edge_added': (node_from_main, node_from_comp)},
                    epsilon_cost=0.00005
                )
    
    def adaptive_epsilon_decay(self, iteration: int, max_iterations: int):
        """
        Decay exploration rate over time
        
        Args:
            iteration: Current iteration
            max_iterations: Total planned iterations
        """
        # Linear decay
        decay_rate = 0.95
        min_epsilon = 0.05
        
        self.epsilon = max(
            min_epsilon,
            self.epsilon * (decay_rate ** (iteration / max_iterations))
        )
        
        logger.info(f"Epsilon decayed to {self.epsilon:.3f}")
    
    def run_rewiring_campaign(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        latency_predictor,
        iterations: int = 20
    ) -> Dict[str, any]:
        """
        Run multiple rewiring iterations
        
        Args:
            graph: Network graph
            node_features: Node features
            latency_predictor: Callable to predict latencies
            iterations: Number of rewiring iterations
            
        Returns:
            Campaign statistics
        """
        start_time = time.time()
        initial_metrics = self._compute_graph_metrics(graph)
        
        for i in range(iterations):
            # Predict current latencies
            adjacency = nx.adjacency_matrix(graph).todense()
            latency_matrix = latency_predictor(adjacency, node_features)
            
            # Perform rewiring
            self.rewire_iteration(graph, node_features, latency_matrix, i)
            
            # Adaptive epsilon decay
            if i > iterations // 2:
                self.adaptive_epsilon_decay(i, iterations)
        
        final_metrics = self._compute_graph_metrics(graph)
        
        campaign_stats = {
            'iterations': iterations,
            'total_rewires': self.total_rewires,
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'improvement': {
                key: final_metrics[key] - initial_metrics[key]
                for key in initial_metrics
            },
            'time': time.time() - start_time
        }
        
        self._generate_receipt(
            operation="rewiring_campaign",
            input_data={'iterations': iterations},
            output_data={'total_rewires': self.total_rewires},
            epsilon_cost=0.0001
        )
        
        logger.info(f"Rewiring campaign completed: {self.total_rewires} total rewires")
        
        return campaign_stats
    
    def _compute_graph_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute graph topology metrics"""
        metrics = {
            'avg_path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf'),
            'clustering': nx.average_clustering(graph),
            'density': nx.density(graph),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else float('inf'),
            'edges': graph.number_of_edges()
        }
        
        try:
            metrics['algebraic_connectivity'] = nx.algebraic_connectivity(graph)
        except:
            metrics['algebraic_connectivity'] = 0.0
        
        return metrics
    
    def get_rewiring_summary(self) -> Dict:
        """Get summary of rewiring operations"""
        if not self.rewiring_history:
            return {'message': 'No rewiring performed yet'}
        
        exploration_count = sum(1 for r in self.rewiring_history if r['mode'] == 'exploration')
        exploitation_count = len(self.rewiring_history) - exploration_count
        
        return {
            'total_iterations': len(self.rewiring_history),
            'total_rewires': self.total_rewires,
            'exploration_iterations': exploration_count,
            'exploitation_iterations': exploitation_count,
            'avg_rewires_per_iteration': self.total_rewires / len(self.rewiring_history),
            'current_epsilon': self.epsilon
        }
    
    def get_podp_summary(self) -> Dict:
        """Get PoDP compliance summary"""
        return {
            'total_receipts': len(self.receipts),
            'epsilon_used': self.epsilon_used,
            'epsilon_budget': self.epsilon_budget,
            'epsilon_remaining': self.epsilon_budget - self.epsilon_used,
            'operations': [r.operation for r in self.receipts[-10:]]  # Last 10 operations
        }