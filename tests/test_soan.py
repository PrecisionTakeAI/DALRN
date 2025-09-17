"""
Comprehensive test suite for Self-Organizing Agent Networks (SOAN).

Tests all SOAN components with PoDP compliance verification and
ε-ledger budget validation.
"""

import pytest
import numpy as np
import networkx as nx
import torch
import json
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import SOAN components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "services", "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "services"))

from topology import WattsStrogatzTopology, NetworkMetrics
from gnn_predictor import GNNLatencyPredictor, PredictionMetrics
from queue_model import MM1Queue, QueueMetrics
from rewiring import EpsilonGreedyRewiring, RewiringMetrics
from orchestrator import SOANOrchestrator, InitializeNetworkRequest, TrainGNNRequest
from common.podp import Receipt, ReceiptChain, keccak


class TestWattsStrogatzTopology:
    """Test suite for Watts-Strogatz topology generator."""

    def test_initialization(self):
        """Test topology generator initialization."""
        topology = WattsStrogatzTopology(N=50, k=4, p=0.1)
        assert topology.N == 50
        assert topology.k == 4
        assert topology.p == 0.1
        assert topology.epsilon_spent > 0
        assert topology.receipt_chain.dispute_id.startswith("soan_topology_")

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError, match="k .* must be less than N"):
            WattsStrogatzTopology(N=10, k=10, p=0.1)

        with pytest.raises(ValueError, match="k .* must be even"):
            WattsStrogatzTopology(N=10, k=3, p=0.1)

        with pytest.raises(ValueError, match="p .* must be in"):
            WattsStrogatzTopology(N=10, k=4, p=1.5)

    def test_network_generation(self):
        """Test network generation with PoDP receipts."""
        topology = WattsStrogatzTopology(N=20, k=4, p=0.2)
        graph, receipt = topology.generate()

        # Verify graph properties
        assert graph.number_of_nodes() == 20
        assert graph.number_of_edges() > 0
        assert nx.is_connected(graph)

        # Verify node attributes
        for node in graph.nodes():
            attrs = graph.nodes[node]
            assert 'queue_length' in attrs
            assert 'service_rate' in attrs
            assert 1.0 <= attrs['service_rate'] <= 2.0

        # Verify PoDP receipt
        assert receipt.step == "network_generation_exit"
        assert receipt.artifacts["node_count"] == 20
        assert receipt.artifacts["epsilon_used"] > 0
        assert "graph_hash" in receipt.artifacts

    def test_metrics_calculation(self):
        """Test network metrics calculation."""
        topology = WattsStrogatzTopology(N=30, k=6, p=0.1)
        graph, _ = topology.generate()
        metrics, receipt = topology.calculate_metrics()

        # Verify metrics
        assert isinstance(metrics, NetworkMetrics)
        assert 0 <= metrics.clustering_coefficient <= 1
        assert metrics.average_path_length > 0
        assert metrics.diameter > 0
        assert 0 <= metrics.density <= 1
        assert metrics.average_degree > 0
        assert metrics.connectivity == nx.is_connected(graph)

        # Small-world property check (high clustering, low path length)
        if metrics.small_world_coefficient is not None:
            assert metrics.small_world_coefficient > 0

        # Verify PoDP receipt
        assert receipt.step == "metrics_calculation_exit"
        assert "metrics" in receipt.artifacts
        assert receipt.artifacts["epsilon_used"] > 0

    def test_export_functionality(self):
        """Test network export to dictionary."""
        topology = WattsStrogatzTopology(N=15, k=4, p=0.15)
        graph, _ = topology.generate()
        metrics, _ = topology.calculate_metrics()
        data, receipt = topology.export_to_dict()

        # Verify exported data
        assert data["parameters"]["N"] == 15
        assert data["parameters"]["k"] == 4
        assert data["parameters"]["p"] == 0.15
        assert len(data["nodes"]) == 15
        assert len(data["edges"]) > 0
        assert data["metrics"] is not None
        assert "receipt_chain" in data

        # Verify PoDP receipt
        assert receipt.step == "network_export_exit"
        assert "export_hash" in receipt.artifacts

    def test_epsilon_budget_tracking(self):
        """Test ε-ledger budget tracking."""
        topology = WattsStrogatzTopology(N=25, k=4, p=0.1)
        initial_epsilon = topology.epsilon_spent

        graph, _ = topology.generate()
        after_gen = topology.epsilon_spent
        assert after_gen > initial_epsilon

        metrics, _ = topology.calculate_metrics()
        after_metrics = topology.epsilon_spent
        assert after_metrics > after_gen

        budget_status = topology.get_epsilon_budget_status()
        assert budget_status["spent"] == after_metrics
        assert budget_status["remaining"] == 4.0 - after_metrics
        assert "breakdown" in budget_status


class TestGNNLatencyPredictor:
    """Test suite for GNN latency predictor."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        topology = WattsStrogatzTopology(N=20, k=4, p=0.1)
        graph, _ = topology.generate()
        return graph

    def test_initialization(self):
        """Test GNN model initialization."""
        model = GNNLatencyPredictor(input_dim=2, hidden_dim=16, output_dim=1)

        assert model.input_dim == 2
        assert model.hidden_dim == 16
        assert model.output_dim == 1
        assert not model.is_trained
        assert model.epsilon_spent > 0

        # Check model architecture
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'dropout')

    def test_forward_pass(self, sample_graph):
        """Test GNN forward pass."""
        model = GNNLatencyPredictor()

        # Prepare input
        node_features = []
        for node in sorted(sample_graph.nodes()):
            attrs = sample_graph.nodes[node]
            features = [
                attrs.get('queue_length', 0),
                attrs.get('service_rate', 1.5)
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float32)

        # Create edge index
        edge_list = list(sample_graph.edges())
        edge_index = torch.tensor(
            [[e[0] for e in edge_list] + [e[1] for e in edge_list],
             [e[1] for e in edge_list] + [e[0] for e in edge_list]],
            dtype=torch.long
        )

        # Forward pass
        output = model.forward(x, edge_index)

        assert output.shape[0] == sample_graph.number_of_nodes()
        assert output.shape[1] == 1

    def test_prediction(self, sample_graph):
        """Test GNN prediction with PoDP receipt."""
        model = GNNLatencyPredictor()
        predictions, receipt = model.predict(sample_graph)

        assert len(predictions) == sample_graph.number_of_nodes()
        assert all(p >= 0 for p in predictions)  # Non-negative latencies

        # Verify PoDP receipt
        assert receipt.step == "prediction_exit"
        assert "predictions_shape" in receipt.artifacts
        assert "mean_prediction" in receipt.artifacts
        assert receipt.artifacts["epsilon_used"] > 0

    def test_training(self, sample_graph):
        """Test GNN model training."""
        model = GNNLatencyPredictor()

        # Generate synthetic target latencies
        target_latencies = np.random.uniform(0, 2, sample_graph.number_of_nodes())

        # Train model
        losses, receipt = model.train_model(
            graph=sample_graph,
            target_latencies=target_latencies,
            epochs=10,
            learning_rate=0.01
        )

        assert len(losses) == 10
        assert model.is_trained
        assert model.training_epochs == 10

        # Loss should generally decrease
        assert losses[-1] <= losses[0] * 2  # Allow some variance

        # Verify PoDP receipt
        assert receipt.step == "training_exit"
        assert "final_loss" in receipt.artifacts
        assert receipt.artifacts["epochs_completed"] == 10

    def test_evaluation(self, sample_graph):
        """Test model evaluation metrics."""
        model = GNNLatencyPredictor()

        # Train the model first
        target_latencies = np.random.uniform(0, 2, sample_graph.number_of_nodes())
        model.train_model(sample_graph, target_latencies, epochs=20)

        # Evaluate
        metrics, receipt = model.evaluate(sample_graph, target_latencies)

        assert isinstance(metrics, PredictionMetrics)
        assert metrics.mae >= 0
        assert metrics.mse >= 0
        assert metrics.rmse >= 0
        assert metrics.max_error >= 0
        assert metrics.inference_time_ms > 0

        # Verify PoDP receipt
        assert receipt.step == "model_evaluation"
        assert "mae" in receipt.artifacts


class TestMM1Queue:
    """Test suite for M/M/1 queue model."""

    def test_initialization(self):
        """Test queue initialization."""
        queue = MM1Queue(arrival_rate=0.5, service_rate=1.5)

        assert queue.arrival_rate == 0.5
        assert queue.service_rate == 1.5
        assert queue.rho == 0.5 / 1.5
        assert queue.is_stable
        assert queue.epsilon_spent > 0

    def test_stability_detection(self):
        """Test queue stability detection."""
        # Stable queue (λ < μ)
        stable_queue = MM1Queue(arrival_rate=0.8, service_rate=1.2)
        assert stable_queue.is_stable

        # Unstable queue (λ >= μ)
        unstable_queue = MM1Queue(arrival_rate=1.5, service_rate=1.0)
        assert not unstable_queue.is_stable

    def test_stability_check(self):
        """Test stability check with metrics."""
        queue = MM1Queue(arrival_rate=0.6, service_rate=1.2)
        is_stable, metrics, receipt = queue.check_stability()

        assert is_stable
        assert metrics["traffic_intensity"] == 0.5
        assert metrics["condition_met"]
        assert "theoretical_avg_queue_length" in metrics
        assert "theoretical_avg_wait_time" in metrics

        # Verify PoDP receipt
        assert receipt.step == "stability_check"
        assert receipt.artifacts["is_stable"]

    def test_theoretical_calculations(self):
        """Test theoretical queue metrics."""
        queue = MM1Queue(arrival_rate=0.5, service_rate=1.0)

        # Theoretical values for M/M/1
        L = queue._theoretical_queue_length()
        W = queue._theoretical_wait_time()
        Ws = queue._theoretical_system_time()

        assert L == 1.0  # ρ/(1-ρ) = 0.5/0.5 = 1
        assert W == 1.0  # ρ/(μ(1-ρ)) = 0.5/(1*0.5) = 1
        assert Ws == 2.0  # 1/(μ-λ) = 1/(1-0.5) = 2

    def test_simulation(self):
        """Test queue simulation."""
        queue = MM1Queue(arrival_rate=0.4, service_rate=1.2)
        metrics, receipt = queue.simulate(duration=50.0, warm_up=0.1)

        assert isinstance(metrics, QueueMetrics)
        assert metrics.average_queue_length >= 0
        assert metrics.average_wait_time >= 0
        assert metrics.average_system_time >= 0
        assert 0 <= metrics.utilization <= 1
        assert metrics.throughput > 0
        assert metrics.is_stable

        # Verify PoDP receipt
        assert receipt.step == "simulation_exit"
        assert "customers_served" in receipt.artifacts
        assert receipt.artifacts["epsilon_used"] > 0

    def test_service_rate_bounds(self):
        """Test service rate is bounded to [1.0, 2.0]."""
        # Service rate below minimum
        queue1 = MM1Queue(arrival_rate=0.5, service_rate=0.5)
        assert queue1.service_rate == 1.0

        # Service rate above maximum
        queue2 = MM1Queue(arrival_rate=0.5, service_rate=3.0)
        assert queue2.service_rate == 2.0

        # Service rate within bounds
        queue3 = MM1Queue(arrival_rate=0.5, service_rate=1.5)
        assert queue3.service_rate == 1.5

    def test_epsilon_budget_tracking(self):
        """Test ε-ledger budget tracking in queue."""
        queue = MM1Queue(arrival_rate=0.3, service_rate=1.0)
        initial_epsilon = queue.epsilon_spent

        queue.simulate(duration=20.0)
        after_sim = queue.epsilon_spent
        assert after_sim > initial_epsilon

        budget_status = queue.get_epsilon_budget_status()
        assert budget_status["spent"] == after_sim
        assert budget_status["remaining"] == 4.0 - after_sim


class TestEpsilonGreedyRewiring:
    """Test suite for ε-greedy rewiring algorithm."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        topology = WattsStrogatzTopology(N=30, k=6, p=0.1)
        graph, _ = topology.generate()
        return graph

    def test_initialization(self):
        """Test rewiring algorithm initialization."""
        rewiring = EpsilonGreedyRewiring(epsilon=0.3, max_iterations=15)

        assert rewiring.epsilon == 0.3
        assert rewiring.max_iterations == 15
        assert rewiring.convergence_threshold == 0.001
        assert rewiring.epsilon_spent > 0

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError, match="Epsilon must be in"):
            EpsilonGreedyRewiring(epsilon=1.5)

        with pytest.raises(ValueError, match="Max iterations must be positive"):
            EpsilonGreedyRewiring(max_iterations=0)

    def test_feature_extraction(self, sample_graph):
        """Test node feature extraction."""
        rewiring = EpsilonGreedyRewiring()

        node = 0
        features = rewiring.extract_node_features(sample_graph, node)

        assert len(features) == 5  # 5 features
        assert features[0] == sample_graph.nodes[node].get('queue_length', 0)
        assert features[1] == sample_graph.nodes[node].get('service_rate', 1.5)

    def test_similarity_matrix(self, sample_graph):
        """Test similarity matrix computation."""
        rewiring = EpsilonGreedyRewiring()
        similarity_matrix, receipt = rewiring.compute_similarity_matrix(sample_graph)

        n_nodes = sample_graph.number_of_nodes()
        assert similarity_matrix.shape == (n_nodes, n_nodes)
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # Symmetric
        assert np.all(similarity_matrix >= -1) and np.all(similarity_matrix <= 1)  # Cosine range

        # Verify PoDP receipt
        assert receipt.step == "similarity_computation_exit"
        assert "matrix_shape" in receipt.artifacts

    def test_exploration_rewiring(self, sample_graph):
        """Test exploration (random) rewiring."""
        rewiring = EpsilonGreedyRewiring()
        edges_to_add, edges_to_remove = rewiring.rewire_exploration(sample_graph)

        # Should suggest changes (if graph has edges)
        if sample_graph.number_of_edges() > 0:
            assert len(edges_to_remove) <= 1
            assert len(edges_to_add) <= 1

        assert rewiring.exploration_count == 1

    def test_exploitation_rewiring(self, sample_graph):
        """Test exploitation (similarity-based) rewiring."""
        rewiring = EpsilonGreedyRewiring()
        similarity_matrix, _ = rewiring.compute_similarity_matrix(sample_graph)

        edges_to_add, edges_to_remove = rewiring.rewire_exploitation(
            sample_graph, similarity_matrix
        )

        # Should suggest changes based on similarity
        if sample_graph.number_of_edges() > 0:
            assert len(edges_to_remove) <= 1
            assert len(edges_to_add) <= 1

        assert rewiring.exploitation_count == 1

    def test_optimization(self, sample_graph):
        """Test network optimization."""
        rewiring = EpsilonGreedyRewiring(epsilon=0.2, max_iterations=5)

        optimized_graph, metrics, receipt = rewiring.optimize(sample_graph)

        # Verify optimization results
        assert isinstance(metrics, RewiringMetrics)
        assert metrics.edges_rewired >= 0
        assert metrics.exploration_count + metrics.exploitation_count <= 5

        # Graph should remain connected
        assert nx.is_connected(optimized_graph)
        assert optimized_graph.number_of_nodes() == sample_graph.number_of_nodes()

        # Verify PoDP receipt
        assert receipt.step == "optimization_exit"
        assert "final_objective" in receipt.artifacts

    def test_adaptive_epsilon_schedule(self):
        """Test adaptive epsilon decay."""
        rewiring = EpsilonGreedyRewiring(epsilon=0.5)

        eps_0 = rewiring.adaptive_epsilon_schedule(0, decay_rate=0.95)
        eps_5 = rewiring.adaptive_epsilon_schedule(5, decay_rate=0.95)
        eps_10 = rewiring.adaptive_epsilon_schedule(10, decay_rate=0.95)

        assert eps_0 == 0.5
        assert eps_5 < eps_0
        assert eps_10 < eps_5
        assert eps_10 == 0.5 * (0.95 ** 10)


class TestSOANOrchestrator:
    """Test suite for SOAN orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return SOANOrchestrator()

    @pytest.mark.asyncio
    async def test_network_initialization(self, orchestrator):
        """Test network initialization through orchestrator."""
        request = InitializeNetworkRequest(
            n_nodes=20,
            k_edges=4,
            p_rewire=0.1
        )

        response = await orchestrator.initialize_network(request)

        assert response.network_id.startswith("soan_net_")
        assert response.status == "success"
        assert "clustering_coefficient" in response.metrics
        assert response.epsilon_used > 0
        assert response.epsilon_remaining < 4.0

    @pytest.mark.asyncio
    async def test_gnn_training(self, orchestrator):
        """Test GNN training through orchestrator."""
        # First initialize a network
        init_request = InitializeNetworkRequest(n_nodes=15, k_edges=4, p_rewire=0.1)
        init_response = await orchestrator.initialize_network(init_request)

        # Train GNN
        train_request = TrainGNNRequest(
            network_id=init_response.network_id,
            epochs=5,
            learning_rate=0.01
        )

        response = await orchestrator.train_gnn_model(train_request)

        assert response.status == "success"
        assert "training_loss" in response.metrics
        assert "evaluation" in response.metrics
        assert response.epsilon_used > init_response.epsilon_used

    @pytest.mark.asyncio
    async def test_network_optimization(self, orchestrator):
        """Test network optimization through orchestrator."""
        # Initialize network
        init_request = InitializeNetworkRequest(n_nodes=20, k_edges=6, p_rewire=0.15)
        init_response = await orchestrator.initialize_network(init_request)

        # Optimize network
        from orchestrator import OptimizeNetworkRequest
        opt_request = OptimizeNetworkRequest(
            network_id=init_response.network_id,
            iterations=3,
            epsilon=0.25
        )

        response = await orchestrator.optimize_network(opt_request)

        assert response.status == "success"
        assert "rewiring" in response.metrics
        assert "network" in response.metrics
        assert response.epsilon_used > init_response.epsilon_used

    @pytest.mark.asyncio
    async def test_queue_simulation(self, orchestrator):
        """Test queue simulation through orchestrator."""
        # Initialize network
        init_request = InitializeNetworkRequest(n_nodes=10, k_edges=4, p_rewire=0.1)
        init_response = await orchestrator.initialize_network(init_request)

        # Simulate queues
        from orchestrator import SimulateQueuesRequest
        sim_request = SimulateQueuesRequest(
            network_id=init_response.network_id,
            duration=10.0,
            arrival_rate=0.4
        )

        response = await orchestrator.simulate_queues(sim_request)

        assert response.status == "success"
        assert "stable_queues" in response.metrics
        assert "avg_queue_length" in response.metrics
        assert response.epsilon_used > init_response.epsilon_used

    @pytest.mark.asyncio
    async def test_epsilon_budget_enforcement(self, orchestrator):
        """Test ε-ledger budget enforcement."""
        # Set a very high epsilon requirement to trigger budget check
        orchestrator.TOTAL_EPSILON_BUDGET = 0.01  # Very small budget

        request = InitializeNetworkRequest(n_nodes=100, k_edges=10, p_rewire=0.2)

        with pytest.raises(Exception):  # Should fail due to budget
            await orchestrator.initialize_network(request)

    @pytest.mark.asyncio
    async def test_network_status(self, orchestrator):
        """Test network status retrieval."""
        # Initialize network
        init_request = InitializeNetworkRequest(n_nodes=25, k_edges=4, p_rewire=0.1)
        init_response = await orchestrator.initialize_network(init_request)

        # Get status
        status = await orchestrator.get_network_status(init_response.network_id)

        assert status["network_id"] == init_response.network_id
        assert status["num_nodes"] == 25
        assert status["num_edges"] > 0
        assert "epsilon_spent" in status
        assert "receipt_count" in status

    @pytest.mark.asyncio
    async def test_receipt_chain_retrieval(self, orchestrator):
        """Test PoDP receipt chain retrieval."""
        # Initialize network
        init_request = InitializeNetworkRequest(n_nodes=15, k_edges=4, p_rewire=0.1)
        init_response = await orchestrator.initialize_network(init_request)

        # Get receipt chain
        chain_data = await orchestrator.get_receipt_chain(init_response.network_id)

        assert "dispute_id" in chain_data
        assert "merkle_root" in chain_data
        assert chain_data["receipt_count"] > 0
        assert len(chain_data["receipts"]) > 0


class TestPoDPCompliance:
    """Test suite for PoDP compliance verification."""

    def test_receipt_generation_in_all_operations(self):
        """Verify all operations generate PoDP receipts."""
        # Topology
        topology = WattsStrogatzTopology(N=10, k=4, p=0.1)
        graph, receipt1 = topology.generate()
        assert receipt1 is not None
        assert receipt1.step.endswith("_exit")

        # GNN
        gnn = GNNLatencyPredictor()
        predictions, receipt2 = gnn.predict(graph)
        assert receipt2 is not None
        assert receipt2.step.endswith("_exit")

        # Queue
        queue = MM1Queue(arrival_rate=0.3, service_rate=1.0)
        metrics, receipt3 = queue.simulate(duration=10.0)
        assert receipt3 is not None
        assert receipt3.step.endswith("_exit")

        # Rewiring
        rewiring = EpsilonGreedyRewiring()
        optimized, metrics, receipt4 = rewiring.optimize(graph)
        assert receipt4 is not None
        assert receipt4.step.endswith("_exit")

    def test_receipt_chain_integrity(self):
        """Test receipt chain maintains integrity."""
        topology = WattsStrogatzTopology(N=15, k=4, p=0.1)
        topology.generate()
        topology.calculate_metrics()

        chain = topology.get_receipt_chain()
        assert len(chain.receipts) >= 4  # init, gen_entry, gen_exit, metrics_entry, metrics_exit

        # Verify merkle root can be calculated
        merkle_root = chain.get_merkle_root()
        assert merkle_root is not None
        assert len(merkle_root) == 64  # SHA256 hex string

    def test_epsilon_never_exceeds_budget(self):
        """Verify ε-ledger budget is never exceeded."""
        components = [
            WattsStrogatzTopology(N=10, k=4, p=0.1),
            GNNLatencyPredictor(),
            MM1Queue(arrival_rate=0.5, service_rate=1.5),
            EpsilonGreedyRewiring()
        ]

        for component in components:
            budget_status = component.get_epsilon_budget_status()
            assert budget_status["spent"] >= 0
            assert budget_status["remaining"] >= 0
            assert budget_status["spent"] + budget_status["remaining"] == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])