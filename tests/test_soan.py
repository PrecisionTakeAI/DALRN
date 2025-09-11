"""
Comprehensive test suite for Self-Organizing Agent Networks (SOAN)

Tests all components with PoDP compliance validation and
ensures >80% code coverage across the module.
"""

import pytest
import numpy as np
import networkx as nx
import torch
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

# Import SOAN components
from services.agents import (
    WattsStrogatzNetwork,
    GNNLatencyPredictor,
    MM1QueueModel,
    EpsilonGreedyRewiring,
    SOANOrchestrator
)
from services.agents.orchestrator import SOANConfig


class TestWattsStrogatzNetwork:
    """Test suite for Watts-Strogatz network topology"""
    
    def test_network_initialization(self):
        """Test network initialization with default parameters"""
        network = WattsStrogatzNetwork(n_nodes=50, k=4, p=0.1)
        
        assert network.n_nodes == 50
        assert network.k == 4
        assert network.p == 0.1
        assert network.epsilon_budget == 0.005
        assert network.epsilon_used == 0.0
        assert len(network.receipts) == 0
    
    def test_network_generation(self):
        """Test Watts-Strogatz network generation"""
        network = WattsStrogatzNetwork(n_nodes=30, k=4, p=0.2)
        graph = network.generate(seed=42)
        
        # Check graph properties
        assert graph.number_of_nodes() == 30
        assert graph.number_of_edges() > 0
        assert nx.is_connected(graph)
        
        # Check node features
        assert network.node_features.shape == (30, 4)
        assert np.all(network.node_features[:, 0] >= 0.5)  # Skills
        assert np.all(network.node_features[:, 1] == 0.0)  # Queue length
        
        # Check PoDP receipt generation
        assert len(network.receipts) > 0
        assert network.receipts[0].operation == "generate_watts_strogatz"
        assert network.receipts[0].verify()
    
    def test_metrics_computation(self):
        """Test network metrics computation"""
        network = WattsStrogatzNetwork(n_nodes=20, k=4, p=0.1)
        graph = network.generate(seed=123)
        
        metrics = network.compute_metrics()
        
        # Check required metrics
        assert 'avg_shortest_path' in metrics
        assert 'clustering_coefficient' in metrics
        assert 'algebraic_connectivity' in metrics
        assert 'degree_mean' in metrics
        assert 'degree_std' in metrics
        assert 'density' in metrics
        
        # Validate metric ranges
        assert metrics['avg_shortest_path'] > 0
        assert 0 <= metrics['clustering_coefficient'] <= 1
        assert metrics['density'] > 0
        
        # Check PoDP compliance
        assert any(r.operation == "compute_metrics" for r in network.receipts)
    
    def test_edge_operations(self):
        """Test edge addition and removal with PoDP tracking"""
        network = WattsStrogatzNetwork(n_nodes=10, k=2, p=0.1)
        graph = network.generate()
        
        initial_edges = graph.number_of_edges()
        
        # Test edge addition
        receipt = network.add_edge(0, 5)
        assert graph.has_edge(0, 5)
        assert receipt.operation == "add_edge"
        assert receipt.verify()
        
        # Test edge removal
        receipt = network.remove_edge(0, 5)
        assert not graph.has_edge(0, 5)
        assert receipt.operation == "remove_edge"
        
        # Check epsilon budget tracking
        assert network.epsilon_used > 0
        assert network.epsilon_used < network.epsilon_budget
    
    def test_epsilon_budget_enforcement(self):
        """Test that epsilon budget is enforced"""
        network = WattsStrogatzNetwork(n_nodes=10, epsilon_budget=0.0001)
        
        # Should succeed
        graph = network.generate()
        
        # Should fail due to budget exhaustion
        with pytest.raises(ValueError, match="ε-ledger budget exceeded"):
            for _ in range(100):
                network.add_edge(0, 1)
                network.remove_edge(0, 1)


class TestGNNLatencyPredictor:
    """Test suite for GNN-based latency prediction"""
    
    def test_gnn_initialization(self):
        """Test GNN predictor initialization"""
        predictor = GNNLatencyPredictor(
            input_dim=4,
            hidden_dim=16,
            learning_rate=0.01
        )
        
        assert predictor.input_dim == 4
        assert predictor.hidden_dim == 16
        assert predictor.learning_rate == 0.01
        assert len(predictor.receipts) == 1
        assert predictor.receipts[0].operation == "initialize_gnn"
    
    def test_data_preparation(self):
        """Test data preparation for PyTorch Geometric"""
        predictor = GNNLatencyPredictor()
        
        # Create simple test graph
        adjacency = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        features = np.random.rand(4, 4)
        latencies = np.random.rand(4, 4)
        
        data = predictor.prepare_data(adjacency, features, latencies)
        
        assert data.x.shape == (4, 4)
        assert data.edge_index.shape[0] == 2
        assert data.y is not None
        
        # Check PoDP receipt
        assert any(r.operation == "prepare_data" for r in predictor.receipts)
    
    def test_model_training(self):
        """Test GNN model training"""
        predictor = GNNLatencyPredictor()
        
        # Prepare training data
        adjacency = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        features = np.random.rand(4, 4)
        latencies = np.random.rand(4, 4)
        
        train_data = predictor.prepare_data(adjacency, features, latencies)
        
        # Train for few epochs
        stats = predictor.train(train_data, epochs=5)
        
        assert 'training_losses' in stats
        assert len(stats['training_losses']) == 5
        assert 'training_time' in stats
        
        # Check PoDP receipts for training
        training_receipts = [r for r in predictor.receipts if 'train_epoch' in r.operation]
        assert len(training_receipts) == 5
    
    def test_latency_prediction(self):
        """Test latency prediction after training"""
        predictor = GNNLatencyPredictor()
        
        # Setup test network
        adjacency = np.eye(5)
        adjacency[0, 1] = adjacency[1, 0] = 1
        adjacency[1, 2] = adjacency[2, 1] = 1
        features = np.random.rand(5, 4)
        
        # Predict without training (should still work)
        latency_matrix = predictor.predict_edge_latency(adjacency, features)
        
        assert latency_matrix.shape == (5, 5)
        assert np.all(latency_matrix >= 0.1)  # Minimum latency constraint
        assert np.all(latency_matrix[adjacency == 0] == 0)  # No latency for non-edges
    
    def test_model_save_load(self):
        """Test model checkpoint save/load"""
        predictor = GNNLatencyPredictor()
        
        # Train briefly
        adjacency = np.eye(3)
        features = np.random.rand(3, 4)
        data = predictor.prepare_data(adjacency, features)
        predictor.train(data, epochs=2)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            predictor.save_model(tmp.name)
            
            # Load into new predictor
            new_predictor = GNNLatencyPredictor()
            new_predictor.load_model(tmp.name)
            
            # Check state preservation
            assert len(new_predictor.training_losses) == 2


class TestMM1QueueModel:
    """Test suite for M/M/1 queue model"""
    
    def test_queue_initialization(self):
        """Test queue model initialization"""
        queue = MM1QueueModel(n_nodes=50)
        
        assert queue.n_nodes == 50
        assert len(queue.service_rates) == 50
        assert np.all(queue.service_rates >= 1.0)
        assert np.all(queue.service_rates <= 2.0)
        assert np.all(queue.arrival_rates == 0)
        assert len(queue.receipts) == 1
    
    def test_single_node_latency(self):
        """Test latency calculation for single node"""
        queue = MM1QueueModel(n_nodes=10)
        
        # Test stable queue
        node_id = 0
        queue.service_rates[node_id] = 2.0
        latency = queue.calculate_latency(node_id, arrival_rate=1.0)
        
        assert latency == 1.0  # 1/(μ-λ) = 1/(2-1) = 1
        assert queue.utilizations[node_id] == 0.5
        
        # Test unstable queue
        latency_unstable = queue.calculate_latency(node_id, arrival_rate=2.5)
        assert latency_unstable == float('inf')
    
    def test_network_latency_calculation(self):
        """Test network-wide latency calculation"""
        queue = MM1QueueModel(n_nodes=3)
        
        # Set known service rates
        queue.service_rates = np.array([2.0, 1.5, 1.8])
        
        # Create traffic matrix
        traffic = np.array([
            [0, 0.5, 0.3],
            [0.2, 0, 0.4],
            [0.1, 0.3, 0]
        ])
        
        latency_matrix = queue.calculate_network_latency(traffic)
        
        assert latency_matrix.shape == (3, 3)
        assert np.all(latency_matrix[traffic == 0] == 0)
        assert np.all(latency_matrix[traffic > 0] > 0)
    
    def test_slo_violation_estimation(self):
        """Test SLO violation estimation"""
        queue = MM1QueueModel(n_nodes=5)
        
        # Create test latency matrix
        latencies = np.array([
            [0, 2, 3, 6, 1],
            [2, 0, 4, 7, 2],
            [3, 4, 0, 8, 3],
            [6, 7, 8, 0, 4],
            [1, 2, 3, 4, 0]
        ])
        
        stats = queue.estimate_slo_violations(latencies, slo_threshold=5.0)
        
        assert 'violation_rate' in stats
        assert 'violating_pairs' in stats
        assert 'percentiles' in stats
        assert stats['violation_rate'] == 4/25  # 4 violations out of 25 entries
    
    def test_service_rate_optimization(self):
        """Test service rate optimization"""
        queue = MM1QueueModel(n_nodes=5)
        
        # Set arrival rates
        queue.arrival_rates = np.array([0.7, 1.2, 0.9, 1.4, 0.8])
        
        # Optimize for target utilization
        optimized_rates = queue.optimize_service_rates(target_utilization=0.7)
        
        assert len(optimized_rates) == 5
        assert np.all(optimized_rates >= 1.0)
        assert np.all(optimized_rates <= 2.0)
        
        # Check target utilization is approximately achieved
        utilizations = queue.arrival_rates / optimized_rates
        assert np.allclose(utilizations[optimized_rates < 2.0], 0.7, atol=0.1)


class TestEpsilonGreedyRewiring:
    """Test suite for ε-greedy rewiring algorithm"""
    
    def test_rewiring_initialization(self):
        """Test rewiring algorithm initialization"""
        rewiring = EpsilonGreedyRewiring(
            epsilon=0.3,
            similarity_threshold=0.7
        )
        
        assert rewiring.epsilon == 0.3
        assert rewiring.similarity_threshold == 0.7
        assert rewiring.total_rewires == 0
        assert len(rewiring.receipts) == 0
    
    def test_feature_similarity_computation(self):
        """Test feature similarity computation"""
        rewiring = EpsilonGreedyRewiring()
        
        features1 = np.array([1, 0, 0, 1])
        features2 = np.array([1, 0, 0, 1])
        features3 = np.array([0, 1, 1, 0])
        
        # Same features should have high similarity
        sim_same = rewiring.compute_feature_similarity(features1, features2)
        assert sim_same == 1.0
        
        # Different features should have lower similarity
        sim_diff = rewiring.compute_feature_similarity(features1, features3)
        assert sim_diff < 1.0
        
        # Check PoDP receipt
        assert len(rewiring.receipts) == 2
    
    def test_similar_node_finding(self):
        """Test finding similar nodes"""
        rewiring = EpsilonGreedyRewiring()
        
        # Create test graph
        graph = nx.complete_graph(5)
        graph.remove_edge(0, 4)  # Remove one edge
        
        # Create features with known similarity
        features = np.array([
            [1, 0, 0, 1],  # Node 0
            [1, 0, 0, 0.9],  # Node 1 - very similar to 0
            [0, 1, 1, 0],  # Node 2 - different
            [0, 1, 0.9, 0],  # Node 3 - different
            [0.9, 0, 0, 1]  # Node 4 - similar to 0
        ])
        
        similar = rewiring.find_similar_nodes(graph, 0, features, top_k=2)
        
        assert len(similar) <= 2
        assert similar[0][0] == 4  # Node 4 should be most similar
    
    def test_bottleneck_identification(self):
        """Test bottleneck edge identification"""
        rewiring = EpsilonGreedyRewiring()
        
        graph = nx.cycle_graph(4)
        latencies = np.array([
            [0, 2, 0, 1],
            [2, 0, 5, 0],  # Edge (1,2) has high latency
            [0, 5, 0, 3],
            [1, 0, 3, 0]
        ])
        
        bottlenecks = rewiring.find_bottleneck_edges(graph, latencies, percentile=50)
        
        assert len(bottlenecks) > 0
        assert bottlenecks[0][2] == 5  # Highest latency edge
    
    def test_rewiring_iteration(self):
        """Test single rewiring iteration"""
        rewiring = EpsilonGreedyRewiring(epsilon=0.5)
        
        # Create test network
        graph = nx.watts_strogatz_graph(10, 4, 0.1)
        features = np.random.rand(10, 4)
        latencies = np.random.rand(10, 10)
        
        initial_edges = graph.number_of_edges()
        
        # Perform rewiring
        stats = rewiring.rewire_iteration(graph, features, latencies, iteration=0)
        
        assert 'mode' in stats
        assert stats['mode'] in ['exploration', 'exploitation']
        assert 'rewires_made' in stats
        assert graph.number_of_edges() == initial_edges  # Edge count preserved
        assert nx.is_connected(graph)  # Connectivity preserved
    
    def test_epsilon_decay(self):
        """Test adaptive epsilon decay"""
        rewiring = EpsilonGreedyRewiring(epsilon=0.5)
        
        initial_epsilon = rewiring.epsilon
        rewiring.adaptive_epsilon_decay(iteration=5, max_iterations=10)
        
        assert rewiring.epsilon < initial_epsilon
        assert rewiring.epsilon >= 0.05  # Minimum epsilon


class TestSOANOrchestrator:
    """Test suite for SOAN orchestrator"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with config"""
        config = SOANConfig(n_nodes=50, gnn_epochs=10)
        orchestrator = SOANOrchestrator(config)
        
        assert orchestrator.config.n_nodes == 50
        assert orchestrator.config.gnn_epochs == 10
        assert orchestrator.topology is not None
        assert orchestrator.gnn_predictor is not None
        assert orchestrator.queue_model is not None
        assert orchestrator.rewiring is not None
    
    def test_network_initialization(self):
        """Test network initialization through orchestrator"""
        config = SOANConfig(n_nodes=20)
        orchestrator = SOANOrchestrator(config)
        
        graph = orchestrator.initialize_network(seed=42)
        
        assert graph.number_of_nodes() == 20
        assert orchestrator.graph is not None
        assert orchestrator.node_features is not None
        assert len(orchestrator.orchestration_receipts) == 1
    
    def test_latency_predictor_training(self):
        """Test GNN training through orchestrator"""
        config = SOANConfig(n_nodes=10, gnn_epochs=5)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        stats = orchestrator.train_latency_predictor(training_samples=100)
        
        assert 'training_losses' in stats
        assert 'training_time' in stats
        assert len(stats['training_losses']) == 5
    
    def test_latency_prediction(self):
        """Test latency prediction through orchestrator"""
        config = SOANConfig(n_nodes=10)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        orchestrator.train_latency_predictor(training_samples=50)
        
        latencies = orchestrator.predict_latencies()
        
        assert latencies.shape == (10, 10)
        assert orchestrator.latency_matrix is not None
    
    def test_topology_optimization(self):
        """Test topology optimization"""
        config = SOANConfig(n_nodes=10, rewiring_iterations=3)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        orchestrator.train_latency_predictor(training_samples=50)
        orchestrator.predict_latencies()
        
        stats = orchestrator.optimize_topology(iterations=2)
        
        assert 'total_rewires' in stats
        assert 'improvement' in stats
        assert stats['iterations'] == 2
    
    def test_slo_tracking(self):
        """Test SLO compliance tracking"""
        config = SOANConfig(n_nodes=10, slo_threshold=3.0)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        orchestrator.predict_latencies()
        
        slo_stats = orchestrator.track_slo_compliance()
        
        assert 'violation_rate' in slo_stats
        assert 'percentiles' in slo_stats
        assert len(orchestrator.slo_violations) == 1
    
    def test_adaptation_cycle(self):
        """Test complete adaptation cycle"""
        config = SOANConfig(n_nodes=10)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        orchestrator.train_latency_predictor(training_samples=50)
        
        cycle_stats = orchestrator.run_adaptation_cycle()
        
        assert 'initial_violation_rate' in cycle_stats
        assert 'final_violation_rate' in cycle_stats
        assert 'improvement' in cycle_stats
        assert 'cycle_time' in cycle_stats
    
    def test_state_persistence(self):
        """Test state save/load functionality"""
        config = SOANConfig(n_nodes=10)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        orchestrator.train_latency_predictor(training_samples=50)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save state
            orchestrator.save_state(tmpdir)
            
            # Load into new orchestrator
            new_orchestrator = SOANOrchestrator()
            new_orchestrator.load_state(tmpdir)
            
            # Verify state restoration
            assert new_orchestrator.config.n_nodes == 10
            assert new_orchestrator.graph is not None
            assert new_orchestrator.graph.number_of_nodes() == 10
    
    def test_epsilon_budget_tracking(self):
        """Test epsilon budget tracking across components"""
        config = SOANConfig(n_nodes=10)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        
        epsilon_usage = orchestrator.get_epsilon_usage()
        
        assert 'orchestrator' in epsilon_usage
        assert 'topology' in epsilon_usage
        assert 'total_used' in epsilon_usage
        assert epsilon_usage['total_used'] > 0
        assert epsilon_usage['total_used'] < epsilon_usage['total_budget']
    
    def test_podp_compliance_report(self):
        """Test PoDP compliance report generation"""
        config = SOANConfig(n_nodes=10)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        
        report = orchestrator.get_podp_compliance_report()
        
        assert 'total_receipts' in report
        assert 'epsilon_summary' in report
        assert 'compliance_status' in report
        assert report['compliance_status'] == 'COMPLIANT'
        assert report['total_receipts'] > 0


# Integration tests
class TestSOANIntegration:
    """Integration tests for complete SOAN workflow"""
    
    def test_full_workflow(self):
        """Test complete SOAN workflow from initialization to optimization"""
        config = SOANConfig(
            n_nodes=20,
            gnn_epochs=10,
            rewiring_iterations=5
        )
        
        orchestrator = SOANOrchestrator(config)
        
        # Initialize network
        graph = orchestrator.initialize_network(seed=42)
        assert graph.number_of_nodes() == 20
        
        # Train predictor
        training_stats = orchestrator.train_latency_predictor(training_samples=100)
        assert len(training_stats['training_losses']) == 10
        
        # Predict latencies
        latencies = orchestrator.predict_latencies()
        assert latencies.shape == (20, 20)
        
        # Check SLO compliance
        slo_stats = orchestrator.track_slo_compliance()
        initial_violations = slo_stats['violation_rate']
        
        # Optimize topology
        if initial_violations > 0.1:
            optimization_stats = orchestrator.optimize_topology(iterations=3)
            assert optimization_stats['total_rewires'] > 0
            
            # Re-check SLO compliance
            new_slo_stats = orchestrator.track_slo_compliance()
            # Should improve or stay same
            assert new_slo_stats['violation_rate'] <= initial_violations + 0.1
        
        # Verify PoDP compliance
        report = orchestrator.get_podp_compliance_report()
        assert report['compliance_status'] == 'COMPLIANT'
        
        # Check epsilon budget not exceeded
        epsilon_usage = orchestrator.get_epsilon_usage()
        assert epsilon_usage['total_used'] < epsilon_usage['total_budget']
    
    def test_adaptive_behavior(self):
        """Test adaptive network behavior under changing conditions"""
        config = SOANConfig(n_nodes=15)
        orchestrator = SOANOrchestrator(config)
        
        orchestrator.initialize_network()
        orchestrator.train_latency_predictor(training_samples=50)
        
        # Run multiple adaptation cycles
        violation_history = []
        
        for i in range(3):
            cycle_stats = orchestrator.run_adaptation_cycle()
            violation_history.append(cycle_stats['final_violation_rate'])
        
        # Network should adapt and maintain or improve performance
        assert len(violation_history) == 3
        # Allow some variance but generally should not degrade significantly
        assert violation_history[-1] <= violation_history[0] + 0.2
    
    def test_scalability(self):
        """Test SOAN scalability with different network sizes"""
        for n_nodes in [10, 50, 100]:
            config = SOANConfig(n_nodes=n_nodes, gnn_epochs=5)
            orchestrator = SOANOrchestrator(config)
            
            # Should handle different sizes
            graph = orchestrator.initialize_network()
            assert graph.number_of_nodes() == n_nodes
            
            # Training should complete
            stats = orchestrator.train_latency_predictor(training_samples=50)
            assert stats is not None
            
            # Predictions should work
            latencies = orchestrator.predict_latencies()
            assert latencies.shape == (n_nodes, n_nodes)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=services.agents", "--cov-report=term-missing"])