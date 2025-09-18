"""
Comprehensive Integration Tests for DALRN System
"""
import pytest
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestSystemIntegration:
    """Test all system components work together"""

    def test_gateway_loads(self):
        """Verify gateway can import without crashing"""
        try:
            from services.gateway.app import app
            assert app is not None
            assert app.title == "DALRN Gateway"
            print("[PASS] Gateway loads successfully")
        except Exception as e:
            pytest.fail(f"Gateway failed to load: {e}")

    def test_soan_topology_works(self):
        """Verify SOAN topology generation"""
        from services.agents.topology import WattsStrogatzTopology

        topology = WattsStrogatzTopology(N=100, k=6, p=0.1)
        graph, receipt = topology.generate()

        assert graph.number_of_nodes() == 100
        assert receipt is not None
        assert receipt.step == "network_generation_exit"
        print("[PASS] SOAN topology generation works")

    def test_gnn_predictor_works(self):
        """Verify GNN can be instantiated and run"""
        from services.agents.gnn_predictor import GNNLatencyPredictor

        predictor = GNNLatencyPredictor()

        # Create dummy data
        num_nodes = 10
        num_features = 10
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        # Test forward pass
        predictor.model.eval()
        with torch.no_grad():
            output = predictor.model(x, edge_index)

        assert output.shape[0] == num_nodes
        print("[PASS] GNN predictor works")

    def test_queue_model_works(self):
        """Verify M/M/1 queue model functions"""
        from services.agents.queue_model import MM1Queue

        queue = MM1Queue(arrival_rate=1.0, service_rate=1.5)

        assert queue.is_stable == True
        assert queue.utilization == pytest.approx(0.667, rel=0.01)

        metrics, receipt = queue.simulate(duration=10.0)
        assert metrics is not None
        assert receipt.step == "queue_simulation"
        print("[PASS] Queue model works")

    def test_rewiring_algorithm_works(self):
        """Verify rewiring algorithm functions"""
        from services.agents.rewiring import EpsilonGreedyRewiring
        from services.agents.topology import WattsStrogatzTopology

        topology = WattsStrogatzTopology(N=20, k=4, p=0.1)
        graph, _ = topology.generate()

        rewirer = EpsilonGreedyRewiring(epsilon=0.2)
        features = {i: np.random.rand(5) for i in range(20)}

        optimized_graph, metrics, receipt = rewirer.optimize(
            graph, features, iterations=5
        )

        assert optimized_graph is not None
        assert metrics is not None
        print("[PASS] Rewiring algorithm works")

    def test_tenseal_encryption(self):
        """Verify TenSEAL actually works"""
        import tenseal as ts

        # Create context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()

        # Test encryption/decryption
        plain = [1.5, 2.3, 3.7]
        encrypted = ts.ckks_vector(context, plain)

        # Perform homomorphic operation
        encrypted_squared = encrypted * encrypted

        # Decrypt
        result = encrypted_squared.decrypt()
        expected = [x*x for x in plain]

        for r, e in zip(result, expected):
            assert abs(r - e) < 0.01, f"Decryption error too large: {abs(r - e)}"

        print("[PASS] TenSEAL encryption works")

    def test_negotiation_service(self):
        """Test negotiation service imports"""
        from services.negotiation.service import compute_nash_equilibrium

        # Simple 2x2 game
        payoff_a = [[3, 0], [5, 1]]
        payoff_b = [[3, 5], [0, 1]]

        result = compute_nash_equilibrium(payoff_a, payoff_b)
        assert result is not None
        print("[PASS] Negotiation service works")

    def test_search_service(self):
        """Test search service components"""
        from services.search.service import VectorIndex

        index = VectorIndex(dimension=768, index_type="HNSW32")

        # Add vectors
        vectors = np.random.randn(100, 768).astype(np.float32)
        index.add(vectors)

        # Search
        query = np.random.randn(1, 768).astype(np.float32)
        results = index.search(query, k=10)

        assert results["ids"].shape == (1, 10)
        print("[PASS] Search service works")

    def test_podp_receipts(self):
        """Test PoDP receipt generation"""
        from services.common.podp import Receipt, ReceiptChain

        receipt = Receipt(
            receipt_id="test_001",
            dispute_id="dispute_001",
            step="test_step",
            inputs={"data": "test"},
            params={"param": "value"},
            artifacts={},
            hashes={},
            signatures=[],
            ts="2025-01-01T00:00:00Z"
        )

        receipt.compute_hash()
        assert receipt.hash is not None

        chain = ReceiptChain(dispute_id="dispute_001")
        chain.add_receipt(receipt)
        chain.build_merkle_tree()

        assert chain.merkle_root is not None
        print("[PASS] PoDP receipts work")

def run_all_tests():
    """Run all integration tests and report results"""
    test_suite = TestSystemIntegration()
    tests = [
        ("Gateway", test_suite.test_gateway_loads),
        ("SOAN Topology", test_suite.test_soan_topology_works),
        ("GNN Predictor", test_suite.test_gnn_predictor_works),
        ("Queue Model", test_suite.test_queue_model_works),
        ("Rewiring", test_suite.test_rewiring_algorithm_works),
        ("TenSEAL", test_suite.test_tenseal_encryption),
        ("Negotiation", test_suite.test_negotiation_service),
        ("Search", test_suite.test_search_service),
        ("PoDP", test_suite.test_podp_receipts)
    ]

    print("\n" + "="*60)
    print("DALRN INTEGRATION TEST SUITE")
    print("="*60 + "\n")

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name} FAILED: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)