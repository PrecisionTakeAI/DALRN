"""
Comprehensive test suite for enhanced DALRN Negotiation Service.
Tests all production features including PoDP receipts, explanations, CIDs, and edge cases.
"""

import pytest
import numpy as np
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Import modules to test
from services.negotiation.service import (
    NegotiationService, NegReq, EnhancedNegotiationRequest,
    NegotiationResponse, app, service
)
from services.negotiation.validation import (
    NegotiationRequest, GameValidator, ValidationError,
    validate_equilibrium, sanitize_strategy, detect_degenerate_game,
    validate_payoff_range
)
from services.negotiation.explanation import (
    ExplanationGenerator, NegotiationContext, generate_concise_explanation
)
from services.negotiation.cid_generator import (
    CausalInfluenceDiagram, CIDNode, CIDEdge,
    NegotiationCIDGenerator, create_enhanced_cid
)
from services.common.podp import Receipt, ReceiptChain

# Test fixtures
@pytest.fixture
def standard_game():
    """Standard prisoner's dilemma game."""
    return {
        'A': [[3, 0], [5, 1]],
        'B': [[3, 5], [0, 1]]
    }

@pytest.fixture
def battle_of_sexes():
    """Battle of the sexes game with multiple equilibria."""
    return {
        'A': [[2, 0], [0, 1]],
        'B': [[1, 0], [0, 2]]
    }

@pytest.fixture
def zero_sum_game():
    """Zero-sum matching pennies game."""
    return {
        'A': [[1, -1], [-1, 1]],
        'B': [[-1, 1], [1, -1]]
    }

@pytest.fixture
def degenerate_game():
    """Degenerate game with dominated strategies."""
    return {
        'A': [[3, 3], [1, 1]],  # Row 1 dominates row 2
        'B': [[3, 1], [3, 1]]
    }

@pytest.fixture
def large_game():
    """Larger 3x3 game for performance testing."""
    return {
        'A': [[3, 0, 2], [5, 1, 4], [2, 3, 1]],
        'B': [[3, 5, 1], [0, 1, 2], [4, 2, 3]]
    }

@pytest.fixture
def negotiation_service():
    """Create a negotiation service instance."""
    return NegotiationService()

# Validation Module Tests
class TestValidation:
    """Test input validation and error handling."""
    
    def test_valid_request(self, standard_game):
        """Test validation of a valid request."""
        req = NegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0)
        )
        assert req.A == standard_game['A']
        assert req.rule == 'nsw'
    
    def test_invalid_matrix_dimensions(self):
        """Test validation catches mismatched matrix dimensions."""
        with pytest.raises(ValueError, match="Matrix dimension mismatch"):
            NegotiationRequest(
                A=[[1, 2], [3, 4]],
                B=[[1, 2, 3], [4, 5, 6]],
                rule='nsw'
            )
    
    def test_empty_matrix(self):
        """Test validation catches empty matrices."""
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            NegotiationRequest(
                A=[],
                B=[[]],
                rule='nsw'
            )
    
    def test_non_rectangular_matrix(self):
        """Test validation catches non-rectangular matrices."""
        with pytest.raises(ValueError, match="Inconsistent row lengths"):
            NegotiationRequest(
                A=[[1, 2], [3]],
                B=[[1, 2], [3, 4]],
                rule='nsw'
            )
    
    def test_invalid_values(self):
        """Test validation catches NaN and Inf values."""
        with pytest.raises(ValueError, match="Invalid value"):
            NegotiationRequest(
                A=[[1, np.nan], [3, 4]],
                B=[[1, 2], [3, 4]],
                rule='nsw'
            )
        
        with pytest.raises(ValueError, match="Invalid value"):
            NegotiationRequest(
                A=[[1, 2], [np.inf, 4]],
                B=[[1, 2], [3, 4]],
                rule='nsw'
            )
    
    def test_invalid_rule(self):
        """Test validation catches invalid selection rules."""
        with pytest.raises(ValueError, match="Invalid rule"):
            NegotiationRequest(
                A=[[1, 2], [3, 4]],
                B=[[1, 2], [3, 4]],
                rule='invalid_rule'
            )
    
    def test_invalid_batna(self):
        """Test validation catches invalid BATNA values."""
        with pytest.raises(ValueError, match="BATNA must have exactly 2 values"):
            NegotiationRequest(
                A=[[1, 2], [3, 4]],
                B=[[1, 2], [3, 4]],
                rule='nsw',
                batna=(0.0,)  # Only one value
            )
    
    def test_equilibrium_validation(self):
        """Test equilibrium validation function."""
        # Valid equilibrium
        eq = (np.array([0.5, 0.5]), np.array([0.3, 0.7]))
        is_valid, error = validate_equilibrium(eq, (2, 2))
        assert is_valid
        assert error is None
        
        # Invalid - doesn't sum to 1
        eq = (np.array([0.5, 0.3]), np.array([0.3, 0.7]))
        is_valid, error = validate_equilibrium(eq, (2, 2))
        assert not is_valid
        assert "doesn't sum to 1" in error
        
        # Invalid - negative values
        eq = (np.array([0.5, 0.5]), np.array([-0.1, 1.1]))
        is_valid, error = validate_equilibrium(eq, (2, 2))
        assert not is_valid
        assert "Negative values" in error
    
    def test_sanitize_strategy(self):
        """Test strategy sanitization."""
        # Small negative values
        strategy = np.array([0.5, 0.5, -1e-10])
        sanitized = sanitize_strategy(strategy)
        assert np.allclose(sanitized.sum(), 1.0)
        assert np.all(sanitized >= 0)
        
        # All zeros
        strategy = np.array([0, 0, 0])
        sanitized = sanitize_strategy(strategy)
        assert np.allclose(sanitized, 1/3)
    
    def test_payoff_range_validation(self):
        """Test payoff range validation."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2], [3, 4]])
        
        is_valid, error = validate_payoff_range(A, B)
        assert is_valid
        
        # Too large values
        A_large = np.array([[1e7, 2], [3, 4]])
        is_valid, error = validate_payoff_range(A_large, B)
        assert not is_valid
        assert "too large" in error
    
    def test_detect_degenerate_game(self, degenerate_game):
        """Test degenerate game detection."""
        A = np.array(degenerate_game['A'])
        B = np.array(degenerate_game['B'])
        
        is_degenerate, desc = detect_degenerate_game(A, B)
        assert is_degenerate
        assert "dominates" in desc
    
    def test_game_validator(self, standard_game, zero_sum_game):
        """Test comprehensive game validation."""
        validator = GameValidator()
        
        # Standard game
        A = np.array(standard_game['A'])
        B = np.array(standard_game['B'])
        result = validator.validate_game(A, B)
        assert result['valid']
        assert result['properties']['shape'] == (2, 2)
        assert not result['properties']['zero_sum']
        
        # Zero-sum game
        A = np.array(zero_sum_game['A'])
        B = np.array(zero_sum_game['B'])
        result = validator.validate_game(A, B)
        assert result['valid']
        assert result['properties']['zero_sum']

# Explanation Module Tests
class TestExplanation:
    """Test explanation memo generation."""
    
    def test_explanation_generator_init(self):
        """Test explanation generator initialization."""
        gen = ExplanationGenerator()
        assert gen.templates is not None
        assert 'header' in gen.templates
    
    def test_generate_concise_explanation(self, standard_game):
        """Test concise explanation generation."""
        A = np.array(standard_game['A'])
        B = np.array(standard_game['B'])
        eq = (np.array([1, 0]), np.array([0, 1]))
        utilities = (1.0, 1.0)
        
        explanation = generate_concise_explanation(A, B, eq, utilities, 'nsw')
        assert "Negotiation Result" in explanation
        assert "Player 1 utility: 1.0000" in explanation
        assert "pure" in explanation
    
    def test_full_memo_generation(self, standard_game):
        """Test full memo generation."""
        gen = ExplanationGenerator()
        A = np.array(standard_game['A'])
        B = np.array(standard_game['B'])
        
        context = NegotiationContext(
            payoff_A=A,
            payoff_B=B,
            equilibrium=(np.array([1, 0]), np.array([0, 1])),
            utilities=(1.0, 1.0),
            selection_rule='nsw',
            batna=(0.0, 0.0),
            all_equilibria=[(np.array([1, 0]), np.array([0, 1]))],
            computation_time=0.1,
            metadata={'iterations': 1}
        )
        
        memo = gen.generate_memo(context, "TEST-001")
        
        assert "# Negotiation Analysis Report" in memo
        assert "TEST-001" in memo
        assert "Nash Social Welfare" in memo
        assert "Game Structure" in memo
        assert "Equilibrium Analysis" in memo
        assert "Fairness Analysis" in memo
        assert "Recommendations" in memo
    
    def test_fairness_metrics_calculation(self):
        """Test fairness metrics calculation."""
        gen = ExplanationGenerator()
        metrics = gen._calculate_fairness_metrics((3.0, 2.0), (1.0, 0.5))
        
        assert metrics['nsw'] == pytest.approx(2.0 * 1.5)
        assert metrics['egal'] == 2.0
        assert metrics['util'] == 5.0
        assert 0 <= metrics['gini'] <= 1
        assert 0 <= metrics['minmax'] <= 1
    
    def test_strategy_interpretation(self):
        """Test strategy interpretation."""
        gen = ExplanationGenerator()
        
        # Pure strategy
        pure_strat = np.array([1, 0, 0])
        interp = gen._interpret_strategy(pure_strat, "row")
        assert "Action 1: 100.0%" in interp
        assert "Pure strategy" in interp
        
        # Mixed strategy
        mixed_strat = np.array([0.3, 0.4, 0.3])
        interp = gen._interpret_strategy(mixed_strat, "row")
        assert "Action 1: 30.0%" in interp
        assert "Action 2: 40.0%" in interp
        assert "Mixed strategy" in interp

# CID Generator Module Tests
class TestCIDGenerator:
    """Test Causal Influence Diagram generation."""
    
    def test_cid_node_creation(self):
        """Test CID node creation."""
        node = CIDNode(
            id="test_node",
            type="decision",
            label="Test Node",
            player=1,
            value=0.5
        )
        
        assert node.id == "test_node"
        assert node.type == "decision"
        assert node.player == 1
        
        node_dict = node.to_dict()
        assert node_dict['id'] == "test_node"
    
    def test_cid_edge_creation(self):
        """Test CID edge creation."""
        edge = CIDEdge(
            source="node1",
            target="node2",
            type="causal",
            weight=0.8
        )
        
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.weight == 0.8
    
    def test_causal_influence_diagram(self):
        """Test CID basic operations."""
        cid = CausalInfluenceDiagram()
        
        # Add nodes
        node1 = CIDNode("n1", "decision", "Decision 1", player=1)
        node2 = CIDNode("n2", "utility", "Utility 1", player=1)
        cid.add_node(node1)
        cid.add_node(node2)
        
        # Add edge
        edge = CIDEdge("n1", "n2", "causal")
        cid.add_edge(edge)
        
        # Test retrieval
        assert len(cid.nodes) == 2
        assert len(cid.edges) == 1
        assert cid.get_children("n1") == ["n2"]
        assert cid.get_parents("n2") == ["n1"]
        
        # Test decision nodes
        decisions = cid.get_decision_nodes(player=1)
        assert len(decisions) == 1
        assert decisions[0].id == "n1"
    
    def test_negotiation_cid_generation(self, standard_game):
        """Test negotiation CID generation."""
        gen = NegotiationCIDGenerator()
        A = np.array(standard_game['A'])
        B = np.array(standard_game['B'])
        eq = (np.array([1, 0]), np.array([0, 1]))
        
        cid = gen.generate_negotiation_cid(A, B, eq, (0.0, 0.0))
        
        assert "decision_p1" in cid.nodes
        assert "decision_p2" in cid.nodes
        assert "utility_p1" in cid.nodes
        assert "utility_p2" in cid.nodes
        assert "outcome" in cid.nodes
        
        # Check JSON serialization
        cid_json = cid.to_json()
        assert isinstance(cid_json, str)
        data = json.loads(cid_json)
        assert 'nodes' in data
        assert 'edges' in data
    
    def test_enhanced_cid_creation(self, standard_game):
        """Test enhanced CID with all features."""
        A = np.array(standard_game['A'])
        B = np.array(standard_game['B'])
        eq = (np.array([1, 0]), np.array([0, 1]))
        fairness = {'nsw': 1.0, 'egal': 1.0}
        
        cid_data = create_enhanced_cid(
            A, B, eq, (0.0, 0.0), fairness,
            metadata={'uncertainty': {'risk': 0.1}}
        )
        
        assert 'cid' in cid_data
        assert 'graphviz' in cid_data
        assert 'strategic_analysis' in cid_data
        assert cid_data['metadata']['num_nodes'] > 5
    
    def test_cid_graphviz_generation(self):
        """Test Graphviz DOT generation."""
        cid = CausalInfluenceDiagram()
        cid.add_node(CIDNode("d1", "decision", "Decision", player=1))
        cid.add_node(CIDNode("u1", "utility", "Utility", player=1))
        cid.add_edge(CIDEdge("d1", "u1", "causal"))
        
        dot = cid.to_graphviz()
        assert "digraph CID" in dot
        assert "Decision" in dot
        assert "Utility" in dot
        assert "->" in dot
    
    def test_influence_paths(self):
        """Test influence path computation."""
        cid = CausalInfluenceDiagram()
        
        # Create a simple path: n1 -> n2 -> n3
        for i in range(1, 4):
            cid.add_node(CIDNode(f"n{i}", "chance", f"Node {i}"))
        
        cid.add_edge(CIDEdge("n1", "n2", "causal"))
        cid.add_edge(CIDEdge("n2", "n3", "causal"))
        
        paths = cid.compute_influence_paths("n1", "n3")
        assert len(paths) == 1
        assert paths[0] == ["n1", "n2", "n3"]

# Service Integration Tests
class TestNegotiationService:
    """Test the integrated negotiation service."""
    
    @pytest.mark.asyncio
    async def test_basic_negotiation(self, negotiation_service, standard_game):
        """Test basic negotiation without extras."""
        req = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert response.dispute_id is not None
        assert 'row_strategy' in response.equilibrium
        assert 'col_strategy' in response.equilibrium
        assert response.utilities['player_1'] is not None
        assert response.utilities['player_2'] is not None
        assert response.num_equilibria >= 1
    
    @pytest.mark.asyncio
    async def test_negotiation_with_explanation(self, negotiation_service, standard_game):
        """Test negotiation with explanation generation."""
        req = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=True,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert response.explanation_summary is not None
        assert "Negotiation Result" in response.explanation_summary
    
    @pytest.mark.asyncio
    async def test_negotiation_with_cid(self, negotiation_service, standard_game):
        """Test negotiation with CID generation."""
        req = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=True,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert response.fairness_metrics is not None
        assert 'nash_social_welfare' in response.fairness_metrics
    
    @pytest.mark.asyncio
    async def test_negotiation_with_receipts(self, negotiation_service, standard_game):
        """Test negotiation with PoDP receipts."""
        req = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=True
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert response.receipt_id is not None
        assert response.receipt_id == response.dispute_id
    
    @pytest.mark.asyncio
    async def test_multiple_equilibria_selection(self, negotiation_service, battle_of_sexes):
        """Test selection among multiple equilibria."""
        req = EnhancedNegotiationRequest(
            A=battle_of_sexes['A'],
            B=battle_of_sexes['B'],
            rule='egal',  # Use egalitarian rule
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert response.num_equilibria >= 2  # Battle of sexes has multiple equilibria
        assert response.selection_rule == 'egal'
    
    @pytest.mark.asyncio
    async def test_different_selection_rules(self, negotiation_service, standard_game):
        """Test different selection rules produce different results."""
        results = {}
        
        for rule in ['nsw', 'egal', 'utilitarian']:
            req = EnhancedNegotiationRequest(
                A=standard_game['A'],
                B=standard_game['B'],
                rule=rule,
                batna=(0.5, 0.5),
                generate_explanation=False,
                generate_cid=False,
                generate_receipt=False
            )
            
            response = await negotiation_service.negotiate_with_receipts(req)
            results[rule] = (
                response.utilities['player_1'],
                response.utilities['player_2']
            )
        
        # Different rules may produce different utilities
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_batna_influence(self, negotiation_service, standard_game):
        """Test BATNA influence on negotiation."""
        # Without BATNA
        req1 = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        # With BATNA
        req2 = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(1.5, 1.5),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response1 = await negotiation_service.negotiate_with_receipts(req1)
        response2 = await negotiation_service.negotiate_with_receipts(req2)
        
        # BATNA should affect fairness metrics
        assert response1.fairness_metrics['surplus_player1'] != response2.fairness_metrics['surplus_player1']
    
    @pytest.mark.asyncio
    async def test_zero_sum_game_handling(self, negotiation_service, zero_sum_game):
        """Test zero-sum game handling."""
        req = EnhancedNegotiationRequest(
            A=zero_sum_game['A'],
            B=zero_sum_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        # Zero-sum game utilities should sum to approximately 0
        total = response.utilities['player_1'] + response.utilities['player_2']
        assert abs(total) < 0.01
    
    @pytest.mark.asyncio
    async def test_degenerate_game_warning(self, negotiation_service, degenerate_game):
        """Test degenerate game produces warnings."""
        req = EnhancedNegotiationRequest(
            A=degenerate_game['A'],
            B=degenerate_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        # Should have warnings about degeneracy
        assert len(response.warnings) > 0
        assert any("degenera" in w.lower() for w in response.warnings)
    
    @pytest.mark.asyncio
    async def test_timeout_protection(self, negotiation_service):
        """Test timeout protection for long computations."""
        # Create a large game that might take long to compute
        large_A = np.random.rand(10, 10).tolist()
        large_B = np.random.rand(10, 10).tolist()
        
        req = EnhancedNegotiationRequest(
            A=large_A,
            B=large_B,
            rule='nsw',
            batna=(0.0, 0.0),
            timeout_seconds=0.1,  # Very short timeout
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        # Should complete (possibly with bargaining fallback)
        response = await negotiation_service.negotiate_with_receipts(req)
        assert response.dispute_id is not None
    
    @pytest.mark.asyncio
    async def test_bargaining_fallback(self, negotiation_service):
        """Test bargaining solution fallback."""
        # Create a game that might not have Nash equilibrium in some cases
        # Using mock to force no equilibrium found
        with patch.object(negotiation_service, '_compute_equilibria_async', return_value=[]):
            with patch.object(negotiation_service, '_compute_support_enumeration_async', return_value=[]):
                req = EnhancedNegotiationRequest(
                    A=[[1, 2], [3, 4]],
                    B=[[4, 3], [2, 1]],
                    rule='nsw',
                    batna=(0.0, 0.0),
                    generate_explanation=False,
                    generate_cid=False,
                    generate_receipt=False
                )
                
                response = await negotiation_service.negotiate_with_receipts(req)
                
                assert response.dispute_id is not None
                assert "bargaining solution" in ' '.join(response.warnings).lower()

# Edge Cases and Error Handling Tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_single_action_game(self, negotiation_service):
        """Test 1x1 game (single action for each player)."""
        req = EnhancedNegotiationRequest(
            A=[[5]],
            B=[[3]],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert response.utilities['player_1'] == 5.0
        assert response.utilities['player_2'] == 3.0
        assert response.equilibrium['row_strategy'] == [1.0]
        assert response.equilibrium['col_strategy'] == [1.0]
    
    @pytest.mark.asyncio
    async def test_asymmetric_game(self, negotiation_service):
        """Test asymmetric game (different dimensions)."""
        req = EnhancedNegotiationRequest(
            A=[[1, 2, 3], [4, 5, 6]],  # 2x3 matrix
            B=[[6, 5, 4], [3, 2, 1]],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert len(response.equilibrium['row_strategy']) == 2
        assert len(response.equilibrium['col_strategy']) == 3
    
    @pytest.mark.asyncio
    async def test_negative_payoffs(self, negotiation_service):
        """Test game with negative payoffs."""
        req = EnhancedNegotiationRequest(
            A=[[-1, -2], [-3, -4]],
            B=[[-4, -3], [-2, -1]],
            rule='egal',
            batna=(-5.0, -5.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        assert response.dispute_id is not None
        # Both players should get negative utilities but better than BATNA
        assert response.utilities['player_1'] > -5.0
        assert response.utilities['player_2'] > -5.0
    
    @pytest.mark.asyncio
    async def test_tied_equilibria(self, negotiation_service):
        """Test tie-breaking when multiple equilibria have same score."""
        # Symmetric game with tied equilibria
        req = EnhancedNegotiationRequest(
            A=[[2, 0], [0, 2]],
            B=[[2, 0], [0, 2]],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=False
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        # Should select deterministically (most balanced)
        assert response.dispute_id is not None
        assert response.num_equilibria >= 2
    
    def test_invalid_edge_in_cid(self):
        """Test CID edge validation."""
        cid = CausalInfluenceDiagram()
        cid.add_node(CIDNode("n1", "decision", "Node 1"))
        
        # Try to add edge with non-existent target
        edge = CIDEdge("n1", "n2", "causal")
        
        with pytest.raises(ValueError, match="Target node n2 not found"):
            cid.add_edge(edge)

# Performance and Stress Tests
class TestPerformance:
    """Test performance and stress scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_game_performance(self, negotiation_service, large_game):
        """Test performance with larger games."""
        import time
        
        req = EnhancedNegotiationRequest(
            A=large_game['A'],
            B=large_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=True,
            generate_cid=True,
            generate_receipt=True
        )
        
        start = time.time()
        response = await negotiation_service.negotiate_with_receipts(req)
        duration = time.time() - start
        
        assert response.dispute_id is not None
        assert duration < 5.0  # Should complete within 5 seconds
        assert response.computation_time < 5.0
    
    @pytest.mark.asyncio
    async def test_concurrent_negotiations(self, negotiation_service, standard_game):
        """Test handling concurrent negotiations."""
        requests = []
        for i in range(5):
            req = EnhancedNegotiationRequest(
                A=standard_game['A'],
                B=standard_game['B'],
                rule='nsw' if i % 2 == 0 else 'egal',
                batna=(0.0, 0.0),
                generate_explanation=False,
                generate_cid=False,
                generate_receipt=False
            )
            requests.append(negotiation_service.negotiate_with_receipts(req))
        
        responses = await asyncio.gather(*requests)
        
        # All should complete successfully
        assert len(responses) == 5
        assert all(r.dispute_id is not None for r in responses)
        
        # Each should have unique dispute_id
        dispute_ids = [r.dispute_id for r in responses]
        assert len(dispute_ids) == len(set(dispute_ids))
    
    def test_validator_caching(self, standard_game):
        """Test game validator caching."""
        validator = GameValidator()
        A = np.array(standard_game['A'])
        B = np.array(standard_game['B'])
        
        # First call
        result1 = validator.validate_game(A, B)
        
        # Second call (should use cache)
        result2 = validator.validate_game(A, B)
        
        assert result1 == result2
        assert len(validator._validation_cache) > 0

# API Endpoint Tests
class TestAPIEndpoints:
    """Test FastAPI endpoints."""
    
    @pytest.mark.asyncio
    async def test_legacy_negotiate_endpoint(self):
        """Test backward compatible /negotiate endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.post("/negotiate", json={
            "A": [[3, 0], [5, 1]],
            "B": [[3, 5], [0, 1]],
            "rule": "nsw",
            "batna": [0.0, 0.0]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "row" in data
        assert "col" in data
        assert "u1" in data
        assert "u2" in data
    
    @pytest.mark.asyncio
    async def test_enhanced_negotiate_endpoint(self):
        """Test enhanced /negotiate/enhanced endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.post("/negotiate/enhanced", json={
            "A": [[3, 0], [5, 1]],
            "B": [[3, 5], [0, 1]],
            "rule": "nsw",
            "batna": [0.0, 0.0],
            "generate_explanation": True,
            "generate_cid": True,
            "generate_receipt": True,
            "tolerance": 1e-9,
            "max_iterations": 1000,
            "timeout_seconds": 30.0
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "dispute_id" in data
        assert "equilibrium" in data
        assert "utilities" in data
        assert "fairness_metrics" in data
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self):
        """Test /health endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "negotiation"
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_get_negotiation_endpoint(self):
        """Test /negotiate/{dispute_id} retrieval endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # First create a negotiation
        create_response = client.post("/negotiate/enhanced", json={
            "A": [[3, 0], [5, 1]],
            "B": [[3, 5], [0, 1]],
            "rule": "nsw",
            "batna": [0.0, 0.0],
            "generate_explanation": False,
            "generate_cid": False,
            "generate_receipt": False,
            "tolerance": 1e-9,
            "max_iterations": 1000,
            "timeout_seconds": 30.0
        })
        
        dispute_id = create_response.json()["dispute_id"]
        
        # Try to retrieve it
        get_response = client.get(f"/negotiate/{dispute_id}")
        
        assert get_response.status_code == 200
        data = get_response.json()
        assert "request" in data
        assert "response" in data
    
    @pytest.mark.asyncio
    async def test_get_receipt_chain_endpoint(self):
        """Test /receipt/{dispute_id} endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Create negotiation with receipt
        create_response = client.post("/negotiate/enhanced", json={
            "A": [[3, 0], [5, 1]],
            "B": [[3, 5], [0, 1]],
            "rule": "nsw",
            "batna": [0.0, 0.0],
            "generate_explanation": False,
            "generate_cid": False,
            "generate_receipt": True,
            "tolerance": 1e-9,
            "max_iterations": 1000,
            "timeout_seconds": 30.0
        })
        
        dispute_id = create_response.json()["dispute_id"]
        
        # Get receipt chain
        receipt_response = client.get(f"/receipt/{dispute_id}")
        
        assert receipt_response.status_code == 200
        data = receipt_response.json()
        assert "dispute_id" in data
        assert "receipts" in data

# Receipt and PoDP Tests
class TestPoDP:
    """Test PoDP receipt generation and validation."""
    
    @pytest.mark.asyncio
    async def test_receipt_generation(self, negotiation_service, standard_game):
        """Test that receipts are properly generated."""
        req = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=True
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        # Check receipt was created
        assert response.receipt_id is not None
        
        # Check receipt chain exists in cache
        from services.negotiation.service import receipt_chains
        assert response.dispute_id in receipt_chains
        
        chain = receipt_chains[response.dispute_id]
        assert len(chain.receipts) >= 3  # Validation, computation, selection
        
        # Check receipt structure
        for receipt in chain.receipts:
            assert receipt.dispute_id == response.dispute_id
            assert receipt.step is not None
            assert receipt.ts is not None
            assert receipt.hash is not None
    
    @pytest.mark.asyncio
    async def test_receipt_chain_merkle_root(self, negotiation_service, standard_game):
        """Test Merkle root computation for receipt chain."""
        req = EnhancedNegotiationRequest(
            A=standard_game['A'],
            B=standard_game['B'],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=True
        )
        
        response = await negotiation_service.negotiate_with_receipts(req)
        
        from services.negotiation.service import receipt_chains
        chain = receipt_chains[response.dispute_id]
        
        # Compute Merkle root
        merkle_root = chain.compute_merkle_root()
        assert merkle_root is not None
        assert len(merkle_root) == 64  # SHA-256 hex string
    
    @pytest.mark.asyncio
    async def test_error_receipt_generation(self, negotiation_service):
        """Test error receipt generation on failure."""
        # Invalid matrices to cause error
        req = EnhancedNegotiationRequest(
            A=[],  # Empty matrix will cause error
            B=[[]],
            rule='nsw',
            batna=(0.0, 0.0),
            generate_explanation=False,
            generate_cid=False,
            generate_receipt=True
        )
        
        with pytest.raises(Exception):
            await negotiation_service.negotiate_with_receipts(req)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])