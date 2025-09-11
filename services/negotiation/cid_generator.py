"""
Causal Influence Diagram (CID) generator for DALRN Negotiation Service.
Creates visual and JSON representations of decision dependencies.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class CIDNode:
    """Represents a node in the Causal Influence Diagram."""
    id: str
    type: str  # 'decision', 'chance', 'utility', 'information'
    label: str
    player: Optional[int] = None
    value: Optional[Any] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = asdict(self)
        if d['metadata'] is None:
            d['metadata'] = {}
        return d

@dataclass
class CIDEdge:
    """Represents an edge in the Causal Influence Diagram."""
    source: str
    target: str
    type: str  # 'causal', 'information', 'utility'
    weight: float = 1.0
    label: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = asdict(self)
        if d['metadata'] is None:
            d['metadata'] = {}
        return d

class CausalInfluenceDiagram:
    """Generates and manages Causal Influence Diagrams for negotiations."""
    
    def __init__(self):
        self.nodes: Dict[str, CIDNode] = {}
        self.edges: List[CIDEdge] = []
        self.metadata: Dict[str, Any] = {
            'created': datetime.utcnow().isoformat(),
            'version': '1.0'
        }
    
    def add_node(self, node: CIDNode) -> None:
        """Add a node to the diagram."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: CIDEdge) -> None:
        """Add an edge to the diagram."""
        # Validate that nodes exist
        if edge.source not in self.nodes:
            raise ValueError(f"Source node {edge.source} not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node {edge.target} not found")
        self.edges.append(edge)
    
    def get_parents(self, node_id: str) -> List[str]:
        """Get parent nodes of a given node."""
        return [e.source for e in self.edges if e.target == node_id]
    
    def get_children(self, node_id: str) -> List[str]:
        """Get child nodes of a given node."""
        return [e.target for e in self.edges if e.source == node_id]
    
    def get_decision_nodes(self, player: Optional[int] = None) -> List[CIDNode]:
        """Get all decision nodes, optionally filtered by player."""
        decisions = [n for n in self.nodes.values() if n.type == 'decision']
        if player is not None:
            decisions = [n for n in decisions if n.player == player]
        return decisions
    
    def get_utility_nodes(self, player: Optional[int] = None) -> List[CIDNode]:
        """Get all utility nodes, optionally filtered by player."""
        utilities = [n for n in self.nodes.values() if n.type == 'utility']
        if player is not None:
            utilities = [n for n in utilities if n.player == player]
        return utilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire diagram to dictionary representation."""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_graphviz(self) -> str:
        """Generate Graphviz DOT representation."""
        lines = ['digraph CID {']
        lines.append('  rankdir=LR;')
        lines.append('  node [fontsize=10];')
        lines.append('  edge [fontsize=9];')
        
        # Define node styles by type
        lines.append('  ')
        lines.append('  // Nodes')
        
        for node in self.nodes.values():
            shape = {
                'decision': 'box',
                'chance': 'ellipse',
                'utility': 'diamond',
                'information': 'hexagon'
            }.get(node.type, 'ellipse')
            
            color = {
                1: 'lightblue',
                2: 'lightgreen'
            }.get(node.player, 'lightgray')
            
            label = node.label.replace('"', '\\"')
            lines.append(f'  {node.id} [label="{label}", shape={shape}, style=filled, fillcolor={color}];')
        
        lines.append('  ')
        lines.append('  // Edges')
        
        for edge in self.edges:
            style = {
                'causal': 'solid',
                'information': 'dashed',
                'utility': 'dotted'
            }.get(edge.type, 'solid')
            
            label = f', label="{edge.label}"' if edge.label else ''
            lines.append(f'  {edge.source} -> {edge.target} [style={style}{label}];')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def compute_influence_paths(self, source: str, target: str) -> List[List[str]]:
        """Compute all influence paths from source to target."""
        paths = []
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path.copy())
                return
            
            visited.add(current)
            for child in self.get_children(current):
                if child not in visited:
                    path.append(child)
                    dfs(child, path, visited.copy())
                    path.pop()
        
        dfs(source, [source], set())
        return paths
    
    def analyze_strategic_relevance(self) -> Dict[str, Any]:
        """Analyze the strategic relevance of nodes."""
        analysis = {
            'decision_influence': {},
            'information_asymmetry': {},
            'utility_dependencies': {}
        }
        
        # Analyze decision influence
        for decision in self.get_decision_nodes():
            influenced_utilities = []
            for utility in self.get_utility_nodes():
                paths = self.compute_influence_paths(decision.id, utility.id)
                if paths:
                    influenced_utilities.append({
                        'utility': utility.id,
                        'paths': len(paths),
                        'min_length': min(len(p) for p in paths)
                    })
            analysis['decision_influence'][decision.id] = influenced_utilities
        
        # Analyze information asymmetry
        for player in [1, 2]:
            player_decisions = self.get_decision_nodes(player)
            accessible_info = set()
            
            for decision in player_decisions:
                for parent in self.get_parents(decision.id):
                    parent_node = self.nodes[parent]
                    if parent_node.type == 'information':
                        accessible_info.add(parent)
            
            analysis['information_asymmetry'][f'player_{player}'] = list(accessible_info)
        
        # Analyze utility dependencies
        for utility in self.get_utility_nodes():
            dependencies = []
            for parent in self.get_parents(utility.id):
                parent_node = self.nodes[parent]
                dependencies.append({
                    'node': parent,
                    'type': parent_node.type,
                    'player': parent_node.player
                })
            analysis['utility_dependencies'][utility.id] = dependencies
        
        return analysis

class NegotiationCIDGenerator:
    """Generate CID for negotiation scenarios."""
    
    def __init__(self):
        self.cid = CausalInfluenceDiagram()
    
    def generate_negotiation_cid(
        self,
        payoff_A: np.ndarray,
        payoff_B: np.ndarray,
        equilibrium: Tuple[np.ndarray, np.ndarray],
        batna: Tuple[float, float] = (0.0, 0.0),
        metadata: Dict[str, Any] = None
    ) -> CausalInfluenceDiagram:
        """
        Generate a CID for a negotiation scenario.
        
        Args:
            payoff_A, payoff_B: Payoff matrices
            equilibrium: Computed equilibrium
            batna: BATNA values
            metadata: Additional metadata
        
        Returns:
            Generated CausalInfluenceDiagram
        """
        n_rows, n_cols = payoff_A.shape
        row_strat, col_strat = equilibrium
        
        # Add game structure nodes
        self.cid.add_node(CIDNode(
            id="game_structure",
            type="chance",
            label="Game Structure",
            value={'shape': (n_rows, n_cols)},
            metadata={'payoff_range_A': (float(payoff_A.min()), float(payoff_A.max())),
                     'payoff_range_B': (float(payoff_B.min()), float(payoff_B.max()))}
        ))
        
        # Add BATNA nodes
        self.cid.add_node(CIDNode(
            id="batna_1",
            type="information",
            label=f"BATNA P1: {batna[0]:.2f}",
            player=1,
            value=batna[0]
        ))
        
        self.cid.add_node(CIDNode(
            id="batna_2",
            type="information",
            label=f"BATNA P2: {batna[1]:.2f}",
            player=2,
            value=batna[1]
        ))
        
        # Add decision nodes for each player
        self.cid.add_node(CIDNode(
            id="decision_p1",
            type="decision",
            label="P1 Strategy",
            player=1,
            value=row_strat.tolist(),
            metadata={'strategy_type': 'pure' if np.max(row_strat) > 0.99 else 'mixed'}
        ))
        
        self.cid.add_node(CIDNode(
            id="decision_p2",
            type="decision",
            label="P2 Strategy",
            player=2,
            value=col_strat.tolist(),
            metadata={'strategy_type': 'pure' if np.max(col_strat) > 0.99 else 'mixed'}
        ))
        
        # Add common knowledge nodes
        self.cid.add_node(CIDNode(
            id="common_knowledge",
            type="information",
            label="Common Knowledge",
            value={'equilibrium_concept': 'Nash', 'selection_rule': metadata.get('rule', 'nsw')}
        ))
        
        # Add utility nodes
        u1 = float(row_strat @ payoff_A @ col_strat)
        u2 = float(row_strat @ payoff_B @ col_strat)
        
        self.cid.add_node(CIDNode(
            id="utility_p1",
            type="utility",
            label=f"U1: {u1:.2f}",
            player=1,
            value=u1,
            metadata={'surplus': u1 - batna[0]}
        ))
        
        self.cid.add_node(CIDNode(
            id="utility_p2",
            type="utility",
            label=f"U2: {u2:.2f}",
            player=2,
            value=u2,
            metadata={'surplus': u2 - batna[1]}
        ))
        
        # Add negotiation outcome node
        self.cid.add_node(CIDNode(
            id="outcome",
            type="chance",
            label="Negotiation Outcome",
            value={'utilities': (u1, u2), 'equilibrium_index': metadata.get('selected_index', 0)}
        ))
        
        # Add edges - Information flow
        self.cid.add_edge(CIDEdge("game_structure", "decision_p1", "information"))
        self.cid.add_edge(CIDEdge("game_structure", "decision_p2", "information"))
        self.cid.add_edge(CIDEdge("batna_1", "decision_p1", "information"))
        self.cid.add_edge(CIDEdge("batna_2", "decision_p2", "information"))
        self.cid.add_edge(CIDEdge("common_knowledge", "decision_p1", "information"))
        self.cid.add_edge(CIDEdge("common_knowledge", "decision_p2", "information"))
        
        # Causal edges - Decisions to outcome
        self.cid.add_edge(CIDEdge("decision_p1", "outcome", "causal", weight=0.5))
        self.cid.add_edge(CIDEdge("decision_p2", "outcome", "causal", weight=0.5))
        
        # Utility edges - Outcome to utilities
        self.cid.add_edge(CIDEdge("outcome", "utility_p1", "utility"))
        self.cid.add_edge(CIDEdge("outcome", "utility_p2", "utility"))
        
        # BATNA influence on utilities
        self.cid.add_edge(CIDEdge("batna_1", "utility_p1", "utility", weight=0.3))
        self.cid.add_edge(CIDEdge("batna_2", "utility_p2", "utility", weight=0.3))
        
        # Add metadata
        self.cid.metadata.update({
            'negotiation_type': 'bilateral',
            'equilibrium_type': 'Nash',
            'num_players': 2,
            'game_dimensions': (n_rows, n_cols)
        })
        
        if metadata:
            self.cid.metadata.update(metadata)
        
        return self.cid
    
    def add_fairness_nodes(
        self,
        cid: CausalInfluenceDiagram,
        fairness_metrics: Dict[str, float]
    ) -> None:
        """Add fairness-related nodes to the CID."""
        # Add fairness assessment node
        cid.add_node(CIDNode(
            id="fairness_assessment",
            type="information",
            label="Fairness Metrics",
            value=fairness_metrics,
            metadata={'primary_metric': 'nsw'}
        ))
        
        # Add social welfare node
        cid.add_node(CIDNode(
            id="social_welfare",
            type="utility",
            label=f"NSW: {fairness_metrics.get('nsw', 0):.2f}",
            value=fairness_metrics.get('nsw', 0)
        ))
        
        # Connect fairness to decisions
        cid.add_edge(CIDEdge("fairness_assessment", "decision_p1", "information", weight=0.2))
        cid.add_edge(CIDEdge("fairness_assessment", "decision_p2", "information", weight=0.2))
        
        # Connect utilities to social welfare
        cid.add_edge(CIDEdge("utility_p1", "social_welfare", "utility", weight=0.5))
        cid.add_edge(CIDEdge("utility_p2", "social_welfare", "utility", weight=0.5))
    
    def add_uncertainty_nodes(
        self,
        cid: CausalInfluenceDiagram,
        uncertainty_params: Dict[str, Any]
    ) -> None:
        """Add nodes representing uncertainty in the negotiation."""
        # Add implementation uncertainty
        cid.add_node(CIDNode(
            id="implementation_uncertainty",
            type="chance",
            label="Implementation Risk",
            value=uncertainty_params.get('implementation_risk', 0.1),
            metadata={'type': 'execution_noise'}
        ))
        
        # Add information uncertainty
        cid.add_node(CIDNode(
            id="information_uncertainty",
            type="chance",
            label="Information Quality",
            value=uncertainty_params.get('info_quality', 0.9),
            metadata={'type': 'observation_noise'}
        ))
        
        # Connect uncertainties
        cid.add_edge(CIDEdge("implementation_uncertainty", "outcome", "causal", weight=0.1))
        cid.add_edge(CIDEdge("information_uncertainty", "decision_p1", "information", weight=0.1))
        cid.add_edge(CIDEdge("information_uncertainty", "decision_p2", "information", weight=0.1))
    
    def add_temporal_structure(
        self,
        cid: CausalInfluenceDiagram,
        stages: List[str]
    ) -> None:
        """Add temporal structure to represent multi-stage negotiation."""
        previous_stage = None
        
        for i, stage in enumerate(stages):
            stage_id = f"stage_{i}_{stage}"
            
            cid.add_node(CIDNode(
                id=stage_id,
                type="information",
                label=f"Stage {i+1}: {stage}",
                value={'stage_index': i, 'stage_name': stage},
                metadata={'temporal_order': i}
            ))
            
            # Connect to decisions
            cid.add_edge(CIDEdge(stage_id, "decision_p1", "information", weight=0.1))
            cid.add_edge(CIDEdge(stage_id, "decision_p2", "information", weight=0.1))
            
            # Connect stages sequentially
            if previous_stage:
                cid.add_edge(CIDEdge(previous_stage, stage_id, "causal", weight=1.0))
            
            previous_stage = stage_id

def create_enhanced_cid(
    payoff_A: np.ndarray,
    payoff_B: np.ndarray,
    equilibrium: Tuple[np.ndarray, np.ndarray],
    batna: Tuple[float, float],
    fairness_metrics: Dict[str, float],
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create an enhanced CID with all features.
    
    Args:
        payoff_A, payoff_B: Payoff matrices
        equilibrium: Selected equilibrium
        batna: BATNA values
        fairness_metrics: Computed fairness metrics
        metadata: Additional metadata
    
    Returns:
        Dictionary containing CID data and visualizations
    """
    generator = NegotiationCIDGenerator()
    
    # Generate base CID
    cid = generator.generate_negotiation_cid(
        payoff_A, payoff_B, equilibrium, batna, metadata
    )
    
    # Add fairness nodes
    generator.add_fairness_nodes(cid, fairness_metrics)
    
    # Add uncertainty if specified
    if metadata and 'uncertainty' in metadata:
        generator.add_uncertainty_nodes(cid, metadata['uncertainty'])
    
    # Add temporal structure if multi-stage
    if metadata and 'stages' in metadata:
        generator.add_temporal_structure(cid, metadata['stages'])
    
    # Analyze strategic relevance
    strategic_analysis = cid.analyze_strategic_relevance()
    
    return {
        'cid': cid.to_dict(),
        'graphviz': cid.to_graphviz(),
        'strategic_analysis': strategic_analysis,
        'metadata': {
            'num_nodes': len(cid.nodes),
            'num_edges': len(cid.edges),
            'has_uncertainty': 'implementation_uncertainty' in cid.nodes,
            'has_fairness': 'fairness_assessment' in cid.nodes,
            'timestamp': datetime.utcnow().isoformat()
        }
    }