"""
Self-Organizing Agent Networks (SOAN) for DALRN

This module implements distributed agent coordination with:
- Watts-Strogatz network topology
- GNN-based latency prediction
- M/M/1 queue modeling
- ε-greedy network rewiring
- Full PoDP compliance and ε-ledger budget tracking
"""

from .topology import WattsStrogatzNetwork
from .gnn_predictor import GNNLatencyPredictor
from .queue_model import MM1QueueModel
from .rewiring import EpsilonGreedyRewiring
from .orchestrator import SOANOrchestrator

__all__ = [
    'WattsStrogatzNetwork',
    'GNNLatencyPredictor',
    'MM1QueueModel',
    'EpsilonGreedyRewiring',
    'SOANOrchestrator'
]

__version__ = '1.0.0'