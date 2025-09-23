"""
Graph Neural Network Latency Predictor for SOAN.

This module implements a 2-layer GCN for predicting node latencies
with full PoDP compliance and ε-ledger budget tracking.
"""

import json
import hashlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Import PoDP utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.common.podp import Receipt, ReceiptChain, keccak


@dataclass
class PredictionMetrics:
    """Metrics for GNN prediction evaluation."""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    max_error: float
    inference_time_ms: float
    epsilon_used: float


class GNNLatencyPredictor(nn.Module):
    """
    2-layer Graph Convolutional Network for latency prediction.

    Architecture:
    - Input: Node features (queue_length, service_rate)
    - Hidden: 16 dimensions with ReLU activation
    - Output: Predicted latency per node
    """

    # ε-ledger budget allocations
    EPSILON_MODEL_INIT = 0.0005
    EPSILON_TRAINING_STEP = 0.001
    EPSILON_PREDICTION = 0.0002

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 1):
        """
        Initialize GNN model with specified architecture.

        Args:
            input_dim: Number of input features per node (default 2)
            hidden_dim: Number of hidden dimensions (default 16)
            output_dim: Number of output features per node (default 1)
        """
        super(GNNLatencyPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

        # Initialize receipt chain
        self.receipt_chain = ReceiptChain(
            dispute_id=f"gnn_predictor_{int(time.time())}"
        )
        self.epsilon_spent = 0.0

        # Track model state
        self.is_trained = False
        self.training_epochs = 0

        # Create initialization receipt
        self._create_init_receipt()

    def _create_init_receipt(self):
        """Create PoDP receipt for model initialization."""
        receipt = Receipt(
            receipt_id=f"gnn_init_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="model_initialization",
            inputs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim
            },
            params={
                "architecture": "2-layer-GCN",
                "activation": "ReLU",
                "dropout": 0.1
            },
            artifacts={
                "parameter_count": sum(p.numel() for p in self.parameters()),
                "epsilon_used": self.EPSILON_MODEL_INIT
            },
            hashes={
                "model_hash": keccak(str(self.state_dict()).encode())
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        self.receipt_chain.add_receipt(receipt)
        self.epsilon_spent += self.EPSILON_MODEL_INIT

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GCN.

        Args:
            x: Node feature tensor [num_nodes, input_dim]
            edge_index: Edge connectivity tensor [2, num_edges]

        Returns:
            Predicted latencies [num_nodes, 1]
        """
        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer (output)
        x = self.conv2(x, edge_index)

        return x

    def predict(
        self,
        graph: nx.Graph,
        node_features: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Receipt]:
        """
        Predict latencies for all nodes in the graph with PoDP receipt.

        Args:
            graph: NetworkX graph
            node_features: Optional custom node features [num_nodes, input_dim]

        Returns:
            Tuple of (predicted latencies, PoDP receipt)
        """
        start_time = time.time()

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"gnn_predict_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="prediction_entry",
            inputs={
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges()
            },
            params={
                "model_trained": self.is_trained,
                "training_epochs": self.training_epochs
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Prepare node features
        if node_features is None:
            node_features = []
            for node in sorted(graph.nodes()):
                attrs = graph.nodes[node]
                features = [
                    attrs.get('queue_length', 0),
                    attrs.get('service_rate', 1.5)
                ]
                node_features.append(features)
            node_features = np.array(node_features, dtype=np.float32)

        # Convert to PyTorch Geometric format
        x = torch.tensor(node_features, dtype=torch.float32)

        # Create edge index
        edge_list = list(graph.edges())
        edge_index = torch.tensor(
            [[e[0] for e in edge_list] + [e[1] for e in edge_list],
             [e[1] for e in edge_list] + [e[0] for e in edge_list]],
            dtype=torch.long
        )

        # Perform prediction
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, edge_index)
            predictions = predictions.numpy().flatten()

        # Apply post-processing (ensure non-negative latencies)
        predictions = np.maximum(predictions, 0)

        # Create exit receipt
        inference_time = (time.time() - start_time) * 1000
        exit_receipt = Receipt(
            receipt_id=f"gnn_predict_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="prediction_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "inference_time_ms": inference_time
            },
            artifacts={
                "predictions_shape": predictions.shape,
                "mean_prediction": float(np.mean(predictions)),
                "std_prediction": float(np.std(predictions)),
                "min_prediction": float(np.min(predictions)),
                "max_prediction": float(np.max(predictions)),
                "epsilon_used": self.EPSILON_PREDICTION,
                "epsilon_total": self.epsilon_spent + self.EPSILON_PREDICTION
            },
            hashes={
                "predictions_hash": keccak(json.dumps(predictions.tolist()).encode())
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Update epsilon budget
        self.epsilon_spent += self.EPSILON_PREDICTION

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        return predictions, exit_receipt

    def train_model(
        self,
        graph: nx.Graph,
        target_latencies: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01
    ) -> Tuple[List[float], Receipt]:
        """
        Train the GNN model with PoDP receipt generation.

        Args:
            graph: NetworkX graph
            target_latencies: Ground truth latencies for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer

        Returns:
            Tuple of (training losses, PoDP receipt)
        """
        start_time = time.time()

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"gnn_train_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="training_entry",
            inputs={
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "target_shape": target_latencies.shape
            },
            params={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer": "Adam"
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Prepare training data
        node_features = []
        for node in sorted(graph.nodes()):
            attrs = graph.nodes[node]
            features = [
                attrs.get('queue_length', 0),
                attrs.get('service_rate', 1.5)
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float32)
        y = torch.tensor(target_latencies, dtype=torch.float32).reshape(-1, 1)

        # Create edge index
        edge_list = list(graph.edges())
        edge_index = torch.tensor(
            [[e[0] for e in edge_list] + [e[1] for e in edge_list],
             [e[1] for e in edge_list] + [e[0] for e in edge_list]],
            dtype=torch.long
        )

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.train()
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.forward(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Update epsilon for each training step
            self.epsilon_spent += self.EPSILON_TRAINING_STEP

        self.is_trained = True
        self.training_epochs += epochs

        # Create exit receipt
        training_time = (time.time() - start_time) * 1000
        exit_receipt = Receipt(
            receipt_id=f"gnn_train_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="training_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "final_loss": losses[-1],
                "training_time_ms": training_time
            },
            artifacts={
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "loss_reduction": losses[0] - losses[-1],
                "epochs_completed": epochs,
                "epsilon_used": self.EPSILON_TRAINING_STEP * epochs,
                "epsilon_total": self.epsilon_spent
            },
            hashes={
                "model_hash": keccak(str(self.state_dict()).encode())
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        return losses, exit_receipt

    def evaluate(
        self,
        graph: nx.Graph,
        true_latencies: np.ndarray
    ) -> Tuple[PredictionMetrics, Receipt]:
        """
        Evaluate model performance with PoDP receipt.

        Args:
            graph: NetworkX graph
            true_latencies: Ground truth latencies

        Returns:
            Tuple of (PredictionMetrics, PoDP receipt)
        """
        start_time = time.time()

        # Get predictions
        predictions, pred_receipt = self.predict(graph)

        # Calculate metrics
        mae = np.mean(np.abs(predictions - true_latencies))
        mse = np.mean((predictions - true_latencies) ** 2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(predictions - true_latencies))
        inference_time = (time.time() - start_time) * 1000

        metrics = PredictionMetrics(
            mae=float(mae),
            mse=float(mse),
            rmse=float(rmse),
            max_error=float(max_error),
            inference_time_ms=inference_time,
            epsilon_used=self.EPSILON_PREDICTION
        )

        # Create evaluation receipt
        eval_receipt = Receipt(
            receipt_id=f"gnn_eval_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="model_evaluation",
            inputs={
                "num_samples": len(true_latencies),
                "prediction_receipt": pred_receipt.receipt_id
            },
            params={},
            artifacts={
                "mae": metrics.mae,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "max_error": metrics.max_error,
                "inference_time_ms": metrics.inference_time_ms
            },
            hashes={
                "metrics_hash": keccak(json.dumps(metrics.__dict__, sort_keys=True).encode())
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        self.receipt_chain.add_receipt(eval_receipt)

        return metrics, eval_receipt

    def get_receipt_chain(self) -> ReceiptChain:
        """Get the complete receipt chain for all operations."""
        return self.receipt_chain

    def get_epsilon_budget_status(self) -> Dict[str, float]:
        """Get current epsilon budget status."""
        return {
            "spent": self.epsilon_spent,
            "remaining": 4.0 - self.epsilon_spent,  # Total budget of 4.0
            "breakdown": {
                "initialization": self.EPSILON_MODEL_INIT,
                "training": self.EPSILON_TRAINING_STEP * self.training_epochs,
                "predictions": self.EPSILON_PREDICTION * len([r for r in self.receipt_chain.receipts if "predict" in r.step])
            }
        }