"""
GNN-based Latency Predictor using Graph Convolutional Networks

Implements 2-layer GCN for predicting edge latencies in the agent network
with full PoDP compliance and ε-ledger budget tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import hashlib
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GNNPoDPReceipt:
    """PoDP receipt for GNN operations"""
    operation: str
    timestamp: float
    model_id: str
    epsilon_used: float
    input_hash: str
    output_hash: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'timestamp': self.timestamp,
            'model_id': self.model_id,
            'epsilon_used': self.epsilon_used,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'metadata': self.metadata
        }


class GCNLatencyModel(nn.Module):
    """
    2-layer Graph Convolutional Network for latency prediction
    
    Architecture:
    - Input: Node features (4 dimensions)
    - Hidden: 16 dimensions with ReLU activation
    - Output: Edge latency predictions
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 1):
        super(GCNLatencyModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x, edge_index):
        """Forward pass through GCN layers"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GNNLatencyPredictor:
    """
    GNN-based latency predictor with PoDP instrumentation
    
    ε-ledger budget allocation:
    - Model initialization: 0.001ε
    - Training epoch: 0.0002ε per epoch
    - Prediction: 0.0001ε per prediction
    - Feature extraction: 0.0001ε
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        epsilon_budget: float = 0.015
    ):
        """
        Initialize GNN latency predictor
        
        Args:
            input_dim: Number of input features per node
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for Adam optimizer
            epsilon_budget: ε-ledger budget for GNN operations
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epsilon_budget = epsilon_budget
        self.epsilon_used = 0.0
        
        # Initialize model
        self.model = GCNLatencyModel(input_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.receipts: List[GNNPoDPReceipt] = []
        
        # Generate initialization receipt
        self._generate_receipt(
            operation="initialize_gnn",
            input_data={'input_dim': input_dim, 'hidden_dim': hidden_dim},
            output_data={'model_params': sum(p.numel() for p in self.model.parameters())},
            epsilon_cost=0.001
        )
        
        logger.info(f"Initialized GNN predictor: input={input_dim}, hidden={hidden_dim}")
    
    def _generate_receipt(
        self,
        operation: str,
        input_data: any,
        output_data: any,
        epsilon_cost: float
    ) -> GNNPoDPReceipt:
        """Generate PoDP receipt for GNN operation"""
        input_hash = hashlib.sha256(
            json.dumps(str(input_data)).encode()
        ).hexdigest()
        output_hash = hashlib.sha256(
            json.dumps(str(output_data)).encode()
        ).hexdigest()
        
        receipt = GNNPoDPReceipt(
            operation=operation,
            timestamp=time.time(),
            model_id=f"gnn_predictor_{id(self)}",
            epsilon_used=epsilon_cost,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'lr': self.learning_rate
            }
        )
        
        self.receipts.append(receipt)
        self.epsilon_used += epsilon_cost
        
        if self.epsilon_used > self.epsilon_budget:
            raise ValueError(f"ε-ledger budget exceeded: {self.epsilon_used:.4f} > {self.epsilon_budget:.4f}")
        
        return receipt
    
    def prepare_data(
        self,
        adjacency_matrix: np.ndarray,
        node_features: np.ndarray,
        edge_latencies: Optional[np.ndarray] = None
    ) -> Data:
        """
        Prepare graph data for PyTorch Geometric
        
        Args:
            adjacency_matrix: Network adjacency matrix
            node_features: Node feature matrix (N x 4)
            edge_latencies: Optional edge latency labels for training
            
        Returns:
            PyTorch Geometric Data object
        """
        # Convert adjacency to edge index format
        edge_index = []
        edge_attr = []
        
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] > 0:
                    edge_index.append([i, j])
                    if edge_latencies is not None:
                        edge_attr.append(edge_latencies[i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index)
        
        if edge_latencies is not None:
            data.y = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        
        self._generate_receipt(
            operation="prepare_data",
            input_data={'nodes': adjacency_matrix.shape[0], 'edges': len(edge_index[0])},
            output_data={'data_ready': True},
            epsilon_cost=0.0001
        )
        
        return data
    
    def train(
        self,
        train_data: Data,
        val_data: Optional[Data] = None,
        epochs: int = 50
    ) -> Dict[str, List[float]]:
        """
        Train the GNN model
        
        Args:
            train_data: Training data
            val_data: Optional validation data
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training history
        """
        start_time = time.time()
        self.model.train()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(train_data.x, train_data.edge_index)
            
            # Compute loss (predict latency for each node, aggregate for edges)
            if hasattr(train_data, 'y'):
                # For edge prediction, aggregate node embeddings
                edge_predictions = self._aggregate_edge_predictions(
                    out, train_data.edge_index
                )
                loss = self.criterion(edge_predictions, train_data.y)
            else:
                # Node-level prediction
                loss = self.criterion(out, torch.zeros_like(out))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.training_losses.append(loss.item())
            
            # Validation
            if val_data is not None:
                val_loss = self._validate(val_data)
                self.validation_losses.append(val_loss)
            
            # PoDP receipt per epoch
            self._generate_receipt(
                operation=f"train_epoch_{epoch}",
                input_data={'epoch': epoch, 'samples': len(train_data.x)},
                output_data={'loss': loss.item()},
                epsilon_cost=0.0002
            )
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                          f"Time: {time.time() - epoch_start:.3f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'training_time': total_time
        }
    
    def _aggregate_edge_predictions(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate node embeddings to predict edge latencies
        
        Uses concatenation and transformation of source/target embeddings
        """
        source_embeds = node_embeddings[edge_index[0]]
        target_embeds = node_embeddings[edge_index[1]]
        
        # Simple aggregation: mean of source and target
        edge_predictions = (source_embeds + target_embeds) / 2
        
        return edge_predictions
    
    def _validate(self, val_data: Data) -> float:
        """Validate model on validation data"""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(val_data.x, val_data.edge_index)
            
            if hasattr(val_data, 'y'):
                edge_predictions = self._aggregate_edge_predictions(
                    out, val_data.edge_index
                )
                loss = self.criterion(edge_predictions, val_data.y)
            else:
                loss = self.criterion(out, torch.zeros_like(out))
        
        self.model.train()
        return loss.item()
    
    def predict(
        self,
        data: Data,
        return_edge_latencies: bool = True
    ) -> np.ndarray:
        """
        Predict latencies for given graph
        
        Args:
            data: Graph data
            return_edge_latencies: If True, return edge latencies; else node embeddings
            
        Returns:
            Predicted latencies or embeddings
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            
            if return_edge_latencies:
                predictions = self._aggregate_edge_predictions(out, data.edge_index)
                predictions = predictions.numpy()
            else:
                predictions = out.numpy()
        
        self._generate_receipt(
            operation="predict",
            input_data={'nodes': len(data.x), 'edges': len(data.edge_index[0])},
            output_data={'predictions_shape': predictions.shape},
            epsilon_cost=0.0001
        )
        
        return predictions
    
    def predict_edge_latency(
        self,
        adjacency_matrix: np.ndarray,
        node_features: np.ndarray
    ) -> np.ndarray:
        """
        Predict latency matrix for entire network
        
        Returns:
            N x N matrix with predicted latencies (0 for non-edges)
        """
        data = self.prepare_data(adjacency_matrix, node_features)
        edge_predictions = self.predict(data, return_edge_latencies=True)
        
        # Reconstruct full latency matrix
        n_nodes = adjacency_matrix.shape[0]
        latency_matrix = np.zeros((n_nodes, n_nodes))
        
        edge_index = data.edge_index.numpy()
        for idx, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
            latency_matrix[i, j] = max(0.1, edge_predictions[idx, 0])  # Minimum latency 0.1
        
        return latency_matrix
    
    def save_model(self, path: str):
        """Save model checkpoint with PoDP receipt"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'epsilon_used': self.epsilon_used
        }, path)
        
        self._generate_receipt(
            operation="save_model",
            input_data={'path': path},
            output_data={'saved': True},
            epsilon_cost=0.0001
        )
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        
        self._generate_receipt(
            operation="load_model",
            input_data={'path': path},
            output_data={'loaded': True},
            epsilon_cost=0.0001
        )
        
        logger.info(f"Model loaded from {path}")
    
    def get_podp_summary(self) -> Dict:
        """Get PoDP compliance summary"""
        return {
            'total_receipts': len(self.receipts),
            'epsilon_used': self.epsilon_used,
            'epsilon_budget': self.epsilon_budget,
            'epsilon_remaining': self.epsilon_budget - self.epsilon_used,
            'operations': [r.operation for r in self.receipts[-10:]]  # Last 10 operations
        }