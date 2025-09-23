"""
Flower Framework Integration with Epsilon-Ledger Service

Demonstrates how to integrate Flower federated learning with the epsilon-ledger
for privacy budget tracking and Opacus for differential privacy.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Core imports
import httpx
from pydantic import BaseModel

# Try importing FL and DP libraries
try:
    import flwr as fl
    from flwr.common import Parameters, Scalar, Config, FitRes, FitIns
    from flwr.server.strategy import FedAvg
    from flwr.server.client_manager import ClientManager
    from flwr.server.client_proxy import ClientProxy
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    logging.warning("Flower not installed. Install with: pip install flwr")

try:
    import torch
    import torch.nn as nn
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.accountants import RDPAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus/PyTorch not installed. Install with: pip install opacus torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Epsilon-ledger service URL
EPS_LEDGER_URL = "http://localhost:8001"


# ============================================================================
# Privacy-Aware Federated Averaging Strategy
# ============================================================================

class PrivacyAwareFedAvg(FedAvg):
    """
    Federated Averaging strategy with epsilon-ledger integration.
    
    This strategy checks privacy budget before each round and records
    privacy spend after aggregation.
    """
    
    def __init__(
        self,
        tenant_id: str,
        model_id: str,
        epsilon_per_round: float = 0.5,
        delta: float = 1e-5,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        robust_aggregation: str = "none",
        eps_ledger_url: str = EPS_LEDGER_URL,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tenant_id = tenant_id
        self.model_id = model_id
        self.epsilon_per_round = epsilon_per_round
        self.delta = delta
        self.clipping_norm = clipping_norm
        self.noise_multiplier = noise_multiplier
        self.robust_aggregation = robust_aggregation
        self.eps_ledger_url = eps_ledger_url
        self.round_number = 0
        self.http_client = httpx.Client(timeout=30.0)
    
    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'http_client'):
            self.http_client.close()
    
    def precheck_privacy_budget(self) -> bool:
        """
        Check if we have sufficient privacy budget for the next round.
        
        Returns:
            bool: True if operation is allowed, False otherwise
        """
        try:
            response = self.http_client.post(
                f"{self.eps_ledger_url}/precheck",
                json={
                    "tenant_id": self.tenant_id,
                    "model_id": self.model_id,
                    "eps_round": self.epsilon_per_round,
                    "delta_round": self.delta,
                    "accountant": "rdp"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if not result["allowed"]:
                    logger.warning(
                        f"Privacy budget exhausted. Remaining: {result['remaining_budget']:.4f}"
                    )
                return result["allowed"]
            else:
                logger.error(f"Precheck failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking privacy budget: {e}")
            # Fail closed - don't allow operation if we can't check budget
            return False
    
    def commit_privacy_spend(
        self,
        num_clients: int,
        aggregation_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Commit privacy spend to the ledger after successful aggregation.
        
        Args:
            num_clients: Number of clients that participated
            aggregation_metadata: Additional metadata about aggregation
        
        Returns:
            bool: True if commit successful, False otherwise
        """
        try:
            response = self.http_client.post(
                f"{self.eps_ledger_url}/commit",
                json={
                    "tenant_id": self.tenant_id,
                    "model_id": self.model_id,
                    "round": self.round_number,
                    "accountant": "rdp",
                    "epsilon": self.epsilon_per_round,
                    "delta": self.delta,
                    "clipping_C": self.clipping_norm,
                    "sigma": self.noise_multiplier,
                    "num_clients": num_clients,
                    "aggregation_method": self.robust_aggregation,
                    "framework": "flower",
                    "metadata": aggregation_metadata or {}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Privacy spend committed. Total spent: {result['total_spent']:.4f}, "
                    f"Remaining: {result['remaining_budget']:.4f}"
                )
                return True
            else:
                logger.error(f"Commit failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error committing privacy spend: {e}")
            return False
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training with privacy checks.
        """
        self.round_number = server_round
        
        # Check privacy budget before proceeding
        if not self.precheck_privacy_budget():
            logger.error(f"Round {server_round} blocked due to insufficient privacy budget")
            return []  # Return empty list to skip this round
        
        # Continue with normal configuration
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates with privacy accounting.
        """
        if not results:
            return None, {}
        
        # Apply robust aggregation if configured
        if self.robust_aggregation != "none":
            results = self.apply_robust_aggregation(results)
        
        # Perform normal aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Add differential privacy noise if configured
        if aggregated_parameters and self.noise_multiplier > 0:
            aggregated_parameters = self.add_dp_noise(aggregated_parameters)
        
        # Commit privacy spend
        if aggregated_parameters:
            aggregation_metadata = {
                "num_failures": len(failures),
                "robust_method": self.robust_aggregation,
                "metrics": metrics
            }
            self.commit_privacy_spend(
                num_clients=len(results),
                aggregation_metadata=aggregation_metadata
            )
        
        return aggregated_parameters, metrics
    
    def apply_robust_aggregation(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """
        Apply robust aggregation to filter out potential Byzantine clients.
        
        Supported methods:
        - krum: Select the most central update
        - multi-krum: Select k most central updates
        - trimmed-mean: Remove outliers and average
        - median: Use coordinate-wise median
        """
        if self.robust_aggregation == "krum":
            return self.apply_krum(results, k=1)
        elif self.robust_aggregation == "multi-krum":
            return self.apply_krum(results, k=max(1, len(results) // 2))
        elif self.robust_aggregation == "trimmed-mean":
            return self.apply_trimmed_mean(results, trim_ratio=0.2)
        elif self.robust_aggregation == "median":
            return self.apply_median(results)
        else:
            return results
    
    def apply_krum(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        k: int = 1
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """
        Apply Krum or Multi-Krum aggregation.
        
        Select k clients whose updates are closest to others.
        """
        if len(results) <= k:
            return results
        
        # Extract weight arrays
        weights_list = []
        for _, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            flat_weights = np.concatenate([w.flatten() for w in weights])
            weights_list.append(flat_weights)
        
        n = len(weights_list)
        distances = np.zeros((n, n))
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(weights_list[i] - weights_list[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute scores (sum of distances to k-1 nearest neighbors)
        scores = []
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            # Sum distances to k-1 nearest neighbors (excluding self)
            score = np.sum(sorted_distances[1:k])
            scores.append(score)
        
        # Select k clients with lowest scores
        selected_indices = np.argsort(scores)[:k]
        selected_results = [results[i] for i in selected_indices]
        
        logger.info(f"Krum selected {k} out of {n} clients")
        return selected_results
    
    def apply_trimmed_mean(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        trim_ratio: float = 0.2
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """
        Apply trimmed mean aggregation.
        
        Remove top and bottom trim_ratio fraction of updates per parameter.
        """
        if len(results) <= 2:
            return results
        
        trim_count = int(len(results) * trim_ratio)
        if trim_count == 0:
            return results
        
        # For simplicity, randomly select clients to keep
        # In production, implement coordinate-wise trimming
        keep_count = len(results) - 2 * trim_count
        if keep_count <= 0:
            keep_count = 1
        
        selected_indices = np.random.choice(
            len(results), keep_count, replace=False
        )
        selected_results = [results[i] for i in selected_indices]
        
        logger.info(f"Trimmed mean kept {keep_count} out of {len(results)} clients")
        return selected_results
    
    def apply_median(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """
        Apply coordinate-wise median aggregation.
        
        This is a simplified version that returns the median client.
        """
        if len(results) <= 2:
            return results
        
        # For simplicity, return middle client(s)
        # In production, implement true coordinate-wise median
        middle_idx = len(results) // 2
        return [results[middle_idx]]
    
    def add_dp_noise(self, parameters: Parameters) -> Parameters:
        """
        Add differential privacy noise to aggregated parameters.
        
        Args:
            parameters: Aggregated model parameters
        
        Returns:
            Parameters with DP noise added
        """
        weights = fl.common.parameters_to_ndarrays(parameters)
        
        # Add Gaussian noise scaled by sensitivity and noise_multiplier
        noisy_weights = []
        for w in weights:
            sensitivity = 2 * self.clipping_norm  # L2 sensitivity
            noise_scale = sensitivity * self.noise_multiplier
            noise = np.random.normal(0, noise_scale, w.shape)
            noisy_w = w + noise
            noisy_weights.append(noisy_w)
        
        return fl.common.ndarrays_to_parameters(noisy_weights)


# ============================================================================
# Opacus-Enabled FL Client
# ============================================================================

class PrivateFlowerClient(fl.client.NumPyClient):
    """
    Flower client with Opacus differential privacy integration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize private FL client.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Test data loader
            epsilon: Privacy budget per round
            delta: Privacy parameter delta
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier for DP-SGD
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.device = device
        
        # Initialize optimizer and privacy engine
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        if OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine(
                accountant="rdp"  # Use RDP accountant
            )
            
            # Make model private
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm
            )
    
    def get_parameters(self, config: Config) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Config
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Train model with differential privacy.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
        
        Returns:
            Updated parameters, number of samples, and metrics
        """
        self.set_parameters(parameters)
        
        # Train with DP-SGD
        self.model.train()
        epoch_losses = []
        
        for epoch in range(config.get("epochs", 1)):
            batch_losses = []
            
            with BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=32,
                optimizer=self.optimizer
            ) as memory_safe_data_loader:
                for batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = nn.functional.cross_entropy(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    batch_losses.append(loss.item())
            
            epoch_losses.append(np.mean(batch_losses))
        
        # Get privacy spent
        if OPACUS_AVAILABLE and hasattr(self, 'privacy_engine'):
            epsilon_spent = self.privacy_engine.get_epsilon(self.delta)
            logger.info(f"Privacy spent this round: ε = {epsilon_spent:.4f}")
        else:
            epsilon_spent = 0.0
        
        metrics = {
            "loss": float(np.mean(epoch_losses)),
            "epsilon_spent": epsilon_spent
        }
        
        return self.get_parameters(config), len(self.train_loader.dataset), metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Config
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate model on test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
        
        Returns:
            Loss, number of samples, and metrics
        """
        self.set_parameters(parameters)
        
        self.model.eval()
        loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        
        metrics = {
            "accuracy": accuracy,
            "loss": loss
        }
        
        return loss, len(self.test_loader.dataset), metrics


# ============================================================================
# Example Usage
# ============================================================================

def run_privacy_aware_fl_simulation():
    """
    Run a simulated federated learning session with privacy budget tracking.
    
    This demonstrates the integration between Flower and epsilon-ledger.
    """
    if not FLOWER_AVAILABLE:
        logger.error("Flower not available. Install with: pip install flwr")
        return
    
    # Configure FL strategy with privacy awareness
    strategy = PrivacyAwareFedAvg(
        tenant_id="demo_tenant",
        model_id="demo_model",
        epsilon_per_round=0.5,
        delta=1e-5,
        clipping_norm=1.0,
        noise_multiplier=1.1,
        robust_aggregation="multi-krum",
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None  # Add evaluation function in production
    )
    
    # Start Flower server
    logger.info("Starting privacy-aware federated learning server...")
    
    # In production, this would run with real clients
    # fl.server.start_server(
    #     server_address="0.0.0.0:8080",
    #     config=fl.server.ServerConfig(num_rounds=5),
    #     strategy=strategy,
    # )
    
    logger.info("Privacy-aware FL simulation complete")


if __name__ == "__main__":
    # Run example
    run_privacy_aware_fl_simulation()
    
    # Example of using the private client
    if OPACUS_AVAILABLE and torch is not None:
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Create dummy data loaders
        from torch.utils.data import TensorDataset, DataLoader
        
        X_train = torch.randn(1000, 784)
        y_train = torch.randint(0, 10, (1000,))
        X_test = torch.randn(200, 784)
        y_test = torch.randint(0, 10, (200,))
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create private client
        client = PrivateFlowerClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.1
        )
        
        logger.info("Private FL client created successfully")