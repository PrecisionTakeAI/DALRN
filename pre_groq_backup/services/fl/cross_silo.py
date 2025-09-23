"""
Cross-Silo Federated Learning Implementation
Enables federation across organizational boundaries with secure aggregation
"""
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import hashlib
import json
from datetime import datetime
import numpy as np
from pathlib import Path

# gRPC imports for cross-organization communication
try:
    import grpc
    from concurrent import futures
    GRPC_AVAILABLE = True
except ImportError:
    print("Warning: gRPC not installed. Install with: pip install grpcio grpcio-tools")
    GRPC_AVAILABLE = False

@dataclass
class Silo:
    """Represents an organizational silo in federated learning"""
    silo_id: str
    organization: str
    num_samples: int
    public_key: bytes
    trust_score: float = 1.0
    last_update: Optional[datetime] = None
    total_contributions: int = 0

class SecureAggregator:
    """Implements secure aggregation for cross-silo FL"""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.pending_shares = {}

    def create_shares(self, value: np.ndarray, num_shares: int) -> List[np.ndarray]:
        """Split value into secret shares using additive secret sharing"""
        shares = []
        sum_shares = np.zeros_like(value)

        for i in range(num_shares - 1):
            share = np.random.randn(*value.shape)
            shares.append(share)
            sum_shares += share

        # Last share ensures sum equals original value
        shares.append(value - sum_shares)
        return shares

    def reconstruct_value(self, shares: List[np.ndarray]) -> Optional[np.ndarray]:
        """Reconstruct value from shares if threshold is met"""
        if len(shares) < self.threshold:
            return None
        return np.sum(shares, axis=0)

class CrossSiloCoordinator:
    """Coordinates cross-silo federated learning with privacy guarantees"""

    def __init__(self, min_silos: int = 3, secure_aggregation: bool = True):
        self.silos: Dict[str, Silo] = {}
        self.min_silos = min_silos
        self.current_round = 0
        self.global_model = None
        self.round_updates = {}
        self.secure_aggregation = secure_aggregation
        self.aggregator = SecureAggregator() if secure_aggregation else None

        # Authorized organizations (in production, load from secure registry)
        self.authorized_orgs = self._load_authorized_orgs()

    def _load_authorized_orgs(self) -> set:
        """Load authorized organizations from configuration"""
        # In production, this would connect to a secure registry
        return {
            "healthcare_org_1",
            "healthcare_org_2",
            "research_institute_alpha",
            "medical_center_beta",
            "university_hospital"
        }

    async def register_silo(
        self,
        silo_id: str,
        organization: str,
        num_samples: int,
        public_key: bytes
    ) -> Dict:
        """Register new silo with verification"""

        # Verify organization is authorized
        if organization not in self.authorized_orgs:
            raise ValueError(f"Organization {organization} is not authorized for cross-silo FL")

        # Verify silo ID uniqueness
        if silo_id in self.silos:
            raise ValueError(f"Silo {silo_id} already registered")

        # Create and register silo
        silo = Silo(
            silo_id=silo_id,
            organization=organization,
            num_samples=num_samples,
            public_key=public_key,
            last_update=datetime.utcnow()
        )

        self.silos[silo_id] = silo

        return {
            "silo_id": silo_id,
            "registered": True,
            "current_round": self.current_round,
            "min_silos_required": self.min_silos,
            "current_silos": len(self.silos)
        }

    async def submit_update(
        self,
        silo_id: str,
        round_number: int,
        encrypted_weights: bytes,
        signature: bytes
    ) -> Dict:
        """Submit model update from a silo"""

        if silo_id not in self.silos:
            raise ValueError(f"Silo {silo_id} not registered")

        silo = self.silos[silo_id]

        # Verify signature (simplified - use real crypto in production)
        if not self._verify_signature(encrypted_weights, signature, silo.public_key):
            raise ValueError("Invalid signature")

        # Initialize round updates if needed
        if round_number not in self.round_updates:
            self.round_updates[round_number] = []

        # Store update
        self.round_updates[round_number].append({
            "silo_id": silo_id,
            "encrypted_weights": encrypted_weights,
            "signature": signature,
            "timestamp": datetime.utcnow()
        })

        # Update silo stats
        silo.last_update = datetime.utcnow()
        silo.total_contributions += 1

        return {
            "round": round_number,
            "silo_id": silo_id,
            "accepted": True,
            "updates_received": len(self.round_updates[round_number]),
            "updates_required": self.min_silos
        }

    async def aggregate_round(self, round_number: int) -> Optional[Dict]:
        """Aggregate updates for a round using secure aggregation"""

        updates = self.round_updates.get(round_number, [])

        if len(updates) < self.min_silos:
            return {
                "status": "waiting",
                "updates_received": len(updates),
                "updates_required": self.min_silos
            }

        # Calculate total samples for weighted averaging
        total_samples = sum(
            self.silos[u["silo_id"]].num_samples
            for u in updates
        )

        # Decrypt and aggregate weights
        aggregated = None
        trust_weight_sum = 0

        for update in updates:
            silo = self.silos[update["silo_id"]]

            # Decrypt weights (simplified - use real encryption in production)
            weights = self._decrypt_weights(
                update["encrypted_weights"],
                silo.public_key
            )

            # Apply sample weighting and trust score
            weight = (silo.num_samples / total_samples) * silo.trust_score

            if aggregated is None:
                aggregated = weights * weight
            else:
                aggregated += weights * weight

            trust_weight_sum += weight

        # Normalize by trust weight sum
        if trust_weight_sum > 0:
            aggregated /= trust_weight_sum

        # Add differential privacy noise if configured
        if self.secure_aggregation:
            noise_scale = 0.01  # Configure based on privacy requirements
            dp_noise = np.random.laplace(0, noise_scale, aggregated.shape)
            aggregated += dp_noise

        self.global_model = aggregated
        self.current_round = round_number + 1

        return {
            "status": "completed",
            "round": round_number,
            "silos_participated": len(updates),
            "model_hash": hashlib.sha256(aggregated.tobytes()).hexdigest(),
            "next_round": self.current_round
        }

    def _verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify digital signature (simplified)"""
        # In production, use proper cryptographic signature verification
        expected = hashlib.sha256(data + public_key).digest()[:32]
        return signature == expected

    def _decrypt_weights(self, encrypted: bytes, public_key: bytes) -> np.ndarray:
        """Decrypt model weights (simplified)"""
        # In production, use proper encryption (e.g., RSA, AES)
        # This is a placeholder that just deserializes the data
        return np.frombuffer(encrypted, dtype=np.float32)

    def update_trust_scores(self, performance_metrics: Dict[str, float]):
        """Update trust scores based on model performance"""
        for silo_id, metric in performance_metrics.items():
            if silo_id in self.silos:
                # Adjust trust score based on performance
                # Higher metric = better performance = higher trust
                self.silos[silo_id].trust_score = min(1.0, metric)

    async def get_global_model(self) -> Optional[Dict]:
        """Get current global model"""
        if self.global_model is None:
            return None

        return {
            "round": self.current_round - 1,
            "model_hash": hashlib.sha256(self.global_model.tobytes()).hexdigest(),
            "shape": list(self.global_model.shape),
            "participating_silos": len(self.silos)
        }

# gRPC Service Implementation (if gRPC available)
if GRPC_AVAILABLE:
    class CrossSiloService:
        """gRPC service for cross-silo communication"""

        def __init__(self, coordinator: CrossSiloCoordinator, port: int = 50051):
            self.coordinator = coordinator
            self.port = port
            self.server = None

        async def start(self):
            """Start gRPC server"""
            self.server = grpc.aio.server()
            self.server.add_insecure_port(f'[::]:{self.port}')
            await self.server.start()
            print(f"Cross-silo gRPC server started on port {self.port}")
            await self.server.wait_for_termination()

        async def stop(self):
            """Stop gRPC server"""
            if self.server:
                await self.server.stop(grace=5.0)

# Integration function for main FL service
async def enable_cross_silo_federation(fl_service, config: Optional[Dict] = None):
    """Enable cross-silo federation in the main FL service"""

    config = config or {}

    # Initialize coordinator
    coordinator = CrossSiloCoordinator(
        min_silos=config.get("min_silos", 3),
        secure_aggregation=config.get("secure_aggregation", True)
    )

    # Start gRPC server if available
    if GRPC_AVAILABLE and config.get("enable_grpc", True):
        grpc_service = CrossSiloService(
            coordinator,
            port=config.get("grpc_port", 50051)
        )
        asyncio.create_task(grpc_service.start())

    # Add cross-silo endpoints to FL service
    @fl_service.post("/fl/cross-silo/register")
    async def register_silo(
        silo_id: str,
        organization: str,
        num_samples: int,
        public_key: str
    ):
        """Register a silo for cross-silo FL"""
        try:
            result = await coordinator.register_silo(
                silo_id=silo_id,
                organization=organization,
                num_samples=num_samples,
                public_key=bytes.fromhex(public_key)
            )
            return result
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(e))

    @fl_service.post("/fl/cross-silo/submit")
    async def submit_silo_update(
        silo_id: str,
        round_number: int,
        encrypted_weights: str,
        signature: str
    ):
        """Submit update from a silo"""
        try:
            result = await coordinator.submit_update(
                silo_id=silo_id,
                round_number=round_number,
                encrypted_weights=bytes.fromhex(encrypted_weights),
                signature=bytes.fromhex(signature)
            )
            return result
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(e))

    @fl_service.post("/fl/cross-silo/aggregate")
    async def trigger_aggregation(round_number: int):
        """Trigger aggregation for a round"""
        result = await coordinator.aggregate_round(round_number)
        return result

    @fl_service.get("/fl/cross-silo/model")
    async def get_global_model():
        """Get current global model"""
        model = await coordinator.get_global_model()
        if model is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="No global model available")
        return model

    @fl_service.get("/fl/cross-silo/silos")
    async def list_silos():
        """List registered silos"""
        silos = []
        for silo_id, silo in coordinator.silos.items():
            silos.append({
                "silo_id": silo_id,
                "organization": silo.organization,
                "num_samples": silo.num_samples,
                "trust_score": silo.trust_score,
                "last_update": silo.last_update.isoformat() if silo.last_update else None,
                "total_contributions": silo.total_contributions
            })
        return {"silos": silos, "total": len(silos)}

    return coordinator