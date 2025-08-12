"""
Proof-of-Deterministic-Processing (PoDP) module for DALRN.
Provides cryptographic receipts for FHE and other deterministic operations.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Receipt:
    """Individual receipt for a single operation."""
    
    dispute_id: str
    step: str
    inputs: Dict[str, Any]
    params: Dict[str, Any]
    artifacts: Dict[str, Any]
    ts: str
    hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of receipt content."""
        content = {
            "dispute_id": self.dispute_id,
            "step": self.step,
            "inputs": self.inputs,
            "params": self.params,
            "artifacts": self.artifacts,
            "ts": self.ts
        }
        serialized = json.dumps(content, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def finalize(self) -> "Receipt":
        """Compute and set the hash for this receipt."""
        self.hash = self.compute_hash()
        return self


@dataclass
class ReceiptChain:
    """Chain of receipts with Merkle root computation."""
    
    dispute_id: str
    receipts: List[Receipt] = field(default_factory=list)
    merkle_root: Optional[str] = None
    
    def compute_merkle_root(self) -> str:
        """Compute Merkle root of all receipts."""
        if not self.receipts:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Get all receipt hashes
        hashes = [r.hash or r.compute_hash() for r in self.receipts]
        
        # Build Merkle tree
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]  # Duplicate last if odd
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level
        
        return hashes[0]
    
    def finalize(self) -> "ReceiptChain":
        """Compute and set the Merkle root."""
        self.merkle_root = self.compute_merkle_root()
        return self
    
    def add_receipt(self, receipt: Receipt) -> None:
        """Add a receipt to the chain."""
        if not receipt.hash:
            receipt.finalize()
        self.receipts.append(receipt)
        self.merkle_root = None  # Invalidate cached root


def create_fhe_receipt(
    operation_id: str,
    tenant_id: str,
    operation_type: str,
    metadata: Dict[str, Any]
) -> Receipt:
    """Create a receipt for an FHE operation."""
    return Receipt(
        dispute_id=operation_id,
        step="FHE_DOT_V1",
        inputs={
            "tenant_id": tenant_id,
            "operation_type": operation_type
        },
        params=metadata.get("params", {}),
        artifacts={
            "computation_metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        },
        ts=datetime.utcnow().isoformat() + "Z"
    ).finalize()