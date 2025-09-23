"""
Proof-of-Deterministic-Processing (PoDP) module for DALRN.
Provides cryptographic receipts for FHE and other deterministic operations.
"""

import json, uuid, hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

KANON = dict(sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def keccak(data: bytes) -> str:
    from eth_hash.auto import keccak as _keccak
    return "0x" + _keccak(data).hex()

class Receipt(BaseModel):
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dispute_id: str
    step: str
    inputs: dict = {}
    params: dict = {}
    artifacts: dict = {}
    hashes: dict = {}
    signatures: list = []
    ts: str
    hash: Optional[str] = None

    @staticmethod
    def new_id(prefix="r_"):
        return prefix + uuid.uuid4().hex[:8]

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of receipt content."""
        content = {
            "receipt_id": self.receipt_id,
            "dispute_id": self.dispute_id,
            "step": self.step,
            "inputs": self.inputs,
            "params": self.params,
            "artifacts": self.artifacts,
            "ts": self.ts
        }
        serialized = json.dumps(content, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def finalize(self):
        canon = json.dumps({
            "receipt_id": self.receipt_id,
            "dispute_id": self.dispute_id,
            "step": self.step,
            "inputs": self.inputs,
            "params": self.params,
            "artifacts": self.artifacts,
            "ts": self.ts,
        }, **KANON)
        self.hashes["canon"] = keccak(canon.encode())
        # Also compute SHA-256 hash for FHE compatibility
        self.hash = self.compute_hash()
        return self

class ReceiptChain(BaseModel):
    dispute_id: str
    receipts: List[Receipt] = []
    merkle_root: Optional[str] = None
    merkle_leaves: List[str] = []

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

    def build_merkle_tree(self):
        leaves = [r.hashes.get("canon", r.compute_hash()) for r in self.receipts]
        if not leaves:
            self.merkle_root = keccak(b"empty")
            return self
        
        nodes = leaves[:]
        while len(nodes) > 1:
            nxt = []
            for i in range(0, len(nodes), 2):
                if i + 1 >= len(nodes):
                    nxt.append(nodes[i])
                    continue
                pair = bytes.fromhex(nodes[i][2:]) + bytes.fromhex(nodes[i+1][2:])
                nxt.append(keccak(pair))
            nodes = nxt
        self.merkle_root = nodes[0]
        self.merkle_leaves = leaves
        return self

    def finalize(self):
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

