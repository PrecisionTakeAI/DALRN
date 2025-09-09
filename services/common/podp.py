import json, uuid
from typing import List, Optional
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

    @staticmethod
    def new_id(prefix="r_"):
        return prefix + uuid.uuid4().hex[:8]

    def finalize(self):
        canon = json.dumps({
            "receipt_id": self.receipt_id,
            "dispute_id": self.dispute_id,
            "step": self.step,
            "inputs": self.inputs,
            "params": self.params,
            "artifacts": self.artifacts,
            "ts": self.ts,
        }, **KANON).encode()
        self.hashes = {
            "inputs_hash": keccak(json.dumps(self.inputs, **KANON).encode()),
            "outputs_hash": keccak(canon),
        }
        return self

class ReceiptChain(BaseModel):
    dispute_id: str
    receipts: List[Receipt]
    algo: str = "keccak256"
    merkle_root: Optional[str] = None
    merkle_leaves: Optional[List[str]] = None

    def finalize(self):
        leaves: List[str] = []
        for r in self.receipts:
            canon = json.dumps(r.model_dump(exclude_none=True), **KANON).encode()
            leaves.append(keccak(canon))
        nodes = leaves[:] or [keccak(b"")]
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            nxt = []
            for i in range(0, len(nodes), 2):
                pair = bytes.fromhex(nodes[i][2:]) + bytes.fromhex(nodes[i+1][2:])
                nxt.append(keccak(pair))
            nodes = nxt
        self.merkle_root = nodes[0]
        self.merkle_leaves = leaves
        return self
