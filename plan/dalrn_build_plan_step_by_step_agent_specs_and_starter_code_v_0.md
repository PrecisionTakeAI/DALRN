# DALRN Build Plan — Step‑by‑Step, Agent Specs, and Starter Code (v0.1)

> Purpose: Give you and your Claude Code agents a clear, actionable path from **Day 0** to a Minimal Verifiable Product (MV\*P) for DALRN, with UAMP as the first app. This includes: phased roadmap, repo structure, service contracts, prompts for agents, acceptance criteria, and starter code.

---

## 0) Strategy: What to build first

- **Build DALRN Core first (thin slice)**: receipts (PoDP), anchor contracts, search service, encrypted-similarity service, negotiation engine, and a simple Gateway.
- **Build UAMP in parallel** as the first consumer app (web UI + workflows). UAMP should exercise the DALRN core through real flows but avoid bespoke logic in MVP.
- **MV\*P** emphasis: everything must be **verifiable** (PoDP receipts + ε‑ledger), not “feature‑rich.”

---

## 1) Phased roadmap (Day 0 → Day 45)

**Phase 0 (Days 0–7): Foundations & baselines**

1. Monorepo + CI skeleton; choose Python 3.11, Node 20, Solidity 0.8.24.
2. Stand up IPFS local node; Postgres for metadata; Prometheus + Grafana for observability.
3. Baseline FAISS GPU recall/latency on a 10k‑doc corpus (store embeddings plaintext for now).
4. TenSEAL demo for **encrypted dot‑product** on unit‑norm vectors; record parameters.
5. Negotiation engine demo (nashpy): 2‑player NE + NSW selection rule; produce explanation memo.

**Exit**: numbers logged; starter services run locally with Docker Compose.

**Phase 1 (Days 8–21): MV\*P pipeline**

1. Implement **PoDP instrumentation** in Gateway + per service; Merkle build; IPFS upload.
2. Deploy **AnchorReceipts** contract on a local L2 (Anvil/Hardhat); anchor roots from pipeline.
3. Build **Gateway** (FastAPI) with endpoints: `/submit-dispute`, `/status/{id}`, `/evidence`.
4. Implement **Search** (FAISS HNSW) service with optional reweighting flag (off by default).
5. Implement **FHE** (CKKS) service for encrypted cosine; client decrypts scores.
6. Implement **Negotiation** service (NE + bargaining fallback + memo output).

**Exit**: end‑to‑end demo → encrypted intake → search → negotiation → on‑chain anchor + UI status page.

**Phase 2 (Days 22–35): ε‑Ledger + FL**

1. ε‑ledger service (RDP accounting, budget enforcement); surface in `/status/{id}`.
2. FL orchestrator (Flower or NV FLARE) + secure aggregation shim; log ε entries per round.
3. Robust aggregation (median/trimmed mean) + participation signing.

**Exit**: federated round with 2–3 “firms” (simulated) producing ε‑ledger entries.

**Phase 3 (Days 36–45): Hardening & UX**

1. AuthN/Z, rate limits, quotas; key‑wrap strategy (crypto‑shredding hooks).
2. Grafana dashboards for latency, error budgets; audit views for PoDP and ε‑ledger.
3. A/B harness (FAISS vs FAISS+reweighting), plots in reports.

---

## 2) Monorepo layout

```
repo/
  services/
    gateway/                # FastAPI + PoDP middleware + status UI (minimal)
    search/                 # FAISS HNSW + optional reweighting
    fhe/                    # TenSEAL CKKS dot-product microservice
    negotiation/            # Nashpy NE + bargaining + memo
    fl/                     # Flower orchestrator + DP (Opacus) + ε-ledger client
    chain/                  # contracts + deployment scripts + abi
    common/                 # protobuf/JSON schemas, PoDP utils, Merkle, clients
  infra/
    docker-compose.yml
    k8s/                    # manifests (phase 2+)
    ipfs/
  docs/
    adr/                    # architecture decision records
  tests/
  Makefile
  pyproject.toml / package.json
```

---

## 3) Service contracts (gRPC/HTTP)

**Gateway (HTTP)**

- `POST /submit-dispute` `{parties[], jurisdiction, cid, enc_meta}` → `{dispute_id}`
- `POST /evidence` `{dispute_id, cid}` → `{evidence_id}`
- `GET  /status/{dispute_id}` → `{phase, receipts[], anchor_tx, eps_budget}`

**Search (gRPC)**

- `Search.Query(Query {dispute_id, query_vec|enc_query, k, reweight_iters})` → `TopK {ids[], scores[]}`

**FHE (gRPC)**

- `FHE.DotProduct(CKKSVec q, CKKSVec v)` → `Cipher` (client decrypts)
- `FHE.BatchedDot(CKKSVec q, CKKSMatrix V)` → `Cipher[]`

**Negotiation (HTTP)**

- `POST /negotiate` `{A, B, rule, batna}` → `{sr, sc, u1, u2, memo_cid}`

**ε‑Ledger (HTTP)**

- `POST /precheck` `{tenant_id, model_id, eps_round}` → `{allowed: bool, remaining}`
- `POST /commit` `{entry}` → `{ok: true}`

---

## 4) Claude Code agent specs (prompts you can drop in)

### 4.1 Orchestrator Agent

**Goal:** plan/coordinate tasks, create branches/PRs, enforce PoDP & ε‑ledger instrumentation. **System Prompt (condensed):**

- You are the DALRN Orchestrator. Every feature must emit PoDP receipts and respect ε‑ledger budgets. Create tasks, branches, tests, and PRs. Refuse pushes without tests. **Outputs:** task graph (YAML), PR descriptions, CI checks.

### 4.2 Gateway Agent

**Goal:** implement FastAPI app + PoDP middleware + status UI stub. **Acceptance:** `/submit-dispute`, `/status/{id}` functional; receipts written; Merkle root built; anchor called; tests with pytest. **Guardrails:** never log plaintext; redact PII.

### 4.3 Search Agent

**Goal:** FAISS HNSW index + gRPC service; optional reweighting flag. **Acceptance:** recall\@10 reported; P95 latency < 600ms on 10k doc baseline. **Guardrails:** reweighting **off by default**; A/B harness included.

### 4.4 FHE Agent

**Goal:** TenSEAL CKKS microservice for encrypted dot-products. **Acceptance:** unit tests validate parity with plaintext cosine within ±2% recall\@k. **Guardrails:** single context per tenant; client-side decrypt only.

### 4.5 Negotiation Agent

**Goal:** NE solver (nashpy) + bargaining + explanation memo. **Acceptance:** deterministic selection rule (NSW/egalitarian) applied; memo CID persisted; tests for multiple equilibria.

### 4.6 FL & Privacy Agent

**Goal:** integrate Flower (or NV FLARE) + DP (Opacus) + ε‑ledger client. **Acceptance:** pre‑round budget check; post‑round ledger commit; robust aggregation option.

### 4.7 Chain Agent

**Goal:** implement `AnchorReceipts` contract and deployment scripts. **Acceptance:** events emitted; `latestRoot` returns last; Foundry tests pass.

> **Tip:** Give each agent a **Feature Ticket** (below) with exact I/O, tests, and acceptance criteria.

---

## 5) Feature tickets (ready to paste into your tracker)

**TKT‑G1: Gateway + PoDP middleware**

- Implement FastAPI server with endpoints; generate receipts per step; build Merkle root; POST to IPFS; call `anchorRoot`.
- Tests: end‑to‑end pipeline with mock services; assert anchor event.

**TKT‑S1: Search service (FAISS)**

- Build HNSW index; gRPC; run baseline; implement reweighting flag and A/B harness.
- Tests: recall/latency benchmarks stored to JSON; CI publishes chart.

**TKT‑H1: FHE service (CKKS)**

- TenSEAL context; batched dot‑product; client decrypt; parity tests vs plaintext.

**TKT‑N1: Negotiation service**

- Lemke–Howson; bargaining; memo generation; CID persistence.

**TKT‑C1: Anchor contract + scripts**

- Implement interface; Foundry tests; deploy to local chain; env wiring in Gateway.

**TKT‑P1: ε‑ledger service**

- Precheck + commit; RDP accounting; UI exposure in `/status/{id}`.

---

## 6) Starter code (minimal, production‑grade skeletons)

### 6.1 Gateway (FastAPI) with PoDP middleware (Python)

```python
# services/gateway/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from common.podp import Receipt, ReceiptChain, build_merkle_root
from common.ipfs import put_json
from chain.client import AnchorClient

app = FastAPI()
anchor = AnchorClient()

class SubmitDispute(BaseModel):
    parties: list[str]
    jurisdiction: str
    cid: str  # IPFS bundle of encrypted docs
    enc_meta: dict

@app.post("/submit-dispute")
async def submit_dispute(body: SubmitDispute):
    dispute_id = Receipt.new_id(prefix="disp_")
    r1 = Receipt(
        dispute_id=dispute_id,
        step="INTAKE_V1",
        inputs={"cid_bundle": body.cid},
        params={"jurisdiction": body.jurisdiction},
        artifacts={},
        ts=datetime.utcnow().isoformat() + "Z",
    ).finalize()

    chain = ReceiptChain(dispute_id=dispute_id, receipts=[r1]).finalize()
    uri = put_json(chain.dict())
    anchor.anchor_root(dispute_id, chain.merkle_root, model_hash=b"\x00"*32, round=0, uri=uri, tags=[b"PoDP"])  # stub

    return {"dispute_id": dispute_id, "anchor_uri": uri}

@app.get("/status/{dispute_id}")
async def status(dispute_id: str):
    # TODO: load latest chain & ε-ledger state from DB and IPFS
    return {"dispute_id": dispute_id, "phase": "INTAKE", "receipts": []}
```

```python
# services/common/podp.py
import json, hashlib, uuid
from pydantic import BaseModel, Field
from typing import List

KANON = dict(sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def keccak(data: bytes) -> str:
    import eth_hash.auto as _
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
    merkle_root: str | None = None
    merkle_leaves: List[str] | None = None

    def finalize(self):
        leaves = []
        for r in self.receipts:
            canon = json.dumps(r.dict(exclude_none=True), **KANON).encode()
            leaves.append(keccak(canon))
        # simple pairwise Merkle (duplicate last if odd)
        nodes = leaves[:]
        if len(nodes) == 0:
            nodes = [keccak(b"")]
        while len(nodes) > 1:
            nxt = []
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            for i in range(0, len(nodes), 2):
                pair = bytes.fromhex(nodes[i][2:]) + bytes.fromhex(nodes[i+1][2:])
                nxt.append(keccak(pair))
            nodes = nxt
        self.merkle_root = nodes[0]
        self.merkle_leaves = leaves
        return self
```

### 6.2 Solidity: minimal `AnchorReceipts` implementation

```solidity
// services/chain/contracts/AnchorReceipts.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract AnchorReceipts {
    event RootAnchored(bytes32 indexed disputeId, bytes32 merkleRoot, bytes32 modelHash, uint256 round, string uri, bytes32[] tags);
    event ReceiptAnchored(bytes32 indexed disputeId, bytes32 receiptHash, uint256 stepIndex, string uri);

    mapping(bytes32 => bytes32) public latestRoots;

    function anchorRoot(bytes32 disputeId, bytes32 merkleRoot, bytes32 modelHash, uint256 round, string calldata uri, bytes32[] calldata tags) external returns (uint256) {
        latestRoots[disputeId] = merkleRoot;
        emit RootAnchored(disputeId, merkleRoot, modelHash, round, uri, tags);
        return block.number;
    }

    function anchorReceipt(bytes32 disputeId, bytes32 receiptHash, uint256 stepIndex, string calldata uri) external returns (uint256) {
        emit ReceiptAnchored(disputeId, receiptHash, stepIndex, uri);
        return block.number;
    }

    function latestRoot(bytes32 disputeId) external view returns (bytes32 merkleRoot, uint256 blockNumber) {
        return (latestRoots[disputeId], block.number);
    }
}
```

### 6.3 Search service (FAISS HNSW) skeleton

```python
# services/search/service.py
import faiss, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
index = faiss.IndexHNSWFlat(768, 32)  # d=768 example, M=32
faiss.normalize_L2

class BuildReq(BaseModel):
    embeddings: list[list[float]]

class QueryReq(BaseModel):
    query: list[float]
    k: int = 10
    reweight_iters: int = 0  # off by default

@app.post("/build")
def build(req: BuildReq):
    vecs = np.asarray(req.embeddings).astype('float32')
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return {"count": index.ntotal}

@app.post("/query")
def query(req: QueryReq):
    q = np.asarray([req.query]).astype('float32')
    faiss.normalize_L2(q)
    D, I = index.search(q, req.k)
    return {"ids": I[0].tolist(), "scores": (1.0 - D[0]).tolist()}
```

### 6.4 FHE service (TenSEAL) skeleton

```python
# services/fhe/service.py
import tenseal as ts
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# In production, contexts and eval keys are per-tenant and persisted
ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60,40,40,60])
ctx.global_scale = 2**40
ctx.generate_galois_keys()

class DotReq(BaseModel):
    q: list[float]  # assume unit-norm
    v: list[float]  # assume unit-norm

@app.post("/dot")
def dot(req: DotReq):
    enc_q = ts.ckks_vector(ctx, req.q)
    enc_v = ts.ckks_vector(ctx, req.v)
    enc = enc_q.dot(enc_v)
    # return serialized ciphertext (bytes/base64); here return context-dependent string for demo
    return {"cipher": enc.serialize().hex()[:128]}  # placeholder for transport; replace with bytes
```

### 6.5 Negotiation service (NE + memo)

```python
# services/negotiation/service.py
import nashpy as nash
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class NegReq(BaseModel):
    A: list[list[float]]
    B: list[list[float]]
    rule: str = "nsw"
    batna: tuple[float, float] = (0.0, 0.0)

@app.post("/negotiate")
def negotiate(req: NegReq):
    A = np.array(req.A)
    B = np.array(req.B)
    game = nash.Game(A, B)
    eqs = list(game.lemke_howson_enumeration())
    def score(eq):
        x, y = eq
        u1 = float(x @ A @ y)
        u2 = float(x @ B @ y)
        if req.rule == "egal":
            return min(u1, u2)
        # default NSW
        return max(u1 - req.batna[0], 0) * max(u2 - req.batna[1], 0)
    best = max(eqs, key=score) if eqs else None
    if not best:
        return {"error": "no_equilibrium"}
    x, y = best
    return {"row": x.tolist(), "col": y.tolist(), "u1": float(x @ A @ y), "u2": float(x @ B @ y)}
```

### 6.6 ε‑Ledger service stub

```python
# services/fl/eps_ledger.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

LEDGERS = {}

class PreCheck(BaseModel):
    tenant_id: str
    model_id: str
    eps_round: float

class Commit(BaseModel):
    tenant_id: str
    model_id: str
    round: int
    accountant: str
    epsilon: float
    delta: float
    clipping_C: float
    sigma: float

@app.post("/precheck")
def precheck(req: PreCheck):
    ledger = LEDGERS.setdefault((req.tenant_id, req.model_id), {"budget": 4.0, "spent": 0.0, "entries": []})
    allowed = ledger["spent"] + req.eps_round <= ledger["budget"]
    return {"allowed": allowed, "remaining": max(ledger["budget"] - ledger["spent"], 0.0)}

@app.post("/commit")
def commit(req: Commit):
    key = (req.tenant_id, req.model_id)
    ledger = LEDGERS.setdefault(key, {"budget": 4.0, "spent": 0.0, "entries": []})
    ledger["entries"].append(req.dict())
    ledger["spent"] += req.epsilon  # placeholder; replace with composed ε from accountant
    return {"ok": True, "spent": ledger["spent"]}
```

---

## 7) Local development (Docker Compose)

```yaml
# infra/docker-compose.yml
version: "3.9"
services:
  ipfs:
    image: ipfs/kubo:latest
    ports: ["5001:5001", "8080:8080"]
  gateway:
    build: ./../services/gateway
    ports: ["8000:8000"]
    environment:
      - IPFS_API=http://ipfs:5001
  search:
    build: ./../services/search
    ports: ["8100:8100"]
  fhe:
    build: ./../services/fhe
    ports: ["8200:8200"]
  negotiation:
    build: ./../services/negotiation
    ports: ["8300:8300"]
```

---

## 8) Acceptance checklist (MV\*P)

-

---

## 9) Risks & cutlines

- **FHE latency**: keep to cosine only; batch; fall back to client‑side if SLA breached.
- **Search claims**: only publish empirical A/B results; keep feature behind flag.
- **FL fragility**: start with simulated firms; move to secure aggregation in phase 2.

---

## 10) What would make this truly unique (stretch)

- **Selective zk‑attestations** in PoDP (e.g., “ε ≤ budget”, “policy X checked”).
- **Negotiation DSL** compiler + policy cards for regulators.
- **Crypto‑shredding receipts**: key‑destroy actions recorded as PoDP steps.

This plan is designed so you can hand off sections to Claude Code agents and get parallelized progress with clean interfaces, tests, and verifiability from day one.

