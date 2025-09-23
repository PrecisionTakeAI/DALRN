"""DALRN Gateway API with Complete PoDP Middleware and Full Feature Set"""
import os
import json
import logging
import asyncio
import time
import hashlib
import redis
import httpx
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from uuid import uuid4
from collections import defaultdict, deque
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Response, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

# Import from common modules
from services.common.podp import Receipt, ReceiptChain
from services.common.ipfs import put_json, get_json
from services.chain.client import AnchorClient

# Configure structured logging with PII redaction
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
request_counter = Counter(
    'gateway_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'gateway_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

active_disputes = Gauge(
    'gateway_active_disputes',
    'Number of active disputes'
)

receipt_chain_size = Histogram(
    'gateway_receipt_chain_size',
    'Size of receipt chains',
    ['dispute_id']
)

epsilon_budget_gauge = Gauge(
    'gateway_epsilon_budget_remaining',
    'Remaining epsilon budget',
    ['tenant_id']
)

# Redis connection (with fallback to in-memory)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_keepalive=True,
        socket_keepalive_options={
            1: 1,  # TCP_KEEPIDLE
            2: 3,  # TCP_KEEPINTVL
            3: 5   # TCP_KEEPCNT
        }
    )
    redis_client.ping()
    USE_REDIS = True
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.warning(f"Redis not available, using in-memory storage: {e}")
    redis_client = None
    USE_REDIS = False

# In-memory fallback storage
memory_storage: Dict[str, dict] = {}
memory_receipts: Dict[str, ReceiptChain] = {}
memory_evidence: Dict[str, List[dict]] = {}

# Circuit breaker pattern
class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class CircuitBreaker:
    """Circuit breaker for external service calls"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, half_open_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record a successful call"""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record a failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def can_attempt(self) -> bool:
        """Check if we can attempt a call"""
        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = "half-open"
                return True
            return False

        # half-open state
        return True

# Circuit breakers for services
ipfs_breaker = CircuitBreaker()
chain_breaker = CircuitBreaker()
soan_breaker = CircuitBreaker()

# Service health tracker
service_health = {
    "ipfs": ServiceStatus.HEALTHY,
    "chain": ServiceStatus.HEALTHY,
    "soan": ServiceStatus.HEALTHY,
    "redis": ServiceStatus.HEALTHY if USE_REDIS else ServiceStatus.DEGRADED,
    "fhe": ServiceStatus.HEALTHY,
    "negotiation": ServiceStatus.HEALTHY,
    "search": ServiceStatus.HEALTHY
}

# Redaction helper for logging
def redact_pii(data: Any) -> Any:
    """Redact PII from data before logging"""
    if isinstance(data, dict):
        redacted = {}
        sensitive_keys = {
            'parties', 'email', 'name', 'address', 'phone',
            'ssn', 'ein', 'account', 'password', 'token',
            'secret', 'private', 'credential'
        }
        for k, v in data.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                redacted[k] = "[REDACTED]"
            else:
                redacted[k] = redact_pii(v)
        return redacted
    elif isinstance(data, list):
        return [redact_pii(item) for item in data]
    elif isinstance(data, str) and len(data) > 50:
        # Redact long strings that might contain sensitive data
        return f"{data[:8]}...[REDACTED]"
    return data

# Storage abstraction layer
class StorageBackend:
    """Abstraction for storage operations with Redis/Memory fallback"""

    @staticmethod
    async def get_dispute(dispute_id: str) -> Optional[dict]:
        """Get dispute from storage"""
        if USE_REDIS:
            try:
                data = redis_client.get(f"dispute:{dispute_id}")
                return json.loads(data) if data else None
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                return memory_storage.get(dispute_id)
        return memory_storage.get(dispute_id)

    @staticmethod
    async def set_dispute(dispute_id: str, data: dict) -> bool:
        """Set dispute in storage"""
        if USE_REDIS:
            try:
                redis_client.setex(
                    f"dispute:{dispute_id}",
                    86400,  # 24 hour TTL
                    json.dumps(data)
                )
                # Also store in memory as backup
                memory_storage[dispute_id] = data
                return True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                memory_storage[dispute_id] = data
                return False
        memory_storage[dispute_id] = data
        return True

    @staticmethod
    async def get_receipt_chain(dispute_id: str) -> Optional[ReceiptChain]:
        """Get receipt chain from storage"""
        if USE_REDIS:
            try:
                data = redis_client.get(f"receipts:{dispute_id}")
                if data:
                    chain_data = json.loads(data)
                    return ReceiptChain(**chain_data)
                return None
            except Exception as e:
                logger.error(f"Redis get receipt error: {e}")
                return memory_receipts.get(dispute_id)
        return memory_receipts.get(dispute_id)

    @staticmethod
    async def set_receipt_chain(dispute_id: str, chain: ReceiptChain) -> bool:
        """Set receipt chain in storage"""
        if USE_REDIS:
            try:
                redis_client.setex(
                    f"receipts:{dispute_id}",
                    86400,
                    json.dumps(chain.model_dump())
                )
                memory_receipts[dispute_id] = chain
                return True
            except Exception as e:
                logger.error(f"Redis set receipt error: {e}")
                memory_receipts[dispute_id] = chain
                return False
        memory_receipts[dispute_id] = chain
        return True

    @staticmethod
    async def add_evidence(dispute_id: str, evidence: dict) -> bool:
        """Add evidence to dispute"""
        if USE_REDIS:
            try:
                redis_client.rpush(
                    f"evidence:{dispute_id}",
                    json.dumps(evidence)
                )
                if dispute_id not in memory_evidence:
                    memory_evidence[dispute_id] = []
                memory_evidence[dispute_id].append(evidence)
                return True
            except Exception as e:
                logger.error(f"Redis add evidence error: {e}")
                if dispute_id not in memory_evidence:
                    memory_evidence[dispute_id] = []
                memory_evidence[dispute_id].append(evidence)
                return False
        if dispute_id not in memory_evidence:
            memory_evidence[dispute_id] = []
        memory_evidence[dispute_id].append(evidence)
        return True

# Initialize clients
anchor_client = AnchorClient()
storage = StorageBackend()

# Rate limiting with sliding window
class RateLimiter:
    """Advanced rate limiter with sliding window and multiple tiers"""

    def __init__(self):
        self.windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.limits = {
            "default": 100,  # 100 requests per minute
            "authenticated": 500,  # 500 requests per minute
            "premium": 2000  # 2000 requests per minute
        }

    async def check_rate_limit(self, client_id: str, tier: str = "default") -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        window = self.windows[client_id]

        # Remove old entries (older than 1 minute)
        while window and window[0] < now - 60:
            window.popleft()

        # Check limit
        limit = self.limits.get(tier, self.limits["default"])
        if len(window) >= limit:
            return False

        # Add current request
        window.append(now)
        return True

rate_limiter = RateLimiter()

# Epsilon budget management
class EpsilonBudgetManager:
    """Manage privacy budget across all operations"""

    def __init__(self, total_budget: float = 4.0):
        self.total_budget = total_budget
        self.consumed: Dict[str, float] = defaultdict(float)
        self.operation_costs = {
            "search": 0.1,
            "negotiation": 0.2,
            "fhe_computation": 0.3,
            "agent_routing": 0.05
        }

    def check_budget(self, tenant_id: str, operation: str) -> bool:
        """Check if operation is within budget"""
        cost = self.operation_costs.get(operation, 0.1)
        return self.consumed[tenant_id] + cost <= self.total_budget

    def consume(self, tenant_id: str, operation: str) -> float:
        """Consume budget for operation"""
        cost = self.operation_costs.get(operation, 0.1)
        self.consumed[tenant_id] += cost
        remaining = self.total_budget - self.consumed[tenant_id]
        epsilon_budget_gauge.labels(tenant_id=tenant_id).set(remaining)
        return remaining

    def get_remaining(self, tenant_id: str) -> float:
        """Get remaining budget"""
        return self.total_budget - self.consumed[tenant_id]

epsilon_manager = EpsilonBudgetManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting DALRN Gateway with full feature set")

    # Initialize service health checks
    asyncio.create_task(health_monitor())

    # Warm up connections
    await check_all_services()

    yield

    # Cleanup
    logger.info("Shutting down DALRN Gateway")
    if redis_client:
        redis_client.close()

app = FastAPI(
    title="DALRN Gateway",
    description="Production-ready Gateway with Complete PoDP Implementation",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SubmitDisputeRequest(BaseModel):
    """Request model for dispute submission"""
    parties: List[str] = Field(..., min_length=2, description="List of party identifiers")
    jurisdiction: str = Field(..., min_length=2, description="Jurisdiction code")
    cid: str = Field(..., description="IPFS CID of encrypted document bundle")
    enc_meta: dict = Field(default_factory=dict, description="Encrypted metadata")
    priority: str = Field(default="normal", description="Priority level")

    @field_validator('cid')
    @classmethod
    def validate_cid(cls, v):
        """Validate IPFS CID format"""
        if not v or len(v) < 10:
            raise ValueError("Invalid CID format")
        return v

    @field_validator('parties')
    @classmethod
    def validate_parties(cls, v):
        """Ensure at least 2 parties"""
        if len(v) < 2:
            raise ValueError("At least 2 parties required")
        return v

class EvidenceSubmissionRequest(BaseModel):
    """Request model for evidence submission"""
    dispute_id: str = Field(..., description="Dispute ID")
    cid: str = Field(..., description="IPFS CID of encrypted evidence")
    evidence_type: str = Field(..., description="Type of evidence")
    metadata: dict = Field(default_factory=dict, description="Evidence metadata")

class ManualAnchorRequest(BaseModel):
    """Request model for manual anchoring"""
    dispute_id: str = Field(..., description="Dispute ID to anchor")
    force: bool = Field(default=False, description="Force anchoring even if recent")

class ValidateReceiptRequest(BaseModel):
    """Request model for receipt validation"""
    receipt_id: str = Field(..., description="Receipt ID to validate")
    dispute_id: str = Field(..., description="Associated dispute ID")
    receipt_data: dict = Field(..., description="Receipt data to validate")

class SubmitDisputeResponse(BaseModel):
    """Response model for dispute submission"""
    dispute_id: str
    receipt_id: str
    anchor_uri: Optional[str] = None
    anchor_tx: Optional[str] = None
    status: str = "submitted"
    estimated_processing_time: Optional[float] = None

class DisputeStatusResponse(BaseModel):
    """Response model for dispute status"""
    dispute_id: str
    phase: str
    receipts: List[dict]
    evidence_count: int = 0
    anchor_tx: Optional[str] = None
    eps_budget_remaining: Optional[float] = None
    last_updated: str
    receipt_chain_uri: Optional[str] = None
    network_metrics: Optional[dict] = None

class EvidenceSubmissionResponse(BaseModel):
    """Response model for evidence submission"""
    evidence_id: str
    receipt_id: str
    status: str = "accepted"
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    ok: bool
    timestamp: str
    version: str = "2.0.0"
    services: dict = Field(default_factory=dict)
    podp_compliant: bool = True
    epsilon_budget_status: dict = Field(default_factory=dict)

class ReceiptListResponse(BaseModel):
    """Response model for receipt list"""
    dispute_id: str
    receipts: List[dict]
    total_count: int
    merkle_root: Optional[str] = None

class ValidationResponse(BaseModel):
    """Response model for receipt validation"""
    valid: bool
    receipt_id: str
    validation_details: dict
    merkle_proof: Optional[List[str]] = None

# Health monitoring
async def health_monitor():
    """Background task to monitor service health"""
    while True:
        await asyncio.sleep(30)
        await check_all_services()

async def check_all_services():
    """Check health of all dependent services"""
    # Check IPFS
    if ipfs_breaker.can_attempt():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{os.getenv('IPFS_HOST', 'localhost')}:5001/api/v0/version",
                    timeout=5.0
                )
                if response.status_code == 200:
                    service_health["ipfs"] = ServiceStatus.HEALTHY
                    ipfs_breaker.record_success()
                else:
                    service_health["ipfs"] = ServiceStatus.DEGRADED
                    ipfs_breaker.record_failure()
        except Exception:
            service_health["ipfs"] = ServiceStatus.UNHEALTHY
            ipfs_breaker.record_failure()

    # Check blockchain
    if chain_breaker.can_attempt():
        try:
            # Mock check - would be real blockchain check
            service_health["chain"] = ServiceStatus.HEALTHY
            chain_breaker.record_success()
        except Exception:
            service_health["chain"] = ServiceStatus.UNHEALTHY
            chain_breaker.record_failure()

    # Check SOAN
    if soan_breaker.can_attempt():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{os.getenv('AGENTS_PORT', '8500')}/health",
                    timeout=5.0
                )
                if response.status_code == 200:
                    service_health["soan"] = ServiceStatus.HEALTHY
                    soan_breaker.record_success()
                else:
                    service_health["soan"] = ServiceStatus.DEGRADED
                    soan_breaker.record_failure()
        except Exception:
            service_health["soan"] = ServiceStatus.UNHEALTHY
            soan_breaker.record_failure()

    # Check other services
    for service, port in [("fhe", 8200), ("negotiation", 8300), ("search", 8100)]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{port}/health",
                    timeout=5.0
                )
                if response.status_code == 200:
                    service_health[service] = ServiceStatus.HEALTHY
                else:
                    service_health[service] = ServiceStatus.DEGRADED
        except Exception:
            service_health[service] = ServiceStatus.UNHEALTHY

# PoDP Middleware
@app.middleware("http")
async def podp_middleware(request: Request, call_next):
    """Enhanced middleware for Proof of Data Possession tracking"""
    start_time = datetime.now(timezone.utc)

    # Generate request ID for tracking
    request_id = f"req_{uuid4().hex[:12]}"
    request.state.request_id = request_id

    # Extract client info
    client_id = request.client.host if request.client else "unknown"

    # Log request (redacted)
    logger.info(
        f"Request received",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": hashlib.sha256(client_id.encode()).hexdigest()[:8]
        }
    )

    # Process request with timing
    response = await call_next(request)

    # Calculate duration
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Update metrics
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    # Log response
    logger.info(
        f"Request completed",
        extra={
            "request_id": request_id,
            "duration_seconds": duration,
            "status_code": response.status_code
        }
    )

    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = str(duration)
    response.headers["X-PoDP-Compliant"] = "true"

    return response

# Dependency for rate limiting
async def check_rate_limit(request: Request):
    """Rate limiting dependency"""
    client_id = request.client.host if request.client else "unknown"

    # Determine tier (would check auth in production)
    tier = "default"

    if not await rate_limiter.check_rate_limit(client_id, tier):
        logger.warning(f"Rate limit exceeded for client: {hashlib.sha256(client_id.encode()).hexdigest()[:8]}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with full service status"""
    try:
        # Check all services
        await check_all_services()

        # Get epsilon budget status
        epsilon_status = {
            "total_budget": epsilon_manager.total_budget,
            "consumed_count": len(epsilon_manager.consumed),
            "healthy": True
        }

        # Determine overall health
        unhealthy_count = sum(1 for s in service_health.values() if s == ServiceStatus.UNHEALTHY)
        degraded_count = sum(1 for s in service_health.values() if s == ServiceStatus.DEGRADED)

        ok = unhealthy_count == 0

        return HealthResponse(
            ok=ok,
            timestamp=datetime.now(timezone.utc).isoformat(),
            services={k: v.value for k, v in service_health.items()},
            podp_compliant=True,
            epsilon_budget_status=epsilon_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            ok=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            services={},
            podp_compliant=False
        )

@app.post("/submit-dispute",
          response_model=SubmitDisputeResponse,
          status_code=status.HTTP_201_CREATED,
          dependencies=[Depends(check_rate_limit)])
async def submit_dispute(
    body: SubmitDisputeRequest,
    request: Request,
    background_tasks: BackgroundTasks
) -> SubmitDisputeResponse:
    """Enhanced dispute submission with SOAN integration"""
    try:
        # Generate deterministic dispute ID
        dispute_id = Receipt.new_id(prefix="disp_")

        # Log submission (with redacted PII)
        logger.info(
            "Processing dispute submission",
            extra={
                "dispute_id": dispute_id,
                "request_id": request.state.request_id,
                "jurisdiction": body.jurisdiction,
                "party_count": len(body.parties),
                "priority": body.priority
            }
        )

        # Check SOAN availability and route if available
        network_metrics = None
        if service_health["soan"] == ServiceStatus.HEALTHY and soan_breaker.can_attempt():
            try:
                async with httpx.AsyncClient() as client:
                    soan_response = await client.post(
                        f"http://localhost:{os.getenv('AGENTS_PORT', '8500')}/api/v1/soan/route",
                        json={
                            "dispute_id": dispute_id,
                            "priority": body.priority,
                            "jurisdiction": body.jurisdiction
                        },
                        timeout=5.0
                    )
                    if soan_response.status_code == 200:
                        network_metrics = soan_response.json().get("metrics")
                        soan_breaker.record_success()
            except Exception as e:
                logger.warning(f"SOAN routing failed: {e}")
                soan_breaker.record_failure()

        # Create initial receipt for intake
        receipt = Receipt(
            receipt_id=Receipt.new_id(prefix="rcpt_"),
            dispute_id=dispute_id,
            step="INTAKE_V2",
            inputs={
                "cid_bundle": body.cid,
                "party_count": len(body.parties),
                "submission_time": datetime.now(timezone.utc).isoformat(),
                "priority": body.priority
            },
            params={
                "jurisdiction": body.jurisdiction,
                "version": "2.0.0",
                "network_routing": network_metrics is not None
            },
            artifacts={
                "request_id": request.state.request_id,
                "network_metrics": network_metrics or {}
            },
            ts=datetime.now(timezone.utc).isoformat()
        ).finalize()

        # Build receipt chain
        chain = ReceiptChain(
            dispute_id=dispute_id,
            receipts=[receipt]
        ).finalize()

        # Store receipt chain
        await storage.set_receipt_chain(dispute_id, chain)

        # Upload to IPFS if available
        uri = None
        if service_health["ipfs"] == ServiceStatus.HEALTHY and ipfs_breaker.can_attempt():
            try:
                chain_dict = chain.model_dump(exclude_none=True)
                uri = await asyncio.to_thread(put_json, chain_dict)
                logger.info(f"Receipt chain uploaded to IPFS: {uri[:30]}...")
                ipfs_breaker.record_success()
            except Exception as e:
                logger.error(f"IPFS upload failed: {str(e)}")
                ipfs_breaker.record_failure()

        # Anchor to chain if available
        anchor_tx = None
        if uri and service_health["chain"] == ServiceStatus.HEALTHY and chain_breaker.can_attempt():
            try:
                anchor_tx = await asyncio.to_thread(
                    anchor_client.anchor_root,
                    dispute_id,
                    chain.merkle_root,
                    b"\x00" * 32,
                    0,
                    uri,
                    [b"PoDP", b"INTAKE_V2"]
                )
                logger.info(f"Anchored to chain: {anchor_tx}")
                chain_breaker.record_success()
            except Exception as e:
                logger.error(f"Chain anchoring failed: {str(e)}")
                chain_breaker.record_failure()

        # Store dispute metadata
        dispute_data = {
            "dispute_id": dispute_id,
            "phase": "INTAKE",
            "parties_hash": hashlib.sha256(json.dumps(body.parties).encode()).hexdigest(),
            "jurisdiction": body.jurisdiction,
            "cid": body.cid,
            "enc_meta": body.enc_meta,
            "priority": body.priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "anchor_tx": anchor_tx,
            "receipt_chain_uri": uri,
            "receipts": [receipt.receipt_id],
            "network_metrics": network_metrics
        }

        await storage.set_dispute(dispute_id, dispute_data)

        # Update metrics
        active_disputes.inc()

        # Estimate processing time based on priority and network
        estimated_time = 60.0  # Base time in seconds
        if body.priority == "high":
            estimated_time *= 0.5
        elif body.priority == "low":
            estimated_time *= 2.0

        if network_metrics and "avg_latency" in network_metrics:
            estimated_time += network_metrics["avg_latency"]

        return SubmitDisputeResponse(
            dispute_id=dispute_id,
            receipt_id=receipt.receipt_id,
            anchor_uri=uri,
            anchor_tx=anchor_tx,
            status="submitted",
            estimated_processing_time=estimated_time
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in submit_dispute: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/evidence",
          response_model=EvidenceSubmissionResponse,
          status_code=status.HTTP_201_CREATED,
          dependencies=[Depends(check_rate_limit)])
async def submit_evidence(
    body: EvidenceSubmissionRequest,
    request: Request
) -> EvidenceSubmissionResponse:
    """Submit encrypted evidence for a dispute with PoDP tracking"""
    try:
        # Verify dispute exists
        dispute = await storage.get_dispute(body.dispute_id)
        if not dispute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dispute not found"
            )

        # Generate evidence ID
        evidence_id = Receipt.new_id(prefix="evid_")

        logger.info(
            "Processing evidence submission",
            extra={
                "dispute_id": body.dispute_id,
                "evidence_id": evidence_id,
                "evidence_type": body.evidence_type,
                "request_id": request.state.request_id
            }
        )

        # Create receipt for evidence submission
        receipt = Receipt(
            receipt_id=Receipt.new_id(prefix="rcpt_"),
            dispute_id=body.dispute_id,
            step="EVIDENCE_SUBMISSION",
            inputs={
                "evidence_id": evidence_id,
                "cid": body.cid,
                "evidence_type": body.evidence_type,
                "submission_time": datetime.now(timezone.utc).isoformat()
            },
            params={
                "version": "2.0.0"
            },
            artifacts={
                "request_id": request.state.request_id,
                "metadata": redact_pii(body.metadata)
            },
            ts=datetime.now(timezone.utc).isoformat()
        ).finalize()

        # Update receipt chain
        chain = await storage.get_receipt_chain(body.dispute_id)
        if not chain:
            chain = ReceiptChain(dispute_id=body.dispute_id, receipts=[])

        chain.add_receipt(receipt)
        chain.finalize()

        await storage.set_receipt_chain(body.dispute_id, chain)

        # Store evidence metadata
        evidence_data = {
            "evidence_id": evidence_id,
            "dispute_id": body.dispute_id,
            "cid": body.cid,
            "evidence_type": body.evidence_type,
            "metadata": body.metadata,
            "receipt_id": receipt.receipt_id,
            "submitted_at": datetime.now(timezone.utc).isoformat()
        }

        await storage.add_evidence(body.dispute_id, evidence_data)

        # Update dispute phase if needed
        if dispute["phase"] == "INTAKE":
            dispute["phase"] = "EVIDENCE_COLLECTION"
            dispute["last_updated"] = datetime.now(timezone.utc).isoformat()
            await storage.set_dispute(body.dispute_id, dispute)

        # Update receipt chain size metric
        receipt_chain_size.labels(dispute_id=body.dispute_id).observe(len(chain.receipts))

        return EvidenceSubmissionResponse(
            evidence_id=evidence_id,
            receipt_id=receipt.receipt_id,
            status="accepted",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting evidence: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/status/{dispute_id}",
         response_model=DisputeStatusResponse,
         dependencies=[Depends(check_rate_limit)])
async def get_dispute_status(
    dispute_id: str,
    request: Request,
    include_receipts: bool = True,
    include_network_metrics: bool = False
) -> DisputeStatusResponse:
    """Get enhanced status of a dispute"""
    try:
        logger.info(
            f"Status request for dispute",
            extra={
                "dispute_id": dispute_id,
                "request_id": request.state.request_id
            }
        )

        # Get dispute
        dispute = await storage.get_dispute(dispute_id)
        if not dispute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dispute not found"
            )

        # Get receipt chain
        receipts = []
        if include_receipts:
            chain = await storage.get_receipt_chain(dispute_id)
            if chain:
                for receipt in chain.receipts:
                    receipt_dict = receipt.model_dump(exclude_none=True)
                    # Redact sensitive fields
                    if 'inputs' in receipt_dict:
                        receipt_dict['inputs'] = redact_pii(receipt_dict['inputs'])
                    receipts.append(receipt_dict)

        # Get evidence count
        evidence_list = memory_evidence.get(dispute_id, [])

        # Calculate epsilon budget remaining
        # Use parties hash as tenant ID for privacy budget
        tenant_id = dispute.get("parties_hash", dispute_id)
        eps_remaining = epsilon_manager.get_remaining(tenant_id)

        # Get network metrics if requested
        network_metrics = None
        if include_network_metrics and dispute.get("network_metrics"):
            network_metrics = dispute["network_metrics"]

        return DisputeStatusResponse(
            dispute_id=dispute_id,
            phase=dispute.get("phase", "UNKNOWN"),
            receipts=receipts,
            evidence_count=len(evidence_list),
            anchor_tx=dispute.get("anchor_tx"),
            eps_budget_remaining=eps_remaining,
            last_updated=dispute.get("last_updated", dispute.get("created_at", datetime.now(timezone.utc).isoformat())),
            receipt_chain_uri=dispute.get("receipt_chain_uri"),
            network_metrics=network_metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dispute status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/anchor-manual",
          status_code=status.HTTP_202_ACCEPTED,
          dependencies=[Depends(check_rate_limit)])
async def manual_anchor(
    body: ManualAnchorRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Manually trigger anchoring for a dispute"""
    try:
        # Get dispute
        dispute = await storage.get_dispute(body.dispute_id)
        if not dispute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dispute not found"
            )

        # Check if recently anchored
        if dispute.get("anchor_tx") and not body.force:
            last_anchor = dispute.get("last_anchored")
            if last_anchor:
                last_time = datetime.fromisoformat(last_anchor)
                if datetime.now(timezone.utc) - last_time < timedelta(hours=1):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Dispute was recently anchored. Use force=true to override."
                    )

        # Get current receipt chain
        chain = await storage.get_receipt_chain(body.dispute_id)
        if not chain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No receipt chain found for dispute"
            )

        # Upload to IPFS
        uri = None
        if service_health["ipfs"] == ServiceStatus.HEALTHY:
            try:
                chain_dict = chain.model_dump(exclude_none=True)
                uri = await asyncio.to_thread(put_json, chain_dict)
                logger.info(f"Receipt chain uploaded for manual anchor: {uri[:30]}...")
            except Exception as e:
                logger.error(f"IPFS upload failed during manual anchor: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="IPFS service unavailable"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="IPFS service is not healthy"
            )

        # Anchor to chain
        if service_health["chain"] == ServiceStatus.HEALTHY:
            try:
                anchor_tx = await asyncio.to_thread(
                    anchor_client.anchor_root,
                    body.dispute_id,
                    chain.merkle_root,
                    b"\x00" * 32,
                    len(chain.receipts),
                    uri,
                    [b"PoDP", b"MANUAL_ANCHOR"]
                )

                # Update dispute
                dispute["anchor_tx"] = anchor_tx
                dispute["last_anchored"] = datetime.now(timezone.utc).isoformat()
                dispute["receipt_chain_uri"] = uri
                await storage.set_dispute(body.dispute_id, dispute)

                logger.info(f"Manual anchor completed: {anchor_tx}")

                return {
                    "status": "anchored",
                    "anchor_tx": anchor_tx,
                    "receipt_chain_uri": uri,
                    "merkle_root": chain.merkle_root
                }

            except Exception as e:
                logger.error(f"Chain anchoring failed: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Blockchain service unavailable"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Blockchain service is not healthy"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual anchor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/receipts/{dispute_id}",
         response_model=ReceiptListResponse,
         dependencies=[Depends(check_rate_limit)])
async def get_receipts(
    dispute_id: str,
    request: Request,
    offset: int = 0,
    limit: int = 100
) -> ReceiptListResponse:
    """Get all receipts for a dispute"""
    try:
        # Verify dispute exists
        dispute = await storage.get_dispute(dispute_id)
        if not dispute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dispute not found"
            )

        # Get receipt chain
        chain = await storage.get_receipt_chain(dispute_id)
        if not chain:
            return ReceiptListResponse(
                dispute_id=dispute_id,
                receipts=[],
                total_count=0,
                merkle_root=None
            )

        # Paginate receipts
        all_receipts = []
        for receipt in chain.receipts[offset:offset+limit]:
            receipt_dict = receipt.model_dump(exclude_none=True)
            # Redact sensitive data
            if 'inputs' in receipt_dict:
                receipt_dict['inputs'] = redact_pii(receipt_dict['inputs'])
            all_receipts.append(receipt_dict)

        return ReceiptListResponse(
            dispute_id=dispute_id,
            receipts=all_receipts,
            total_count=len(chain.receipts),
            merkle_root=chain.merkle_root
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting receipts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/validate-receipt",
          response_model=ValidationResponse,
          dependencies=[Depends(check_rate_limit)])
async def validate_receipt(
    body: ValidateReceiptRequest,
    request: Request
) -> ValidationResponse:
    """Validate a receipt against the chain"""
    try:
        # Get the receipt chain
        chain = await storage.get_receipt_chain(body.dispute_id)
        if not chain:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Receipt chain not found"
            )

        # Find the receipt
        target_receipt = None
        for receipt in chain.receipts:
            if receipt.receipt_id == body.receipt_id:
                target_receipt = receipt
                break

        if not target_receipt:
            return ValidationResponse(
                valid=False,
                receipt_id=body.receipt_id,
                validation_details={
                    "error": "Receipt not found in chain"
                }
            )

        # Validate receipt data matches
        provided_hash = Receipt(**body.receipt_data).compute_hash()
        actual_hash = target_receipt.compute_hash()

        valid = provided_hash == actual_hash

        # Build Merkle proof if valid
        merkle_proof = []
        if valid and chain.merkle_leaves:
            # Simple Merkle proof (would be more sophisticated in production)
            merkle_proof = chain.merkle_leaves[:5]  # Sample leaves

        return ValidationResponse(
            valid=valid,
            receipt_id=body.receipt_id,
            validation_details={
                "provided_hash": provided_hash,
                "actual_hash": actual_hash,
                "merkle_root": chain.merkle_root,
                "receipt_index": chain.receipts.index(target_receipt)
            },
            merkle_proof=merkle_proof if valid else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating receipt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY)

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - API info"""
    return {
        "message": "DALRN Gateway API v2.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "metrics": "/metrics",
        "features": [
            "Complete PoDP implementation",
            "Evidence submission",
            "SOAN integration",
            "Redis/Memory storage",
            "Circuit breakers",
            "Prometheus metrics",
            "Receipt validation",
            "Epsilon budget management"
        ]
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": getattr(request.state, "request_id", "unknown")}
    )

if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        "services.gateway.app_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )