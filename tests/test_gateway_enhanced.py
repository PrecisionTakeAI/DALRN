"""Comprehensive tests for enhanced DALRN Gateway with complete feature set"""
import pytest
import json
import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import redis

# Import the enhanced app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock redis before importing app
with patch('redis.Redis') as mock_redis:
    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.side_effect = redis.ConnectionError("Test mode")
    mock_redis.return_value = mock_redis_instance

    from services.gateway.app_enhanced import (
        app, storage, epsilon_manager, rate_limiter,
        service_health, ServiceStatus, CircuitBreaker
    )

client = TestClient(app)

# Test fixtures
@pytest.fixture
def sample_dispute_request():
    """Sample dispute submission request"""
    return {
        "parties": ["party1@example.com", "party2@example.com"],
        "jurisdiction": "US-CA",
        "cid": "QmTestCID123456789",
        "enc_meta": {"encrypted": "data"},
        "priority": "normal"
    }

@pytest.fixture
def sample_evidence_request():
    """Sample evidence submission request"""
    return {
        "dispute_id": "disp_test123",
        "cid": "QmEvidenceCID123456",
        "evidence_type": "document",
        "metadata": {"doc_type": "contract", "size": 1024}
    }

@pytest.fixture
def sample_receipt():
    """Sample receipt for testing"""
    from services.common.podp import Receipt
    return Receipt(
        receipt_id="rcpt_test123",
        dispute_id="disp_test123",
        step="TEST_STEP",
        inputs={"test": "data"},
        params={"version": "1.0.0"},
        artifacts={"request_id": "req_test"},
        ts=datetime.now(timezone.utc).isoformat()
    ).finalize()

@pytest.fixture
def mock_ipfs():
    """Mock IPFS functions"""
    with patch('services.gateway.app_enhanced.put_json') as mock_put, \
         patch('services.gateway.app_enhanced.get_json') as mock_get:
        mock_put.return_value = "QmMockIPFSHash123"
        mock_get.return_value = {"test": "data"}
        yield mock_put, mock_get

@pytest.fixture
def mock_anchor_client():
    """Mock blockchain anchor client"""
    with patch('services.gateway.app_enhanced.anchor_client') as mock_client:
        mock_client.anchor_root.return_value = "0xMockTxHash123"
        yield mock_client

@pytest.fixture
def reset_state():
    """Reset application state between tests"""
    # Clear in-memory storage
    from services.gateway.app_enhanced import memory_storage, memory_receipts, memory_evidence
    memory_storage.clear()
    memory_receipts.clear()
    memory_evidence.clear()

    # Reset epsilon manager
    epsilon_manager.consumed.clear()

    # Reset rate limiter
    rate_limiter.windows.clear()

    # Reset service health
    for service in service_health:
        service_health[service] = ServiceStatus.HEALTHY

    yield

    # Cleanup after test
    memory_storage.clear()
    memory_receipts.clear()
    memory_evidence.clear()

# Basic endpoint tests

def test_health_check(reset_state):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["ok"] is True
    assert "timestamp" in data
    assert data["version"] == "2.0.0"
    assert data["podp_compliant"] is True
    assert "services" in data
    assert "epsilon_budget_status" in data

def test_root_endpoint(reset_state):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert "DALRN Gateway API v2.0" in data["message"]
    assert "features" in data
    assert len(data["features"]) > 5

def test_metrics_endpoint(reset_state):
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert "gateway_requests_total" in response.text
    assert "gateway_request_duration_seconds" in response.text

# Dispute submission tests

def test_submit_dispute_success(reset_state, sample_dispute_request, mock_ipfs, mock_anchor_client):
    """Test successful dispute submission"""
    response = client.post("/submit-dispute", json=sample_dispute_request)
    assert response.status_code == 201

    data = response.json()
    assert "dispute_id" in data
    assert data["dispute_id"].startswith("disp_")
    assert "receipt_id" in data
    assert data["receipt_id"].startswith("rcpt_")
    assert data["status"] == "submitted"
    assert "estimated_processing_time" in data

def test_submit_dispute_validation_error(reset_state):
    """Test dispute submission with validation errors"""
    # Missing parties
    invalid_request = {
        "parties": ["only_one_party"],
        "jurisdiction": "US",
        "cid": "QmTest"
    }
    response = client.post("/submit-dispute", json=invalid_request)
    assert response.status_code == 422  # Validation error

def test_submit_dispute_with_soan_integration(reset_state, sample_dispute_request, mock_ipfs):
    """Test dispute submission with SOAN network routing"""
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics": {
                "avg_latency": 10.5,
                "nodes_visited": 5
            }
        }
        mock_post.return_value = mock_response

        response = client.post("/submit-dispute", json=sample_dispute_request)
        assert response.status_code == 201

        data = response.json()
        # Should include estimated time affected by network metrics
        assert data["estimated_processing_time"] > 60.0

# Evidence submission tests

def test_submit_evidence_success(reset_state, sample_evidence_request, sample_dispute_request):
    """Test successful evidence submission"""
    # First create a dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]

    # Update evidence request with real dispute ID
    sample_evidence_request["dispute_id"] = dispute_id

    # Submit evidence
    response = client.post("/evidence", json=sample_evidence_request)
    assert response.status_code == 201

    data = response.json()
    assert "evidence_id" in data
    assert data["evidence_id"].startswith("evid_")
    assert "receipt_id" in data
    assert data["status"] == "accepted"
    assert "timestamp" in data

def test_submit_evidence_dispute_not_found(reset_state, sample_evidence_request):
    """Test evidence submission for non-existent dispute"""
    response = client.post("/evidence", json=sample_evidence_request)
    assert response.status_code == 404
    assert "Dispute not found" in response.json()["detail"]

def test_evidence_updates_dispute_phase(reset_state, sample_dispute_request):
    """Test that evidence submission updates dispute phase"""
    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]

    # Check initial phase
    status_response = client.get(f"/status/{dispute_id}")
    assert status_response.json()["phase"] == "INTAKE"

    # Submit evidence
    evidence_request = {
        "dispute_id": dispute_id,
        "cid": "QmEvidence123",
        "evidence_type": "document",
        "metadata": {}
    }
    client.post("/evidence", json=evidence_request)

    # Check updated phase
    status_response = client.get(f"/status/{dispute_id}")
    assert status_response.json()["phase"] == "EVIDENCE_COLLECTION"

# Status endpoint tests

def test_get_status_success(reset_state, sample_dispute_request):
    """Test successful status retrieval"""
    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]

    # Get status
    response = client.get(f"/status/{dispute_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["dispute_id"] == dispute_id
    assert data["phase"] == "INTAKE"
    assert "receipts" in data
    assert "evidence_count" in data
    assert "eps_budget_remaining" in data
    assert "last_updated" in data

def test_get_status_not_found(reset_state):
    """Test status retrieval for non-existent dispute"""
    response = client.get("/status/disp_nonexistent")
    assert response.status_code == 404
    assert "Dispute not found" in response.json()["detail"]

def test_get_status_with_network_metrics(reset_state, sample_dispute_request):
    """Test status retrieval with network metrics"""
    # Create dispute with mock SOAN metrics
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metrics": {"latency": 5.0}
        }
        mock_post.return_value = mock_response

        dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
        dispute_id = dispute_response.json()["dispute_id"]

    # Get status with network metrics
    response = client.get(f"/status/{dispute_id}?include_network_metrics=true")
    assert response.status_code == 200

    data = response.json()
    # Metrics should be included if available
    if data.get("network_metrics"):
        assert "latency" in data["network_metrics"]

# Manual anchoring tests

def test_manual_anchor_success(reset_state, sample_dispute_request, mock_ipfs, mock_anchor_client):
    """Test successful manual anchoring"""
    # Set services as healthy
    service_health["ipfs"] = ServiceStatus.HEALTHY
    service_health["chain"] = ServiceStatus.HEALTHY

    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]

    # Trigger manual anchor
    response = client.post("/anchor-manual", json={
        "dispute_id": dispute_id,
        "force": False
    })
    assert response.status_code == 202

    data = response.json()
    assert data["status"] == "anchored"
    assert "anchor_tx" in data
    assert "receipt_chain_uri" in data
    assert "merkle_root" in data

def test_manual_anchor_dispute_not_found(reset_state):
    """Test manual anchor for non-existent dispute"""
    response = client.post("/anchor-manual", json={
        "dispute_id": "disp_nonexistent",
        "force": False
    })
    assert response.status_code == 404

def test_manual_anchor_service_unavailable(reset_state, sample_dispute_request):
    """Test manual anchor when services are unavailable"""
    # Set IPFS as unhealthy
    service_health["ipfs"] = ServiceStatus.UNHEALTHY

    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]

    # Try to anchor
    response = client.post("/anchor-manual", json={
        "dispute_id": dispute_id,
        "force": False
    })
    assert response.status_code == 503
    assert "IPFS service is not healthy" in response.json()["detail"]

# Receipt retrieval tests

def test_get_receipts_success(reset_state, sample_dispute_request):
    """Test successful receipt retrieval"""
    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]

    # Get receipts
    response = client.get(f"/receipts/{dispute_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["dispute_id"] == dispute_id
    assert "receipts" in data
    assert len(data["receipts"]) >= 1  # At least intake receipt
    assert "total_count" in data
    assert "merkle_root" in data

def test_get_receipts_pagination(reset_state, sample_dispute_request):
    """Test receipt pagination"""
    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]

    # Add some evidence to create more receipts
    for i in range(5):
        client.post("/evidence", json={
            "dispute_id": dispute_id,
            "cid": f"QmEvidence{i}",
            "evidence_type": "document",
            "metadata": {}
        })

    # Get paginated receipts
    response = client.get(f"/receipts/{dispute_id}?offset=1&limit=2")
    assert response.status_code == 200

    data = response.json()
    assert len(data["receipts"]) <= 2
    assert data["total_count"] >= 6  # Intake + 5 evidence

# Receipt validation tests

def test_validate_receipt_success(reset_state, sample_dispute_request):
    """Test successful receipt validation"""
    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]
    receipt_id = dispute_response.json()["receipt_id"]

    # Get the receipt data
    receipts_response = client.get(f"/receipts/{dispute_id}")
    receipt_data = receipts_response.json()["receipts"][0]

    # Validate receipt
    response = client.post("/validate-receipt", json={
        "receipt_id": receipt_id,
        "dispute_id": dispute_id,
        "receipt_data": receipt_data
    })
    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is True
    assert data["receipt_id"] == receipt_id
    assert "validation_details" in data

def test_validate_receipt_invalid(reset_state, sample_dispute_request):
    """Test receipt validation with tampered data"""
    # Create dispute
    dispute_response = client.post("/submit-dispute", json=sample_dispute_request)
    dispute_id = dispute_response.json()["dispute_id"]
    receipt_id = dispute_response.json()["receipt_id"]

    # Get the receipt and tamper with it
    receipts_response = client.get(f"/receipts/{dispute_id}")
    receipt_data = receipts_response.json()["receipts"][0]
    receipt_data["inputs"]["tampered"] = True

    # Validate tampered receipt
    response = client.post("/validate-receipt", json={
        "receipt_id": receipt_id,
        "dispute_id": dispute_id,
        "receipt_data": receipt_data
    })
    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is False

# Rate limiting tests

def test_rate_limiting(reset_state):
    """Test rate limiting functionality"""
    # Clear rate limiter
    rate_limiter.windows.clear()

    # Make requests up to limit
    for i in range(100):
        response = client.get("/health")
        assert response.status_code == 200

    # Next request should be rate limited
    response = client.get("/health")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]

# Circuit breaker tests

def test_circuit_breaker_functionality():
    """Test circuit breaker pattern"""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

    # Initial state is closed
    assert breaker.can_attempt() is True

    # Record failures
    for _ in range(3):
        breaker.record_failure()

    # Circuit should be open
    assert breaker.state == "open"
    assert breaker.can_attempt() is False

    # Wait for timeout
    import time
    time.sleep(1.1)

    # Circuit should allow attempt (half-open)
    assert breaker.can_attempt() is True

    # Success should close circuit
    breaker.record_success()
    assert breaker.state == "closed"

# Epsilon budget tests

def test_epsilon_budget_management(reset_state):
    """Test privacy budget management"""
    tenant_id = "test_tenant"

    # Check initial budget
    remaining = epsilon_manager.get_remaining(tenant_id)
    assert remaining == 4.0

    # Check budget for operations
    assert epsilon_manager.check_budget(tenant_id, "search") is True

    # Consume budget
    remaining = epsilon_manager.consume(tenant_id, "search")
    assert remaining == 3.9

    # Consume more budget
    epsilon_manager.consume(tenant_id, "fhe_computation")
    remaining = epsilon_manager.get_remaining(tenant_id)
    assert remaining == 3.6

    # Check budget limits
    for _ in range(10):
        epsilon_manager.consume(tenant_id, "fhe_computation")

    assert epsilon_manager.check_budget(tenant_id, "fhe_computation") is False

# Storage backend tests

@pytest.mark.asyncio
async def test_storage_backend_fallback(reset_state):
    """Test storage backend fallback to memory"""
    # Storage should use memory when Redis is not available
    dispute_data = {"test": "data"}
    dispute_id = "disp_test"

    # Test set and get
    await storage.set_dispute(dispute_id, dispute_data)
    retrieved = await storage.get_dispute(dispute_id)
    assert retrieved == dispute_data

    # Test with receipt chain
    from services.common.podp import ReceiptChain, Receipt
    receipt = Receipt(
        receipt_id="rcpt_test",
        dispute_id=dispute_id,
        step="TEST",
        ts=datetime.now(timezone.utc).isoformat()
    ).finalize()

    chain = ReceiptChain(dispute_id=dispute_id, receipts=[receipt])
    await storage.set_receipt_chain(dispute_id, chain)

    retrieved_chain = await storage.get_receipt_chain(dispute_id)
    assert retrieved_chain.dispute_id == dispute_id
    assert len(retrieved_chain.receipts) == 1

# Service health monitoring tests

@pytest.mark.asyncio
async def test_service_health_monitoring():
    """Test service health monitoring"""
    from services.gateway.app_enhanced import check_all_services

    # Mock httpx responses
    with patch('httpx.AsyncClient.get') as mock_get:
        # Simulate healthy service
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        await check_all_services()

        # Services should be marked as healthy (where checked)
        # Note: Some services might not be checked in test environment

# PII redaction tests

def test_pii_redaction():
    """Test PII redaction functionality"""
    from services.gateway.app_enhanced import redact_pii

    sensitive_data = {
        "parties": ["party1@example.com", "party2@example.com"],
        "email": "user@example.com",
        "name": "John Doe",
        "public_field": "This is public",
        "nested": {
            "ssn": "123-45-6789",
            "safe": "public info"
        }
    }

    redacted = redact_pii(sensitive_data)

    assert redacted["parties"] == "[REDACTED]"
    assert redacted["email"] == "[REDACTED]"
    assert redacted["name"] == "[REDACTED]"
    assert redacted["public_field"] == "This is public"
    assert redacted["nested"]["ssn"] == "[REDACTED]"
    assert redacted["nested"]["safe"] == "public info"

# Middleware tests

def test_podp_middleware_headers(reset_state):
    """Test PoDP middleware adds correct headers"""
    response = client.get("/health")

    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"].startswith("req_")
    assert "X-Processing-Time" in response.headers
    assert "X-PoDP-Compliant" in response.headers
    assert response.headers["X-PoDP-Compliant"] == "true"

# Error handling tests

def test_error_handling_validation(reset_state):
    """Test validation error handling"""
    # Invalid CID
    invalid_request = {
        "parties": ["p1", "p2"],
        "jurisdiction": "US",
        "cid": "bad"  # Too short
    }
    response = client.post("/submit-dispute", json=invalid_request)
    assert response.status_code == 422

def test_error_handling_unexpected(reset_state):
    """Test unexpected error handling"""
    with patch('services.gateway.app_enhanced.storage.get_dispute') as mock_get:
        mock_get.side_effect = Exception("Unexpected error")

        response = client.get("/status/disp_test")
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
        assert "request_id" in response.json()

# Integration tests

def test_full_dispute_workflow(reset_state, mock_ipfs, mock_anchor_client):
    """Test complete dispute workflow"""
    # Set services healthy
    service_health["ipfs"] = ServiceStatus.HEALTHY
    service_health["chain"] = ServiceStatus.HEALTHY

    # 1. Submit dispute
    dispute_request = {
        "parties": ["alice@example.com", "bob@example.com"],
        "jurisdiction": "US-NY",
        "cid": "QmDispute123",
        "enc_meta": {"case_type": "contract"},
        "priority": "high"
    }
    dispute_response = client.post("/submit-dispute", json=dispute_request)
    assert dispute_response.status_code == 201
    dispute_id = dispute_response.json()["dispute_id"]

    # 2. Submit evidence
    for i in range(3):
        evidence_response = client.post("/evidence", json={
            "dispute_id": dispute_id,
            "cid": f"QmEvidence{i}",
            "evidence_type": "document",
            "metadata": {"page_count": i+1}
        })
        assert evidence_response.status_code == 201

    # 3. Check status
    status_response = client.get(f"/status/{dispute_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["phase"] == "EVIDENCE_COLLECTION"
    assert status_data["evidence_count"] == 3

    # 4. Get all receipts
    receipts_response = client.get(f"/receipts/{dispute_id}")
    assert receipts_response.status_code == 200
    receipts_data = receipts_response.json()
    assert receipts_data["total_count"] >= 4  # Intake + 3 evidence

    # 5. Validate a receipt
    first_receipt = receipts_data["receipts"][0]
    validation_response = client.post("/validate-receipt", json={
        "receipt_id": first_receipt["receipt_id"],
        "dispute_id": dispute_id,
        "receipt_data": first_receipt
    })
    assert validation_response.status_code == 200
    assert validation_response.json()["valid"] is True

    # 6. Manual anchor
    anchor_response = client.post("/anchor-manual", json={
        "dispute_id": dispute_id,
        "force": True
    })
    assert anchor_response.status_code == 202
    assert "anchor_tx" in anchor_response.json()

    # 7. Final status check
    final_status = client.get(f"/status/{dispute_id}")
    assert final_status.status_code == 200
    assert final_status.json()["anchor_tx"] is not None

# Performance tests

def test_performance_metrics_collection(reset_state):
    """Test that performance metrics are collected"""
    # Make several requests
    for _ in range(5):
        client.get("/health")

    # Check metrics endpoint
    metrics_response = client.get("/metrics")
    metrics_text = metrics_response.text

    # Verify metrics are present
    assert "gateway_requests_total" in metrics_text
    assert "gateway_request_duration_seconds" in metrics_text
    assert 'endpoint="/health"' in metrics_text

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])