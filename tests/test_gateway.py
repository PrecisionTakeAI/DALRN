"""Comprehensive test suite for DALRN Gateway with PoDP"""
import json
import logging
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from fastapi.testclient import TestClient

from services.gateway.app import app, dispute_storage, receipt_chains, redact_pii, request_counts
from services.common.podp import Receipt, ReceiptChain, KANON

client = TestClient(app)

# Test fixtures
@pytest.fixture
def valid_dispute_request():
    """Valid dispute submission request"""
    return {
        "parties": ["party1@example.com", "party2@example.com"],
        "jurisdiction": "US-CA",
        "cid": "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
        "enc_meta": {
            "algorithm": "AES-256-GCM",
            "key_derivation": "PBKDF2"
        }
    }

@pytest.fixture
def mock_ipfs():
    """Mock IPFS client"""
    with patch('services.gateway.app.put_json') as mock_put:
        mock_put.return_value = "ipfs://QmTestHash/receipt_chain.json"
        yield mock_put

@pytest.fixture
def mock_anchor_client():
    """Mock blockchain anchor client"""
    with patch('services.gateway.app.anchor_client.anchor_root') as mock_anchor:
        mock_anchor.return_value = "0x" + "a" * 64
        yield mock_anchor

@pytest.fixture(autouse=True)
def clear_storage():
    """Clear storage between tests"""
    dispute_storage.clear()
    receipt_chains.clear()
    request_counts.clear()  # Clear rate limit counts
    yield
    dispute_storage.clear()
    receipt_chains.clear()
    request_counts.clear()

# Health check tests
def test_healthz():
    """Test health check endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "timestamp" in data
    assert "services" in data
    assert data["version"] == "1.0.0"

# Happy path tests
def test_submit_dispute_happy_path(valid_dispute_request, mock_ipfs, mock_anchor_client):
    """Test successful dispute submission with full PoDP flow"""
    response = client.post("/submit-dispute", json=valid_dispute_request)
    
    assert response.status_code == 201
    data = response.json()
    
    # Verify response structure
    assert "dispute_id" in data
    assert "receipt_id" in data
    assert "anchor_uri" in data
    assert "anchor_tx" in data
    assert data["status"] == "submitted"
    
    # Verify dispute ID format
    assert data["dispute_id"].startswith("disp_")
    assert len(data["dispute_id"]) == 13  # prefix + 8 hex chars
    
    # Verify receipt ID format
    assert data["receipt_id"].startswith("rcpt_")
    
    # Verify IPFS was called
    mock_ipfs.assert_called_once()
    
    # Verify anchor was called
    mock_anchor_client.assert_called_once()
    anchor_call_args = mock_anchor_client.call_args[0]
    assert anchor_call_args[0] == data["dispute_id"]  # dispute_id
    assert anchor_call_args[1].startswith("0x")  # merkle_root
    assert len(anchor_call_args[1]) == 66  # 0x + 64 hex chars
    
    # Verify storage
    assert data["dispute_id"] in dispute_storage
    assert data["dispute_id"] in receipt_chains

def test_get_status_after_submission(valid_dispute_request, mock_ipfs, mock_anchor_client):
    """Test getting status after successful submission"""
    # Submit dispute
    submit_response = client.post("/submit-dispute", json=valid_dispute_request)
    assert submit_response.status_code == 201
    dispute_id = submit_response.json()["dispute_id"]
    
    # Get status
    status_response = client.get(f"/status/{dispute_id}")
    assert status_response.status_code == 200
    
    data = status_response.json()
    assert data["dispute_id"] == dispute_id
    assert data["phase"] == "INTAKE"
    assert len(data["receipts"]) == 1
    assert data["anchor_tx"] is not None
    assert data["eps_budget"] == 10.0  # Default placeholder value
    assert "last_updated" in data
    assert "receipt_chain_uri" in data
    
    # Verify receipt structure (should be redacted)
    receipt = data["receipts"][0]
    assert "receipt_id" in receipt
    assert "dispute_id" in receipt
    assert "step" in receipt
    assert receipt["step"] == "INTAKE_V1"

def test_get_status_not_found():
    """Test getting status for non-existent dispute"""
    response = client.get("/status/nonexistent_id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Dispute not found"

# Validation tests
def test_submit_dispute_invalid_parties():
    """Test submission with invalid parties list"""
    request = {
        "parties": ["only_one_party"],  # Need at least 2
        "jurisdiction": "US-CA",
        "cid": "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
        "enc_meta": {}
    }
    response = client.post("/submit-dispute", json=request)
    assert response.status_code == 422  # Validation error

def test_submit_dispute_invalid_cid():
    """Test submission with invalid CID"""
    request = {
        "parties": ["party1", "party2"],
        "jurisdiction": "US-CA",
        "cid": "invalid",  # Too short
        "enc_meta": {}
    }
    response = client.post("/submit-dispute", json=request)
    assert response.status_code == 422

def test_submit_dispute_missing_fields():
    """Test submission with missing required fields"""
    request = {
        "parties": ["party1", "party2"]
        # Missing jurisdiction and cid
    }
    response = client.post("/submit-dispute", json=request)
    assert response.status_code == 422

# Merkle tree correctness tests
def test_merkle_root_changes_with_receipts():
    """Test that Merkle root changes when receipts change"""
    # Create first receipt chain
    receipt1 = Receipt(
        dispute_id="test_dispute",
        step="STEP1",
        inputs={"data": "value1"},
        params={},
        artifacts={},
        ts=datetime.utcnow().isoformat() + "Z"
    ).finalize()
    
    chain1 = ReceiptChain(
        dispute_id="test_dispute",
        receipts=[receipt1]
    ).finalize()
    
    # Create second receipt chain with different data
    receipt2 = Receipt(
        dispute_id="test_dispute",
        step="STEP1",
        inputs={"data": "value2"},  # Different value
        params={},
        artifacts={},
        ts=datetime.utcnow().isoformat() + "Z"
    ).finalize()
    
    chain2 = ReceiptChain(
        dispute_id="test_dispute",
        receipts=[receipt2]
    ).finalize()
    
    # Merkle roots should be different
    assert chain1.merkle_root != chain2.merkle_root
    
    # Both should be valid hex strings
    assert chain1.merkle_root.startswith("0x")
    assert chain2.merkle_root.startswith("0x")
    assert len(chain1.merkle_root) == 66
    assert len(chain2.merkle_root) == 66

def test_merkle_root_deterministic():
    """Test that Merkle root is deterministic for same data"""
    # Create identical receipts with same receipt_id
    receipt_data = {
        "receipt_id": "rcpt_test123",  # Fixed receipt_id for determinism
        "dispute_id": "test_dispute",
        "step": "STEP1",
        "inputs": {"data": "value1"},
        "params": {"param": "value"},
        "artifacts": {},
        "ts": "2024-01-01T00:00:00Z"
    }
    
    receipt1 = Receipt(**receipt_data).finalize()
    receipt2 = Receipt(**receipt_data).finalize()
    
    chain1 = ReceiptChain(
        dispute_id="test_dispute",
        receipts=[receipt1]
    ).finalize()
    
    chain2 = ReceiptChain(
        dispute_id="test_dispute",
        receipts=[receipt2]
    ).finalize()
    
    # Same data should produce same Merkle root
    assert chain1.merkle_root == chain2.merkle_root

def test_merkle_tree_multiple_receipts():
    """Test Merkle tree with multiple receipts"""
    receipts = []
    for i in range(5):
        receipt = Receipt(
            dispute_id="test_dispute",
            step=f"STEP{i}",
            inputs={"index": i},
            params={},
            artifacts={},
            ts=datetime.utcnow().isoformat() + "Z"
        ).finalize()
        receipts.append(receipt)
    
    chain = ReceiptChain(
        dispute_id="test_dispute",
        receipts=receipts
    ).finalize()
    
    # Should have valid Merkle root
    assert chain.merkle_root.startswith("0x")
    assert len(chain.merkle_root) == 66
    
    # Should have correct number of leaves
    assert len(chain.merkle_leaves) == 5
    
    # Each leaf should be a valid hash
    for leaf in chain.merkle_leaves:
        assert leaf.startswith("0x")
        assert len(leaf) == 66

# Canonicalization tests
def test_json_canonicalization():
    """Test deterministic JSON canonicalization"""
    data = {
        "z": "last",
        "a": "first",
        "nested": {
            "y": 2,
            "x": 1
        },
        "list": [3, 2, 1]
    }
    
    # Canonicalize multiple times
    canon1 = json.dumps(data, **KANON)
    canon2 = json.dumps(data, **KANON)
    
    # Should be identical
    assert canon1 == canon2
    
    # Should have sorted keys
    assert canon1.index('"a"') < canon1.index('"z"')
    assert canon1.index('"x"') < canon1.index('"y"')
    
    # Should have no whitespace
    assert ' ' not in canon1
    assert '\n' not in canon1

def test_canonicalization_with_unicode():
    """Test canonicalization preserves Unicode"""
    data = {
        "name": "José",
        "city": "São Paulo",
        "emoji": "🔒"
    }
    
    canon = json.dumps(data, **KANON)
    
    # Should preserve Unicode (ensure_ascii=False)
    assert "José" in canon
    assert "São Paulo" in canon
    assert "🔒" in canon
    
    # Should not have escaped Unicode
    assert "\\u" not in canon

# IPFS failure handling tests
@patch('services.gateway.app.put_json')
def test_ipfs_failure_handling(mock_put_json, valid_dispute_request, mock_anchor_client):
    """Test handling of IPFS failures"""
    # Make IPFS fail
    mock_put_json.side_effect = Exception("IPFS connection failed")
    
    response = client.post("/submit-dispute", json=valid_dispute_request)
    
    # Should still succeed but without IPFS URI
    assert response.status_code == 201
    data = response.json()
    assert data["anchor_uri"] is None
    assert data["anchor_tx"] is None  # No anchor without IPFS
    
    # Dispute should still be stored
    assert data["dispute_id"] in dispute_storage

# Chain failure handling tests
@patch('services.gateway.app.anchor_client.anchor_root')
def test_chain_failure_handling(mock_anchor, valid_dispute_request, mock_ipfs):
    """Test handling of blockchain failures"""
    # Make chain anchoring fail
    mock_anchor.side_effect = Exception("Chain connection failed")
    
    response = client.post("/submit-dispute", json=valid_dispute_request)
    
    # Should still succeed but without anchor tx
    assert response.status_code == 201
    data = response.json()
    assert data["anchor_uri"] is not None  # IPFS should work
    assert data["anchor_tx"] is None  # No anchor tx due to failure
    
    # Dispute should still be stored
    assert data["dispute_id"] in dispute_storage

# Rate limiting tests
def test_rate_limiting(valid_dispute_request, mock_ipfs, mock_anchor_client):
    """Test rate limiting functionality"""
    # Should allow up to 30 requests per minute
    for i in range(30):
        response = client.post("/submit-dispute", json=valid_dispute_request)
        assert response.status_code == 201
    
    # 31st request should be rate limited
    response = client.post("/submit-dispute", json=valid_dispute_request)
    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded"

# PII redaction tests
def test_pii_redaction():
    """Test PII redaction in logging"""
    sensitive_data = {
        "parties": ["alice@example.com", "bob@example.com"],
        "email": "user@example.com",
        "name": "John Doe",
        "address": "123 Main St, City, State",
        "phone": "555-123-4567",
        "ssn": "123-45-6789",
        "safe_field": "This is safe"
    }
    
    redacted = redact_pii(sensitive_data)
    
    # Sensitive fields should be redacted
    assert redacted["parties"] == "[REDACTED]"
    assert redacted["email"] == "[REDACTED]"
    assert redacted["name"] == "[REDACTED]"
    assert redacted["address"] == "[REDACTED]"
    assert redacted["phone"] == "[REDACTED]"
    assert redacted["ssn"] == "[REDACTED]"
    
    # Non-sensitive fields should remain
    assert redacted["safe_field"] == "This is safe"

def test_pii_redaction_nested():
    """Test PII redaction in nested structures"""
    data = {
        "metadata": {
            "parties": ["party1", "party2"],
            "timestamp": "2024-01-01"
        },
        "list": [
            {"name": "Alice"},
            {"safe": "data"}
        ]
    }
    
    redacted = redact_pii(data)
    
    # Nested sensitive data should be redacted
    assert redacted["metadata"]["parties"] == "[REDACTED]"
    assert redacted["metadata"]["timestamp"] == "2024-01-01"
    assert redacted["list"][0]["name"] == "[REDACTED]"
    assert redacted["list"][1]["safe"] == "data"

# Middleware tests
def test_podp_middleware_adds_request_id():
    """Test that PoDP middleware adds request ID to responses"""
    response = client.get("/healthz")
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"].startswith("req_")

def test_middleware_logs_requests(caplog):
    """Test that middleware logs requests properly"""
    with caplog.at_level(logging.INFO):
        response = client.get("/healthz")
        
        # Should have request and response logs
        assert "Request received" in caplog.text
        assert "Request completed" in caplog.text

# Integration tests
def test_full_dispute_lifecycle(valid_dispute_request, mock_ipfs, mock_anchor_client):
    """Test complete dispute lifecycle from submission to status check"""
    # Submit dispute
    submit_response = client.post("/submit-dispute", json=valid_dispute_request)
    assert submit_response.status_code == 201
    
    dispute_data = submit_response.json()
    dispute_id = dispute_data["dispute_id"]
    
    # Check initial status
    status_response = client.get(f"/status/{dispute_id}")
    assert status_response.status_code == 200
    
    status_data = status_response.json()
    assert status_data["phase"] == "INTAKE"
    assert len(status_data["receipts"]) == 1
    
    # Verify receipt chain is stored
    assert dispute_id in receipt_chains
    chain = receipt_chains[dispute_id]
    assert chain.merkle_root is not None
    assert len(chain.receipts) == 1
    
    # Verify first receipt
    first_receipt = chain.receipts[0]
    assert first_receipt.step == "INTAKE_V1"
    assert first_receipt.dispute_id == dispute_id

def test_multiple_disputes_isolation(mock_ipfs, mock_anchor_client):
    """Test that multiple disputes are properly isolated"""
    # Submit first dispute
    dispute1_request = {
        "parties": ["alice@example.com", "bob@example.com"],
        "jurisdiction": "US-CA",
        "cid": "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
        "enc_meta": {"version": "1"}
    }
    response1 = client.post("/submit-dispute", json=dispute1_request)
    assert response1.status_code == 201
    dispute1_id = response1.json()["dispute_id"]
    
    # Submit second dispute
    dispute2_request = {
        "parties": ["charlie@example.com", "david@example.com"],
        "jurisdiction": "US-NY",
        "cid": "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
        "enc_meta": {"version": "2"}
    }
    response2 = client.post("/submit-dispute", json=dispute2_request)
    assert response2.status_code == 201
    dispute2_id = response2.json()["dispute_id"]
    
    # Verify disputes are different
    assert dispute1_id != dispute2_id
    
    # Check both statuses
    status1 = client.get(f"/status/{dispute1_id}")
    status2 = client.get(f"/status/{dispute2_id}")
    
    assert status1.status_code == 200
    assert status2.status_code == 200
    
    # Verify different receipt chains
    assert receipt_chains[dispute1_id].merkle_root != receipt_chains[dispute2_id].merkle_root

# Development endpoint tests
def test_list_disputes_endpoint(valid_dispute_request, mock_ipfs, mock_anchor_client):
    """Test development endpoint for listing disputes"""
    # Submit some disputes
    for i in range(3):
        request = valid_dispute_request.copy()
        request["cid"] = f"QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uc{i}"
        response = client.post("/submit-dispute", json=request)
        assert response.status_code == 201
    
    # List disputes (only works in non-production)
    response = client.get("/disputes")
    assert response.status_code == 200
    
    data = response.json()
    assert data["count"] == 3
    assert len(data["disputes"]) == 3
    
    # Verify dispute info is redacted/minimal
    for dispute in data["disputes"]:
        assert "dispute_id" in dispute
        assert "phase" in dispute
        assert "created_at" in dispute
        assert "has_anchor" in dispute
        # Should not have sensitive data
        assert "parties" not in dispute
        assert "cid" not in dispute

@patch.dict('os.environ', {'ENVIRONMENT': 'production'})
def test_list_disputes_blocked_in_production():
    """Test that list disputes endpoint is blocked in production"""
    response = client.get("/disputes")
    assert response.status_code == 403
    assert response.json()["detail"] == "Not available in production"

# Error handling tests
def test_500_error_handling():
    """Test internal server error handling"""
    with patch('services.gateway.app.Receipt.new_id') as mock_new_id:
        mock_new_id.side_effect = Exception("Unexpected error")
        
        request = {
            "parties": ["party1", "party2"],
            "jurisdiction": "US-CA",
            "cid": "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
            "enc_meta": {}
        }
        response = client.post("/submit-dispute", json=request)
        
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"

def test_root_endpoint():
    """Test root endpoint provides API info"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["message"] == "DALRN Gateway API"
    assert data["docs"] == "/docs"
    assert data["health"] == "/healthz"

# Performance tests
def test_receipt_generation_performance():
    """Test that receipt generation is reasonably fast"""
    import time
    
    start = time.time()
    for i in range(100):
        receipt = Receipt(
            dispute_id=f"test_{i}",
            step="PERF_TEST",
            inputs={"index": i},
            params={"test": True},
            artifacts={},
            ts=datetime.utcnow().isoformat() + "Z"
        ).finalize()
    end = time.time()
    
    # Should generate 100 receipts in under 1 second
    assert (end - start) < 1.0

def test_merkle_tree_performance():
    """Test Merkle tree generation performance"""
    import time
    
    # Create many receipts
    receipts = []
    for i in range(1000):
        receipt = Receipt(
            dispute_id="perf_test",
            step=f"STEP_{i}",
            inputs={"index": i},
            params={},
            artifacts={},
            ts=datetime.utcnow().isoformat() + "Z"
        ).finalize()
        receipts.append(receipt)
    
    # Time Merkle tree generation
    start = time.time()
    chain = ReceiptChain(
        dispute_id="perf_test",
        receipts=receipts
    ).finalize()
    end = time.time()
    
    # Should build tree for 1000 receipts in under 1 second
    assert (end - start) < 1.0
    assert chain.merkle_root is not None
    assert len(chain.merkle_leaves) == 1000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])