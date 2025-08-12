"""
Comprehensive test suite for the Epsilon-Ledger Service.

Tests privacy budget tracking, FL integration, and Gateway integration.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

# Import the epsilon ledger service
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'fl'))
from eps_ledger import (
    app, EpsilonLedger, PreCheckRequest, CommitRequest,
    AccountantType, DEFAULT_BUDGET, ADMIN_TOKEN
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def ledger():
    """Create a test epsilon ledger instance with temporary storage."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        ledger = EpsilonLedger(storage_path=temp_path)
        yield ledger
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.fixture
def sample_precheck_request():
    """Sample pre-check request."""
    return {
        "tenant_id": "tenant_001",
        "model_id": "model_001",
        "eps_round": 0.5,
        "delta_round": 1e-5,
        "accountant": "rdp"
    }


@pytest.fixture
def sample_commit_request():
    """Sample commit request."""
    return {
        "tenant_id": "tenant_001",
        "model_id": "model_001",
        "round": 1,
        "accountant": "rdp",
        "epsilon": 0.5,
        "delta": 1e-5,
        "clipping_C": 1.0,
        "sigma": 1.2,
        "batch_size": 32,
        "dataset_size": 10000,
        "num_clients": 10,
        "aggregation_method": "fedavg",
        "framework": "flower",
        "metadata": {"test": "data"}
    }


# ============================================================================
# Unit Tests - EpsilonLedger Class
# ============================================================================

class TestEpsilonLedger:
    """Test the EpsilonLedger class functionality."""
    
    def test_initialization(self, ledger):
        """Test ledger initialization."""
        assert ledger.ledgers == {}
        assert ledger.storage_path is not None
        assert ledger.lock is not None
    
    def test_precheck_new_ledger(self, ledger):
        """Test pre-check for a new tenant/model pair."""
        request = PreCheckRequest(
            tenant_id="test_tenant",
            model_id="test_model",
            eps_round=1.0
        )
        
        response = ledger.precheck(request)
        
        assert response.allowed is True
        assert response.remaining_budget == DEFAULT_BUDGET
        assert response.total_spent == 0.0
        assert response.projected_spend == 1.0
    
    def test_precheck_budget_exceeded(self, ledger):
        """Test pre-check when budget would be exceeded."""
        request = PreCheckRequest(
            tenant_id="test_tenant",
            model_id="test_model",
            eps_round=DEFAULT_BUDGET + 1.0
        )
        
        response = ledger.precheck(request)
        
        assert response.allowed is False
        assert response.remaining_budget == DEFAULT_BUDGET
        assert "denied" in response.message.lower()
    
    def test_commit_success(self, ledger):
        """Test successful commit of privacy spend."""
        request = CommitRequest(
            tenant_id="test_tenant",
            model_id="test_model",
            round=1,
            accountant=AccountantType.RDP,
            epsilon=1.0,
            delta=1e-5
        )
        
        response = ledger.commit(request)
        
        assert response.success is True
        assert response.total_spent == 1.0
        assert response.remaining_budget == DEFAULT_BUDGET - 1.0
        assert response.entry_id is not None
    
    def test_commit_exceeds_budget(self, ledger):
        """Test commit that would exceed budget."""
        from fastapi import HTTPException
        
        request = CommitRequest(
            tenant_id="test_tenant",
            model_id="test_model",
            round=1,
            accountant=AccountantType.RDP,
            epsilon=DEFAULT_BUDGET + 1.0,
            delta=1e-5
        )
        
        with pytest.raises(HTTPException) as exc_info:
            ledger.commit(request)
        
        assert exc_info.value.status_code == 400
        assert "exceed budget" in exc_info.value.detail
    
    def test_multiple_commits_composition(self, ledger):
        """Test privacy composition across multiple commits."""
        tenant_id = "test_tenant"
        model_id = "test_model"
        
        # Commit multiple rounds
        for round_num in range(3):
            request = CommitRequest(
                tenant_id=tenant_id,
                model_id=model_id,
                round=round_num,
                accountant=AccountantType.RDP,
                epsilon=0.5,
                delta=1e-5
            )
            response = ledger.commit(request)
            assert response.success is True
        
        # Check total spent
        status = ledger.get_budget_status(tenant_id, model_id)
        assert status.total_spent == 1.5  # 3 * 0.5
        assert status.num_rounds == 3
        assert status.entries_count == 3
    
    def test_get_history(self, ledger):
        """Test retrieving privacy spend history."""
        tenant_id = "test_tenant"
        model_id = "test_model"
        
        # Create some history
        for i in range(5):
            request = CommitRequest(
                tenant_id=tenant_id,
                model_id=model_id,
                round=i,
                accountant=AccountantType.RDP,
                epsilon=0.1 * (i + 1),
                delta=1e-5
            )
            ledger.commit(request)
        
        # Get history
        history = ledger.get_history(tenant_id, model_id, limit=3)
        
        assert len(history) == 3
        assert history[-1].round == 4
        assert history[-1].epsilon == 0.5
    
    def test_reset_budget(self, ledger):
        """Test resetting privacy budget."""
        tenant_id = "test_tenant"
        model_id = "test_model"
        
        # Spend some budget
        request = CommitRequest(
            tenant_id=tenant_id,
            model_id=model_id,
            round=1,
            accountant=AccountantType.RDP,
            epsilon=2.0,
            delta=1e-5
        )
        ledger.commit(request)
        
        # Reset budget
        success = ledger.reset_budget(tenant_id, model_id, new_budget=10.0)
        assert success is True
        
        # Check reset worked
        status = ledger.get_budget_status(tenant_id, model_id)
        assert status.total_budget == 10.0
        assert status.total_spent == 0.0
        assert status.entries_count == 0
    
    def test_persistence(self):
        """Test ledger persistence to storage."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create ledger and add data
            ledger1 = EpsilonLedger(storage_path=temp_path)
            request = CommitRequest(
                tenant_id="test_tenant",
                model_id="test_model",
                round=1,
                accountant=AccountantType.RDP,
                epsilon=1.5,
                delta=1e-5
            )
            ledger1.commit(request)
            
            # Create new ledger instance and load data
            ledger2 = EpsilonLedger(storage_path=temp_path)
            
            # Verify data was persisted
            status = ledger2.get_budget_status("test_tenant", "test_model")
            assert status.total_spent == 1.5
            assert status.entries_count == 1
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# ============================================================================
# Integration Tests - FastAPI Endpoints
# ============================================================================

class TestAPIEndpoints:
    """Test the FastAPI endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "epsilon-ledger"
    
    def test_precheck_endpoint(self, client, sample_precheck_request):
        """Test pre-check endpoint."""
        response = client.post("/precheck", json=sample_precheck_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "allowed" in data
        assert "remaining_budget" in data
        assert "total_spent" in data
        assert "projected_spend" in data
    
    def test_commit_endpoint(self, client, sample_commit_request):
        """Test commit endpoint."""
        response = client.post("/commit", json=sample_commit_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "total_spent" in data
        assert "remaining_budget" in data
        assert "entry_id" in data
    
    def test_budget_status_endpoint(self, client):
        """Test budget status endpoint."""
        response = client.get("/budget/tenant_001/model_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["tenant_id"] == "tenant_001"
        assert data["model_id"] == "model_001"
        assert "total_budget" in data
        assert "total_spent" in data
        assert "remaining_budget" in data
    
    def test_history_endpoint(self, client, sample_commit_request):
        """Test history endpoint."""
        # Create some history
        for i in range(3):
            sample_commit_request["round"] = i
            client.post("/commit", json=sample_commit_request)
        
        # Get history
        response = client.get("/history/tenant_001/model_001?limit=2")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        assert all("entry_id" in entry for entry in data)
        assert all("round" in entry for entry in data)
    
    def test_reset_endpoint_unauthorized(self, client):
        """Test reset endpoint without admin token."""
        response = client.post("/reset/tenant_001/model_001")
        assert response.status_code == 422  # Missing header
    
    def test_reset_endpoint_authorized(self, client):
        """Test reset endpoint with admin token."""
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        response = client.post(
            "/reset/tenant_001/model_001?new_budget=8.0",
            headers=headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["new_budget"] == 8.0


# ============================================================================
# Federated Learning Integration Tests
# ============================================================================

class TestFederatedLearningIntegration:
    """Test FL framework integration endpoints."""
    
    def test_fl_preround_flower(self, client):
        """Test FL pre-round check for Flower framework."""
        request = {
            "framework": "flower",
            "tenant_id": "fl_tenant",
            "model_id": "fl_model",
            "num_rounds": 10,
            "batch_size": 32,
            "noise_multiplier": 1.0
        }
        
        response = client.post("/fl/preround", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["framework"] == "flower"
        assert "allowed" in data
        assert "remaining_budget" in data
        assert "recommended_params" in data
        assert "max_rounds" in data["recommended_params"]
    
    def test_fl_preround_nvflare(self, client):
        """Test FL pre-round check for NV-FLARE framework."""
        request = {
            "framework": "nvflare",
            "tenant_id": "fl_tenant",
            "project_name": "fl_project",
            "privacy_config": {
                "epsilon_per_round": 0.3
            }
        }
        
        response = client.post("/fl/preround", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["framework"] == "nvflare"
        assert "allowed" in data
    
    def test_fl_postround_flower(self, client):
        """Test FL post-round commit for Flower framework."""
        request = {
            "framework": "flower",
            "tenant_id": "fl_tenant",
            "model_id": "fl_model",
            "round": 1,
            "epsilon": 0.5,
            "delta": 1e-5,
            "clipping_norm": 1.0,
            "noise_multiplier": 1.2,
            "batch_size": 32,
            "dataset_size": 10000,
            "num_clients": 5,
            "aggregation_method": "fedavg"
        }
        
        response = client.post("/fl/postround", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["framework"] == "flower"
        assert data["success"] is True
        assert "total_spent" in data
        assert "remaining_budget" in data


# ============================================================================
# Budget Exhaustion Tests
# ============================================================================

class TestBudgetExhaustion:
    """Test budget exhaustion scenarios."""
    
    def test_budget_exhaustion_detection(self, ledger):
        """Test that budget exhaustion is properly detected."""
        tenant_id = "exhaust_tenant"
        model_id = "exhaust_model"
        
        # Spend most of the budget
        request = CommitRequest(
            tenant_id=tenant_id,
            model_id=model_id,
            round=1,
            accountant=AccountantType.RDP,
            epsilon=DEFAULT_BUDGET - 0.1,
            delta=1e-5
        )
        response = ledger.commit(request)
        assert response.success is True
        
        # Try to exceed budget
        precheck = PreCheckRequest(
            tenant_id=tenant_id,
            model_id=model_id,
            eps_round=0.2
        )
        response = ledger.precheck(precheck)
        assert response.allowed is False
        assert "denied" in response.message.lower()
    
    def test_concurrent_access_safety(self, ledger):
        """Test thread-safe concurrent access."""
        import threading
        import time
        
        tenant_id = "concurrent_tenant"
        model_id = "concurrent_model"
        errors = []
        
        def commit_epsilon(round_num):
            try:
                request = CommitRequest(
                    tenant_id=tenant_id,
                    model_id=model_id,
                    round=round_num,
                    accountant=AccountantType.RDP,
                    epsilon=0.1,
                    delta=1e-5
                )
                ledger.commit(request)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=commit_epsilon, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0
        status = ledger.get_budget_status(tenant_id, model_id)
        assert status.total_spent == 1.0  # 10 * 0.1


# ============================================================================
# Robust Aggregation Tests
# ============================================================================

class TestRobustAggregation:
    """Test robust aggregation support."""
    
    def test_aggregation_metadata_tracking(self, ledger):
        """Test that aggregation metadata is properly tracked."""
        request = CommitRequest(
            tenant_id="agg_tenant",
            model_id="agg_model",
            round=1,
            accountant=AccountantType.RDP,
            epsilon=0.5,
            delta=1e-5,
            aggregation_method="krum",
            num_clients=10,
            metadata={
                "byzantine_threshold": 2,
                "outliers_removed": 1
            }
        )
        
        response = ledger.commit(request)
        assert response.success is True
        
        # Check history includes aggregation info
        history = ledger.get_history("agg_tenant", "agg_model")
        assert len(history) == 1
        assert history[0].metadata.get("byzantine_threshold") == 2
    
    def test_different_aggregation_methods(self, client):
        """Test support for different aggregation methods."""
        methods = ["fedavg", "krum", "multi-krum", "trimmed-mean", "median"]
        
        for i, method in enumerate(methods):
            request = {
                "tenant_id": "agg_test",
                "model_id": f"model_{method}",
                "round": i,
                "accountant": "rdp",
                "epsilon": 0.3,
                "delta": 1e-5,
                "aggregation_method": method,
                "num_clients": 10
            }
            
            response = client.post("/commit", json=request)
            assert response.status_code == 200
            assert response.json()["success"] is True


# ============================================================================
# Privacy Accounting Tests
# ============================================================================

class TestPrivacyAccounting:
    """Test privacy accounting functionality."""
    
    def test_different_accountants(self, client):
        """Test support for different privacy accountants."""
        accountants = ["rdp", "gaussian", "basic"]
        
        for accountant in accountants:
            request = {
                "tenant_id": "acc_test",
                "model_id": f"model_{accountant}",
                "round": 1,
                "accountant": accountant,
                "epsilon": 0.5,
                "delta": 1e-5
            }
            
            response = client.post("/commit", json=request)
            assert response.status_code == 200
            assert response.json()["success"] is True
    
    def test_privacy_parameters_tracking(self, ledger):
        """Test that all privacy parameters are tracked."""
        request = CommitRequest(
            tenant_id="param_tenant",
            model_id="param_model",
            round=1,
            accountant=AccountantType.RDP,
            epsilon=0.5,
            delta=1e-5,
            clipping_C=1.0,
            sigma=1.2,
            batch_size=32,
            dataset_size=10000
        )
        
        response = ledger.commit(request)
        assert response.success is True
        
        # Verify parameters are stored
        key = ("param_tenant", "param_model")
        assert key in ledger.ledgers
        entry = ledger.ledgers[key]["entries"][0]
        assert entry["clipping_C"] == 1.0
        assert entry["sigma"] == 1.2
        assert entry["batch_size"] == 32
        assert entry["dataset_size"] == 10000


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_epsilon_value(self, client):
        """Test handling of invalid epsilon values."""
        request = {
            "tenant_id": "test",
            "model_id": "test",
            "eps_round": -1.0  # Invalid negative epsilon
        }
        
        response = client.post("/precheck", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_delta_value(self, client):
        """Test handling of invalid delta values."""
        request = {
            "tenant_id": "test",
            "model_id": "test",
            "round": 1,
            "accountant": "rdp",
            "epsilon": 0.5,
            "delta": 2.0  # Invalid delta > 1
        }
        
        response = client.post("/commit", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        request = {
            "tenant_id": "test"
            # Missing model_id and other required fields
        }
        
        response = client.post("/precheck", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_unsupported_framework(self, client):
        """Test handling of unsupported FL framework."""
        request = {
            "framework": "unknown_framework",
            "tenant_id": "test",
            "model_id": "test"
        }
        
        response = client.post("/fl/preround", json=request)
        assert response.status_code == 500
        assert "Unsupported framework" in response.json()["detail"]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])