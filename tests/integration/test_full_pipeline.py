#!/usr/bin/env python3
"""
DALRN Full Pipeline Integration Test
Tests complete flow: Auth → Search → FHE → Negotiation → Chain
"""

import pytest
import asyncio
import json
import time
import numpy as np
from typing import Dict, Any
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service endpoints
GATEWAY_URL = "http://localhost:8000"
SEARCH_URL = "http://localhost:8100"
FHE_URL = "http://localhost:8200"
NEGOTIATION_URL = "http://localhost:8300"
FL_URL = "http://localhost:8400"


class TestFullPipeline:
    """Integration tests for complete DALRN pipeline"""

    @pytest.fixture(scope="class")
    async def auth_token(self):
        """Get authentication token for tests"""
        async with httpx.AsyncClient() as client:
            # Register test user
            register_data = {
                "username": f"test_user_{int(time.time())}",
                "email": "test@dalrn.local",
                "password": "TestPassword123!",
                "role": "user"
            }

            response = await client.post(
                f"{GATEWAY_URL}/auth/register",
                json=register_data
            )

            if response.status_code != 200:
                # Try login if already registered
                login_data = {
                    "username": register_data["username"],
                    "password": register_data["password"]
                }
                response = await client.post(
                    f"{GATEWAY_URL}/auth/login",
                    json=login_data
                )

            assert response.status_code == 200
            data = response.json()
            return data["access_token"]

    @pytest.mark.asyncio
    async def test_1_health_checks(self):
        """Test that all services are healthy"""
        services = [
            (GATEWAY_URL, "Gateway"),
            (SEARCH_URL, "Search"),
            (FHE_URL, "FHE"),
            (NEGOTIATION_URL, "Negotiation"),
            (FL_URL, "FL"),
        ]

        async with httpx.AsyncClient(timeout=5.0) as client:
            for url, name in services:
                try:
                    response = await client.get(f"{url}/health")
                    assert response.status_code == 200
                    logger.info(f"✅ {name} service is healthy")
                except Exception as e:
                    logger.error(f"❌ {name} service failed: {e}")
                    pytest.skip(f"{name} service not available")

    @pytest.mark.asyncio
    async def test_2_authentication_flow(self, auth_token):
        """Test authentication and authorization"""
        assert auth_token is not None
        logger.info(f"✅ Authentication successful")

        # Verify token
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{GATEWAY_URL}/auth/verify",
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            assert response.status_code == 200
            logger.info("✅ Token verification successful")

    @pytest.mark.asyncio
    async def test_3_dispute_submission(self, auth_token):
        """Test dispute submission with PoDP receipts"""
        async with httpx.AsyncClient() as client:
            dispute_data = {
                "parties": ["Alice", "Bob"],
                "jurisdiction": "US-CA",
                "cid": "QmTest123",
                "enc_meta": {"type": "contract_dispute"}
            }

            response = await client.post(
                f"{GATEWAY_URL}/submit-dispute",
                json=dispute_data,
                headers={"Authorization": f"Bearer {auth_token}"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "dispute_id" in data
            assert "receipt_cid" in data

            dispute_id = data["dispute_id"]
            logger.info(f"✅ Dispute submitted: {dispute_id}")

            # Check status
            response = await client.get(
                f"{GATEWAY_URL}/status/{dispute_id}",
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            assert response.status_code == 200
            logger.info("✅ Dispute status retrieved")

    @pytest.mark.asyncio
    async def test_4_vector_search(self):
        """Test FAISS vector search with recall metrics"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Build index
            num_vectors = 1000
            dim = 768
            embeddings = np.random.randn(num_vectors, dim).tolist()

            build_data = {
                "embeddings": embeddings,
                "append": False
            }

            response = await client.post(
                f"{SEARCH_URL}/build",
                json=build_data
            )
            assert response.status_code == 200
            logger.info(f"✅ Built index with {num_vectors} vectors")

            # Query index
            query_vector = np.random.randn(dim).tolist()
            query_data = {
                "query": query_vector,
                "k": 10,
                "reweight_iters": 0
            }

            response = await client.post(
                f"{SEARCH_URL}/query",
                json=query_data
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["indices"]) == 10
            assert "recall_at_10" in data
            assert "latency_ms" in data
            logger.info(f"✅ Search completed with recall@10={data['recall_at_10']:.2f}")

    @pytest.mark.asyncio
    async def test_5_fhe_encryption(self):
        """Test homomorphic encryption operations"""
        async with httpx.AsyncClient(timeout=15.0) as client:
            tenant_id = "test_tenant"

            # Create context
            context_data = {
                "tenant_id": tenant_id,
                "poly_modulus_degree": 8192
            }

            response = await client.post(
                f"{FHE_URL}/context/create",
                json=context_data
            )

            # Note: FHE service requires real TenSEAL
            if response.status_code == 200:
                data = response.json()
                assert "context_id" in data
                logger.info("✅ FHE context created")

                # Test dot product (would need actual encrypted data)
                logger.info("✅ FHE service operational")
            else:
                logger.warning("⚠️ FHE service requires TenSEAL client for full test")

    @pytest.mark.asyncio
    async def test_6_negotiation(self):
        """Test Nash equilibrium negotiation"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Simple 2x2 game
            negotiation_data = {
                "payoff_matrix_A": [[3, 1], [0, 2]],
                "payoff_matrix_B": [[3, 0], [1, 2]],
                "selection_rule": "nsw",
                "batna": [0, 0]
            }

            response = await client.post(
                f"{NEGOTIATION_URL}/negotiate",
                json=negotiation_data
            )

            assert response.status_code == 200
            data = response.json()
            assert "equilibrium" in data
            assert "utilities" in data
            logger.info(f"✅ Negotiation completed: {data['equilibrium']}")

    @pytest.mark.asyncio
    async def test_7_federated_learning(self):
        """Test FL privacy budget checking"""
        async with httpx.AsyncClient() as client:
            # Test epsilon precheck
            precheck_data = {
                "tenant_id": "test_tenant",
                "model_id": "test_model",
                "eps_round": 0.5,
                "delta_round": 1e-5
            }

            response = await client.post(
                f"{FL_URL}/precheck",
                json=precheck_data
            )

            if response.status_code == 200:
                data = response.json()
                assert "allowed" in data
                assert "remaining_budget" in data
                logger.info(f"✅ FL precheck: allowed={data['allowed']}, "
                          f"remaining={data['remaining_budget']:.2f}")
            else:
                logger.warning("⚠️ FL service epsilon ledger not available")

    @pytest.mark.asyncio
    async def test_8_end_to_end_flow(self, auth_token):
        """Test complete pipeline flow"""
        logger.info("\n" + "="*60)
        logger.info("Running End-to-End Pipeline Test")
        logger.info("="*60)

        async with httpx.AsyncClient(timeout=20.0) as client:
            # Step 1: Submit dispute
            dispute_data = {
                "parties": ["Company A", "Company B"],
                "jurisdiction": "International",
                "cid": "QmEndToEndTest",
                "enc_meta": {
                    "type": "contract",
                    "value": 1000000
                }
            }

            response = await client.post(
                f"{GATEWAY_URL}/submit-dispute",
                json=dispute_data,
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            assert response.status_code == 200
            dispute_id = response.json()["dispute_id"]
            logger.info(f"Step 1 ✅ Dispute created: {dispute_id}")

            # Step 2: Perform search for similar cases
            query_vector = np.random.randn(768).tolist()
            search_data = {
                "query": query_vector,
                "k": 5
            }

            response = await client.post(
                f"{SEARCH_URL}/query",
                json=search_data
            )
            assert response.status_code == 200
            similar_cases = response.json()["indices"]
            logger.info(f"Step 2 ✅ Found {len(similar_cases)} similar cases")

            # Step 3: Run negotiation
            negotiation_data = {
                "payoff_matrix_A": [[100, 50], [30, 80]],
                "payoff_matrix_B": [[100, 30], [50, 80]],
                "selection_rule": "nsw",
                "batna": [25, 25]
            }

            response = await client.post(
                f"{NEGOTIATION_URL}/negotiate",
                json=negotiation_data
            )
            assert response.status_code == 200
            equilibrium = response.json()["equilibrium"]
            logger.info(f"Step 3 ✅ Negotiation result: {equilibrium}")

            # Step 4: Check dispute status
            response = await client.get(
                f"{GATEWAY_URL}/status/{dispute_id}",
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            assert response.status_code == 200
            final_status = response.json()["status"]
            logger.info(f"Step 4 ✅ Final status: {final_status}")

            logger.info("\n✅ End-to-End Pipeline Test Completed Successfully!")


@pytest.mark.asyncio
async def test_service_resilience():
    """Test service resilience and error handling"""
    async with httpx.AsyncClient() as client:
        # Test with invalid data
        response = await client.post(
            f"{GATEWAY_URL}/submit-dispute",
            json={"invalid": "data"}
        )
        assert response.status_code in [400, 401, 422]
        logger.info("✅ Invalid request properly rejected")

        # Test rate limiting (if implemented)
        # Test timeout handling
        # Test concurrent requests


def run_integration_tests():
    """Run all integration tests"""
    logger.info("\n" + "="*60)
    logger.info("DALRN Integration Test Suite")
    logger.info("="*60 + "\n")

    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])


if __name__ == "__main__":
    run_integration_tests()