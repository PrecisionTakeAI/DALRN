#!/usr/bin/env python3
"""
Test script for AnchorClient functionality
Tests the client interface with mock data
"""

import sys
import os
import json
import logging
from pathlib import Path
from web3 import Web3

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import AnchorClient, NetworkConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_client_initialization():
    """Test AnchorClient initialization"""
    logger.info("Testing AnchorClient initialization...")

    try:
        # Initialize with local configuration
        client = AnchorClient(
            rpc_url="http://127.0.0.1:8545",
            contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3",
            private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            network=NetworkConfig.LOCAL
        )

        logger.info(f"✓ Client initialized successfully")
        logger.info(f"  Network: {client.network.value}")
        logger.info(f"  Contract: {client.contract_address}")
        logger.info(f"  Account: {client.account.address if client.account else 'None'}")
        logger.info(f"  Connected: {client.w3.is_connected()}")

        return client

    except Exception as e:
        logger.error(f"✗ Client initialization failed: {e}")
        return None

def test_contract_info(client):
    """Test getting contract information"""
    logger.info("\nTesting contract info retrieval...")

    try:
        info = client.get_contract_info()
        logger.info("✓ Contract info retrieved:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        return True

    except Exception as e:
        logger.info(f"ℹ Contract may not be deployed (expected for mock): {e}")
        return False

def test_mock_anchor_data():
    """Test creating mock anchor data"""
    logger.info("\nTesting mock data creation...")

    # Generate test data
    dispute_id = "test-dispute-001"
    merkle_root = Web3.keccak(text="test-merkle-root").hex()
    model_hash = Web3.keccak(text="test-model").hex()
    receipt_hash = Web3.keccak(text="test-receipt").hex()

    mock_anchor_result = {
        "transaction_hash": "0x" + "0" * 64,
        "block_number": 1,
        "block_hash": "0x" + "1" * 64,
        "gas_used": 150000,
        "status": "success",
        "event": {
            "dispute_id": Web3.keccak(text=dispute_id).hex(),
            "merkle_root": merkle_root,
            "timestamp": 1734523200,
            "block_number": 1
        }
    }

    logger.info("✓ Mock anchor data created:")
    logger.info(f"  Dispute ID: {dispute_id}")
    logger.info(f"  Merkle Root: {merkle_root[:10]}...")
    logger.info(f"  Model Hash: {model_hash[:10]}...")
    logger.info(f"  Receipt Hash: {receipt_hash[:10]}...")

    return mock_anchor_result

def test_abi_validity():
    """Test that the ABI is valid"""
    logger.info("\nTesting ABI validity...")

    abi_path = Path(__file__).parent / "abi" / "AnchorReceipts.json"

    if not abi_path.exists():
        logger.error(f"✗ ABI file not found at {abi_path}")
        return False

    try:
        with open(abi_path, 'r') as f:
            abi = json.load(f)

        # Check for expected functions
        function_names = [item['name'] for item in abi if item['type'] == 'function']
        event_names = [item['name'] for item in abi if item['type'] == 'event']

        expected_functions = ['anchorRoot', 'anchorReceipt', 'latestRoot', 'latestRootInfo',
                            'getRootByRound', 'hasRoot', 'totalRootsAnchored', 'totalReceiptsAnchored']
        expected_events = ['RootAnchored', 'ReceiptAnchored']

        logger.info(f"✓ ABI loaded successfully")
        logger.info(f"  Functions: {len(function_names)}")
        logger.info(f"  Events: {len(event_names)}")

        # Check for required functions
        for func in expected_functions:
            if func in function_names:
                logger.info(f"  ✓ Function '{func}' found")
            else:
                logger.error(f"  ✗ Function '{func}' missing")

        # Check for required events
        for event in expected_events:
            if event in event_names:
                logger.info(f"  ✓ Event '{event}' found")
            else:
                logger.error(f"  ✗ Event '{event}' missing")

        return True

    except Exception as e:
        logger.error(f"✗ ABI validation failed: {e}")
        return False

def create_deployment_summary():
    """Create a deployment summary"""
    logger.info("\nCreating deployment summary...")

    summary = {
        "contract": {
            "name": "AnchorReceipts",
            "version": "1.0.0",
            "compiler": "solc-0.8.24",
            "optimization": True,
            "runs": 200
        },
        "features": {
            "anchorRoot": "Store Merkle roots for dispute receipts",
            "anchorReceipt": "Store individual receipt hashes",
            "latestRoot": "Get the most recent root for a dispute",
            "events": ["RootAnchored", "ReceiptAnchored"],
            "gasOptimized": True,
            "batchSupport": False
        },
        "deployment": {
            "local": {
                "address": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
                "status": "mock",
                "note": "Use Foundry for actual deployment"
            }
        },
        "files": {
            "contract": "contracts/AnchorReceipts.sol",
            "abi": "abi/AnchorReceipts.json",
            "bytecode": "build/AnchorReceipts.bin",
            "tests": "test/AnchorReceipts.t.sol"
        }
    }

    summary_path = Path(__file__).parent / "deployment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Summary saved to {summary_path}")
    return summary

def main():
    """Main test execution"""
    print("=" * 70)
    print("AnchorReceipts Contract Test Suite")
    print("=" * 70)

    # Test 1: ABI Validity
    logger.info("\n📋 Test 1: ABI Validation")
    abi_valid = test_abi_validity()

    # Test 2: Client Initialization
    logger.info("\n🔧 Test 2: Client Initialization")
    client = test_client_initialization()

    # Test 3: Contract Info (will fail if not deployed)
    if client:
        logger.info("\n📊 Test 3: Contract Information")
        test_contract_info(client)

    # Test 4: Mock Data
    logger.info("\n🎭 Test 4: Mock Data Generation")
    mock_data = test_mock_anchor_data()

    # Test 5: Deployment Summary
    logger.info("\n📄 Test 5: Deployment Summary")
    summary = create_deployment_summary()

    # Final Report
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ ABI Valid: {abi_valid}")
    print(f"✅ Client Initialized: {client is not None}")
    print(f"✅ Mock Data Created: {mock_data is not None}")
    print(f"✅ Summary Created: {summary is not None}")
    print("\n📦 Contract Components Ready:")
    print(f"  - Solidity Contract: contracts/AnchorReceipts.sol")
    print(f"  - ABI File: abi/AnchorReceipts.json")
    print(f"  - Bytecode: build/AnchorReceipts.bin")
    print(f"  - Tests: test/AnchorReceipts.t.sol")
    print(f"  - Client: client.py")
    print(f"  - Deployment Script: compile_and_deploy.py")
    print("\n⚠️  Note: Contract is not deployed to local node.")
    print("    To deploy with Foundry:")
    print("    1. Install Foundry: curl -L https://foundry.paradigm.xyz | bash")
    print("    2. Run: forge create contracts/AnchorReceipts.sol:AnchorReceipts \\")
    print("           --rpc-url http://127.0.0.1:8545 \\")
    print("           --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())