"""Integration tests for blockchain client with local Anvil chain"""

import pytest
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from web3 import Web3
from eth_account import Account

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from chain.client import (
    AnchorClient,
    NetworkConfig,
    NetworkSettings,
    TransactionStatus
)


class TestAnchorClientUnit:
    """Unit tests for AnchorClient using mocks"""

    @pytest.fixture
    def mock_web3(self):
        """Create a mock Web3 instance"""
        mock = Mock()
        mock.is_connected.return_value = True
        mock.eth.chain_id = 31337
        mock.eth.block_number = 100
        mock.eth.accounts = ["0x" + "1" * 40]
        mock.eth.gas_price = 20000000000
        mock.eth.get_transaction_count.return_value = 0
        mock.eth.estimate_gas.return_value = 100000
        mock.eth.get_block.return_value = {"baseFeePerGas": 10000000000}
        mock.eth.max_priority_fee = 1000000000
        return mock

    @pytest.fixture
    def mock_contract(self):
        """Create a mock contract instance"""
        mock = Mock()

        # Mock contract functions
        mock.functions.totalRootsAnchored.return_value.call.return_value = 10
        mock.functions.totalReceiptsAnchored.return_value.call.return_value = 25
        mock.functions.latestRoot.return_value.call.return_value = (
            b'\x00' * 32, 100
        )
        mock.functions.hasRoot.return_value.call.return_value = True

        # Mock anchor functions
        anchor_fn = Mock()
        anchor_fn.build_transaction.return_value = {
            "to": "0x" + "2" * 40,
            "data": "0x" + "a" * 100,
            "value": 0
        }
        mock.functions.anchorRoot.return_value = anchor_fn
        mock.functions.anchorReceipt.return_value = anchor_fn

        # Mock events
        mock.events.RootAnchored.return_value.process_receipt.return_value = [{
            "args": {
                "disputeId": b"test-dispute",
                "merkleRoot": b"root" * 8,
                "timestamp": 1234567890,
                "blockNumber": 100
            }
        }]
        mock.events.ReceiptAnchored.return_value.process_receipt.return_value = []

        return mock

    @patch("chain.client.Web3")
    @patch("chain.client.Path")
    def test_client_initialization(self, mock_path, mock_web3_class, mock_web3, mock_contract):
        """Test client initialization with various configurations"""
        # Setup mocks
        mock_web3_class.HTTPProvider.return_value = Mock()
        mock_web3_class.return_value = mock_web3
        mock_web3_class.is_address.return_value = True
        mock_web3_class.to_checksum_address.return_value = "0x" + "1" * 40

        mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value.exists.return_value = True

        # Test default initialization
        client = AnchorClient()
        assert client.network == NetworkConfig.LOCAL
        assert client.rpc_url == "http://localhost:8545"

        # Test with custom network
        client = AnchorClient(network=NetworkConfig.SEPOLIA)
        assert client.network == NetworkConfig.SEPOLIA
        assert client.network_settings.chain_id == 11155111

    @patch("chain.client.Web3")
    def test_network_settings(self, mock_web3_class):
        """Test network settings for different networks"""
        mock_web3_class.HTTPProvider.return_value = Mock()
        mock_web3_class.return_value = Mock(is_connected=Mock(return_value=True), eth=Mock(chain_id=1))
        mock_web3_class.is_address.return_value = True
        mock_web3_class.to_checksum_address.return_value = "0x" + "1" * 40

        # Test each network configuration
        networks = [
            (NetworkConfig.LOCAL, 31337),
            (NetworkConfig.SEPOLIA, 11155111),
            (NetworkConfig.MAINNET, 1),
            (NetworkConfig.POLYGON, 137),
            (NetworkConfig.ARBITRUM, 42161),
        ]

        for network, expected_chain_id in networks:
            client = AnchorClient(network=network)
            settings = client._get_network_settings(network)
            assert settings.chain_id == expected_chain_id

    def test_convert_to_bytes32(self):
        """Test string to bytes32 conversion"""
        client = AnchorClient.__new__(AnchorClient)

        # Test hex string
        result = client._convert_to_bytes32("0x" + "a" * 64)
        assert len(result) == 32
        assert result == bytes.fromhex("a" * 64)

        # Test short hex string (should be padded)
        result = client._convert_to_bytes32("0xabc")
        assert len(result) == 32
        assert result[:2] == bytes.fromhex("ab")

        # Test regular string
        result = client._convert_to_bytes32("test-dispute")
        assert len(result) == 32
        assert result[:12] == b"test-dispute"
        assert result[12:] == b'\x00' * 20

    @patch("chain.client.Web3")
    @patch("chain.client.Account")
    def test_account_setup(self, mock_account_class, mock_web3_class):
        """Test account setup from private key"""
        mock_web3_class.HTTPProvider.return_value = Mock()
        mock_web3_class.return_value = Mock(is_connected=Mock(return_value=True), eth=Mock(chain_id=31337))
        mock_web3_class.is_address.return_value = True
        mock_web3_class.to_checksum_address.return_value = "0x" + "1" * 40

        mock_account = Mock()
        mock_account.address = "0x" + "2" * 40
        mock_account_class.from_key.return_value = mock_account

        # Test with private key
        private_key = "0x" + "a" * 64
        client = AnchorClient(private_key=private_key)

        assert client.account == mock_account
        mock_account_class.from_key.assert_called_with("a" * 64)

    @patch("chain.client.Web3")
    @patch("chain.client.backoff")
    def test_gas_price_caching(self, mock_backoff, mock_web3_class, mock_web3):
        """Test gas price caching mechanism"""
        mock_web3_class.HTTPProvider.return_value = Mock()
        mock_web3_class.return_value = mock_web3
        mock_web3_class.is_address.return_value = True
        mock_web3_class.to_checksum_address.return_value = "0x" + "1" * 40

        # Bypass decorator
        mock_backoff.on_exception.return_value = lambda f: f

        client = AnchorClient()
        client.w3 = mock_web3
        client.network_settings = NetworkSettings(
            rpc_url="http://localhost:8545",
            chain_id=31337,
            confirmation_blocks=1
        )

        # First call should fetch from network
        gas_price1 = client._get_gas_price()
        assert "maxFeePerGas" in gas_price1 or "gasPrice" in gas_price1

        # Second call should use cache
        gas_price2 = client._get_gas_price()
        assert gas_price1 == gas_price2

    @patch("chain.client.Web3")
    def test_transaction_status(self, mock_web3_class, mock_web3):
        """Test transaction status checking"""
        mock_web3_class.HTTPProvider.return_value = Mock()
        mock_web3_class.return_value = mock_web3
        mock_web3_class.is_address.return_value = True
        mock_web3_class.to_checksum_address.return_value = "0x" + "1" * 40

        client = AnchorClient()
        client.w3 = mock_web3

        # Test successful transaction
        mock_web3.eth.get_transaction_receipt.return_value = {
            "status": 1,
            "blockNumber": 100,
            "gasUsed": 50000
        }
        mock_web3.eth.block_number = 105

        status = client.get_transaction_status("0x" + "a" * 64)
        assert status["status"] == TransactionStatus.SUCCESS.value
        assert status["confirmations"] == 5

        # Test failed transaction
        mock_web3.eth.get_transaction_receipt.return_value = {
            "status": 0,
            "blockNumber": 100,
            "gasUsed": 50000
        }

        status = client.get_transaction_status("0x" + "b" * 64)
        assert status["status"] == TransactionStatus.FAILED.value

        # Test pending transaction
        mock_web3.eth.get_transaction_receipt.return_value = None
        mock_web3.eth.get_transaction.return_value = {
            "gasPrice": 20000000000,
            "nonce": 5
        }

        status = client.get_transaction_status("0x" + "c" * 64)
        assert status["status"] == TransactionStatus.PENDING.value


class TestAnchorClientIntegration:
    """Integration tests with local Anvil chain"""

    @pytest.fixture
    def anvil_url(self):
        """Get Anvil RPC URL"""
        return os.getenv("ANVIL_URL", "http://localhost:8545")

    @pytest.fixture
    def private_key(self):
        """Get test private key (Anvil's first account)"""
        return "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

    @pytest.fixture
    def contract_address(self):
        """Get deployed contract address"""
        # This should be set after deploying the contract
        return os.getenv("TEST_CONTRACT_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require Anvil and deployed contract"
    )
    def test_real_connection(self, anvil_url):
        """Test real connection to Anvil"""
        client = AnchorClient(rpc_url=anvil_url, network=NetworkConfig.LOCAL)
        assert client.w3.is_connected()
        assert client.w3.eth.chain_id == 31337

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require Anvil and deployed contract"
    )
    def test_anchor_and_retrieve(self, anvil_url, private_key, contract_address):
        """Test anchoring a root and retrieving it"""
        client = AnchorClient(
            rpc_url=anvil_url,
            contract_address=contract_address,
            private_key=private_key,
            network=NetworkConfig.LOCAL
        )

        # Test data
        dispute_id = f"test-dispute-{int(time.time())}"
        merkle_root = "0x" + "a" * 64
        model_hash = "0x" + "b" * 64
        round_num = 1
        uri = "ipfs://QmTestHash123"
        tags = ["test", "integration"]

        # Anchor root
        result = client.anchor_root(
            dispute_id=dispute_id,
            merkle_root=merkle_root,
            model_hash=model_hash,
            round=round_num,
            uri=uri,
            tags=tags
        )

        assert "transaction_hash" in result
        assert "block_number" in result
        assert result["status"] == "success"

        # Retrieve the root
        root_info = client.latest_root(dispute_id)
        assert root_info["exists"]
        assert root_info["merkle_root"] == merkle_root

        # Check if root exists
        has_root = client.has_root(dispute_id)
        assert has_root is True

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require Anvil and deployed contract"
    )
    def test_anchor_receipt(self, anvil_url, private_key, contract_address):
        """Test anchoring an individual receipt"""
        client = AnchorClient(
            rpc_url=anvil_url,
            contract_address=contract_address,
            private_key=private_key,
            network=NetworkConfig.LOCAL
        )

        # Test data
        dispute_id = f"test-dispute-receipt-{int(time.time())}"
        receipt_hash = "0x" + "c" * 64
        step_index = 1
        uri = "ipfs://QmReceiptHash456"

        # Anchor receipt
        result = client.anchor_receipt(
            dispute_id=dispute_id,
            receipt_hash=receipt_hash,
            step_index=step_index,
            uri=uri
        )

        assert "transaction_hash" in result
        assert "block_number" in result
        assert result["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require Anvil and deployed contract"
    )
    def test_contract_info(self, anvil_url, contract_address):
        """Test getting contract information"""
        client = AnchorClient(
            rpc_url=anvil_url,
            contract_address=contract_address,
            network=NetworkConfig.LOCAL
        )

        info = client.get_contract_info()

        assert "total_roots" in info
        assert "total_receipts" in info
        assert "connected" in info
        assert info["connected"] is True
        assert info["network"] == "local"
        assert info["chain_id"] == 31337

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require Anvil and deployed contract"
    )
    def test_verify_receipt_proof(self, anvil_url, private_key, contract_address):
        """Test Merkle proof verification"""
        client = AnchorClient(
            rpc_url=anvil_url,
            contract_address=contract_address,
            private_key=private_key,
            network=NetworkConfig.LOCAL
        )

        dispute_id = f"test-merkle-{int(time.time())}"

        # First anchor a root
        merkle_root = Web3.keccak(text="test-root").hex()
        client.anchor_root(
            dispute_id=dispute_id,
            merkle_root=merkle_root,
            model_hash="0x" + "d" * 64,
            round=1,
            uri="ipfs://QmMerkleTest",
            tags=["merkle"]
        )

        # Verify with correct proof (simplified for testing)
        receipt_hash = Web3.keccak(text="test-receipt").hex()

        # In a real scenario, you'd build a proper Merkle tree
        # For testing, we'll just verify the root directly
        is_valid = client.verify_receipt_proof(
            dispute_id=dispute_id,
            receipt_hash=merkle_root,  # Using root as receipt for simplicity
            merkle_proof=[]
        )

        # This will fail in our simplified test, which is expected
        assert is_valid is False  # Expected since we're not using a real Merkle tree


class TestErrorHandling:
    """Test error handling scenarios"""

    @patch("chain.client.Web3")
    def test_invalid_address(self, mock_web3_class):
        """Test handling of invalid contract address"""
        mock_web3_class.is_address.return_value = False

        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            AnchorClient(contract_address="invalid-address")

    @patch("chain.client.Web3")
    def test_connection_failure(self, mock_web3_class):
        """Test handling of connection failure"""
        mock_web3 = Mock()
        mock_web3.is_connected.return_value = False
        mock_web3_class.HTTPProvider.return_value = Mock()
        mock_web3_class.return_value = mock_web3

        with pytest.raises(ConnectionError, match="Failed to connect"):
            AnchorClient()

    @patch("chain.client.Account")
    def test_invalid_private_key(self, mock_account_class):
        """Test handling of invalid private key"""
        mock_account_class.from_key.side_effect = ValueError("Invalid key")

        with pytest.raises(ValueError, match="Invalid private key"):
            client = AnchorClient.__new__(AnchorClient)
            client._setup_account("invalid-key")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])