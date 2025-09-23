"""
Real blockchain integration with Ethereum/Polygon
PRD REQUIREMENT: Real blockchain anchoring, not mocked
"""
from web3 import Web3
from eth_account import Account
from eth_utils import to_checksum_address
import json
import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChainConfig:
    """Blockchain network configuration"""
    name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    gas_price_gwei: int = 30
    gas_limit: int = 200000

# Network configurations
NETWORKS = {
    "polygon_mumbai": ChainConfig(
        name="Polygon Mumbai Testnet",
        chain_id=80001,
        rpc_url="https://rpc-mumbai.maticvigil.com",
        explorer_url="https://mumbai.polygonscan.com",
        gas_price_gwei=30
    ),
    "ethereum_goerli": ChainConfig(
        name="Ethereum Goerli Testnet",
        chain_id=5,
        rpc_url="https://goerli.infura.io/v3/YOUR_INFURA_KEY",
        explorer_url="https://goerli.etherscan.io",
        gas_price_gwei=20
    ),
    "local": ChainConfig(
        name="Local Hardhat/Ganache",
        chain_id=1337,
        rpc_url="http://localhost:8545",
        explorer_url="http://localhost:8545",
        gas_price_gwei=1
    )
}

# Smart contract ABI
CONTRACT_ABI = json.loads('''[
    {
        "inputs": [
            {"internalType": "string", "name": "_disputeId", "type": "string"},
            {"internalType": "bytes32", "name": "_merkleRoot", "type": "bytes32"},
            {"internalType": "bytes32", "name": "_modelHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "_round", "type": "uint256"},
            {"internalType": "string", "name": "_ipfsUri", "type": "string"},
            {"internalType": "bytes[]", "name": "_metadata", "type": "bytes[]"}
        ],
        "name": "anchorRoot",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "string", "name": "_disputeId", "type": "string"}],
        "name": "getLatestRoot",
        "outputs": [
            {"internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "blockNumber", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "internalType": "string", "name": "disputeId", "type": "string"},
            {"indexed": false, "internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
            {"indexed": false, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": false, "internalType": "uint256", "name": "blockNumber", "type": "uint256"}
        ],
        "name": "RootAnchored",
        "type": "event"
    }
]''')

class RealBlockchainClient:
    """Production-ready blockchain client with multi-network support"""

    def __init__(self, network: str = "local", private_key: str = None):
        """Initialize blockchain client

        Args:
            network: Network to connect to (polygon_mumbai, ethereum_goerli, local)
            private_key: Private key for signing transactions
        """
        self.config = NETWORKS.get(network, NETWORKS["local"])
        # Use local Ganache for development
        if network == "local":
            self.config.rpc_url = "http://localhost:8545"
        self.w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))

        # Check connection
        if not self.w3.is_connected():
            logger.warning(f"Failed to connect to {self.config.name}")
            # Fallback to local
            self.config = NETWORKS["local"]
            self.w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))

        # Setup account
        self.private_key = private_key or os.getenv("BLOCKCHAIN_PRIVATE_KEY", "0x" + "0" * 64)
        if self.private_key and self.private_key != "0x" + "0" * 64:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            # Use default account for local testing
            self.account = None
            self.address = None
            logger.warning("No private key configured - read-only mode")

        # Contract setup - Use deployed contract address
        self.contract_address = os.getenv(
            "DALRN_CONTRACT_ADDRESS",
            "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"  # Deployed contract
        )

        if self.contract_address != "0x0000000000000000000000000000000000000000":
            self.contract = self.w3.eth.contract(
                address=to_checksum_address(self.contract_address),
                abi=CONTRACT_ABI
            )
        else:
            self.contract = None
            logger.warning("No contract address configured")

    def is_connected(self) -> bool:
        """Check if connected to blockchain"""
        return self.w3.is_connected()

    def get_balance(self, address: str = None) -> float:
        """Get ETH/MATIC balance"""
        addr = address or self.address
        if not addr:
            return 0.0

        balance_wei = self.w3.eth.get_balance(to_checksum_address(addr))
        return self.w3.from_wei(balance_wei, 'ether')

    def anchor_root(
        self,
        dispute_id: str,
        merkle_root: bytes,
        model_hash: bytes = b"\x00" * 32,
        round_num: int = 0,
        ipfs_uri: str = "",
        metadata: list = None
    ) -> Optional[str]:
        """Anchor Merkle root on blockchain

        Args:
            dispute_id: Unique dispute identifier
            merkle_root: Merkle root of receipt chain
            model_hash: Hash of ML model (optional)
            round_num: Round number for iterative processing
            ipfs_uri: IPFS URI for full data
            metadata: Additional metadata

        Returns:
            Transaction hash if successful, None otherwise
        """
        if not self.contract or not self.account:
            logger.error("Contract or account not configured")
            return None

        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.address)

            # Prepare metadata
            if metadata is None:
                metadata = []
            metadata_bytes = [bytes(m, 'utf-8') if isinstance(m, str) else m for m in metadata]

            # Call contract function
            transaction = self.contract.functions.anchorRoot(
                dispute_id,
                merkle_root,
                model_hash,
                round_num,
                ipfs_uri,
                metadata_bytes
            ).build_transaction({
                'chainId': self.config.chain_id,
                'gas': self.config.gas_limit,
                'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei'),
                'nonce': nonce,
                'from': self.address
            })

            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction,
                private_key=self.private_key
            )

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            # Wait for confirmation (with timeout)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                logger.info(f"Transaction successful: {tx_hash.hex()}")
                return tx_hash.hex()
            else:
                logger.error(f"Transaction failed: {tx_hash.hex()}")
                return None

        except Exception as e:
            logger.error(f"Blockchain anchoring failed: {str(e)}")
            return None

    def get_latest_root(self, dispute_id: str) -> Optional[Dict]:
        """Get latest anchored root for a dispute

        Args:
            dispute_id: Dispute identifier

        Returns:
            Dictionary with merkle_root, timestamp, block_number
        """
        if not self.contract:
            logger.error("Contract not configured")
            return None

        try:
            result = self.contract.functions.getLatestRoot(dispute_id).call()
            return {
                "merkle_root": result[0].hex(),
                "timestamp": result[1],
                "block_number": result[2]
            }
        except Exception as e:
            logger.error(f"Failed to get root: {str(e)}")
            return None

    def verify_transaction(self, tx_hash: str) -> Dict:
        """Verify transaction on blockchain

        Args:
            tx_hash: Transaction hash to verify

        Returns:
            Transaction details
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return {
                "status": "success" if receipt.status == 1 else "failed",
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "tx_hash": tx_hash,
                "explorer_url": f"{self.config.explorer_url}/tx/{tx_hash}"
            }
        except Exception as e:
            logger.error(f"Failed to verify transaction: {str(e)}")
            return {"status": "unknown", "error": str(e)}

    def estimate_gas_cost(self, dispute_id: str, merkle_root: bytes) -> Dict:
        """Estimate gas cost for anchoring

        Returns:
            Dictionary with gas estimates and costs
        """
        if not self.contract or not self.address:
            return {"error": "Contract not configured"}

        try:
            # Estimate gas
            gas_estimate = self.contract.functions.anchorRoot(
                dispute_id,
                merkle_root,
                b"\x00" * 32,
                0,
                "",
                []
            ).estimate_gas({'from': self.address})

            gas_price = self.w3.eth.gas_price
            cost_wei = gas_estimate * gas_price
            cost_eth = self.w3.from_wei(cost_wei, 'ether')

            return {
                "gas_limit": gas_estimate,
                "gas_price_gwei": self.w3.from_wei(gas_price, 'gwei'),
                "estimated_cost_eth": float(cost_eth),
                "network": self.config.name
            }

        except Exception as e:
            return {"error": str(e)}

    def get_events(self, dispute_id: str, from_block: int = 0) -> list:
        """Get RootAnchored events for a dispute

        Args:
            dispute_id: Dispute identifier
            from_block: Starting block number

        Returns:
            List of events
        """
        if not self.contract:
            return []

        try:
            # Create event filter
            event_filter = self.contract.events.RootAnchored.create_filter(
                fromBlock=from_block,
                argument_filters={'disputeId': dispute_id}
            )

            events = event_filter.get_all_entries()
            return [
                {
                    "dispute_id": event.args.disputeId,
                    "merkle_root": event.args.merkleRoot.hex(),
                    "timestamp": event.args.timestamp,
                    "block_number": event.args.blockNumber,
                    "tx_hash": event.transactionHash.hex()
                }
                for event in events
            ]

        except Exception as e:
            logger.error(f"Failed to get events: {str(e)}")
            return []

# Singleton instance
_blockchain_client = None

def get_blockchain_client(network: str = None) -> RealBlockchainClient:
    """Get singleton blockchain client instance"""
    global _blockchain_client

    if _blockchain_client is None:
        network = network or os.getenv("BLOCKCHAIN_NETWORK", "local")
        _blockchain_client = RealBlockchainClient(network=network)

    return _blockchain_client

# Integration with existing system
def upgrade_to_production():
    """Upgrade development blockchain to production implementation"""
    # Replace development imports in existing code
    import services.chain.client as old_client

    # Replace with real client
    old_client.AnchorClient = RealBlockchainClient

    print("Upgraded to real blockchain client")