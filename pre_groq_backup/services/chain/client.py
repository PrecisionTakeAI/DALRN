"""Chain client for anchoring Merkle roots to blockchain"""
import os
import json
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from web3 import Web3
from web3.types import TxReceipt, HexBytes, Wei
from web3.exceptions import ContractLogicError, TimeExhausted, TransactionNotFound
try:
    # Web3 v6 import
    from web3.middleware import geth_poa_middleware
except ImportError:
    # Web3 v7+ import
    try:
        from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
    except ImportError:
        # Fallback if neither works
        geth_poa_middleware = None
from eth_account import Account
from eth_typing import HexStr, Address, ChecksumAddress
import backoff

logger = logging.getLogger(__name__)

class NetworkConfig(Enum):
    """Supported network configurations"""
    LOCAL = "local"
    SEPOLIA = "sepolia"
    MAINNET = "mainnet"
    POLYGON = "polygon"
    POLYGON_MUMBAI = "polygon_mumbai"
    ARBITRUM = "arbitrum"
    ARBITRUM_SEPOLIA = "arbitrum_sepolia"


@dataclass
class NetworkSettings:
    """Network-specific settings"""
    rpc_url: str
    chain_id: int
    explorer_url: Optional[str] = None
    gas_limit_multiplier: float = 1.2
    max_priority_fee: Optional[Wei] = None
    max_fee_per_gas: Optional[Wei] = None
    confirmation_blocks: int = 1


class TransactionStatus(Enum):
    """Transaction status states"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AnchorClient:
    """Client for interacting with the DALRN anchor contract"""
    
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        network: Optional[NetworkConfig] = None,
        connection_pool_size: int = 5,
        request_timeout: int = 30
    ):
        """Initialize the anchor client with production configuration

        Args:
            rpc_url: Ethereum RPC endpoint URL
            contract_address: Address of the anchor contract
            private_key: Private key for signing transactions
            network: Network configuration preset
            connection_pool_size: Number of connections in pool
            request_timeout: Request timeout in seconds
        """
        # Network configuration
        self.network = network or NetworkConfig.LOCAL
        self.network_settings = self._get_network_settings(self.network)

        # Override with provided values
        self.rpc_url = rpc_url or self.network_settings.rpc_url or os.getenv("ETH_RPC_URL", "http://localhost:8545")
        self.contract_address = self._validate_address(
            contract_address or os.getenv("ANCHOR_CONTRACT_ADDRESS", "0x0000000000000000000000000000000000000000")
        )

        # Initialize Web3 with connection pooling
        self.w3 = self._initialize_web3(self.rpc_url, connection_pool_size, request_timeout)

        # Setup account for transactions
        self.account = None
        if private_key:
            self._setup_account(private_key)
        elif os.getenv("PRIVATE_KEY"):
            self._setup_account(os.getenv("PRIVATE_KEY"))

        # Load contract ABI and initialize contract
        self.contract = self._load_contract()

        # Transaction tracking
        self.pending_transactions: Dict[str, Dict[str, Any]] = {}

        # Gas price cache (TTL: 30 seconds)
        self._gas_price_cache: Optional[Tuple[Wei, float]] = None
        self._gas_price_cache_ttl = 30

        logger.info(
            "AnchorClient initialized",
            extra={
                "network": self.network.value,
                "rpc_url": self.rpc_url,
                "contract_address": self.contract_address,
                "account": self.account.address if self.account else None,
                "connected": self.w3.is_connected()
            }
        )

    def _get_network_settings(self, network: NetworkConfig) -> NetworkSettings:
        """Get network-specific settings"""
        settings_map = {
            NetworkConfig.LOCAL: NetworkSettings(
                rpc_url="http://localhost:8545",
                chain_id=31337,
                confirmation_blocks=1
            ),
            NetworkConfig.SEPOLIA: NetworkSettings(
                rpc_url=os.getenv("SEPOLIA_RPC_URL", "https://sepolia.infura.io/v3/YOUR_PROJECT_ID"),
                chain_id=11155111,
                explorer_url="https://sepolia.etherscan.io",
                confirmation_blocks=2
            ),
            NetworkConfig.MAINNET: NetworkSettings(
                rpc_url=os.getenv("MAINNET_RPC_URL", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"),
                chain_id=1,
                explorer_url="https://etherscan.io",
                confirmation_blocks=6
            ),
            NetworkConfig.POLYGON: NetworkSettings(
                rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
                chain_id=137,
                explorer_url="https://polygonscan.com",
                confirmation_blocks=30
            ),
            NetworkConfig.POLYGON_MUMBAI: NetworkSettings(
                rpc_url=os.getenv("POLYGON_MUMBAI_RPC_URL", "https://rpc-mumbai.maticvigil.com"),
                chain_id=80001,
                explorer_url="https://mumbai.polygonscan.com",
                confirmation_blocks=10
            ),
            NetworkConfig.ARBITRUM: NetworkSettings(
                rpc_url=os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"),
                chain_id=42161,
                explorer_url="https://arbiscan.io",
                confirmation_blocks=1
            ),
            NetworkConfig.ARBITRUM_SEPOLIA: NetworkSettings(
                rpc_url=os.getenv("ARBITRUM_SEPOLIA_RPC_URL", "https://sepolia-rollup.arbitrum.io/rpc"),
                chain_id=421614,
                explorer_url="https://sepolia.arbiscan.io",
                confirmation_blocks=1
            )
        }
        return settings_map[network]

    def _initialize_web3(self, rpc_url: str, pool_size: int, timeout: int) -> Web3:
        """Initialize Web3 with connection pooling and middleware"""
        # Web3 v7 doesn't support pool_connections - use only timeout
        provider = Web3.HTTPProvider(
            rpc_url,
            request_kwargs={
                'timeout': timeout
            }
        )

        w3 = Web3(provider)

        # Add POA middleware for networks that need it (like Polygon)
        if self.network in [NetworkConfig.POLYGON, NetworkConfig.POLYGON_MUMBAI] and geth_poa_middleware:
            try:
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except Exception as e:
                logger.warning(f"Could not inject POA middleware: {e}")

        # Verify connection
        if not w3.is_connected():
            raise ConnectionError(f"Failed to connect to {rpc_url}")

        # Verify chain ID matches expected
        actual_chain_id = w3.eth.chain_id
        expected_chain_id = self.network_settings.chain_id
        if actual_chain_id != expected_chain_id:
            logger.warning(
                f"Chain ID mismatch: expected {expected_chain_id}, got {actual_chain_id}"
            )

        return w3

    def _validate_address(self, address: str) -> ChecksumAddress:
        """Validate and convert address to checksum format"""
        if not Web3.is_address(address):
            raise ValueError(f"Invalid Ethereum address: {address}")
        return Web3.to_checksum_address(address)

    def _setup_account(self, private_key: str):
        """Setup account from private key"""
        try:
            # Remove 0x prefix if present
            if private_key.startswith("0x"):
                private_key = private_key[2:]

            self.account = Account.from_key(private_key)
            logger.info(f"Account configured: {self.account.address}")
        except Exception as e:
            raise ValueError(f"Invalid private key: {str(e)}")

    def _load_contract(self):
        """Load contract ABI and initialize contract instance"""
        try:
            # Load ABI from file
            abi_path = Path(__file__).parent / "abi" / "AnchorReceipts.json"
            if not abi_path.exists():
                raise FileNotFoundError(f"Contract ABI not found at {abi_path}")

            with open(abi_path, 'r') as f:
                abi = json.load(f)

            # Create contract instance
            contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=abi
            )

            # Verify contract exists (will throw if not deployed)
            try:
                _ = contract.functions.totalRootsAnchored().call()
                logger.info(f"Contract verified at {self.contract_address}")
            except Exception as e:
                logger.warning(f"Contract may not be deployed at {self.contract_address}: {e}")

            return contract

        except Exception as e:
            logger.error(f"Failed to load contract: {e}")
            # Return None but don't fail initialization
            return None

    @backoff.on_exception(
        backoff.expo,
        (ConnectionError, TimeoutError, TransactionNotFound),
        max_tries=3,
        max_value=10
    )
    def _get_gas_price(self) -> Dict[str, Wei]:
        """Get current gas price with caching and retry logic"""
        # Check cache
        if self._gas_price_cache:
            price, timestamp = self._gas_price_cache
            if time.time() - timestamp < self._gas_price_cache_ttl:
                return {"maxFeePerGas": price, "maxPriorityFeePerGas": Wei(int(price * 0.1))}

        # Get fresh gas price
        try:
            # For EIP-1559 networks
            if self.network_settings.max_fee_per_gas:
                gas_price = {
                    "maxFeePerGas": self.network_settings.max_fee_per_gas,
                    "maxPriorityFeePerGas": self.network_settings.max_priority_fee or Wei(int(self.network_settings.max_fee_per_gas * 0.1))
                }
            else:
                # Get from network
                base_fee = self.w3.eth.get_block('latest')['baseFeePerGas']
                priority_fee = self.w3.eth.max_priority_fee
                max_fee = Wei(int(base_fee * 2 + priority_fee))

                gas_price = {
                    "maxFeePerGas": max_fee,
                    "maxPriorityFeePerGas": priority_fee
                }

            # Cache the result
            self._gas_price_cache = (gas_price["maxFeePerGas"], time.time())

            return gas_price

        except Exception:
            # Fallback to legacy gas price
            gas_price = self.w3.eth.gas_price
            self._gas_price_cache = (gas_price, time.time())
            return {"gasPrice": gas_price}

    @backoff.on_exception(
        backoff.expo,
        (ConnectionError, TimeoutError),
        max_tries=3,
        max_value=10
    )
    def _estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas with retry logic and multiplier"""
        try:
            estimated = self.w3.eth.estimate_gas(transaction)
            # Add buffer for safety
            return int(estimated * self.network_settings.gas_limit_multiplier)
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}, using default")
            return 500000  # Default fallback

    def _sign_and_send_transaction(self, transaction: Dict[str, Any]) -> HexBytes:
        """Sign and send transaction with proper error handling"""
        if not self.account:
            raise ValueError("No account configured for signing transactions")

        # Add gas price
        gas_params = self._get_gas_price()
        transaction.update(gas_params)

        # Estimate gas
        transaction['gas'] = self._estimate_gas(transaction)

        # Add nonce
        transaction['nonce'] = self.w3.eth.get_transaction_count(
            self.account.address,
            'pending'  # Include pending transactions
        )

        # Add chain ID
        transaction['chainId'] = self.network_settings.chain_id

        # Sign transaction
        signed_tx = self.account.sign_transaction(transaction)

        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        logger.info(
            f"Transaction sent: {tx_hash.hex()}",
            extra={
                "from": self.account.address,
                "to": transaction.get('to'),
                "gas": transaction['gas'],
                "nonce": transaction['nonce']
            }
        )

        return tx_hash

    @backoff.on_exception(
        backoff.expo,
        TransactionNotFound,
        max_tries=5,
        max_value=5
    )
    def _wait_for_transaction(self, tx_hash: HexBytes, timeout: int = 120) -> TxReceipt:
        """Wait for transaction confirmation with retry logic"""
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=timeout,
                poll_latency=2
            )

            if receipt['status'] == 0:
                raise ContractLogicError(f"Transaction failed: {tx_hash.hex()}")

            # Wait for additional confirmations if required
            if self.network_settings.confirmation_blocks > 1:
                target_block = receipt['blockNumber'] + self.network_settings.confirmation_blocks
                while self.w3.eth.block_number < target_block:
                    time.sleep(2)

            return receipt

        except TimeExhausted:
            logger.error(f"Transaction timeout: {tx_hash.hex()}")
            raise

    def _convert_to_bytes32(self, value: str) -> bytes:
        """Convert string to bytes32 format"""
        if value.startswith("0x"):
            # Already hex, just ensure it's 32 bytes
            hex_value = value[2:]
            if len(hex_value) < 64:
                hex_value = hex_value.ljust(64, '0')
            return bytes.fromhex(hex_value[:64])
        else:
            # Convert string to bytes32
            encoded = value.encode('utf-8')
            if len(encoded) > 32:
                encoded = encoded[:32]
            return encoded.ljust(32, b'\0')
        
    def anchor_root(
        self,
        dispute_id: str,
        merkle_root: str,
        model_hash: str,
        round: int,
        uri: str,
        tags: List[str]
    ) -> Dict[str, Any]:
        """Anchor a Merkle root to the blockchain

        Args:
            dispute_id: Unique identifier for the dispute
            merkle_root: Merkle root of the receipt chain (hex string)
            model_hash: Hash of the model used (hex string)
            round: Round number for this anchor
            uri: IPFS URI of the receipt chain
            tags: List of tags for categorization (strings)

        Returns:
            Dictionary with transaction details including hash and block number
        """
        try:
            # Validate inputs
            if not self.contract:
                raise ValueError("Contract not initialized")

            # Convert inputs to proper formats
            dispute_id_bytes = self._convert_to_bytes32(dispute_id)
            merkle_root_bytes = self._convert_to_bytes32(merkle_root)
            model_hash_bytes = self._convert_to_bytes32(model_hash)

            # Convert tags to bytes32[]
            tags_bytes = [self._convert_to_bytes32(tag) for tag in tags[:10]]  # Max 10 tags

            # Log operation (with redacted sensitive data)
            logger.info(
                "Anchoring root for dispute",
                extra={
                    "dispute_id": dispute_id[:8] + "...",
                    "merkle_root": merkle_root[:10] + "...",
                    "round": round,
                    "uri_prefix": uri[:20] if uri else None,
                    "tags_count": len(tags_bytes)
                }
            )

            # Build transaction
            function = self.contract.functions.anchorRoot(
                dispute_id_bytes,
                merkle_root_bytes,
                model_hash_bytes,
                round,
                uri,
                tags_bytes
            )

            # Create transaction dict
            transaction = function.build_transaction({
                'from': self.account.address if self.account else self.w3.eth.accounts[0]
            })

            # Send transaction
            tx_hash = self._sign_and_send_transaction(transaction)

            # Track pending transaction
            self.pending_transactions[tx_hash.hex()] = {
                "type": "anchor_root",
                "dispute_id": dispute_id,
                "round": round,
                "timestamp": time.time()
            }

            # Wait for confirmation
            receipt = self._wait_for_transaction(tx_hash)

            # Parse events from receipt
            events = self.contract.events.RootAnchored().process_receipt(receipt)

            # Build result
            result = {
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt['blockNumber'],
                "block_hash": receipt['blockHash'].hex(),
                "gas_used": receipt['gasUsed'],
                "status": "success" if receipt['status'] == 1 else "failed"
            }

            # Add event data if available
            if events:
                event_data = events[0]['args']
                result["event"] = {
                    "dispute_id": event_data.get('disputeId', b'').hex(),
                    "merkle_root": event_data.get('merkleRoot', b'').hex(),
                    "timestamp": event_data.get('timestamp', 0),
                    "block_number": event_data.get('blockNumber', 0)
                }

            # Generate explorer URL if available
            if self.network_settings.explorer_url:
                result["explorer_url"] = f"{self.network_settings.explorer_url}/tx/{tx_hash.hex()}"

            logger.info(
                f"Root anchored successfully",
                extra={
                    "tx_hash": tx_hash.hex(),
                    "block": receipt['blockNumber'],
                    "gas_used": receipt['gasUsed']
                }
            )

            return result

        except ContractLogicError as e:
            logger.error(f"Contract logic error during anchor: {str(e)}")
            raise
        except TimeExhausted as e:
            logger.error(f"Transaction timeout during anchor: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during anchor: {str(e)}")
            raise
    
    def anchor_receipt(
        self,
        dispute_id: str,
        receipt_hash: str,
        step_index: int,
        uri: str
    ) -> Dict[str, Any]:
        """Anchor an individual receipt to the blockchain

        Args:
            dispute_id: Unique identifier for the dispute
            receipt_hash: Hash of the receipt (hex string)
            step_index: Index of the step in computation
            uri: IPFS URI for receipt data

        Returns:
            Dictionary with transaction details
        """
        try:
            if not self.contract:
                raise ValueError("Contract not initialized")

            # Convert inputs
            dispute_id_bytes = self._convert_to_bytes32(dispute_id)
            receipt_hash_bytes = self._convert_to_bytes32(receipt_hash)

            # Build transaction
            function = self.contract.functions.anchorReceipt(
                dispute_id_bytes,
                receipt_hash_bytes,
                step_index,
                uri
            )

            transaction = function.build_transaction({
                'from': self.account.address if self.account else self.w3.eth.accounts[0]
            })

            # Send transaction
            tx_hash = self._sign_and_send_transaction(transaction)

            # Wait for confirmation
            receipt = self._wait_for_transaction(tx_hash)

            # Parse events
            events = self.contract.events.ReceiptAnchored().process_receipt(receipt)

            result = {
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt['blockNumber'],
                "gas_used": receipt['gasUsed'],
                "status": "success" if receipt['status'] == 1 else "failed"
            }

            if events:
                event_data = events[0]['args']
                result["event"] = {
                    "receipt_hash": event_data.get('receiptHash', b'').hex(),
                    "step_index": event_data.get('stepIndex', 0),
                    "timestamp": event_data.get('timestamp', 0)
                }

            logger.info(f"Receipt anchored: {tx_hash.hex()}")
            return result

        except Exception as e:
            logger.error(f"Error anchoring receipt: {e}")
            raise

    @backoff.on_exception(
        backoff.constant,
        (ConnectionError,),
        interval=2,
        max_tries=3
    )
    def latest_root(self, dispute_id: str) -> Dict[str, Any]:
        """Get the latest root for a dispute

        Args:
            dispute_id: Unique identifier for the dispute

        Returns:
            Dictionary with merkle root and block number
        """
        try:
            if not self.contract:
                raise ValueError("Contract not initialized")

            dispute_id_bytes = self._convert_to_bytes32(dispute_id)

            # Call view function
            merkle_root, block_number = self.contract.functions.latestRoot(
                dispute_id_bytes
            ).call()

            return {
                "dispute_id": dispute_id,
                "merkle_root": merkle_root.hex() if merkle_root else "0x" + "0" * 64,
                "block_number": block_number,
                "exists": merkle_root != b'\x00' * 32
            }

        except Exception as e:
            logger.error(f"Error getting latest root: {e}")
            raise

    def latest_root_info(self, dispute_id: str) -> Dict[str, Any]:
        """Get detailed information about the latest root

        Args:
            dispute_id: Unique identifier for the dispute

        Returns:
            Dictionary with root info including timestamp
        """
        try:
            if not self.contract:
                raise ValueError("Contract not initialized")

            dispute_id_bytes = self._convert_to_bytes32(dispute_id)

            # Call view function
            merkle_root, timestamp, block_number = self.contract.functions.latestRootInfo(
                dispute_id_bytes
            ).call()

            return {
                "dispute_id": dispute_id,
                "merkle_root": merkle_root.hex() if merkle_root else "0x" + "0" * 64,
                "timestamp": timestamp,
                "block_number": block_number,
                "exists": merkle_root != b'\x00' * 32
            }

        except Exception as e:
            logger.error(f"Error getting latest root info: {e}")
            raise

    def get_root_by_round(self, dispute_id: str, round: int) -> Dict[str, Any]:
        """Get root information for a specific round

        Args:
            dispute_id: Unique identifier for the dispute
            round: Round number

        Returns:
            Dictionary with root information for that round
        """
        try:
            if not self.contract:
                raise ValueError("Contract not initialized")

            dispute_id_bytes = self._convert_to_bytes32(dispute_id)

            merkle_root, timestamp, block_number = self.contract.functions.getRootByRound(
                dispute_id_bytes,
                round
            ).call()

            return {
                "dispute_id": dispute_id,
                "round": round,
                "merkle_root": merkle_root.hex() if merkle_root else "0x" + "0" * 64,
                "timestamp": timestamp,
                "block_number": block_number,
                "exists": merkle_root != b'\x00' * 32
            }

        except Exception as e:
            logger.error(f"Error getting root by round: {e}")
            raise

    def has_root(self, dispute_id: str) -> bool:
        """Check if a dispute has any anchored root

        Args:
            dispute_id: Unique identifier for the dispute

        Returns:
            True if root exists, False otherwise
        """
        try:
            if not self.contract:
                return False

            dispute_id_bytes = self._convert_to_bytes32(dispute_id)
            return self.contract.functions.hasRoot(dispute_id_bytes).call()

        except Exception:
            return False

    def get_contract_info(self) -> Dict[str, Any]:
        """Get general contract information

        Returns:
            Dictionary with contract stats
        """
        try:
            if not self.contract:
                raise ValueError("Contract not initialized")

            total_roots = self.contract.functions.totalRootsAnchored().call()
            total_receipts = self.contract.functions.totalReceiptsAnchored().call()

            return {
                "contract_address": self.contract_address,
                "network": self.network.value,
                "chain_id": self.network_settings.chain_id,
                "total_roots": total_roots,
                "total_receipts": total_receipts,
                "connected": self.w3.is_connected(),
                "latest_block": self.w3.eth.block_number
            }

        except Exception as e:
            logger.error(f"Error getting contract info: {e}")
            raise

    def verify_receipt_proof(
        self,
        dispute_id: str,
        receipt_hash: str,
        merkle_proof: List[str]
    ) -> bool:
        """Verify a receipt is part of an anchored Merkle tree

        Args:
            dispute_id: Unique identifier for the dispute
            receipt_hash: Hash of the receipt to verify
            merkle_proof: Merkle proof path

        Returns:
            True if proof is valid, False otherwise
        """
        try:
            # Get the anchored root
            root_info = self.latest_root(dispute_id)
            if not root_info["exists"]:
                return False

            anchored_root = root_info["merkle_root"]

            # Verify the proof locally
            computed_root = receipt_hash
            for proof_element in merkle_proof:
                # Sort to ensure consistent ordering
                if computed_root < proof_element:
                    combined = computed_root + proof_element
                else:
                    combined = proof_element + computed_root

                # Hash the combination
                computed_root = Web3.keccak(hexstr=combined).hex()

            return computed_root.lower() == anchored_root.lower()

        except Exception as e:
            logger.error(f"Error verifying receipt proof: {e}")
            return False

    def get_pending_transactions(self) -> Dict[str, Any]:
        """Get list of pending transactions

        Returns:
            Dictionary of pending transactions
        """
        # Clean old entries (older than 5 minutes)
        current_time = time.time()
        self.pending_transactions = {
            tx_hash: info
            for tx_hash, info in self.pending_transactions.items()
            if current_time - info["timestamp"] < 300
        }

        return self.pending_transactions

    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get status of a specific transaction

        Args:
            tx_hash: Transaction hash to check

        Returns:
            Dictionary with transaction status
        """
        try:
            # Try to get receipt
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)

            if receipt:
                return {
                    "status": TransactionStatus.SUCCESS.value if receipt['status'] == 1 else TransactionStatus.FAILED.value,
                    "block_number": receipt['blockNumber'],
                    "gas_used": receipt['gasUsed'],
                    "confirmations": self.w3.eth.block_number - receipt['blockNumber']
                }
            else:
                # Check if transaction exists
                tx = self.w3.eth.get_transaction(tx_hash)
                if tx:
                    return {
                        "status": TransactionStatus.PENDING.value,
                        "gas_price": tx.get('gasPrice', 0),
                        "nonce": tx['nonce']
                    }
                else:
                    return {"status": "not_found"}

        except Exception as e:
            logger.error(f"Error getting transaction status: {e}")
            return {"status": "error", "message": str(e)}
