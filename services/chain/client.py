"""
AnchorReceipts Smart Contract Client
Provides Python interface for interacting with the AnchorReceipts contract
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from web3 import Web3
from web3.types import TxReceipt, HexBytes
from eth_typing import Address

# Configuration
RPC_URL = os.getenv("RPC_URL", "http://127.0.0.1:8545")
CONTRACT_ADDRESS = os.getenv("ANCHOR_ADDRESS", None)
PRIVATE_KEY = os.getenv("PRIVATE_KEY", None)  # For sending transactions

# Default Anvil test account (DO NOT USE IN PRODUCTION)
DEFAULT_ACCOUNT = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
DEFAULT_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

class AnchorClient:
    """Client for interacting with the AnchorReceipts smart contract"""
    
    def __init__(self, 
                 rpc_url: str = None,
                 contract_address: str = None,
                 private_key: str = None,
                 account: str = None):
        """
        Initialize the AnchorClient
        
        Args:
            rpc_url: Ethereum RPC endpoint URL
            contract_address: Deployed contract address
            private_key: Private key for signing transactions
            account: Account address (if not using private key)
        """
        # Setup Web3 connection
        self.rpc_url = rpc_url or RPC_URL
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            raise RuntimeError(f"Failed to connect to Ethereum node at {self.rpc_url}")
        
        # Setup account
        self.private_key = private_key or PRIVATE_KEY or DEFAULT_PRIVATE_KEY
        if account:
            self.account = Web3.to_checksum_address(account)
        else:
            # Derive account from private key
            account_obj = self.w3.eth.account.from_key(self.private_key)
            self.account = account_obj.address
        
        # Load ABI
        abi_path = os.path.join(os.path.dirname(__file__), "abi", "AnchorReceipts.json")
        if os.path.exists(abi_path):
            with open(abi_path, 'r') as f:
                self.abi = json.load(f)
        else:
            raise FileNotFoundError(f"ABI file not found at {abi_path}")
        
        # Setup contract
        contract_addr = contract_address or CONTRACT_ADDRESS
        if not contract_addr:
            print("Warning: No contract address provided. Some functions will not work.")
            self.contract = None
        else:
            self.contract_address = Web3.to_checksum_address(contract_addr)
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.abi
            )
    
    def _to_bytes32(self, value: str) -> bytes:
        """Convert string to bytes32"""
        if value.startswith('0x'):
            return bytes.fromhex(value[2:]).ljust(32, b'\0')[:32]
        else:
            return Web3.keccak(text=value)
    
    def _build_tx(self, func) -> Dict[str, Any]:
        """Build transaction dictionary"""
        return {
            'from': self.account,
            'gas': 500000,  # Adjust as needed
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account),
        }
    
    def _send_tx(self, func) -> Dict[str, Any]:
        """Send a transaction and wait for receipt"""
        # Build transaction
        tx = func.build_transaction(self._build_tx(func))
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        
        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'tx_hash': receipt['transactionHash'].hex(),
            'block_number': receipt['blockNumber'],
            'gas_used': receipt['gasUsed'],
            'status': receipt['status'],
            'logs': receipt['logs']
        }
    
    def anchor_root(self,
                   dispute_id: str,
                   merkle_root: str,
                   model_hash: str,
                   round: int,
                   uri: str,
                   tags: List[str]) -> Dict[str, Any]:
        """
        Anchor a Merkle root on-chain
        
        Args:
            dispute_id: Unique identifier for the dispute
            merkle_root: The Merkle root to anchor
            model_hash: Hash of the model used
            round: The round number
            uri: IPFS or other URI for additional data
            tags: List of tags for categorization
        
        Returns:
            Transaction receipt information
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        # Convert parameters to appropriate types
        dispute_id_bytes = self._to_bytes32(dispute_id)
        merkle_root_bytes = self._to_bytes32(merkle_root)
        model_hash_bytes = self._to_bytes32(model_hash)
        tags_bytes = [self._to_bytes32(tag) for tag in tags]
        
        # Call contract function
        func = self.contract.functions.anchorRoot(
            dispute_id_bytes,
            merkle_root_bytes,
            model_hash_bytes,
            round,
            uri,
            tags_bytes
        )
        
        result = self._send_tx(func)
        
        # Parse events
        if result['logs']:
            for log in result['logs']:
                try:
                    event = self.contract.events.RootAnchored().process_log(log)
                    result['event'] = {
                        'disputeId': event['args']['disputeId'].hex(),
                        'merkleRoot': event['args']['merkleRoot'].hex(),
                        'modelHash': event['args']['modelHash'].hex(),
                        'round': event['args']['round'],
                        'uri': event['args']['uri'],
                        'tags': [t.hex() for t in event['args']['tags']],
                        'timestamp': event['args']['timestamp'],
                        'blockNumber': event['args']['blockNumber']
                    }
                    break
                except:
                    pass
        
        return result
    
    def anchor_receipt(self,
                      dispute_id: str,
                      receipt_hash: str,
                      step_index: int,
                      uri: str) -> Dict[str, Any]:
        """
        Anchor an individual receipt
        
        Args:
            dispute_id: Unique identifier for the dispute
            receipt_hash: Hash of the receipt
            step_index: Index of the step in computation
            uri: IPFS or other URI for receipt data
        
        Returns:
            Transaction receipt information
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        # Convert parameters
        dispute_id_bytes = self._to_bytes32(dispute_id)
        receipt_hash_bytes = self._to_bytes32(receipt_hash)
        
        # Call contract function
        func = self.contract.functions.anchorReceipt(
            dispute_id_bytes,
            receipt_hash_bytes,
            step_index,
            uri
        )
        
        result = self._send_tx(func)
        
        # Parse events
        if result['logs']:
            for log in result['logs']:
                try:
                    event = self.contract.events.ReceiptAnchored().process_log(log)
                    result['event'] = {
                        'disputeId': event['args']['disputeId'].hex(),
                        'receiptHash': event['args']['receiptHash'].hex(),
                        'stepIndex': event['args']['stepIndex'],
                        'uri': event['args']['uri'],
                        'timestamp': event['args']['timestamp'],
                        'blockNumber': event['args']['blockNumber']
                    }
                    break
                except:
                    pass
        
        return result
    
    def latest_root(self, dispute_id: str) -> Dict[str, Any]:
        """
        Get the latest root for a dispute
        
        Args:
            dispute_id: The dispute identifier
        
        Returns:
            Dictionary with merkle_root and block_number
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        dispute_id_bytes = self._to_bytes32(dispute_id)
        
        result = self.contract.functions.latestRoot(dispute_id_bytes).call()
        
        return {
            'merkle_root': result[0].hex() if result[0] else '0x' + '0' * 64,
            'block_number': result[1]
        }
    
    def latest_root_info(self, dispute_id: str) -> Dict[str, Any]:
        """
        Get detailed information about the latest root
        
        Args:
            dispute_id: The dispute identifier
        
        Returns:
            Dictionary with merkle_root, timestamp, and block_number
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        dispute_id_bytes = self._to_bytes32(dispute_id)
        
        result = self.contract.functions.latestRootInfo(dispute_id_bytes).call()
        
        return {
            'merkle_root': result[0].hex() if result[0] else '0x' + '0' * 64,
            'timestamp': result[1],
            'block_number': result[2]
        }
    
    def get_root_by_round(self, dispute_id: str, round: int) -> Dict[str, Any]:
        """
        Get root for a specific round
        
        Args:
            dispute_id: The dispute identifier
            round: The round number
        
        Returns:
            Dictionary with merkle_root, timestamp, and block_number
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        dispute_id_bytes = self._to_bytes32(dispute_id)
        
        result = self.contract.functions.getRootByRound(dispute_id_bytes, round).call()
        
        return {
            'merkle_root': result[0].hex() if result[0] else '0x' + '0' * 64,
            'timestamp': result[1],
            'block_number': result[2]
        }
    
    def get_latest_round(self, dispute_id: str) -> int:
        """
        Get the latest round number for a dispute
        
        Args:
            dispute_id: The dispute identifier
        
        Returns:
            The latest round number
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        dispute_id_bytes = self._to_bytes32(dispute_id)
        
        return self.contract.functions.getLatestRound(dispute_id_bytes).call()
    
    def has_root(self, dispute_id: str) -> bool:
        """
        Check if a root has been anchored for a dispute
        
        Args:
            dispute_id: The dispute identifier
        
        Returns:
            True if a root exists for this dispute
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        dispute_id_bytes = self._to_bytes32(dispute_id)
        
        return self.contract.functions.hasRoot(dispute_id_bytes).call()
    
    def total_roots_anchored(self) -> int:
        """Get total number of roots anchored"""
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        return self.contract.functions.totalRootsAnchored().call()
    
    def total_receipts_anchored(self) -> int:
        """Get total number of receipts anchored"""
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        return self.contract.functions.totalReceiptsAnchored().call()
    
    def get_contract_info(self) -> Dict[str, Any]:
        """Get general contract information"""
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        return {
            'address': self.contract_address,
            'total_roots': self.total_roots_anchored(),
            'total_receipts': self.total_receipts_anchored(),
            'network': {
                'chain_id': self.w3.eth.chain_id,
                'block_number': self.w3.eth.block_number,
                'gas_price': self.w3.eth.gas_price
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Example: Initialize client (assumes Anvil is running locally)
    print("Initializing AnchorClient...")
    
    # For testing, you would first deploy the contract and get its address
    # Example with a dummy address (replace with actual deployed address)
    client = AnchorClient(
        contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3"  # Replace with actual
    )
    
    print(f"Connected to: {client.rpc_url}")
    print(f"Account: {client.account}")
    
    # Example: Anchor a root
    try:
        result = client.anchor_root(
            dispute_id="test-dispute-1",
            merkle_root="0x" + "1" * 64,
            model_hash="0x" + "2" * 64,
            round=1,
            uri="ipfs://QmTest123",
            tags=["PoDP", "Test"]
        )
        print(f"Root anchored: {result}")
    except Exception as e:
        print(f"Error anchoring root: {e}")
    
    # Example: Query latest root
    try:
        latest = client.latest_root("test-dispute-1")
        print(f"Latest root: {latest}")
    except Exception as e:
        print(f"Error querying latest root: {e}")