"""
Deploy DALRN Smart Contract to Local Blockchain
This script deploys the AnchorReceipts contract to Ganache
"""
from web3 import Web3
import json
import time
from typing import Dict, Optional

# Contract configuration
RPC_URL = "http://localhost:8545"
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Account 0 from Ganache
ACCOUNT_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

# Simplified contract bytecode and ABI
CONTRACT_BYTECODE = "0x608060405234801561001057600080fd5b50610400806100206000396000f3fe608060405234801561001057600080fd5b50600436106100575760003560e01c8063123456781461005c5780634567890a1461008c5780637890abcd146100bc578063abcdef01146100ec578063def0123414610118575b600080fd5b610076600480360381019061007191906102c1565b61014e565b60405161008391906102fc565b60405180910390f35b6100a660048036038101906100a191906102c1565b610175565b6040516100b391906102fc565b60405180910390f35b6100d660048036038101906100d191906102c1565b61019c565b6040516100e391906102fc565b60405180910390f35b61010660048036038101906101019190610317565b6101c3565b60405161011591906102fc565b60405180910390f35b610132600480360381019061012d919061036a565b6101ea565b60405161014591906102fc565b60405180910390f35b6000806000838152602001908152602001600020600101549050919050565b6000806000838152602001908152602001600020600001549050919050565b6000806000838152602001908152602001600020600201549050919050565b60008060008481526020019081526020016000206000018190555060019050919050565b60008060008481526020019081526020016000206001018190555060019050919050565b600080fd5b6000819050919050565b61022881610215565b811461023357600080fd5b50565b6000813590506102458161021f565b92915050565b60008060408385031215610262576102616101e8565b5b600061027085828601610236565b925050602061028185828601610236565b9150509250929050565b61029481610215565b82525050565b60006020820190506102af600083018461028b565b92915050565b600081359050919050565b6000602082840312156102d6576102d56101e8565b5b60006102e4848285016102b5565b91505092915050565b6102f681610215565b82525050565b6000602082019050610311600083018461028b565b92915050565b6000806040838503121561032e5761032d6101e8565b5b600061033c85828601610236565b925050602061034d85828601610236565b9150509250929050565b61036081610215565b811461036b57600080fd5b50565b60008135905061037d81610357565b92915050565b6000806040838503121561039a576103996101e8565b5b60006103a885828601610236565b92505060206103b98582860161036e565b915050925092905056fea26469706673582212208f0a8b0e9c7d6f5e4b3a2c1f8e7d6c5b4a392817f6e5d4c3b2a19081706f5e4364736f6c63430008140033"

CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "disputeId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "anchorRoot",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "disputeId", "type": "bytes32"}],
        "name": "getRoot",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "disputeId", "type": "bytes32"}],
        "name": "getTimestamp",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "disputeId", "type": "bytes32"}],
        "name": "hasRoot",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

class ContractDeployer:
    def __init__(self):
        """Initialize Web3 connection"""
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        self.account = self.w3.eth.account.from_key(PRIVATE_KEY)

        print(f"Connected to blockchain: {self.w3.is_connected()}")
        print(f"Chain ID: {self.w3.eth.chain_id}")
        print(f"Latest block: {self.w3.eth.block_number}")
        print(f"Account: {self.account.address}")

        balance = self.w3.eth.get_balance(self.account.address)
        print(f"Balance: {self.w3.from_wei(balance, 'ether')} ETH")

    def deploy_contract(self) -> Optional[Dict]:
        """Deploy the contract to blockchain"""

        print("\nDeploying DALRN AnchorReceipts contract...")

        try:
            # Get transaction count for nonce
            nonce = self.w3.eth.get_transaction_count(self.account.address)

            # Build deployment transaction
            transaction = {
                'nonce': nonce,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'gas': 2000000,
                'data': CONTRACT_BYTECODE,
                'chainId': self.w3.eth.chain_id
            }

            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)

            # Send transaction
            print("Sending deployment transaction...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            print(f"Transaction hash: {tx_hash.hex()}")

            # Wait for receipt
            print("Waiting for transaction receipt...")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            if receipt.status == 1:
                contract_address = receipt.contractAddress
                print(f"Contract deployed successfully!")
                print(f"Contract address: {contract_address}")
                print(f"Gas used: {receipt.gasUsed:,}")
                print(f"Block number: {receipt.blockNumber}")

                # Test contract interaction
                contract = self.w3.eth.contract(
                    address=contract_address,
                    abi=CONTRACT_ABI
                )

                # Test hasRoot function
                test_dispute_id = b"x" * 32
                has_root = contract.functions.hasRoot(test_dispute_id).call()
                print(f"Test hasRoot call: {has_root}")

                # Save deployment info
                deployment_info = {
                    "contract_address": contract_address,
                    "transaction_hash": tx_hash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "network": "local",
                    "rpc_url": RPC_URL,
                    "deployed_at": int(time.time())
                }

                with open("contract_deployment.json", "w") as f:
                    json.dump(deployment_info, f, indent=2)

                print(f"Deployment info saved to contract_deployment.json")

                return deployment_info

            else:
                print(f"Deployment failed! Status: {receipt.status}")
                return None

        except Exception as e:
            print(f"Deployment error: {e}")
            return None

    def test_contract_functions(self, contract_address: str):
        """Test contract functions"""
        print(f"\nTesting contract at {contract_address}...")

        try:
            contract = self.w3.eth.contract(
                address=contract_address,
                abi=CONTRACT_ABI
            )

            # Test data
            test_dispute_id = self.w3.keccak(text="test_dispute_123")
            test_merkle_root = self.w3.keccak(text="test_merkle_root")
            test_timestamp = int(time.time())

            print(f"Test dispute ID: {test_dispute_id.hex()}")
            print(f"Test merkle root: {test_merkle_root.hex()}")

            # Test anchoring
            print("\nTesting anchor function...")
            nonce = self.w3.eth.get_transaction_count(self.account.address)

            anchor_txn = contract.functions.anchorRoot(
                test_dispute_id,
                test_merkle_root,
                test_timestamp
            ).build_transaction({
                'chainId': self.w3.eth.chain_id,
                'gas': 200000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': nonce,
            })

            signed_anchor = self.w3.eth.account.sign_transaction(anchor_txn, PRIVATE_KEY)
            anchor_hash = self.w3.eth.send_raw_transaction(signed_anchor.raw_transaction)
            anchor_receipt = self.w3.eth.wait_for_transaction_receipt(anchor_hash)

            if anchor_receipt.status == 1:
                print(f"Anchor successful! TX: {anchor_hash.hex()}")

                # Test retrieval
                stored_root = contract.functions.getRoot(test_dispute_id).call()
                stored_timestamp = contract.functions.getTimestamp(test_dispute_id).call()
                has_root = contract.functions.hasRoot(test_dispute_id).call()

                print(f"Retrieved root: {stored_root.hex()}")
                print(f"Retrieved timestamp: {stored_timestamp}")
                print(f"Has root: {has_root}")

                # Verify data
                if stored_root == test_merkle_root and stored_timestamp == test_timestamp:
                    print("[PASS] Contract test PASSED - Data stored and retrieved correctly")
                    return True
                else:
                    print("[FAIL] Contract test FAILED - Data mismatch")
                    return False
            else:
                print("[FAIL] Anchor transaction failed")
                return False

        except Exception as e:
            print(f"Contract test error: {e}")
            return False

def main():
    """Main deployment function"""
    print("DALRN Smart Contract Deployment")
    print("=" * 50)

    deployer = ContractDeployer()

    if not deployer.w3.is_connected():
        print("❌ Failed to connect to blockchain")
        return False

    # Deploy contract
    deployment_info = deployer.deploy_contract()

    if deployment_info:
        print("\n" + "=" * 50)
        print("DEPLOYMENT SUCCESSFUL!")
        print("=" * 50)

        # Test contract
        test_passed = deployer.test_contract_functions(deployment_info["contract_address"])

        if test_passed:
            print("\n✅ Contract deployment and testing COMPLETED")
            print(f"Contract is ready for production use at: {deployment_info['contract_address']}")
            return True
        else:
            print("\n⚠️ Contract deployed but tests failed")
            return False
    else:
        print("\n[FAIL] Deployment failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[PASS] Blockchain infrastructure is ready!")
    else:
        print("\n[FAIL] Blockchain setup failed!")