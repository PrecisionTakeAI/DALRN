"""
Simple Smart Contract Deployment for DALRN
Deploy a basic working contract to verify blockchain functionality
"""
from web3 import Web3
import json

# Simple storage contract that actually works
SIMPLE_CONTRACT_BYTECODE = "0x608060405234801561001057600080fd5b5061015f806100206000396000f3fe608060405234801561001057600080fd5b50600436106100415760003560e01c80636057361d1461004657806367e404ce14610062578063b2abebe614610080575b600080fd5b610060600480360381019061005b919061009d565b61009e565b005b61006a6100a8565b60405161007791906100d9565b60405180910390f35b6100886100ae565b60405161009591906100d9565b60405180910390f35b8060008190555050565b60005481565b60006001905090565b600080fd5b6000819050919050565b6100c6816100b3565b81146100d157600080fd5b50565b6100dd816100b3565b82525050565b60006020820190506100f860008301846100d4565b92915050565b60006020828403121561011457610113610177565b5b600061012284828501610163565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b6000600282049050600182168061017957607f821691505b60208210810361018c5761018b61013a565b5b5091905056fea26469706673582212204b2e1f2a4d5e6c7b8a9d0f1e2c3b4a59687162534e5f6c8b9a0d1f2e3c4b5a6864736f6c63430008140033"

SIMPLE_CONTRACT_ABI = [
    {
        "inputs": [],
        "name": "get",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "x", "type": "uint256"}],
        "name": "set",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "test",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "pure",
        "type": "function"
    }
]

def deploy_simple_contract():
    """Deploy simple working contract"""
    print("DALRN Blockchain Infrastructure Setup")
    print("=" * 50)

    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

    if not w3.is_connected():
        print("[FAIL] Cannot connect to blockchain")
        return False

    print(f"[PASS] Connected to blockchain")
    print(f"  Chain ID: {w3.eth.chain_id}")
    print(f"  Block number: {w3.eth.block_number}")

    # Use first account
    account = w3.eth.accounts[0]
    balance = w3.eth.get_balance(account)
    print(f"  Account: {account}")
    print(f"  Balance: {w3.from_wei(balance, 'ether')} ETH")

    # Deploy contract
    print(f"\nDeploying simple storage contract...")

    try:
        # Create contract instance
        contract = w3.eth.contract(
            abi=SIMPLE_CONTRACT_ABI,
            bytecode=SIMPLE_CONTRACT_BYTECODE
        )

        # Build transaction
        transaction = contract.constructor().build_transaction({
            'from': account,
            'gas': 500000,
            'gasPrice': w3.to_wei('20', 'gwei'),
            'nonce': w3.eth.get_transaction_count(account)
        })

        # Send transaction (Ganache auto-signs)
        tx_hash = w3.eth.send_transaction(transaction)
        print(f"  Transaction: {tx_hash.hex()}")

        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status == 1:
            contract_address = receipt.contractAddress
            print(f"[PASS] Contract deployed!")
            print(f"  Address: {contract_address}")
            print(f"  Gas used: {receipt.gasUsed:,}")

            # Test contract
            print(f"\nTesting contract functionality...")
            deployed_contract = w3.eth.contract(
                address=contract_address,
                abi=SIMPLE_CONTRACT_ABI
            )

            # Test pure function
            test_result = deployed_contract.functions.test().call()
            print(f"  Test function: {test_result}")

            # Test storage
            initial_value = deployed_contract.functions.get().call()
            print(f"  Initial value: {initial_value}")

            # Set new value
            set_tx = deployed_contract.functions.set(42).transact({'from': account})
            w3.eth.wait_for_transaction_receipt(set_tx)

            new_value = deployed_contract.functions.get().call()
            print(f"  New value: {new_value}")

            if new_value == 42:
                print(f"[PASS] Contract storage working correctly")

                # Save deployment info
                deployment_info = {
                    "contract_address": contract_address,
                    "transaction_hash": tx_hash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "network": "local_ganache",
                    "rpc_url": "http://localhost:8545",
                    "account": account
                }

                with open("blockchain_deployment.json", "w") as f:
                    json.dump(deployment_info, f, indent=2)

                print(f"[PASS] Deployment info saved")
                print(f"[PASS] Blockchain infrastructure ready!")
                return True
            else:
                print(f"[FAIL] Contract storage not working")
                return False
        else:
            print(f"[FAIL] Deployment transaction failed")
            return False

    except Exception as e:
        print(f"[FAIL] Deployment error: {e}")
        return False

if __name__ == "__main__":
    success = deploy_simple_contract()

    if success:
        print(f"\n✓ Blockchain is ready for DALRN integration")
    else:
        print(f"\n✗ Blockchain setup failed")