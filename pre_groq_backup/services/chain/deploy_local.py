#!/usr/bin/env python3
"""
Local Deployment Helper Script
Helps deploy and test the AnchorReceipts contract locally with Anvil
"""

import subprocess
import json
import time
import sys
import os
from web3 import Web3
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import AnchorClient


def compile_contract():
    """Compile the contract using solc"""
    print("Compiling contract...")
    
    contract_path = Path(__file__).parent / "contracts" / "AnchorReceipts.sol"
    
    # Simple compilation command (requires solc installed)
    # For production, use Foundry's forge build
    cmd = [
        "solc",
        "--optimize",
        "--bin",
        "--abi",
        "--overwrite",
        "-o", str(Path(__file__).parent / "build"),
        str(contract_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return False
        print("Contract compiled successfully")
        return True
    except FileNotFoundError:
        print("solc not found. Please install solc or use Foundry for compilation.")
        print("Assuming contract is already compiled...")
        return True


def deploy_with_web3():
    """Deploy contract using Web3.py"""
    print("\nDeploying contract with Web3.py...")
    
    # Connect to local node
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    
    if not w3.is_connected():
        print("Error: Cannot connect to Ethereum node. Is Anvil running?")
        print("Start Anvil with: anvil")
        return None
    
    # Use first Anvil account
    account = w3.eth.accounts[0]
    print(f"Deploying from account: {account}")
    
    # Read ABI
    abi_path = Path(__file__).parent / "abi" / "AnchorReceipts.json"
    with open(abi_path, 'r') as f:
        abi = json.load(f)
    
    # Simple bytecode for deployment (simplified version)
    # In production, read from compiled output
    bytecode = "0x608060405234801561001057600080fd5b50612000806100206000396000f3fe"  # Placeholder
    
    # Note: For actual deployment, you need the real bytecode from compilation
    # This is a simplified example
    print("Note: Using placeholder bytecode. For real deployment, compile with Foundry first.")
    
    # Create contract instance
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Build transaction
    tx = Contract.constructor().build_transaction({
        'from': account,
        'gas': 2000000,
        'gasPrice': w3.eth.gas_price,
        'nonce': w3.eth.get_transaction_count(account),
    })
    
    # For Anvil, we can use eth_sendTransaction directly without signing
    try:
        tx_hash = w3.eth.send_transaction(tx)
        print(f"Transaction sent: {tx_hash.hex()}")
        
        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = receipt['contractAddress']
        
        print(f"Contract deployed at: {contract_address}")
        print(f"Gas used: {receipt['gasUsed']}")
        
        return contract_address
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        return None


def test_contract(contract_address):
    """Test the deployed contract"""
    print(f"\nTesting contract at {contract_address}...")
    
    # Initialize client
    client = AnchorClient(
        contract_address=contract_address,
        private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    )
    
    print("1. Testing anchor_root...")
    try:
        result = client.anchor_root(
            dispute_id="test-dispute-local",
            merkle_root="0x" + "a" * 64,
            model_hash="0x" + "b" * 64,
            round=1,
            uri="ipfs://QmLocalTest123",
            tags=["PoDP", "LocalTest"]
        )
        print(f"   ✓ Root anchored in block {result['block_number']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("2. Testing latest_root...")
    try:
        result = client.latest_root("test-dispute-local")
        print(f"   ✓ Latest root: {result['merkle_root'][:10]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("3. Testing anchor_receipt...")
    try:
        result = client.anchor_receipt(
            dispute_id="test-dispute-local",
            receipt_hash="0x" + "c" * 64,
            step_index=1,
            uri="ipfs://QmReceipt123"
        )
        print(f"   ✓ Receipt anchored in block {result['block_number']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("4. Testing contract info...")
    try:
        info = client.get_contract_info()
        print(f"   ✓ Total roots: {info['total_roots']}")
        print(f"   ✓ Total receipts: {info['total_receipts']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True


def save_deployment_info(contract_address):
    """Save deployment information"""
    deployment_info = {
        "network": "localhost",
        "contract_address": contract_address,
        "rpc_url": "http://127.0.0.1:8545",
        "deployer": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        "timestamp": int(time.time())
    }
    
    deploy_file = Path(__file__).parent / "deployment_local.json"
    with open(deploy_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\nDeployment info saved to: {deploy_file}")
    
    # Create .env file for easy use
    env_file = Path(__file__).parent / ".env.local"
    with open(env_file, 'w') as f:
        f.write(f"# Local deployment configuration\n")
        f.write(f"RPC_URL=http://127.0.0.1:8545\n")
        f.write(f"ANCHOR_ADDRESS={contract_address}\n")
        f.write(f"PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80\n")
    
    print(f"Environment file saved to: {env_file}")


def main():
    """Main deployment flow"""
    print("=" * 60)
    print("AnchorReceipts Local Deployment Script")
    print("=" * 60)
    
    # Check if Anvil is running
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    if not w3.is_connected():
        print("\n⚠ Anvil is not running!")
        print("Please start Anvil in another terminal with:")
        print("  anvil")
        print("\nOr if you don't have Foundry installed:")
        print("  curl -L https://foundry.paradigm.xyz | bash")
        print("  foundryup")
        sys.exit(1)
    
    print(f"\n✓ Connected to Anvil at http://127.0.0.1:8545")
    print(f"  Chain ID: {w3.eth.chain_id}")
    print(f"  Latest block: {w3.eth.block_number}")
    
    # Note about compilation
    print("\n" + "=" * 60)
    print("NOTE: This script uses a simplified deployment process.")
    print("For production deployment, use Foundry's forge script:")
    print("  forge script scripts/Deploy.s.sol --broadcast --rpc-url http://127.0.0.1:8545")
    print("=" * 60)
    
    # Deploy contract
    print("\nWould you like to proceed with simplified deployment? (y/n): ", end="")
    response = input().strip().lower()
    
    if response != 'y':
        print("Deployment cancelled.")
        sys.exit(0)
    
    # For this simplified version, we'll show how to structure it
    # In reality, you'd need the actual compiled bytecode
    print("\n⚠ This is a demonstration of the deployment structure.")
    print("For actual deployment, please use Foundry's forge tool.")
    
    # Create example deployment info
    example_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    
    print(f"\nExample deployment address: {example_address}")
    save_deployment_info(example_address)
    
    print("\n" + "=" * 60)
    print("Deployment structure created successfully!")
    print("\nTo actually deploy the contract:")
    print("1. Install Foundry: curl -L https://foundry.paradigm.xyz | bash")
    print("2. Run: foundryup")
    print("3. Start Anvil: anvil")
    print("4. Deploy: forge script services/chain/scripts/Deploy.s.sol --broadcast --rpc-url http://127.0.0.1:8545")
    print("\nThen update the ANCHOR_ADDRESS in .env.local with the deployed address.")
    print("=" * 60)


if __name__ == "__main__":
    main()