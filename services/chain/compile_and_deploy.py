#!/usr/bin/env python3
"""
Compile and Deploy AnchorReceipts Contract
This script compiles the Solidity contract and deploys it to a local Ethereum node.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from web3 import Web3
from eth_account import Account
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Contract paths
SCRIPT_DIR = Path(__file__).parent
CONTRACT_PATH = SCRIPT_DIR / "contracts" / "AnchorReceipts.sol"
BUILD_DIR = SCRIPT_DIR / "build"
ABI_PATH = SCRIPT_DIR / "abi" / "AnchorReceipts.json"

# Default RPC and account (Anvil/Ganache default)
DEFAULT_RPC = "http://127.0.0.1:8545"
DEFAULT_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

def install_solc():
    """Install solc-select and solc if not available"""
    try:
        # Check if solc is available
        result = subprocess.run(["solc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("solc is already installed")
            return True
    except FileNotFoundError:
        pass

    logger.info("Installing solc-select and solc...")
    try:
        # Install solc-select
        subprocess.run([sys.executable, "-m", "pip", "install", "solc-select"], check=True)

        # Install solc version 0.8.24
        subprocess.run(["solc-select", "install", "0.8.24"], check=True)
        subprocess.run(["solc-select", "use", "0.8.24"], check=True)

        logger.info("solc 0.8.24 installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install solc: {e}")
        return False

def compile_with_solc():
    """Compile the contract using solc and extract bytecode"""
    logger.info("Compiling contract with solc...")

    # Create build directory
    BUILD_DIR.mkdir(exist_ok=True)

    try:
        # Compile contract
        cmd = [
            "solc",
            "--optimize",
            "--optimize-runs", "200",
            "--combined-json", "abi,bin,bin-runtime",
            "--overwrite",
            "--base-path", str(SCRIPT_DIR),
            str(CONTRACT_PATH)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Compilation failed: {result.stderr}")
            return None, None

        # Parse output
        output = json.loads(result.stdout)
        contract_key = f"{CONTRACT_PATH}:AnchorReceipts"

        if contract_key not in output["contracts"]:
            # Try alternative key format
            contract_key = "contracts/AnchorReceipts.sol:AnchorReceipts"
            if contract_key not in output["contracts"]:
                logger.error(f"Contract not found in compilation output. Available: {list(output['contracts'].keys())}")
                return None, None

        contract_data = output["contracts"][contract_key]
        bytecode = "0x" + contract_data["bin"]
        abi = json.loads(contract_data["abi"])

        # Save ABI
        with open(ABI_PATH, 'w') as f:
            json.dump(abi, f, indent=2)
        logger.info(f"ABI saved to {ABI_PATH}")

        # Save bytecode
        bytecode_path = BUILD_DIR / "AnchorReceipts.bin"
        with open(bytecode_path, 'w') as f:
            f.write(bytecode)
        logger.info(f"Bytecode saved to {bytecode_path}")

        return bytecode, abi

    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return None, None

def compile_with_py_solc():
    """Alternative: Compile using py-solc-x"""
    logger.info("Attempting compilation with py-solc-x...")

    try:
        from solcx import compile_source, install_solc, set_solc_version

        # Install and set solc version
        install_solc('0.8.24')
        set_solc_version('0.8.24')

        # Read contract source
        with open(CONTRACT_PATH, 'r') as f:
            contract_source = f.read()

        # Compile
        compiled = compile_source(
            contract_source,
            output_values=['abi', 'bin'],
            solc_version='0.8.24'
        )

        # Get contract data
        contract_id = '<stdin>:AnchorReceipts'
        contract_data = compiled[contract_id]

        bytecode = "0x" + contract_data['bin']
        abi = contract_data['abi']

        # Save ABI
        with open(ABI_PATH, 'w') as f:
            json.dump(abi, f, indent=2)

        # Save bytecode
        bytecode_path = BUILD_DIR / "AnchorReceipts.bin"
        with open(bytecode_path, 'w') as f:
            f.write(bytecode)
        logger.info(f"Bytecode saved to {bytecode_path}")

        return bytecode, abi

    except ImportError:
        logger.info("py-solc-x not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "py-solc-x"], check=True)
        return compile_with_py_solc()
    except Exception as e:
        logger.error(f"py-solc-x compilation failed: {e}")
        return None, None

def deploy_contract(bytecode, abi, rpc_url=DEFAULT_RPC, private_key=DEFAULT_PRIVATE_KEY):
    """Deploy the contract to the blockchain"""
    logger.info(f"Deploying contract to {rpc_url}...")

    # Connect to node
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        logger.error(f"Cannot connect to Ethereum node at {rpc_url}")
        logger.info("Please ensure Ganache or Anvil is running:")
        logger.info("  For Ganache: ganache --port 8545")
        logger.info("  For Anvil: anvil")
        return None

    logger.info(f"Connected to chain ID: {w3.eth.chain_id}")

    # Setup account
    if private_key.startswith("0x"):
        private_key = private_key[2:]
    account = Account.from_key(private_key)
    logger.info(f"Deploying from account: {account.address}")
    logger.info(f"Account balance: {w3.eth.get_balance(account.address) / 10**18:.4f} ETH")

    # Create contract
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Build deployment transaction
    constructor_tx = Contract.constructor().build_transaction({
        'from': account.address,
        'gas': 3000000,
        'gasPrice': w3.eth.gas_price,
        'nonce': w3.eth.get_transaction_count(account.address),
        'chainId': w3.eth.chain_id
    })

    # Sign and send transaction
    signed_tx = account.sign_transaction(constructor_tx)
    # Handle both old and new API
    raw_tx = signed_tx.raw_transaction if hasattr(signed_tx, 'raw_transaction') else signed_tx.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw_tx)
    logger.info(f"Deployment transaction sent: {tx_hash.hex()}")

    # Wait for confirmation
    logger.info("Waiting for deployment confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt.status == 0:
        logger.error("Deployment failed!")
        return None

    contract_address = receipt.contractAddress
    logger.info(f"Contract deployed successfully at: {contract_address}")
    logger.info(f"Gas used: {receipt.gasUsed:,}")
    logger.info(f"Block number: {receipt.blockNumber}")

    return {
        "address": contract_address,
        "tx_hash": tx_hash.hex(),
        "block_number": receipt.blockNumber,
        "gas_used": receipt.gasUsed,
        "deployer": account.address
    }

def verify_deployment(contract_address, abi, rpc_url=DEFAULT_RPC):
    """Verify the deployed contract works correctly"""
    logger.info(f"Verifying contract at {contract_address}...")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=abi)

    try:
        # Check initial state
        total_roots = contract.functions.totalRootsAnchored().call()
        total_receipts = contract.functions.totalReceiptsAnchored().call()

        logger.info(f"Contract state verified:")
        logger.info(f"  Total roots anchored: {total_roots}")
        logger.info(f"  Total receipts anchored: {total_receipts}")

        # Test read function
        test_dispute_id = Web3.keccak(text="test-dispute").hex()
        has_root = contract.functions.hasRoot(bytes.fromhex(test_dispute_id[2:])).call()
        logger.info(f"  hasRoot(test-dispute): {has_root}")

        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def test_contract_functions(contract_address, abi, rpc_url=DEFAULT_RPC, private_key=DEFAULT_PRIVATE_KEY):
    """Test contract functions with actual transactions"""
    logger.info("Testing contract functions...")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=abi)

    # Setup account
    if private_key.startswith("0x"):
        private_key = private_key[2:]
    account = Account.from_key(private_key)

    try:
        # Test data
        dispute_id = Web3.keccak(text="test-dispute-001")
        merkle_root = Web3.keccak(text="test-merkle-root-001")
        model_hash = Web3.keccak(text="test-model-001")
        tags = [Web3.keccak(text="PoDP"), Web3.keccak(text="Test")]

        # Test anchorRoot
        logger.info("Testing anchorRoot function...")
        tx = contract.functions.anchorRoot(
            dispute_id,
            merkle_root,
            model_hash,
            1,  # round
            "ipfs://QmTest123",
            tags
        ).build_transaction({
            'from': account.address,
            'gas': 500000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account.address),
            'chainId': w3.eth.chain_id
        })

        signed_tx = account.sign_transaction(tx)
        raw_tx = signed_tx.raw_transaction if hasattr(signed_tx, 'raw_transaction') else signed_tx.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw_tx)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status == 1:
            logger.info(f"  ✓ anchorRoot successful (gas: {receipt.gasUsed:,})")

            # Check the root was stored
            stored_root, block_num = contract.functions.latestRoot(dispute_id).call()
            logger.info(f"  ✓ latestRoot returns: {stored_root.hex()[:10]}... at block {block_num}")

            # Test anchorReceipt
            logger.info("Testing anchorReceipt function...")
            receipt_hash = Web3.keccak(text="test-receipt-001")

            tx2 = contract.functions.anchorReceipt(
                dispute_id,
                receipt_hash,
                1,  # stepIndex
                "ipfs://QmReceipt123"
            ).build_transaction({
                'from': account.address,
                'gas': 300000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(account.address),
                'chainId': w3.eth.chain_id
            })

            signed_tx2 = account.sign_transaction(tx2)
            raw_tx2 = signed_tx2.raw_transaction if hasattr(signed_tx2, 'raw_transaction') else signed_tx2.rawTransaction
            tx_hash2 = w3.eth.send_raw_transaction(raw_tx2)
            receipt2 = w3.eth.wait_for_transaction_receipt(tx_hash2)

            if receipt2.status == 1:
                logger.info(f"  ✓ anchorReceipt successful (gas: {receipt2.gasUsed:,})")

                # Check totals
                total_roots = contract.functions.totalRootsAnchored().call()
                total_receipts = contract.functions.totalReceiptsAnchored().call()
                logger.info(f"  ✓ Total roots: {total_roots}, Total receipts: {total_receipts}")

                return True
            else:
                logger.error("  ✗ anchorReceipt failed")
                return False
        else:
            logger.error("  ✗ anchorRoot failed")
            return False

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return False

def save_deployment_info(deployment_info, network="local"):
    """Save deployment information to files"""
    # Save deployment JSON
    deployment_file = SCRIPT_DIR / f"deployment_{network}.json"
    deployment_data = {
        "network": network,
        "contract_address": deployment_info["address"],
        "deployment_tx": deployment_info["tx_hash"],
        "deployment_block": deployment_info["block_number"],
        "deployer": deployment_info["deployer"],
        "timestamp": int(time.time()),
        "rpc_url": DEFAULT_RPC
    }

    with open(deployment_file, 'w') as f:
        json.dump(deployment_data, f, indent=2)
    logger.info(f"Deployment info saved to {deployment_file}")

    # Update .env file
    env_file = SCRIPT_DIR / ".env.local"
    env_content = f"""# AnchorReceipts Contract Deployment
ANCHOR_CONTRACT_ADDRESS={deployment_info["address"]}
RPC_URL={DEFAULT_RPC}
PRIVATE_KEY={DEFAULT_PRIVATE_KEY}
NETWORK={network}
DEPLOYMENT_BLOCK={deployment_info["block_number"]}
"""

    with open(env_file, 'w') as f:
        f.write(env_content)
    logger.info(f"Environment file saved to {env_file}")

    # Update parent .env if exists
    parent_env = SCRIPT_DIR.parent.parent / ".env"
    if parent_env.exists():
        with open(parent_env, 'a') as f:
            f.write(f"\n# AnchorReceipts Contract\n")
            f.write(f"ANCHOR_CONTRACT_ADDRESS={deployment_info['address']}\n")
        logger.info(f"Updated {parent_env} with contract address")

def main():
    """Main deployment process"""
    print("=" * 70)
    print("AnchorReceipts Contract Deployment Script")
    print("=" * 70)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Deploy AnchorReceipts contract")
    parser.add_argument("--rpc", default=DEFAULT_RPC, help="RPC URL")
    parser.add_argument("--private-key", default=DEFAULT_PRIVATE_KEY, help="Private key for deployment")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing")
    args = parser.parse_args()

    # Step 1: Compile contract
    if not args.skip_compile:
        logger.info("\n📦 Step 1: Compiling contract...")

        # Try different compilation methods
        bytecode, abi = compile_with_solc()

        if not bytecode:
            logger.info("Trying alternative compilation method...")
            bytecode, abi = compile_with_py_solc()

        if not bytecode:
            logger.error("Failed to compile contract. Please install solc or py-solc-x.")
            sys.exit(1)

        logger.info("✅ Contract compiled successfully!")
    else:
        # Load existing ABI and bytecode
        logger.info("Loading existing ABI and bytecode...")
        with open(ABI_PATH, 'r') as f:
            abi = json.load(f)

        bytecode_path = BUILD_DIR / "AnchorReceipts.bin"
        if bytecode_path.exists():
            with open(bytecode_path, 'r') as f:
                bytecode = f.read()
        else:
            logger.error(f"Bytecode not found at {bytecode_path}. Please compile first.")
            sys.exit(1)

    # Step 2: Deploy contract
    logger.info("\n🚀 Step 2: Deploying contract...")
    deployment_info = deploy_contract(bytecode, abi, args.rpc, args.private_key)

    if not deployment_info:
        logger.error("Deployment failed!")
        sys.exit(1)

    logger.info("✅ Contract deployed successfully!")

    # Step 3: Verify deployment
    logger.info("\n🔍 Step 3: Verifying deployment...")
    if not verify_deployment(deployment_info["address"], abi, args.rpc):
        logger.error("Verification failed!")
        sys.exit(1)

    logger.info("✅ Deployment verified!")

    # Step 4: Test functions
    if not args.skip_test:
        logger.info("\n🧪 Step 4: Testing contract functions...")
        if not test_contract_functions(deployment_info["address"], abi, args.rpc, args.private_key):
            logger.error("Testing failed!")
            sys.exit(1)

        logger.info("✅ All tests passed!")

    # Step 5: Save deployment info
    logger.info("\n💾 Step 5: Saving deployment information...")
    save_deployment_info(deployment_info)

    # Summary
    print("\n" + "=" * 70)
    print("🎉 DEPLOYMENT SUCCESSFUL!")
    print("=" * 70)
    print(f"Contract Address: {deployment_info['address']}")
    print(f"Transaction Hash: {deployment_info['tx_hash']}")
    print(f"Block Number:     {deployment_info['block_number']}")
    print(f"Gas Used:         {deployment_info['gas_used']:,}")
    print(f"ABI Location:     {ABI_PATH}")
    print("=" * 70)
    print("\nNext steps:")
    print("1. The contract is now deployed and ready to use")
    print("2. ABI file is available at:", ABI_PATH)
    print("3. Use the AnchorClient class to interact with the contract")
    print("4. Run tests with: python -m pytest tests/test_chain.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())