#!/usr/bin/env python3
"""
Smart Contract Deployment Script for DALRN
Deploys AnchorReceipts contract to specified network
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any
from web3 import Web3
from eth_account import Account
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Contract artifacts path
ARTIFACTS_PATH = Path(__file__).parent / "abi"
CONTRACT_NAME = "AnchorReceipts"


class ContractDeployer:
    """Handles smart contract deployment to various networks"""

    def __init__(self, network: str = "local"):
        """
        Initialize deployer for specified network.

        Args:
            network: Network to deploy to (local, testnet, mainnet)
        """
        self.network = network
        self.w3 = None
        self.account = None
        self.contract_address = None
        self.setup_network()

    def setup_network(self):
        """Configure network connection based on environment"""
        networks = {
            "local": {
                "url": os.getenv("WEB3_PROVIDER_URL", "http://localhost:8545"),
                "chain_id": 31337,
                "name": "Anvil Local"
            },
            "sepolia": {
                "url": os.getenv("SEPOLIA_RPC_URL", "https://rpc.sepolia.org"),
                "chain_id": 11155111,
                "name": "Sepolia Testnet"
            },
            "mainnet": {
                "url": os.getenv("MAINNET_RPC_URL", "https://eth.llamarpc.com"),
                "chain_id": 1,
                "name": "Ethereum Mainnet"
            }
        }

        if self.network not in networks:
            raise ValueError(f"Unknown network: {self.network}")

        config = networks[self.network]
        logger.info(f"Connecting to {config['name']}...")

        # Connect to network
        self.w3 = Web3(Web3.HTTPProvider(config["url"]))

        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to {config['name']} at {config['url']}")

        logger.info(f"Connected to {config['name']} (Chain ID: {config['chain_id']})")

        # Setup account
        self.setup_account()

    def setup_account(self):
        """Setup deployment account"""
        if self.network == "local":
            # Use first Anvil account for local deployment
            if len(self.w3.eth.accounts) == 0:
                raise ValueError("No accounts available. Is Anvil running?")
            self.account = self.w3.eth.accounts[0]
            logger.info(f"Using Anvil account: {self.account}")
        else:
            # Use private key for other networks
            private_key = os.getenv("DEPLOYER_PRIVATE_KEY")
            if not private_key:
                raise ValueError("DEPLOYER_PRIVATE_KEY environment variable not set")

            account = Account.from_key(private_key)
            self.account = account.address
            self.private_key = private_key
            logger.info(f"Using deployer account: {self.account}")

        # Check balance
        balance = self.w3.eth.get_balance(self.account)
        balance_eth = self.w3.from_wei(balance, 'ether')
        logger.info(f"Account balance: {balance_eth:.4f} ETH")

        if balance == 0:
            raise ValueError("Deployment account has no ETH")

    def load_contract_artifacts(self) -> tuple[str, list]:
        """Load contract ABI and bytecode"""
        # Load ABI
        abi_file = ARTIFACTS_PATH / f"{CONTRACT_NAME}.json"
        if not abi_file.exists():
            raise FileNotFoundError(f"ABI file not found: {abi_file}")

        with open(abi_file, 'r') as f:
            abi = json.load(f)

        # Load bytecode
        bytecode_file = ARTIFACTS_PATH / f"{CONTRACT_NAME}.bytecode"
        if bytecode_file.exists():
            with open(bytecode_file, 'r') as f:
                bytecode = f.read().strip()
        else:
            # Use compiled bytecode from Solidity compiler output
            # This is a placeholder - in production, get from forge build output
            logger.warning("Using placeholder bytecode. Run forge build for actual bytecode.")
            bytecode = self.get_compiled_bytecode()

        return bytecode, abi

    def get_compiled_bytecode(self) -> str:
        """Get compiled bytecode from Foundry output"""
        foundry_out = Path(__file__).parent.parent.parent / "out" / f"{CONTRACT_NAME}.sol" / f"{CONTRACT_NAME}.json"

        if foundry_out.exists():
            with open(foundry_out, 'r') as f:
                data = json.load(f)
                bytecode = data.get("bytecode", {}).get("object", "")
                if bytecode:
                    logger.info("Loaded bytecode from Foundry output")
                    return bytecode

        # Return minimal bytecode for testing
        # This deploys a minimal contract but won't have full functionality
        return "0x608060405234801561001057600080fd5b50610150806100206000396000f3fe"

    def estimate_gas_price(self) -> int:
        """Estimate appropriate gas price for network"""
        if self.network == "local":
            return self.w3.eth.gas_price

        # For other networks, use dynamic gas pricing
        base_fee = self.w3.eth.get_block('latest')['baseFeePerGas']
        priority_fee = self.w3.to_wei(2, 'gwei')  # 2 gwei priority

        max_fee = base_fee * 2 + priority_fee
        logger.info(f"Gas price: Base={self.w3.from_wei(base_fee, 'gwei'):.2f} gwei, "
                   f"Max={self.w3.from_wei(max_fee, 'gwei'):.2f} gwei")

        return max_fee

    def deploy_contract(self) -> str:
        """
        Deploy the AnchorReceipts contract.

        Returns:
            Deployed contract address
        """
        logger.info(f"Deploying {CONTRACT_NAME} contract...")

        # Load artifacts
        bytecode, abi = self.load_contract_artifacts()

        # Create contract instance
        Contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)

        # Estimate gas
        gas_estimate = Contract.constructor().estimate_gas({'from': self.account})
        gas_limit = int(gas_estimate * 1.2)  # Add 20% buffer
        logger.info(f"Estimated gas: {gas_estimate}, using limit: {gas_limit}")

        # Build transaction
        tx_params = {
            'from': self.account,
            'gas': gas_limit,
            'nonce': self.w3.eth.get_transaction_count(self.account),
        }

        if self.network == "local":
            # Simple gas price for local network
            tx_params['gasPrice'] = self.w3.eth.gas_price
        else:
            # EIP-1559 transaction for other networks
            tx_params['maxFeePerGas'] = self.estimate_gas_price()
            tx_params['maxPriorityFeePerGas'] = self.w3.to_wei(2, 'gwei')

        # Build constructor transaction
        tx = Contract.constructor().build_transaction(tx_params)

        # Sign and send transaction
        if self.network == "local":
            # Anvil allows sending without signing
            tx_hash = self.w3.eth.send_transaction(tx)
        else:
            # Sign transaction for other networks
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        logger.info(f"Transaction sent: {tx_hash.hex()}")
        logger.info("Waiting for confirmation...")

        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt['status'] != 1:
            raise Exception(f"Deployment failed! Transaction reverted: {tx_hash.hex()}")

        self.contract_address = receipt['contractAddress']
        logger.info(f"✅ Contract deployed at: {self.contract_address}")
        logger.info(f"Gas used: {receipt['gasUsed']:,}")
        logger.info(f"Block number: {receipt['blockNumber']}")

        return self.contract_address

    def verify_deployment(self) -> bool:
        """Verify the deployed contract is functional"""
        if not self.contract_address:
            raise ValueError("No contract deployed yet")

        logger.info("Verifying deployment...")

        # Load ABI
        _, abi = self.load_contract_artifacts()
        contract = self.w3.eth.contract(address=self.contract_address, abi=abi)

        # Check contract code exists
        code = self.w3.eth.get_code(self.contract_address)
        if code == b'':
            logger.error("No code at contract address!")
            return False

        # Try to read a view function
        try:
            total_roots = contract.functions.totalRootsAnchored().call()
            logger.info(f"✅ Contract verified. Total roots anchored: {total_roots}")
            return True
        except Exception as e:
            logger.error(f"Contract verification failed: {e}")
            return False

    def save_deployment_info(self):
        """Save deployment information to file"""
        deployment_info = {
            "network": self.network,
            "contract_name": CONTRACT_NAME,
            "contract_address": self.contract_address,
            "deployer": self.account,
            "timestamp": int(time.time()),
            "block_number": self.w3.eth.block_number
        }

        # Save to deployments directory
        deployments_dir = Path(__file__).parent / "deployments"
        deployments_dir.mkdir(exist_ok=True)

        deployment_file = deployments_dir / f"{self.network}.json"

        # Load existing deployments
        existing = {}
        if deployment_file.exists():
            with open(deployment_file, 'r') as f:
                existing = json.load(f)

        # Add new deployment
        existing[CONTRACT_NAME] = deployment_info

        # Save updated deployments
        with open(deployment_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Deployment info saved to {deployment_file}")

        # Update .env file if local deployment
        if self.network == "local":
            self.update_env_file()

    def update_env_file(self):
        """Update .env file with contract address"""
        env_file = Path(__file__).parent.parent.parent / ".env"

        if not env_file.exists():
            # Copy from .env.example
            env_example = Path(__file__).parent.parent.parent / ".env.example"
            if env_example.exists():
                import shutil
                shutil.copy(env_example, env_file)
                logger.info("Created .env from .env.example")

        # Update ANCHOR_CONTRACT_ADDRESS
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = f.readlines()

            updated = False
            for i, line in enumerate(lines):
                if line.startswith("ANCHOR_CONTRACT_ADDRESS="):
                    lines[i] = f"ANCHOR_CONTRACT_ADDRESS={self.contract_address}\n"
                    updated = True
                    break

            if not updated:
                lines.append(f"\n# Auto-updated by deployment script\n")
                lines.append(f"ANCHOR_CONTRACT_ADDRESS={self.contract_address}\n")

            with open(env_file, 'w') as f:
                f.writelines(lines)

            logger.info("Updated .env file with contract address")


def main():
    """Main deployment function"""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy DALRN smart contracts")
    parser.add_argument(
        "--network",
        choices=["local", "sepolia", "mainnet"],
        default="local",
        help="Network to deploy to"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify contract on Etherscan (testnet/mainnet only)"
    )

    args = parser.parse_args()

    try:
        # Create deployer
        deployer = ContractDeployer(network=args.network)

        # Deploy contract
        contract_address = deployer.deploy_contract()

        # Verify deployment
        if deployer.verify_deployment():
            # Save deployment info
            deployer.save_deployment_info()

            logger.info("")
            logger.info("=" * 60)
            logger.info("🎉 Deployment Successful!")
            logger.info(f"Contract: {CONTRACT_NAME}")
            logger.info(f"Network: {args.network}")
            logger.info(f"Address: {contract_address}")
            logger.info("=" * 60)

            # Etherscan verification reminder
            if args.network in ["sepolia", "mainnet"] and args.verify:
                logger.info("")
                logger.info("To verify on Etherscan, run:")
                logger.info(f"forge verify-contract {contract_address} {CONTRACT_NAME}")
        else:
            logger.error("Deployment verification failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()