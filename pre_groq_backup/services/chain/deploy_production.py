#!/usr/bin/env python3
"""
Production Deployment Script for AnchorReceipts Contract
Supports multiple networks and includes verification
"""

import os
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from web3 import Web3
from eth_account import Account
from client import AnchorClient, NetworkConfig


class ContractDeployer:
    """Handles contract deployment across different networks"""

    def __init__(
        self,
        network: NetworkConfig,
        private_key: str,
        rpc_url: Optional[str] = None
    ):
        """Initialize deployer with network configuration"""
        self.network = network
        self.private_key = private_key

        # Initialize client for network checks
        self.client = AnchorClient(
            network=network,
            private_key=private_key,
            rpc_url=rpc_url
        )

        self.w3 = self.client.w3
        self.account = self.client.account

        print(f"Deployer initialized for {network.value}")
        print(f"Account: {self.account.address}")
        print(f"Balance: {self.w3.from_wei(self.w3.eth.get_balance(self.account.address), 'ether')} ETH")

    def load_bytecode(self) -> str:
        """Load compiled contract bytecode"""
        # For production, this should load from Foundry's compiled output
        build_path = Path(__file__).parent / "out" / "AnchorReceipts.sol" / "AnchorReceipts.json"

        if not build_path.exists():
            print(f"Error: Compiled contract not found at {build_path}")
            print("Please compile with: forge build")
            sys.exit(1)

        with open(build_path, 'r') as f:
            artifact = json.load(f)

        return artifact["bytecode"]["object"]

    def deploy_contract(self) -> str:
        """Deploy the AnchorReceipts contract"""
        print("\nDeploying AnchorReceipts contract...")

        # Load ABI
        abi_path = Path(__file__).parent / "abi" / "AnchorReceipts.json"
        with open(abi_path, 'r') as f:
            abi = json.load(f)

        # Load bytecode
        bytecode = self.load_bytecode()

        # Create contract factory
        Contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)

        # Estimate gas
        try:
            gas_estimate = Contract.constructor().estimate_gas({
                'from': self.account.address
            })
            gas_limit = int(gas_estimate * 1.2)
            print(f"Estimated gas: {gas_estimate} (using {gas_limit})")
        except Exception as e:
            print(f"Gas estimation failed: {e}")
            gas_limit = 3000000

        # Get gas price
        if self.network in [NetworkConfig.MAINNET, NetworkConfig.POLYGON, NetworkConfig.ARBITRUM]:
            # For mainnet, be more careful with gas
            base_fee = self.w3.eth.get_block('latest')['baseFeePerGas']
            priority_fee = self.w3.eth.max_priority_fee
            max_fee = int(base_fee * 1.5 + priority_fee)

            tx_params = {
                'from': self.account.address,
                'gas': gas_limit,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': priority_fee,
                'nonce': self.w3.eth.get_transaction_count(self.account.address, 'pending')
            }
        else:
            # For testnets, use simpler gas pricing
            tx_params = {
                'from': self.account.address,
                'gas': gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address, 'pending')
            }

        # Build deployment transaction
        deployment_tx = Contract.constructor().build_transaction(tx_params)

        # Sign and send transaction
        signed_tx = self.account.sign_transaction(deployment_tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        print(f"Deployment transaction sent: {tx_hash.hex()}")
        print("Waiting for confirmation...")

        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)

        if receipt['status'] == 0:
            print("Error: Deployment failed!")
            return None

        contract_address = receipt['contractAddress']
        print(f"✓ Contract deployed at: {contract_address}")
        print(f"  Block: {receipt['blockNumber']}")
        print(f"  Gas used: {receipt['gasUsed']}")

        return contract_address

    def verify_deployment(self, contract_address: str) -> bool:
        """Verify the deployed contract is functioning"""
        print(f"\nVerifying contract at {contract_address}...")

        try:
            # Create client with the deployed contract
            client = AnchorClient(
                network=self.network,
                contract_address=contract_address,
                private_key=self.private_key
            )

            # Get contract info
            info = client.get_contract_info()
            print(f"✓ Contract verified:")
            print(f"  Total roots: {info['total_roots']}")
            print(f"  Total receipts: {info['total_receipts']}")

            # Test anchoring
            print("\nTesting anchor function...")
            result = client.anchor_root(
                dispute_id=f"deploy-test-{int(time.time())}",
                merkle_root="0x" + "1" * 64,
                model_hash="0x" + "2" * 64,
                round=1,
                uri="ipfs://deployment-test",
                tags=["deployment", "test"]
            )

            if result["status"] == "success":
                print(f"✓ Test anchor successful: {result['transaction_hash']}")
                return True
            else:
                print(f"✗ Test anchor failed: {result}")
                return False

        except Exception as e:
            print(f"✗ Verification failed: {e}")
            return False

    def save_deployment(self, contract_address: str):
        """Save deployment information"""
        deployment_info = {
            "network": self.network.value,
            "contract_address": contract_address,
            "deployer": self.account.address,
            "timestamp": int(time.time()),
            "block_number": self.w3.eth.block_number,
            "chain_id": self.w3.eth.chain_id
        }

        # Add explorer URL if available
        if self.client.network_settings.explorer_url:
            deployment_info["explorer_url"] = f"{self.client.network_settings.explorer_url}/address/{contract_address}"

        # Save to file
        output_file = Path(__file__).parent / f"deployment_{self.network.value}.json"
        with open(output_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)

        print(f"\nDeployment info saved to: {output_file}")

        # Create .env file
        env_file = Path(__file__).parent / f".env.{self.network.value}"
        with open(env_file, 'w') as f:
            f.write(f"# {self.network.value.upper()} deployment configuration\n")
            f.write(f"NETWORK={self.network.value}\n")
            f.write(f"ANCHOR_CONTRACT_ADDRESS={contract_address}\n")
            f.write(f"RPC_URL={self.client.rpc_url}\n")
            f.write(f"CHAIN_ID={self.w3.eth.chain_id}\n")
            if self.client.network_settings.explorer_url:
                f.write(f"EXPLORER_URL={self.client.network_settings.explorer_url}\n")

        print(f"Environment file saved to: {env_file}")

    def generate_verification_script(self, contract_address: str):
        """Generate Etherscan verification script"""
        if not self.client.network_settings.explorer_url:
            print("Skipping verification script (no explorer for this network)")
            return

        script = f"""#!/bin/bash
# Etherscan verification script for {self.network.value}

CONTRACT_ADDRESS={contract_address}
NETWORK={self.network.value}

# Verify on Etherscan (requires ETHERSCAN_API_KEY env var)
forge verify-contract \\
    --chain {self.network.value} \\
    --num-of-optimizations 200 \\
    --watch \\
    $CONTRACT_ADDRESS \\
    services/chain/contracts/AnchorReceipts.sol:AnchorReceipts

echo "Verification submitted. Check status at:"
echo "{self.client.network_settings.explorer_url}/address/$CONTRACT_ADDRESS#code"
"""

        script_file = Path(__file__).parent / f"verify_{self.network.value}.sh"
        with open(script_file, 'w') as f:
            f.write(script)

        os.chmod(script_file, 0o755)
        print(f"Verification script saved to: {script_file}")


def main():
    """Main deployment flow"""
    parser = argparse.ArgumentParser(description="Deploy AnchorReceipts contract")
    parser.add_argument(
        "--network",
        type=str,
        choices=[n.value for n in NetworkConfig],
        default="local",
        help="Target network for deployment"
    )
    parser.add_argument(
        "--private-key",
        type=str,
        help="Private key for deployment (or set PRIVATE_KEY env var)"
    )
    parser.add_argument(
        "--rpc-url",
        type=str,
        help="Custom RPC URL (optional, uses network defaults)"
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        help="Only verify existing contract at this address"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test transaction after deployment"
    )

    args = parser.parse_args()

    # Get private key
    private_key = args.private_key or os.getenv("PRIVATE_KEY")
    if not private_key:
        print("Error: Private key required (--private-key or PRIVATE_KEY env var)")
        sys.exit(1)

    # Get network
    network = NetworkConfig[args.network.upper()]

    print("=" * 60)
    print(f"AnchorReceipts Deployment - {network.value}")
    print("=" * 60)

    # Initialize deployer
    deployer = ContractDeployer(
        network=network,
        private_key=private_key,
        rpc_url=args.rpc_url
    )

    # Verify only mode
    if args.verify_only:
        success = deployer.verify_deployment(args.verify_only)
        sys.exit(0 if success else 1)

    # Confirmation for mainnet
    if network == NetworkConfig.MAINNET:
        print("\n⚠ WARNING: You are about to deploy to MAINNET!")
        print("This will cost real ETH. Are you sure? (yes/no): ", end="")
        response = input().strip().lower()
        if response != "yes":
            print("Deployment cancelled.")
            sys.exit(0)

    # Deploy contract
    contract_address = deployer.deploy_contract()
    if not contract_address:
        print("Deployment failed!")
        sys.exit(1)

    # Verify deployment
    if not args.skip_test:
        success = deployer.verify_deployment(contract_address)
        if not success:
            print("Warning: Verification failed, but contract is deployed")

    # Save deployment info
    deployer.save_deployment(contract_address)

    # Generate verification script
    deployer.generate_verification_script(contract_address)

    print("\n" + "=" * 60)
    print("✓ Deployment complete!")
    print(f"  Network: {network.value}")
    print(f"  Contract: {contract_address}")
    if deployer.client.network_settings.explorer_url:
        print(f"  Explorer: {deployer.client.network_settings.explorer_url}/address/{contract_address}")
    print("=" * 60)


if __name__ == "__main__":
    main()