# AnchorReceipts Contract Deployment Guide

## Overview

The AnchorReceipts contract has been upgraded with a production-ready blockchain client that replaces the previous stub implementation. This guide covers deployment, testing, and usage across multiple networks.

## Key Features of the New Implementation

### 1. Production-Ready Client (`client.py`)

The new implementation provides:

- **Multi-network Support**: Preconfigured for Local, Sepolia, Mainnet, Polygon, and Arbitrum
- **Connection Pooling**: Efficient connection management with configurable pool size
- **Retry Logic**: Exponential backoff with `@backoff` decorators for resilient operations
- **Gas Optimization**: Smart gas estimation with network-specific multipliers
- **Transaction Management**: Proper nonce handling, EIP-1559 support, and confirmation tracking
- **Error Handling**: Comprehensive error catching with detailed logging
- **Merkle Proof Verification**: Local verification of receipt inclusion proofs

### 2. Network Configurations

```python
class NetworkConfig(Enum):
    LOCAL = "local"              # Anvil/Hardhat (Chain ID: 31337)
    SEPOLIA = "sepolia"          # Ethereum Testnet (Chain ID: 11155111)
    MAINNET = "mainnet"          # Ethereum Mainnet (Chain ID: 1)
    POLYGON = "polygon"          # Polygon Mainnet (Chain ID: 137)
    POLYGON_MUMBAI = "polygon_mumbai"  # Polygon Testnet (Chain ID: 80001)
    ARBITRUM = "arbitrum"        # Arbitrum One (Chain ID: 42161)
    ARBITRUM_SEPOLIA = "arbitrum_sepolia"  # Arbitrum Testnet (Chain ID: 421614)
```

## Deployment Instructions

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install web3 eth-account backoff
   ```

2. **Install Foundry** (for contract compilation):
   ```bash
   curl -L https://foundry.paradigm.xyz | bash
   foundryup
   ```

3. **Set Environment Variables**:
   ```bash
   export PRIVATE_KEY="your-private-key-here"
   export SEPOLIA_RPC_URL="your-sepolia-rpc-url"
   export MAINNET_RPC_URL="your-mainnet-rpc-url"
   export ETHERSCAN_API_KEY="your-etherscan-api-key"
   ```

### Local Development Deployment

1. **Start Anvil** (local blockchain):
   ```bash
   anvil
   ```

2. **Compile Contract**:
   ```bash
   cd services/chain
   forge build
   ```

3. **Deploy Locally**:
   ```bash
   python deploy_production.py --network local --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
   ```

### Testnet Deployment (Sepolia)

1. **Ensure you have Sepolia ETH** in your deployment account

2. **Deploy to Sepolia**:
   ```bash
   python deploy_production.py \
     --network sepolia \
     --private-key $PRIVATE_KEY \
     --rpc-url $SEPOLIA_RPC_URL
   ```

3. **Verify on Etherscan**:
   ```bash
   ./verify_sepolia.sh
   ```

### Mainnet Deployment

⚠️ **WARNING**: Mainnet deployment costs real ETH. Double-check everything!

1. **Review deployment parameters** carefully

2. **Deploy to Mainnet**:
   ```bash
   python deploy_production.py \
     --network mainnet \
     --private-key $PRIVATE_KEY \
     --rpc-url $MAINNET_RPC_URL
   ```

3. **Verify on Etherscan**:
   ```bash
   forge verify-contract \
     --chain mainnet \
     --num-of-optimizations 200 \
     --watch \
     $CONTRACT_ADDRESS \
     services/chain/contracts/AnchorReceipts.sol:AnchorReceipts
   ```

## Client Usage Examples

### Basic Initialization

```python
from services.chain.client import AnchorClient, NetworkConfig

# Local development
client = AnchorClient(
    network=NetworkConfig.LOCAL,
    private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)

# Sepolia testnet
client = AnchorClient(
    network=NetworkConfig.SEPOLIA,
    contract_address="0xYourDeployedContractAddress",
    private_key=os.getenv("PRIVATE_KEY")
)
```

### Anchoring a Merkle Root

```python
result = client.anchor_root(
    dispute_id="dispute-123",
    merkle_root="0x" + "a" * 64,
    model_hash="0x" + "b" * 64,
    round=1,
    uri="ipfs://QmExampleHash",
    tags=["PoDP", "v1.0"]
)

print(f"Transaction: {result['transaction_hash']}")
print(f"Block: {result['block_number']}")
print(f"Gas Used: {result['gas_used']}")
print(f"Explorer: {result.get('explorer_url', 'N/A')}")
```

### Retrieving Latest Root

```python
root_info = client.latest_root("dispute-123")

if root_info["exists"]:
    print(f"Merkle Root: {root_info['merkle_root']}")
    print(f"Block Number: {root_info['block_number']}")
else:
    print("No root anchored for this dispute yet")
```

### Anchoring Individual Receipts

```python
receipt_result = client.anchor_receipt(
    dispute_id="dispute-123",
    receipt_hash="0x" + "c" * 64,
    step_index=1,
    uri="ipfs://QmReceiptData"
)

print(f"Receipt anchored in tx: {receipt_result['transaction_hash']}")
```

### Verifying Merkle Proofs

```python
is_valid = client.verify_receipt_proof(
    dispute_id="dispute-123",
    receipt_hash="0xReceiptHashHere",
    merkle_proof=["0xProof1", "0xProof2", "0xProof3"]
)

print(f"Proof valid: {is_valid}")
```

### Monitoring Transaction Status

```python
status = client.get_transaction_status("0xTransactionHashHere")

print(f"Status: {status['status']}")  # pending, success, or failed
if status['status'] == 'success':
    print(f"Confirmations: {status['confirmations']}")
```

## Testing

### Running Unit Tests

```bash
cd tests
pytest test_chain_integration.py::TestAnchorClientUnit -v
```

### Running Integration Tests (Requires Anvil)

1. Start Anvil:
   ```bash
   anvil
   ```

2. Deploy test contract:
   ```bash
   python services/chain/deploy_production.py --network local
   export TEST_CONTRACT_ADDRESS="deployed-address-here"
   ```

3. Run integration tests:
   ```bash
   export RUN_INTEGRATION_TESTS=1
   pytest test_chain_integration.py::TestAnchorClientIntegration -v
   ```

## Configuration Files

After deployment, the following files are created:

### `deployment_{network}.json`
Contains deployment details:
```json
{
  "network": "sepolia",
  "contract_address": "0x...",
  "deployer": "0x...",
  "timestamp": 1234567890,
  "block_number": 12345,
  "chain_id": 11155111,
  "explorer_url": "https://sepolia.etherscan.io/address/0x..."
}
```

### `.env.{network}`
Environment configuration:
```bash
NETWORK=sepolia
ANCHOR_CONTRACT_ADDRESS=0x...
RPC_URL=https://sepolia.infura.io/v3/...
CHAIN_ID=11155111
EXPLORER_URL=https://sepolia.etherscan.io
```

## Network-Specific Considerations

### Polygon Networks
- POA middleware is automatically injected
- Higher confirmation blocks required (30 for mainnet)
- Different gas pricing model

### Arbitrum Networks
- Very fast confirmations (1 block)
- Lower gas costs
- Different RPC endpoints

### Local Development
- No real ETH required
- Instant confirmations
- Perfect for testing

## Troubleshooting

### Connection Issues
```python
# Check connection
if not client.w3.is_connected():
    print("Not connected to network!")
```

### Gas Estimation Failures
The client automatically falls back to 500,000 gas if estimation fails. You can override:
```python
client.network_settings.gas_limit_multiplier = 1.5  # Increase buffer
```

### Transaction Timeouts
Adjust timeout in wait_for_transaction:
```python
receipt = client._wait_for_transaction(tx_hash, timeout=300)  # 5 minutes
```

### Chain ID Mismatches
The client warns but continues if chain ID doesn't match expected. Check your RPC URL.

## Security Best Practices

1. **Never commit private keys** - Use environment variables
2. **Use hardware wallets** for mainnet deployments
3. **Test thoroughly** on testnets before mainnet
4. **Monitor gas prices** before large deployments
5. **Implement access controls** in production environments
6. **Use secure RPC endpoints** (Infura, Alchemy, etc.)
7. **Enable rate limiting** for public-facing services

## API Reference

### Core Methods

- `anchor_root()` - Anchor a Merkle root on-chain
- `anchor_receipt()` - Anchor an individual receipt
- `latest_root()` - Get the latest anchored root for a dispute
- `latest_root_info()` - Get detailed root information including timestamp
- `get_root_by_round()` - Get root for a specific round
- `has_root()` - Check if a dispute has any anchored roots
- `get_contract_info()` - Get contract statistics
- `verify_receipt_proof()` - Verify Merkle proof for a receipt
- `get_transaction_status()` - Check transaction status
- `get_pending_transactions()` - List pending transactions

## Performance Metrics

The production client achieves:
- **Connection pooling**: Up to 5 concurrent connections
- **Retry logic**: 3 attempts with exponential backoff
- **Gas optimization**: 20% buffer on estimates
- **Cache TTL**: 30 seconds for gas prices
- **Transaction tracking**: 5-minute retention

## Next Steps

1. **Deploy to testnet** and verify functionality
2. **Run integration tests** with real network
3. **Set up monitoring** for production deployments
4. **Implement automated testing** in CI/CD pipeline
5. **Create deployment scripts** for your specific use case

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test files for usage examples
3. Check contract events in block explorer
4. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

---

**Note**: This implementation replaces the previous stub with a fully functional blockchain client. All mock transaction hashes have been replaced with real blockchain interactions.