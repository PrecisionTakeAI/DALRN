// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Script.sol";
import "../contracts/AnchorReceipts.sol";

/**
 * @title DeployAnchorReceipts
 * @notice Deployment script for the AnchorReceipts contract
 * @dev Run with: forge script scripts/Deploy.s.sol:DeployAnchorReceipts --rpc-url $RPC_URL --broadcast
 */
contract DeployAnchorReceipts is Script {
    // Contract instances
    AnchorReceipts public anchorReceipts;
    
    // Deployment configuration
    struct DeployConfig {
        uint256 deployerPrivateKey;
        string networkName;
        bool verify;
    }
    
    function run() external returns (AnchorReceipts) {
        // Get configuration
        DeployConfig memory config = getConfig();
        
        // Start broadcast
        vm.startBroadcast(config.deployerPrivateKey);
        
        // Deploy the contract
        console.log("Deploying AnchorReceipts contract...");
        anchorReceipts = new AnchorReceipts();
        console.log("AnchorReceipts deployed at:", address(anchorReceipts));
        
        // Stop broadcast
        vm.stopBroadcast();
        
        // Post-deployment actions
        postDeploy(config);
        
        return anchorReceipts;
    }
    
    function getConfig() internal view returns (DeployConfig memory) {
        DeployConfig memory config;
        
        // Get deployer private key from environment
        config.deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)); // Default anvil key
        
        // Determine network
        uint256 chainId = block.chainid;
        if (chainId == 31337) {
            config.networkName = "localhost";
            config.verify = false;
        } else if (chainId == 1) {
            config.networkName = "mainnet";
            config.verify = true;
        } else if (chainId == 11155111) {
            config.networkName = "sepolia";
            config.verify = true;
        } else if (chainId == 5) {
            config.networkName = "goerli";
            config.verify = true;
        } else {
            config.networkName = "unknown";
            config.verify = false;
        }
        
        console.log("Network:", config.networkName);
        console.log("Chain ID:", chainId);
        
        return config;
    }
    
    function postDeploy(DeployConfig memory config) internal {
        // Export deployment info
        string memory deploymentInfo = string(abi.encodePacked(
            '{"network":"', config.networkName,
            '","address":"', vm.toString(address(anchorReceipts)),
            '","blockNumber":', vm.toString(block.number),
            '","timestamp":', vm.toString(block.timestamp),
            '","deployer":"', vm.toString(msg.sender),
            '"}'
        ));
        
        console.log("Deployment Info:");
        console.log(deploymentInfo);
        
        // Write deployment info to file
        string memory path = string(abi.encodePacked("deployments/", config.networkName, "-latest.json"));
        vm.writeFile(path, deploymentInfo);
        
        // Export ABI
        exportABI();
        
        // Verify contract if needed
        if (config.verify) {
            console.log("Contract verification enabled. Run verification command manually.");
        }
    }
    
    function exportABI() internal {
        console.log("Exporting ABI...");
        // Note: In a real deployment, you would extract the ABI from the compiled artifacts
        // For now, we'll create a marker file to indicate ABI export is needed
        vm.writeFile("services/chain/abi/.export-needed", "true");
    }
}

/**
 * @title DeployAnchorReceiptsWithInit
 * @notice Extended deployment script with initialization
 */
contract DeployAnchorReceiptsWithInit is DeployAnchorReceipts {
    function run() external override returns (AnchorReceipts) {
        // Deploy contract
        AnchorReceipts deployed = super.run();
        
        // Get configuration
        DeployConfig memory config = getConfig();
        
        // Initialize with test data if on localhost
        if (keccak256(bytes(config.networkName)) == keccak256(bytes("localhost"))) {
            vm.startBroadcast(config.deployerPrivateKey);
            initializeTestData(deployed);
            vm.stopBroadcast();
        }
        
        return deployed;
    }
    
    function initializeTestData(AnchorReceipts anchor) internal {
        console.log("Initializing test data...");
        
        // Create test dispute
        bytes32 testDisputeId = keccak256("test-dispute-1");
        bytes32 testMerkleRoot = keccak256("test-root-1");
        bytes32 testModelHash = keccak256("test-model-1");
        
        // Create test tags
        bytes32[] memory testTags = new bytes32[](2);
        testTags[0] = keccak256("PoDP");
        testTags[1] = keccak256("Test");
        
        // Anchor test root
        uint256 blockNumber = anchor.anchorRoot(
            testDisputeId,
            testMerkleRoot,
            testModelHash,
            1,
            "ipfs://QmTestData123",
            testTags
        );
        
        console.log("Test root anchored at block:", blockNumber);
        
        // Anchor test receipt
        bytes32 testReceiptHash = keccak256("test-receipt-1");
        blockNumber = anchor.anchorReceipt(
            testDisputeId,
            testReceiptHash,
            1,
            "ipfs://QmTestReceipt123"
        );
        
        console.log("Test receipt anchored at block:", blockNumber);
        console.log("Test data initialization complete");
    }
}