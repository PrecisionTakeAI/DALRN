// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../contracts/AnchorReceipts.sol";

contract AnchorReceiptsTest is Test {
    AnchorReceipts public anchor;
    
    // Test data
    bytes32 constant DISPUTE_ID = keccak256("dispute1");
    bytes32 constant MERKLE_ROOT = keccak256("root1");
    bytes32 constant MODEL_HASH = keccak256("model1");
    bytes32 constant RECEIPT_HASH = keccak256("receipt1");
    string constant URI = "ipfs://QmTest123";
    string constant LONG_URI = "ipfs://QmVeryLongHashThatExceedsTheMaximumAllowedLengthForURIsInThisContractImplementationWhichIsSetTo512CharactersToPreventStorageAbuseAndEnsureReasonableGasConsumptionDuringTransactionExecutionThisStringNeedsToBeReallyLongToTestTheValidationLogicProperlyAndMakeSureItFailsWhenTheURIIsTooLongForTheContractToHandleEfficientlyWithoutCausingExcessiveGasConsumptionOrStorageIssuesThatCouldLeadToDoSAttacksOrOtherVulnerabilitiesInTheSystemSoWeMakeItReallyReallyLongToExceed512Characters";
    
    event RootAnchored(
        bytes32 indexed disputeId, 
        bytes32 indexed merkleRoot, 
        bytes32 modelHash, 
        uint256 round, 
        string uri, 
        bytes32[] tags,
        uint256 timestamp,
        uint256 blockNumber
    );
    
    event ReceiptAnchored(
        bytes32 indexed disputeId, 
        bytes32 indexed receiptHash, 
        uint256 stepIndex, 
        string uri,
        uint256 timestamp,
        uint256 blockNumber
    );

    function setUp() public {
        anchor = new AnchorReceipts();
    }

    // ============ AnchorRoot Tests ============

    function testAnchorRoot() public {
        bytes32[] memory tags = new bytes32[](2);
        tags[0] = keccak256("PoDP");
        tags[1] = keccak256("FL");
        
        // Test event emission
        vm.expectEmit(true, true, true, true);
        emit RootAnchored(
            DISPUTE_ID,
            MERKLE_ROOT,
            MODEL_HASH,
            1,
            URI,
            tags,
            block.timestamp,
            block.number
        );
        
        uint256 blockNo = anchor.anchorRoot(
            DISPUTE_ID,
            MERKLE_ROOT,
            MODEL_HASH,
            1,
            URI,
            tags
        );
        
        // Verify return value
        assertEq(blockNo, block.number);
        
        // Verify storage
        (bytes32 storedRoot, uint256 storedBlock) = anchor.latestRoot(DISPUTE_ID);
        assertEq(storedRoot, MERKLE_ROOT);
        assertEq(storedBlock, block.number);
        
        // Verify counters
        assertEq(anchor.totalRootsAnchored(), 1);
    }

    function testAnchorRootMultipleRounds() public {
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        
        // Anchor root for round 1
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, URI, tags);
        
        // Anchor root for round 2
        bytes32 root2 = keccak256("root2");
        anchor.anchorRoot(DISPUTE_ID, root2, MODEL_HASH, 2, URI, tags);
        
        // Verify latest root is from round 2
        (bytes32 latestRoot,) = anchor.latestRoot(DISPUTE_ID);
        assertEq(latestRoot, root2);
        
        // Verify we can still access round 1
        (bytes32 round1Root,,) = anchor.getRootByRound(DISPUTE_ID, 1);
        assertEq(round1Root, MERKLE_ROOT);
        
        // Verify latest round tracking
        assertEq(anchor.getLatestRound(DISPUTE_ID), 2);
    }

    function testAnchorRootEmptyTags() public {
        bytes32[] memory emptyTags = new bytes32[](0);
        
        uint256 blockNo = anchor.anchorRoot(
            DISPUTE_ID,
            MERKLE_ROOT,
            MODEL_HASH,
            1,
            URI,
            emptyTags
        );
        
        assertEq(blockNo, block.number);
        (bytes32 storedRoot,) = anchor.latestRoot(DISPUTE_ID);
        assertEq(storedRoot, MERKLE_ROOT);
    }

    function testAnchorRootMaxTags() public {
        bytes32[] memory maxTags = new bytes32[](10);
        for (uint i = 0; i < 10; i++) {
            maxTags[i] = keccak256(abi.encodePacked("tag", i));
        }
        
        uint256 blockNo = anchor.anchorRoot(
            DISPUTE_ID,
            MERKLE_ROOT,
            MODEL_HASH,
            1,
            URI,
            maxTags
        );
        
        assertEq(blockNo, block.number);
    }

    function testAnchorRootInvalidDisputeId() public {
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        
        vm.expectRevert(AnchorReceipts.InvalidDisputeId.selector);
        anchor.anchorRoot(bytes32(0), MERKLE_ROOT, MODEL_HASH, 1, URI, tags);
    }

    function testAnchorRootInvalidMerkleRoot() public {
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        
        vm.expectRevert(AnchorReceipts.InvalidMerkleRoot.selector);
        anchor.anchorRoot(DISPUTE_ID, bytes32(0), MODEL_HASH, 1, URI, tags);
    }

    function testAnchorRootURITooLong() public {
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        
        vm.expectRevert(AnchorReceipts.URITooLong.selector);
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, LONG_URI, tags);
    }

    function testAnchorRootTooManyTags() public {
        bytes32[] memory tooManyTags = new bytes32[](11);
        for (uint i = 0; i < 11; i++) {
            tooManyTags[i] = keccak256(abi.encodePacked("tag", i));
        }
        
        vm.expectRevert(AnchorReceipts.TooManyTags.selector);
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, URI, tooManyTags);
    }

    // ============ AnchorReceipt Tests ============

    function testAnchorReceipt() public {
        // Test event emission
        vm.expectEmit(true, true, true, true);
        emit ReceiptAnchored(
            DISPUTE_ID,
            RECEIPT_HASH,
            5,
            URI,
            block.timestamp,
            block.number
        );
        
        uint256 blockNo = anchor.anchorReceipt(
            DISPUTE_ID,
            RECEIPT_HASH,
            5,
            URI
        );
        
        assertEq(blockNo, block.number);
        assertEq(anchor.totalReceiptsAnchored(), 1);
    }

    function testAnchorReceiptMultiple() public {
        anchor.anchorReceipt(DISPUTE_ID, RECEIPT_HASH, 1, URI);
        anchor.anchorReceipt(DISPUTE_ID, keccak256("receipt2"), 2, URI);
        anchor.anchorReceipt(DISPUTE_ID, keccak256("receipt3"), 3, URI);
        
        assertEq(anchor.totalReceiptsAnchored(), 3);
    }

    function testAnchorReceiptInvalidDisputeId() public {
        vm.expectRevert(AnchorReceipts.InvalidDisputeId.selector);
        anchor.anchorReceipt(bytes32(0), RECEIPT_HASH, 1, URI);
    }

    function testAnchorReceiptInvalidReceiptHash() public {
        vm.expectRevert(AnchorReceipts.InvalidReceiptHash.selector);
        anchor.anchorReceipt(DISPUTE_ID, bytes32(0), 1, URI);
    }

    function testAnchorReceiptURITooLong() public {
        vm.expectRevert(AnchorReceipts.URITooLong.selector);
        anchor.anchorReceipt(DISPUTE_ID, RECEIPT_HASH, 1, LONG_URI);
    }

    // ============ View Function Tests ============

    function testLatestRoot() public {
        // Initially no root
        (bytes32 root, uint256 blockNum) = anchor.latestRoot(DISPUTE_ID);
        assertEq(root, bytes32(0));
        assertEq(blockNum, 0);
        
        // Anchor a root
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, URI, tags);
        
        // Verify latest root
        (root, blockNum) = anchor.latestRoot(DISPUTE_ID);
        assertEq(root, MERKLE_ROOT);
        assertEq(blockNum, block.number);
    }

    function testLatestRootInfo() public {
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        
        uint256 anchorBlock = block.number;
        uint256 anchorTime = block.timestamp;
        
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, URI, tags);
        
        (bytes32 root, uint256 timestamp, uint256 blockNum) = anchor.latestRootInfo(DISPUTE_ID);
        assertEq(root, MERKLE_ROOT);
        assertEq(timestamp, anchorTime);
        assertEq(blockNum, anchorBlock);
    }

    function testGetRootByRound() public {
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        
        // Anchor multiple rounds
        bytes32 root1 = keccak256("root1");
        bytes32 root2 = keccak256("root2");
        bytes32 root3 = keccak256("root3");
        
        anchor.anchorRoot(DISPUTE_ID, root1, MODEL_HASH, 1, URI, tags);
        
        vm.roll(block.number + 10);
        anchor.anchorRoot(DISPUTE_ID, root2, MODEL_HASH, 2, URI, tags);
        
        vm.roll(block.number + 10);
        anchor.anchorRoot(DISPUTE_ID, root3, MODEL_HASH, 3, URI, tags);
        
        // Verify each round
        (bytes32 retrievedRoot1,,) = anchor.getRootByRound(DISPUTE_ID, 1);
        assertEq(retrievedRoot1, root1);
        
        (bytes32 retrievedRoot2,,) = anchor.getRootByRound(DISPUTE_ID, 2);
        assertEq(retrievedRoot2, root2);
        
        (bytes32 retrievedRoot3,,) = anchor.getRootByRound(DISPUTE_ID, 3);
        assertEq(retrievedRoot3, root3);
    }

    function testHasRoot() public {
        // Initially no root
        assertFalse(anchor.hasRoot(DISPUTE_ID));
        
        // Anchor a root
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, URI, tags);
        
        // Now has root
        assertTrue(anchor.hasRoot(DISPUTE_ID));
    }

    // ============ Fuzz Tests ============

    function testFuzzAnchorRoot(
        bytes32 disputeId,
        bytes32 merkleRoot,
        bytes32 modelHash,
        uint256 round,
        string memory uri,
        uint8 tagCount
    ) public {
        // Bound inputs
        vm.assume(disputeId != bytes32(0));
        vm.assume(merkleRoot != bytes32(0));
        vm.assume(bytes(uri).length <= 512);
        tagCount = uint8(bound(tagCount, 0, 10));
        
        // Create tags
        bytes32[] memory tags = new bytes32[](tagCount);
        for (uint i = 0; i < tagCount; i++) {
            tags[i] = keccak256(abi.encodePacked("tag", i));
        }
        
        // Anchor root
        uint256 blockNo = anchor.anchorRoot(
            disputeId,
            merkleRoot,
            modelHash,
            round,
            uri,
            tags
        );
        
        // Verify
        assertEq(blockNo, block.number);
        (bytes32 storedRoot,) = anchor.latestRoot(disputeId);
        assertEq(storedRoot, merkleRoot);
    }

    function testFuzzAnchorReceipt(
        bytes32 disputeId,
        bytes32 receiptHash,
        uint256 stepIndex,
        string memory uri
    ) public {
        // Bound inputs
        vm.assume(disputeId != bytes32(0));
        vm.assume(receiptHash != bytes32(0));
        vm.assume(bytes(uri).length <= 512);
        
        // Anchor receipt
        uint256 blockNo = anchor.anchorReceipt(
            disputeId,
            receiptHash,
            stepIndex,
            uri
        );
        
        assertEq(blockNo, block.number);
    }

    // ============ Gas Tests ============

    function testGasAnchorRoot() public {
        bytes32[] memory tags = new bytes32[](3);
        tags[0] = keccak256("PoDP");
        tags[1] = keccak256("FL");
        tags[2] = keccak256("Test");
        
        uint256 gasBefore = gasleft();
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, URI, tags);
        uint256 gasUsed = gasBefore - gasleft();
        
        // Log gas usage for optimization tracking
        emit log_named_uint("Gas used for anchorRoot", gasUsed);
        
        // Ensure reasonable gas usage (adjust threshold as needed)
        assertLt(gasUsed, 150000);
    }

    function testGasAnchorReceipt() public {
        uint256 gasBefore = gasleft();
        anchor.anchorReceipt(DISPUTE_ID, RECEIPT_HASH, 1, URI);
        uint256 gasUsed = gasBefore - gasleft();
        
        // Log gas usage
        emit log_named_uint("Gas used for anchorReceipt", gasUsed);
        
        // Ensure reasonable gas usage
        assertLt(gasUsed, 100000);
    }

    function testGasLatestRoot() public {
        // First anchor a root
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("PoDP");
        anchor.anchorRoot(DISPUTE_ID, MERKLE_ROOT, MODEL_HASH, 1, URI, tags);
        
        // Measure gas for view function
        uint256 gasBefore = gasleft();
        anchor.latestRoot(DISPUTE_ID);
        uint256 gasUsed = gasBefore - gasleft();
        
        // Log gas usage
        emit log_named_uint("Gas used for latestRoot", gasUsed);
        
        // View function should be very cheap
        assertLt(gasUsed, 10000);
    }

    // ============ Integration Tests ============

    function testIntegrationCompleteFlow() public {
        // Simulate a complete dispute resolution flow
        bytes32 disputeId = keccak256("integration-dispute");
        bytes32[] memory tags = new bytes32[](2);
        tags[0] = keccak256("PoDP");
        tags[1] = keccak256("Integration");
        
        // Round 1: Initial anchoring
        bytes32 root1 = keccak256("root-round-1");
        anchor.anchorRoot(disputeId, root1, MODEL_HASH, 1, "ipfs://round1", tags);
        
        // Anchor some receipts for round 1
        anchor.anchorReceipt(disputeId, keccak256("receipt1"), 1, "ipfs://receipt1");
        anchor.anchorReceipt(disputeId, keccak256("receipt2"), 2, "ipfs://receipt2");
        anchor.anchorReceipt(disputeId, keccak256("receipt3"), 3, "ipfs://receipt3");
        
        // Round 2: Update after some computation
        vm.roll(block.number + 100);
        bytes32 root2 = keccak256("root-round-2");
        anchor.anchorRoot(disputeId, root2, MODEL_HASH, 2, "ipfs://round2", tags);
        
        // More receipts
        anchor.anchorReceipt(disputeId, keccak256("receipt4"), 4, "ipfs://receipt4");
        anchor.anchorReceipt(disputeId, keccak256("receipt5"), 5, "ipfs://receipt5");
        
        // Verify final state
        (bytes32 latestRoot,) = anchor.latestRoot(disputeId);
        assertEq(latestRoot, root2);
        assertEq(anchor.getLatestRound(disputeId), 2);
        assertEq(anchor.totalRootsAnchored(), 2);
        assertEq(anchor.totalReceiptsAnchored(), 5);
        
        // Verify historical data is preserved
        (bytes32 historicalRoot,,) = anchor.getRootByRound(disputeId, 1);
        assertEq(historicalRoot, root1);
    }

    function testIntegrationMultipleDisputes() public {
        bytes32[] memory tags = new bytes32[](1);
        tags[0] = keccak256("Multi");
        
        // Create multiple disputes
        bytes32 dispute1 = keccak256("dispute1");
        bytes32 dispute2 = keccak256("dispute2");
        bytes32 dispute3 = keccak256("dispute3");
        
        bytes32 root1 = keccak256("root1");
        bytes32 root2 = keccak256("root2");
        bytes32 root3 = keccak256("root3");
        
        // Anchor roots for different disputes
        anchor.anchorRoot(dispute1, root1, MODEL_HASH, 1, URI, tags);
        anchor.anchorRoot(dispute2, root2, MODEL_HASH, 1, URI, tags);
        anchor.anchorRoot(dispute3, root3, MODEL_HASH, 1, URI, tags);
        
        // Verify each dispute has its own root
        (bytes32 stored1,) = anchor.latestRoot(dispute1);
        (bytes32 stored2,) = anchor.latestRoot(dispute2);
        (bytes32 stored3,) = anchor.latestRoot(dispute3);
        
        assertEq(stored1, root1);
        assertEq(stored2, root2);
        assertEq(stored3, root3);
        
        // Update one dispute
        bytes32 root1Updated = keccak256("root1-updated");
        anchor.anchorRoot(dispute1, root1Updated, MODEL_HASH, 2, URI, tags);
        
        // Verify only dispute1 was updated
        (stored1,) = anchor.latestRoot(dispute1);
        (stored2,) = anchor.latestRoot(dispute2);
        
        assertEq(stored1, root1Updated);
        assertEq(stored2, root2);
    }
}