// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title AnchorReceipts
 * @author DALRN
 * @notice Contract for anchoring receipt Merkle roots and individual receipts on-chain
 * @dev Implements gas-efficient storage patterns for receipt anchoring with comprehensive event emission
 */
contract AnchorReceipts {
    
    // Custom errors for gas optimization
    error InvalidDisputeId();
    error InvalidMerkleRoot();
    error InvalidReceiptHash();
    error URITooLong();
    error TooManyTags();
    
    // Constants
    uint256 private constant MAX_URI_LENGTH = 512;
    uint256 private constant MAX_TAGS = 10;
    
    // Events
    /**
     * @notice Emitted when a new Merkle root is anchored
     * @param disputeId Unique identifier for the dispute
     * @param merkleRoot The Merkle root being anchored
     * @param modelHash Hash of the model used
     * @param round The round number
     * @param uri IPFS or other URI for additional data
     * @param tags Array of tags for categorization
     * @param timestamp Block timestamp when anchored
     * @param blockNumber Block number when anchored
     */
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
    
    /**
     * @notice Emitted when an individual receipt is anchored
     * @param disputeId Unique identifier for the dispute
     * @param receiptHash Hash of the receipt
     * @param stepIndex Index of the step in the computation
     * @param uri IPFS or other URI for receipt data
     * @param timestamp Block timestamp when anchored
     * @param blockNumber Block number when anchored
     */
    event ReceiptAnchored(
        bytes32 indexed disputeId, 
        bytes32 indexed receiptHash, 
        uint256 stepIndex, 
        string uri,
        uint256 timestamp,
        uint256 blockNumber
    );
    
    // Storage
    /**
     * @dev Struct to store root information with timestamp
     */
    struct RootInfo {
        bytes32 merkleRoot;
        uint256 timestamp;
        uint256 blockNumber;
    }
    
    /// @notice Mapping from disputeId to the latest root information
    mapping(bytes32 => RootInfo) private _latestRoots;
    
    /// @notice Mapping to track all roots for a dispute (disputeId => round => RootInfo)
    mapping(bytes32 => mapping(uint256 => RootInfo)) private _rootHistory;
    
    /// @notice Mapping to track the latest round for each dispute
    mapping(bytes32 => uint256) private _latestRound;
    
    /// @notice Total number of roots anchored
    uint256 public totalRootsAnchored;
    
    /// @notice Total number of receipts anchored
    uint256 public totalReceiptsAnchored;
    
    /**
     * @notice Anchors a Merkle root for a dispute
     * @dev Stores the root and emits an event with all relevant information
     * @param disputeId Unique identifier for the dispute
     * @param merkleRoot The Merkle root to anchor
     * @param modelHash Hash of the model used in computation
     * @param round The round number for this root
     * @param uri IPFS or other URI for additional data
     * @param tags Array of tags for categorization (max 10)
     * @return blockNumber The block number when the root was anchored
     */
    function anchorRoot(
        bytes32 disputeId,
        bytes32 merkleRoot,
        bytes32 modelHash,
        uint256 round,
        string calldata uri,
        bytes32[] calldata tags
    ) external returns (uint256 blockNumber) {
        // Validations
        if (disputeId == bytes32(0)) revert InvalidDisputeId();
        if (merkleRoot == bytes32(0)) revert InvalidMerkleRoot();
        if (bytes(uri).length > MAX_URI_LENGTH) revert URITooLong();
        if (tags.length > MAX_TAGS) revert TooManyTags();
        
        blockNumber = block.number;
        uint256 timestamp = block.timestamp;
        
        // Create root info
        RootInfo memory rootInfo = RootInfo({
            merkleRoot: merkleRoot,
            timestamp: timestamp,
            blockNumber: blockNumber
        });
        
        // Update storage
        _latestRoots[disputeId] = rootInfo;
        _rootHistory[disputeId][round] = rootInfo;
        
        // Update latest round if this is newer
        if (round > _latestRound[disputeId]) {
            _latestRound[disputeId] = round;
        }
        
        // Increment counter
        unchecked {
            totalRootsAnchored++;
        }
        
        // Emit event
        emit RootAnchored(
            disputeId,
            merkleRoot,
            modelHash,
            round,
            uri,
            tags,
            timestamp,
            blockNumber
        );
        
        return blockNumber;
    }
    
    /**
     * @notice Anchors an individual receipt
     * @dev Emits an event with receipt information
     * @param disputeId Unique identifier for the dispute
     * @param receiptHash Hash of the receipt
     * @param stepIndex Index of the step in the computation
     * @param uri IPFS or other URI for receipt data
     * @return blockNumber The block number when the receipt was anchored
     */
    function anchorReceipt(
        bytes32 disputeId,
        bytes32 receiptHash,
        uint256 stepIndex,
        string calldata uri
    ) external returns (uint256 blockNumber) {
        // Validations
        if (disputeId == bytes32(0)) revert InvalidDisputeId();
        if (receiptHash == bytes32(0)) revert InvalidReceiptHash();
        if (bytes(uri).length > MAX_URI_LENGTH) revert URITooLong();
        
        blockNumber = block.number;
        uint256 timestamp = block.timestamp;
        
        // Increment counter
        unchecked {
            totalReceiptsAnchored++;
        }
        
        // Emit event
        emit ReceiptAnchored(
            disputeId,
            receiptHash,
            stepIndex,
            uri,
            timestamp,
            blockNumber
        );
        
        return blockNumber;
    }
    
    /**
     * @notice Gets the latest root for a dispute
     * @dev Returns zero values if no root has been anchored
     * @param disputeId The dispute identifier
     * @return merkleRoot The latest Merkle root
     * @return blockNumber The block number when it was anchored
     */
    function latestRoot(bytes32 disputeId) 
        external 
        view 
        returns (bytes32 merkleRoot, uint256 blockNumber) 
    {
        RootInfo memory info = _latestRoots[disputeId];
        return (info.merkleRoot, info.blockNumber);
    }
    
    /**
     * @notice Gets detailed information about the latest root
     * @param disputeId The dispute identifier
     * @return merkleRoot The latest Merkle root
     * @return timestamp The timestamp when anchored
     * @return blockNumber The block number when anchored
     */
    function latestRootInfo(bytes32 disputeId)
        external
        view
        returns (
            bytes32 merkleRoot,
            uint256 timestamp,
            uint256 blockNumber
        )
    {
        RootInfo memory info = _latestRoots[disputeId];
        return (info.merkleRoot, info.timestamp, info.blockNumber);
    }
    
    /**
     * @notice Gets a root for a specific round
     * @param disputeId The dispute identifier
     * @param round The round number
     * @return merkleRoot The Merkle root for that round
     * @return timestamp The timestamp when anchored
     * @return blockNumber The block number when anchored
     */
    function getRootByRound(bytes32 disputeId, uint256 round)
        external
        view
        returns (
            bytes32 merkleRoot,
            uint256 timestamp,
            uint256 blockNumber
        )
    {
        RootInfo memory info = _rootHistory[disputeId][round];
        return (info.merkleRoot, info.timestamp, info.blockNumber);
    }
    
    /**
     * @notice Gets the latest round number for a dispute
     * @param disputeId The dispute identifier
     * @return The latest round number
     */
    function getLatestRound(bytes32 disputeId) external view returns (uint256) {
        return _latestRound[disputeId];
    }
    
    /**
     * @notice Checks if a root has been anchored for a dispute
     * @param disputeId The dispute identifier
     * @return True if a root exists for this dispute
     */
    function hasRoot(bytes32 disputeId) external view returns (bool) {
        return _latestRoots[disputeId].merkleRoot != bytes32(0);
    }
}
