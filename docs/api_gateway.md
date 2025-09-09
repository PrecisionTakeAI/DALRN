# DALRN Gateway API Documentation

## Overview

The DALRN Gateway API provides a secure, privacy-preserving interface for submitting and tracking disputes in the Decentralized Alternative Legal Resolution Network. It implements Proof of Data Possession (PoDP) middleware to ensure cryptographic verifiability and data integrity throughout the dispute resolution process.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API uses rate limiting (30 requests per minute per IP) but does not require authentication. Production deployments should implement appropriate authentication mechanisms.

## Endpoints

### Health Check

Check the health status of the gateway and its dependencies.

**Endpoint:** `GET /healthz`

**Response:**
```json
{
  "ok": true,
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "services": {
    "ipfs": "healthy",
    "chain": "healthy"
  }
}
```

### Submit Dispute

Submit a new dispute for resolution with encrypted evidence and metadata.

**Endpoint:** `POST /submit-dispute`

**Request Body:**
```json
{
  "parties": ["Alice Ltd", "Bob Pty"],
  "jurisdiction": "NSW-AU",
  "cid": "bafy2bzacecpzldootsmg7p3xszmlnpuiexdj2piwp7e6rbmznqsnu7jyqgjg",
  "enc_meta": {
    "embedding_dim": 768,
    "tenant_id": "t_demo",
    "algorithm": "AES-256-GCM"
  }
}
```

**Request Fields:**
- `parties` (array, required): List of party identifiers (minimum 2)
- `jurisdiction` (string, required): Jurisdiction code (e.g., "NSW-AU", "US-CA", "EU-DE")
- `cid` (string, required): IPFS CID of the encrypted document bundle
- `enc_meta` (object, optional): Metadata about encryption and processing parameters

**Response (201 Created):**
```json
{
  "dispute_id": "disp_9f3ea1c2",
  "receipt_id": "rcpt_e5f6g7h8",
  "anchor_uri": "ipfs://bafy.../receipt_chain.json",
  "anchor_tx": "0xaaaa...aaaa",
  "status": "submitted"
}
```

**Response Fields:**
- `dispute_id`: Unique identifier for the dispute
- `receipt_id`: PoDP receipt ID for this submission
- `anchor_uri`: IPFS URI of the receipt chain
- `anchor_tx`: Blockchain transaction hash for the anchor
- `status`: Current status of the submission

**Error Responses:**
- `400 Bad Request`: Invalid input data
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Get Dispute Status

Retrieve the current status and receipt chain for a dispute.

**Endpoint:** `GET /status/{dispute_id}`

**Path Parameters:**
- `dispute_id`: The unique dispute identifier

**Response (200 OK):**
```json
{
  "dispute_id": "disp_9f3ea1c2",
  "phase": "INTAKE",
  "receipts": [
    {
      "receipt_id": "rcpt_12ab34cd",
      "dispute_id": "disp_9f3ea1c2",
      "step": "INTAKE_V1",
      "inputs": {
        "cid_bundle": "[REDACTED]",
        "party_count": 2,
        "submission_time": "2024-01-01T00:00:00Z"
      },
      "params": {
        "jurisdiction": "NSW-AU",
        "version": "1.0.0"
      },
      "artifacts": {
        "request_id": "req_12345678"
      },
      "hashes": {
        "inputs_hash": "0x1234...5678",
        "outputs_hash": "0xabcd...efgh"
      },
      "signatures": [],
      "ts": "2024-01-01T00:00:00Z"
    }
  ],
  "anchor_tx": "0xaaaa...aaaa",
  "eps_budget": 10.0,
  "last_updated": "2024-01-01T00:00:00Z",
  "receipt_chain_uri": "ipfs://bafy.../receipt_chain.json"
}
```

**Response Fields:**
- `dispute_id`: The dispute identifier
- `phase`: Current phase (INTAKE | SEARCHING | NEGOTIATING | COMPLETE | ERROR)
- `receipts`: Array of PoDP receipts (with PII redacted)
- `anchor_tx`: Blockchain anchor transaction hash
- `eps_budget`: Remaining differential privacy epsilon budget
- `last_updated`: Timestamp of last update
- `receipt_chain_uri`: IPFS URI of the complete receipt chain

**Error Responses:**
- `404 Not Found`: Dispute not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## PoDP Receipt Structure

Each receipt in the system follows this structure:

```json
{
  "receipt_id": "rcpt_unique_id",
  "dispute_id": "disp_unique_id",
  "step": "STEP_NAME",
  "inputs": {},
  "params": {},
  "artifacts": {},
  "hashes": {
    "inputs_hash": "0x...",
    "outputs_hash": "0x..."
  },
  "signatures": [],
  "ts": "2024-01-01T00:00:00Z"
}
```

### Receipt Fields

- `receipt_id`: Unique identifier for this receipt
- `dispute_id`: Associated dispute identifier
- `step`: Processing step name (e.g., "INTAKE_V1", "SEARCH_V1", "NEGOTIATION_V1")
- `inputs`: Input data for this step (PII redacted in responses)
- `params`: Parameters used for processing
- `artifacts`: Additional artifacts or metadata
- `hashes`: Cryptographic hashes for verification
- `signatures`: Digital signatures (if applicable)
- `ts`: ISO 8601 timestamp

## Merkle Tree Structure

The system builds Merkle trees from receipts using the following algorithm:

1. Each receipt is canonicalized to JSON with sorted keys and no whitespace
2. Receipts are hashed using Keccak-256
3. Hashes form the leaves of the Merkle tree
4. Tree is built bottom-up, duplicating the last node if odd number
5. Root hash is anchored to blockchain for immutability

### Canonicalization Rules

- Keys are sorted alphabetically
- No whitespace between elements
- Unicode is preserved (not escaped)
- Format: `{"key1":"value1","key2":"value2"}`

## Privacy and Security

### PII Protection

- **No plaintext PII in logs**: All personally identifiable information is redacted before logging
- **Hashed party identifiers**: Party information is hashed in storage
- **Encrypted evidence**: All evidence is encrypted before submission
- **Redacted responses**: API responses automatically redact sensitive fields

### Rate Limiting

- Default: 30 requests per minute per IP address
- Configurable via environment variables
- Returns 429 status when exceeded

### Request Tracking

Every request receives a unique request ID for tracking:
- Header: `X-Request-ID`
- Format: `req_[8-char-hex]`
- Used for correlation in logs and debugging

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error description"
}
```

### Common Error Codes

- `400`: Bad Request - Invalid input format
- `404`: Not Found - Resource doesn't exist
- `422`: Unprocessable Entity - Validation error
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server-side error

## Development Endpoints

These endpoints are only available in non-production environments:

### List All Disputes

**Endpoint:** `GET /disputes`

**Response:**
```json
{
  "disputes": [
    {
      "dispute_id": "disp_9f3ea1c2",
      "phase": "INTAKE",
      "created_at": "2024-01-01T00:00:00Z",
      "has_anchor": true
    }
  ],
  "count": 1
}
```

## Example Requests

### Submit a Dispute

```bash
curl -X POST http://localhost:8000/submit-dispute \
  -H "Content-Type: application/json" \
  -d '{
    "parties": ["Alice Ltd", "Bob Pty"],
    "jurisdiction": "NSW-AU",
    "cid": "bafy2bzacecpzldootsmg7p3xszmlnpuiexdj2piwp7e6rbmznqsnu7jyqgjg",
    "enc_meta": {
      "embedding_dim": 768,
      "tenant_id": "t_demo"
    }
  }'
```

### Check Dispute Status

```bash
curl http://localhost:8000/status/disp_9f3ea1c2
```

### Health Check

```bash
curl http://localhost:8000/healthz
```

## Environment Variables

The gateway can be configured using the following environment variables:

- `ETH_RPC_URL`: Ethereum RPC endpoint (default: http://localhost:8545)
- `ANCHOR_CONTRACT`: Address of the anchor contract
- `IPFS_API`: IPFS API endpoint (default: http://127.0.0.1:5001)
- `ENVIRONMENT`: Environment name (production/development)
- `LOG_LEVEL`: Logging level (INFO/WARNING/ERROR)

## Running the Gateway

### Development

```bash
python -m services.gateway.app
```

Or with uvicorn directly:

```bash
uvicorn services.gateway.app:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn services.gateway.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Docker

```bash
docker build -t dalrn-gateway .
docker run -p 8000:8000 dalrn-gateway
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_gateway.py -v
```

Run with coverage:

```bash
pytest tests/test_gateway.py --cov=services.gateway --cov-report=html
```

### Test Coverage Areas

- **Happy Path**: End-to-end dispute submission and status retrieval
- **Merkle Correctness**: Deterministic root generation and verification
- **IPFS Handling**: Failure recovery and retry logic
- **Chain Anchoring**: Blockchain interaction and failure handling
- **PII Redaction**: Verification of data privacy measures
- **Rate Limiting**: Request throttling and limits
- **Error Handling**: Graceful degradation and error responses

## API Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Schema: http://localhost:8000/openapi.json

## Architecture Notes

### PoDP Middleware Flow

1. **Request Reception**: Middleware assigns unique request ID
2. **Input Validation**: Pydantic models validate structure
3. **Receipt Generation**: Create cryptographic receipt for operation
4. **Chain Building**: Add receipt to Merkle tree
5. **IPFS Storage**: Upload receipt chain to distributed storage
6. **Blockchain Anchoring**: Anchor Merkle root on-chain
7. **Response Formation**: Return receipt ID and anchor details

### Data Flow

```
Client Request → Gateway → PoDP Middleware → Receipt Generation
                    ↓                              ↓
              Validation                    Merkle Tree Building
                    ↓                              ↓
              Processing                      IPFS Upload
                    ↓                              ↓
              Storage                      Chain Anchoring
                    ↓                              ↓
              Response ← Receipt ID ← Anchor TX
```

### Security Considerations

- All sensitive data must be encrypted before submission
- Party identifiers should use pseudonyms or hashes
- Logs must never contain plaintext PII
- IPFS content should be encrypted
- Merkle roots provide tamper-evidence
- Blockchain anchoring ensures immutability