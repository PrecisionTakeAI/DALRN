# DALRN API Documentation

## Base URLs
- **Production**: `https://api.dalrn.ai`
- **Development**: `http://localhost:8000`

## Authentication
JWT Bearer token required for most endpoints.

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "alice",
  "password": "alice123"
}

Response: {
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "expires_in": 1800
}
```

## Core Endpoints

### Submit Dispute
```http
POST /submit-dispute
Authorization: Bearer {token}

{
  "parties": ["alice@example.com", "bob@example.com"],
  "jurisdiction": "US",
  "cid": "QmT78zSu...",
  "enc_meta": {...}
}
```

### Get Status
```http
GET /status/{dispute_id}
Authorization: Bearer {token}
```

### Search
```http
POST /search
Authorization: Bearer {token}

{
  "query": "contract breach",
  "limit": 10,
  "offset": 0
}
```

## Rate Limits
- **Free**: 30 req/min, 500/hr, 5000/day
- **Basic**: 60 req/min, 1000/hr, 10000/day
- **Premium**: 120 req/min, 5000/hr, 50000/day

## Response Codes
- `200`: Success
- `401`: Unauthorized
- `403`: Forbidden
- `429`: Rate limit exceeded
- `500`: Internal error