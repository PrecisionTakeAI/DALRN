"""
Input Validation and Sanitization Module
Prevents injection attacks and validates all inputs
"""
import re
import html
import bleach
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Comprehensive input validation and sanitization"""

    # Regex patterns for validation
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        'ipfs_cid': r'^Qm[a-zA-Z0-9]{44}$',
        'eth_address': r'^0x[a-fA-F0-9]{40}$',
        'safe_string': r'^[a-zA-Z0-9\s\-_\.@]+$',
        'jurisdiction': r'^[A-Z]{2,3}$'
    }

    # SQL injection patterns
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
        r"(-{2}|\/\*|\*\/)",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)"
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<embed",
        r"<object"
    ]

    @staticmethod
    def validate_pattern(value: str, pattern_name: str) -> bool:
        """Validate string against pattern"""
        pattern = InputValidator.PATTERNS.get(pattern_name)
        if not pattern:
            return False
        return bool(re.match(pattern, value))

    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML content"""
        allowed_tags = ['p', 'br', 'span', 'strong', 'em', 'ul', 'ol', 'li']
        allowed_attrs = {'*': ['class']}
        return bleach.clean(text, tags=allowed_tags, attributes=allowed_attrs)

    @staticmethod
    def escape_special_chars(text: str) -> str:
        """Escape special characters"""
        return html.escape(text)

    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check for SQL injection patterns"""
        for pattern in InputValidator.SQL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"SQL injection attempt detected: {value[:50]}...")
                return False
        return True

    @staticmethod
    def check_xss(value: str) -> bool:
        """Check for XSS patterns"""
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"XSS attempt detected: {value[:50]}...")
                return False
        return True

    @staticmethod
    def validate_dispute_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dispute submission data"""
        validated = {}

        # Validate parties
        if 'parties' not in data or len(data['parties']) < 2:
            raise ValueError("At least 2 parties required")

        validated['parties'] = []
        for party in data['parties'][:10]:  # Max 10 parties
            if not InputValidator.validate_pattern(party, 'email'):
                raise ValueError(f"Invalid party email: {party}")
            validated['parties'].append(party)

        # Validate jurisdiction
        if 'jurisdiction' in data:
            if not InputValidator.validate_pattern(data['jurisdiction'], 'jurisdiction'):
                raise ValueError("Invalid jurisdiction code")
            validated['jurisdiction'] = data['jurisdiction']

        # Validate CID
        if 'cid' in data:
            if not InputValidator.validate_pattern(data['cid'], 'ipfs_cid'):
                raise ValueError("Invalid IPFS CID")
            validated['cid'] = data['cid']

        # Sanitize description
        if 'description' in data:
            desc = str(data['description'])[:5000]  # Max length
            if not InputValidator.check_sql_injection(desc):
                raise ValueError("Invalid description content")
            if not InputValidator.check_xss(desc):
                raise ValueError("Invalid description content")
            validated['description'] = InputValidator.sanitize_html(desc)

        return validated

    @staticmethod
    def validate_file_upload(filename: str, content_type: str, size: int) -> bool:
        """Validate file upload"""
        # Allowed extensions
        ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt', '.json', '.png', '.jpg', '.jpeg'}
        MAX_SIZE = 10 * 1024 * 1024  # 10MB

        # Check size
        if size > MAX_SIZE:
            raise ValueError(f"File too large: {size} bytes (max {MAX_SIZE})")

        # Check extension
        import os
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type not allowed: {ext}")

        # Check content type
        ALLOWED_CONTENT_TYPES = {
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'application/json',
            'image/png',
            'image/jpeg'
        }

        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError(f"Content type not allowed: {content_type}")

        return True

# Pydantic models with validation
class SecureDisputeRequest(BaseModel):
    """Secure dispute request with validation"""
    parties: List[str] = Field(..., min_items=2, max_items=10)
    jurisdiction: str = Field(..., min_length=2, max_length=3)
    cid: Optional[str] = Field(None, regex=r'^Qm[a-zA-Z0-9]{44}$')
    description: Optional[str] = Field(None, max_length=5000)

    @validator('parties')
    def validate_parties(cls, v):
        for party in v:
            if not InputValidator.validate_pattern(party, 'email'):
                raise ValueError(f"Invalid email: {party}")
        return v

    @validator('jurisdiction')
    def validate_jurisdiction(cls, v):
        if not v.isupper() or not v.isalpha():
            raise ValueError("Jurisdiction must be uppercase letters")
        return v

    @validator('description')
    def validate_description(cls, v):
        if v:
            if not InputValidator.check_sql_injection(v):
                raise ValueError("Invalid content detected")
            if not InputValidator.check_xss(v):
                raise ValueError("Invalid content detected")
            return InputValidator.sanitize_html(v)
        return v

class SecureSearchRequest(BaseModel):
    """Secure search request"""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0, le=10000)

    @validator('query')
    def validate_query(cls, v):
        if not InputValidator.check_sql_injection(v):
            raise ValueError("Invalid query")
        if not InputValidator.check_xss(v):
            raise ValueError("Invalid query")
        return InputValidator.escape_special_chars(v)

# Middleware for automatic validation
async def input_validation_middleware(request, call_next):
    """Middleware to validate all inputs"""

    # Check Content-Type for POST/PUT
    if request.method in ["POST", "PUT"]:
        content_type = request.headers.get("Content-Type", "")

        # Validate content type
        allowed_types = ["application/json", "multipart/form-data"]
        if not any(t in content_type for t in allowed_types):
            return HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported content type"
            )

        # Check Content-Length
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            return HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request body too large"
            )

    # Validate headers
    for header, value in request.headers.items():
        if len(value) > 8192:  # Max header size
            return HTTPException(
                status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE,
                detail=f"Header too large: {header}"
            )

    response = await call_next(request)
    return response

if __name__ == "__main__":
    # Test validation
    validator = InputValidator()

    # Test patterns
    print("Testing email:", validator.validate_pattern("user@example.com", "email"))
    print("Testing UUID:", validator.validate_pattern("123e4567-e89b-12d3-a456-426614174000", "uuid"))

    # Test SQL injection
    print("\nTesting SQL injection detection:")
    print("Clean:", validator.check_sql_injection("normal text"))
    print("Injection:", validator.check_sql_injection("'; DROP TABLE users; --"))

    # Test XSS
    print("\nTesting XSS detection:")
    print("Clean:", validator.check_xss("normal text"))
    print("XSS:", validator.check_xss("<script>alert('xss')</script>"))

    # Test dispute validation
    print("\nTesting dispute validation:")
    dispute_data = {
        "parties": ["alice@example.com", "bob@example.com"],
        "jurisdiction": "US",
        "cid": "QmT78zSuBmuS4z925WZfrqQ1qHaJ56DQaTfyMUF7F8ff5o",
        "description": "Contract dispute <b>details</b>"
    }

    try:
        validated = validator.validate_dispute_data(dispute_data)
        print("Valid dispute:", validated)
    except ValueError as e:
        print("Invalid dispute:", e)