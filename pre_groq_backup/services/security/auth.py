"""
DALRN Security Module - Authentication & Authorization
Implements JWT authentication, role-based access control, and security middleware
"""
import os
import secrets
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
API_KEY = os.getenv("API_KEY", secrets.token_hex(32))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

# User roles
class UserRole:
    """User role definitions"""
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"

# Permission definitions
PERMISSIONS = {
    UserRole.ADMIN: ["*"],  # All permissions
    UserRole.USER: ["dispute:create", "dispute:read", "dispute:update", "evidence:create"],
    UserRole.SERVICE: ["dispute:read", "metrics:write", "agent:register"],
    UserRole.READONLY: ["dispute:read", "metrics:read"]
}

@dataclass
class AuthUser:
    """Authenticated user information"""
    user_id: str
    username: str
    email: str
    role: str
    permissions: List[str]
    session_id: Optional[str] = None

class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    username: str
    role: str = UserRole.USER
    exp: Optional[datetime] = None
    session_id: Optional[str] = None

class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: Dict[str, Any]

class AuthService:
    """Authentication and authorization service"""

    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        # Session storage (use Redis in production)
        self.active_sessions = {}

        # API key validation
        self.api_keys = {
            API_KEY: {
                "name": "default",
                "role": UserRole.SERVICE,
                "rate_limit": 1000
            }
        }

    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + self.access_token_expire
        to_encode.update({"exp": expire, "type": "access"})

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + self.refresh_token_expire
        to_encode.update({"exp": expire, "type": "refresh"})

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def decode_token(self, token: str) -> Optional[TokenData]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("type") not in ["access", "refresh"]:
                return None

            return TokenData(
                user_id=payload.get("user_id"),
                username=payload.get("username"),
                role=payload.get("role", UserRole.USER),
                session_id=payload.get("session_id")
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT error: {e}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password"""
        # In production, query from database
        # For demo, use hardcoded users
        demo_users = {
            "admin": {
                "user_id": "usr_admin_001",
                "username": "admin",
                "email": "admin@dalrn.ai",
                "password_hash": self.hash_password("admin123"),
                "role": UserRole.ADMIN
            },
            "alice": {
                "user_id": "usr_alice_002",
                "username": "alice",
                "email": "alice@example.com",
                "password_hash": self.hash_password("alice123"),
                "role": UserRole.USER
            }
        }

        user = demo_users.get(username)
        if not user:
            return None

        if not self.verify_password(password, user["password_hash"]):
            return None

        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }

    def login(self, username: str, password: str) -> Optional[LoginResponse]:
        """Process login and return tokens"""
        user = self.authenticate_user(username, password)
        if not user:
            return None

        # Create session
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            "user_id": user["user_id"],
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc)
        }

        # Create tokens
        token_data = {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "session_id": session_id
        }

        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user={
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"]
            }
        )

    def validate_session(self, session_id: str) -> bool:
        """Validate if session is active"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        # Check session timeout (30 minutes of inactivity)
        last_activity = session.get("last_activity")
        if datetime.now(timezone.utc) - last_activity > timedelta(minutes=30):
            del self.active_sessions[session_id]
            return False

        # Update last activity
        session["last_activity"] = datetime.now(timezone.utc)
        return True

    def get_user_permissions(self, role: str) -> List[str]:
        """Get permissions for a role"""
        return PERMISSIONS.get(role, [])

    def has_permission(self, user: AuthUser, permission: str) -> bool:
        """Check if user has specific permission"""
        if "*" in user.permissions:
            return True
        return permission in user.permissions

# Singleton instance
auth_service = AuthService()

# FastAPI dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> AuthUser:
    """Get current authenticated user"""
    token = credentials.credentials

    token_data = auth_service.decode_token(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Validate session
    if token_data.session_id:
        if not auth_service.validate_session(token_data.session_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired",
                headers={"WWW-Authenticate": "Bearer"}
            )

    # Create user object
    permissions = auth_service.get_user_permissions(token_data.role)

    return AuthUser(
        user_id=token_data.user_id,
        username=token_data.username,
        email=f"{token_data.username}@dalrn.ai",
        role=token_data.role,
        permissions=permissions,
        session_id=token_data.session_id
    )

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if not auth_service.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission}"
            )
        return user
    return permission_checker

async def validate_api_key(request: Request) -> Optional[str]:
    """Validate API key from headers"""
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        return None

    key_info = auth_service.api_keys.get(api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return key_info.get("name")

# Security headers middleware
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses"""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response

if __name__ == "__main__":
    # Test authentication
    auth = AuthService()

    # Test password hashing
    password = "test_password_123"
    hashed = auth.hash_password(password)
    print(f"Password hash: {hashed}")
    print(f"Verification: {auth.verify_password(password, hashed)}")

    # Test login
    login_result = auth.login("alice", "alice123")
    if login_result:
        print(f"\nLogin successful!")
        print(f"Access token: {login_result.access_token[:20]}...")
        print(f"User: {login_result.user}")

        # Decode token
        token_data = auth.decode_token(login_result.access_token)
        print(f"\nDecoded token: {token_data}")
    else:
        print("Login failed")