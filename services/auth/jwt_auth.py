"""
JWT Authentication module for DALRN.
Provides secure authentication with JWT tokens and role-based access control.
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

from fastapi import HTTPException, Security, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Import our database connection
from services.database.connection import db, get_user_by_username, create_user

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dalrn-secret-key-change-in-production-" + os.urandom(16).hex())
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security scheme
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Create auth router for endpoints
auth_router = APIRouter(prefix="/auth", tags=["authentication"])


# Pydantic models
class UserCreate(BaseModel):
    """User registration model."""
    username: str
    email: str  # Changed from EmailStr to str
    password: str
    role: str = "user"


class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AuthService:
    """Authentication service handling JWT operations."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Check token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}"
                )

            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )


# Dependency for getting current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    payload = AuthService.verify_token(token, "access")

    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )

    # Get user from database
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


# Dependency for role-based access control
def require_role(required_role: str):
    """Create a dependency that requires a specific user role."""
    async def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        user_role = current_user.get("role", "user")

        # Admin can access everything
        if user_role == "admin":
            return current_user

        # Check if user has required role
        if user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )

        return current_user

    return role_checker


# Auth endpoints
@auth_router.post("/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    """Register a new user."""
    # Check if user already exists
    existing_user = get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    # Hash password and create user
    password_hash = AuthService.hash_password(user_data.password)

    try:
        user_id = create_user(
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            role=user_data.role
        )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not create user. Email might already exist."
        )

    # Create tokens
    token_data = {"sub": user_data.username, "user_id": user_id}
    access_token = AuthService.create_access_token(token_data)
    refresh_token = AuthService.create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@auth_router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """Login user and return tokens."""
    # Get user from database
    user = get_user_by_username(user_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Verify password
    if not AuthService.verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Create tokens
    token_data = {"sub": user["username"], "user_id": user["id"]}
    access_token = AuthService.create_access_token(token_data)
    refresh_token = AuthService.create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Refresh access token using refresh token."""
    token = credentials.credentials
    payload = AuthService.verify_token(token, "refresh")

    username = payload.get("sub")
    user_id = payload.get("user_id")

    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Create new tokens
    token_data = {"sub": username, "user_id": user_id}
    access_token = AuthService.create_access_token(token_data)
    refresh_token = AuthService.create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@auth_router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    # Remove password hash from response
    user_info = current_user.copy()
    user_info.pop("password_hash", None)
    return user_info


@auth_router.get("/verify")
async def verify_token_endpoint(current_user: dict = Depends(get_current_user)):
    """Verify if token is valid."""
    return {"valid": True, "username": current_user["username"]}


# Export commonly used components
__all__ = [
    "AuthService",
    "get_current_user",
    "require_role",
    "auth_router",
    "UserCreate",
    "UserLogin",
    "TokenResponse"
]


if __name__ == "__main__":
    # Test authentication
    print(f"JWT Secret Key configured: {'YES' if SECRET_KEY != 'dalrn-secret-key-change-in-production' else 'NO (using default)'}")
    print(f"Access token expiry: {ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    print(f"Refresh token expiry: {REFRESH_TOKEN_EXPIRE_DAYS} days")

    # Test password hashing
    test_password = "test123"
    hashed = AuthService.hash_password(test_password)
    print(f"\nPassword hash test:")
    print(f"  Original: {test_password}")
    print(f"  Hashed: {hashed[:20]}...")
    print(f"  Verify: {AuthService.verify_password(test_password, hashed)}")