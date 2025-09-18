"""
JWT Authentication implementation for DALRN
PRD REQUIREMENT: JWT-based authentication and authorization
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
import bcrypt
import os
import hashlib
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Import database models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.models import User, DatabaseService

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dalrn-secret-key-change-in-production-" + os.urandom(32).hex())
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security scheme
security = HTTPBearer()

class UserCreate(BaseModel):
    """User registration model"""
    username: str
    email: str
    password: str
    role: str = "user"

class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str

class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Token payload model"""
    user_id: str
    username: str
    role: str
    exp: datetime

class AuthService:
    """JWT Authentication service"""

    def __init__(self):
        self.db = DatabaseService()

    def generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{hashlib.sha256(os.urandom(32)).hexdigest()[:12]}"

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )

    def create_user(self, user_data: UserCreate) -> User:
        """Create new user with hashed password"""
        # Check if user exists
        existing = self.db.session.query(User).filter(
            (User.username == user_data.username) |
            (User.email == user_data.email)
        ).first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )

        # Create user
        user = User(
            id=self.generate_user_id(),
            username=user_data.username,
            email=user_data.email,
            password_hash=self.hash_password(user_data.password),
            role=user_data.role
        )

        self.db.session.add(user)
        self.db.session.commit()
        self.db.session.refresh(user)

        return user

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user and return user object"""
        user = self.db.session.query(User).filter_by(username=username).first()

        if not user:
            return None

        if not self.verify_password(password, user.password_hash):
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        self.db.session.commit()

        return user

    def create_access_token(self, user_id: str, username: str, role: str) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        payload = {
            "sub": user_id,  # Subject (user ID)
            "username": username,
            "role": role,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }

        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }

        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Check token type
            if payload.get("type") != token_type:
                return None

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    def get_current_user(self, token: str) -> User:
        """Get current user from token"""
        payload = self.verify_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

        user = self.db.session.query(User).filter_by(id=payload["sub"]).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated"
            )

        return user

    def refresh_access_token(self, refresh_token: str) -> str:
        """Generate new access token from refresh token"""
        payload = self.verify_token(refresh_token, token_type="refresh")
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        user = self.db.session.query(User).filter_by(id=payload["sub"]).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        return self.create_access_token(user.id, user.username, user.role)

# Dependency for protected routes
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    """FastAPI dependency to get current user from JWT token"""
    auth_service = AuthService()
    token = credentials.credentials
    return auth_service.get_current_user(token)

async def require_role(required_role: str):
    """FastAPI dependency to require specific role"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role} role"
            )
        return current_user
    return role_checker

# Auth endpoints for FastAPI
from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register new user"""
    auth_service = AuthService()

    # Create user
    user = auth_service.create_user(user_data)

    # Generate tokens
    access_token = auth_service.create_access_token(user.id, user.username, user.role)
    refresh_token = auth_service.create_refresh_token(user.id)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )

@auth_router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login and get JWT tokens"""
    auth_service = AuthService()

    # Authenticate user
    user = auth_service.authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Generate tokens
    access_token = auth_service.create_access_token(user.id, user.username, user.role)
    refresh_token = auth_service.create_refresh_token(user.id)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )

@auth_router.post("/refresh")
async def refresh(refresh_token: str):
    """Refresh access token"""
    auth_service = AuthService()
    access_token = auth_service.refresh_access_token(refresh_token)

    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat()
    }

@auth_router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout user (client should discard tokens)"""
    # In a production system, we would blacklist the token
    return {"message": "Successfully logged out"}

# Example of protected endpoint
@auth_router.get("/admin-only")
async def admin_only(current_user: User = Depends(require_role("admin"))):
    """Admin-only endpoint example"""
    return {"message": f"Hello admin {current_user.username}"}