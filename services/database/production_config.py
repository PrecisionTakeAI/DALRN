"""
Production Database Configuration
Supports both SQLite (development) and PostgreSQL (production)
"""
import os
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.sql import func, text
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Production-ready database configuration
class DatabaseConfig:
    def __init__(self):
        self.environment = os.getenv("DALRN_ENV", "production")
        self.database_url = self._get_database_url()

    def _get_database_url(self) -> str:
        """Get database URL based on environment"""

        # Check for explicit DATABASE_URL first
        if os.getenv("DATABASE_URL"):
            return os.getenv("DATABASE_URL")

        # Production PostgreSQL
        if self.environment == "production":
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "dalrn_production")
            user = os.getenv("POSTGRES_USER", "dalrn_user")
            password = os.getenv("POSTGRES_PASSWORD", "dalrn_secure_pass")

            return f"postgresql://{user}:{password}@{host}:{port}/{db}"

        # Development SQLite
        else:
            db_path = os.getenv("SQLITE_PATH", "dalrn_dev.db")
            return f"sqlite:///{db_path}"

    def get_engine_config(self) -> Dict:
        """Get SQLAlchemy engine configuration"""

        base_config = {
            "echo": os.getenv("SQL_DEBUG", "false").lower() == "true",
            "future": True,
        }

        # PostgreSQL specific optimizations
        if "postgresql" in self.database_url:
            base_config.update({
                "pool_size": int(os.getenv("DB_POOL_SIZE", "20")),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "30")),
                "pool_pre_ping": True,
                "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
                "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
            })

        return base_config

# Enhanced models for production
class User(Base):
    """Enhanced user model with proper indexing"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)  # UUID
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user", index=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))

    # Additional fields for production
    email_verified = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))

    # Relationships
    disputes = relationship("Dispute", back_populates="submitter")
    receipts = relationship("Receipt", back_populates="user")

class Dispute(Base):
    """Enhanced dispute model with proper indexing and auditing"""
    __tablename__ = "disputes"

    id = Column(String(36), primary_key=True, index=True)
    submitter_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # Core dispute data
    parties = Column(JSON)  # List of party identifiers
    jurisdiction = Column(String(10), nullable=False, index=True)
    dispute_type = Column(String(50), index=True)
    priority = Column(String(10), default="normal", index=True)

    # Status tracking
    phase = Column(String(20), default="INTAKE", index=True)
    status = Column(String(20), default="submitted", index=True)
    resolution_type = Column(String(50))

    # Content
    title = Column(String(200))
    description = Column(Text)
    cid = Column(String(100), index=True)  # IPFS content ID
    enc_meta = Column(JSON)  # Encrypted metadata

    # Blockchain integration
    anchor_tx = Column(String(66), index=True)  # Transaction hash
    merkle_root = Column(String(66), index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True))
    deadline = Column(DateTime(timezone=True))

    # Financial
    dispute_amount = Column(Float)
    resolution_amount = Column(Float)
    currency = Column(String(10), default="USD")

    # Relationships
    submitter = relationship("User", back_populates="disputes")
    receipts = relationship("Receipt", back_populates="dispute", cascade="all, delete-orphan")
    evidences = relationship("Evidence", back_populates="dispute", cascade="all, delete-orphan")
    settlements = relationship("Settlement", back_populates="dispute", cascade="all, delete-orphan")

class Receipt(Base):
    """Enhanced PoDP receipt with versioning"""
    __tablename__ = "receipts"

    id = Column(String(36), primary_key=True, index=True)  # receipt_id
    dispute_id = Column(String(36), ForeignKey("disputes.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), index=True)

    # Receipt data
    step = Column(String(50), nullable=False, index=True)
    version = Column(String(10), default="1.0")
    inputs = Column(JSON)
    params = Column(JSON)
    artifacts = Column(JSON)
    hashes = Column(JSON)
    signatures = Column(JSON)

    # Chain data
    merkle_root = Column(String(66), index=True)
    parent_receipt_id = Column(String(36), ForeignKey("receipts.id"))
    chain_position = Column(Integer)

    # Metadata
    processor_id = Column(String(100))  # Agent/service that created this
    processing_time_ms = Column(Integer)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    dispute = relationship("Dispute", back_populates="receipts")
    user = relationship("User", back_populates="receipts")
    parent = relationship("Receipt", remote_side=[id])

class Evidence(Base):
    """Evidence submissions with versioning"""
    __tablename__ = "evidences"

    id = Column(String(36), primary_key=True, index=True)
    dispute_id = Column(String(36), ForeignKey("disputes.id"), nullable=False, index=True)
    submitter_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # Evidence data
    evidence_type = Column(String(50), nullable=False, index=True)
    title = Column(String(200))
    description = Column(Text)
    cid = Column(String(100), nullable=False, index=True)  # IPFS content ID
    file_hash = Column(String(66))
    file_size = Column(Integer)
    mime_type = Column(String(100))

    # Encryption and privacy
    encryption_meta = Column(JSON)
    access_level = Column(String(20), default="parties")  # public, parties, mediator, court

    # Timestamps
    submitted_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    verified_at = Column(DateTime(timezone=True))

    # Relationships
    dispute = relationship("Dispute", back_populates="evidences")
    submitter = relationship("User", foreign_keys=[submitter_id])

class Settlement(Base):
    """Settlement records and agreements"""
    __tablename__ = "settlements"

    id = Column(String(36), primary_key=True, index=True)
    dispute_id = Column(String(36), ForeignKey("disputes.id"), nullable=False, index=True)

    # Settlement data
    settlement_type = Column(String(50), nullable=False)  # mediation, arbitration, agreement
    status = Column(String(20), default="proposed", index=True)
    terms = Column(JSON)

    # Financial
    total_amount = Column(Float)
    currency = Column(String(10), default="USD")
    payment_terms = Column(JSON)

    # Signatures and approval
    signatures = Column(JSON)  # Digital signatures from parties
    approved_by = Column(JSON)  # List of approving parties

    # Smart contract
    contract_address = Column(String(42))  # Ethereum address
    execution_tx = Column(String(66))  # Transaction hash

    # Timestamps
    proposed_at = Column(DateTime(timezone=True), server_default=func.now())
    agreed_at = Column(DateTime(timezone=True))
    executed_at = Column(DateTime(timezone=True))

    # Relationships
    dispute = relationship("Dispute", back_populates="settlements")

class Agent(Base):
    """Enhanced agent registry with performance tracking"""
    __tablename__ = "agents"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False, index=True)
    version = Column(String(20), default="1.0")
    status = Column(String(20), default="active", index=True)

    # Network and capabilities
    network_id = Column(String(100), index=True)
    capabilities = Column(JSON)
    specializations = Column(JSON)  # Areas of expertise

    # Performance metrics
    performance_metrics = Column(JSON)
    success_rate = Column(Float, default=0.0)
    avg_processing_time = Column(Float)
    load_capacity = Column(Integer, default=100)
    current_load = Column(Integer, default=0)

    # Registration and heartbeat
    registered_by = Column(String(36), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_heartbeat = Column(DateTime(timezone=True))
    last_task_at = Column(DateTime(timezone=True))

class SystemMetrics(Base):
    """System-wide performance and health metrics"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    metric_type = Column(String(50), nullable=False, index=True)

    # Core metrics
    active_agents = Column(Integer, default=0)
    disputes_pending = Column(Integer, default=0)
    disputes_resolved = Column(Integer, default=0)
    disputes_total = Column(Integer, default=0)

    # Performance metrics
    avg_resolution_time = Column(Float)
    p95_response_time = Column(Float)
    throughput_per_hour = Column(Float)
    error_rate = Column(Float, default=0.0)

    # Network metrics
    clustering_coefficient = Column(Float)
    network_efficiency = Column(Float)

    # Resource utilization
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)

    # Privacy budget
    epsilon_budget_used = Column(Float, default=0.0)
    epsilon_budget_total = Column(Float, default=4.0)

class ProductionDatabaseService:
    """Production-ready database service with comprehensive features"""

    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()

        # Create engine with production settings
        engine_config = self.config.get_engine_config()
        self.engine = create_engine(self.config.database_url, **engine_config)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Initialize database
        self.create_tables()

        logger.info(f"Database initialized: {self._get_db_type()}")

    def _get_db_type(self) -> str:
        """Get database type for logging"""
        if "postgresql" in self.config.database_url:
            return "PostgreSQL (Production)"
        elif "sqlite" in self.config.database_url:
            return "SQLite (Development)"
        else:
            return "Unknown"

    def create_tables(self):
        """Create all tables with proper error handling"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def get_session(self) -> Session:
        """Get database session with proper error handling"""
        try:
            return self.SessionLocal()
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        try:
            with self.get_session() as session:
                # Test basic connectivity
                result = session.execute(text("SELECT 1")).scalar()

                # Get table counts
                counts = {}
                for table_name in ["users", "disputes", "receipts", "evidences", "agents"]:
                    try:
                        count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                        counts[table_name] = count
                    except:
                        counts[table_name] = "N/A"

                # Check database size (PostgreSQL only)
                db_size = "N/A"
                if "postgresql" in self.config.database_url:
                    try:
                        size_query = text("SELECT pg_size_pretty(pg_database_size(current_database()))")
                        db_size = session.execute(size_query).scalar()
                    except:
                        pass

                return {
                    "status": "healthy",
                    "database_type": self._get_db_type(),
                    "connection": "ok",
                    "response_time_ms": self._measure_response_time(),
                    "table_counts": counts,
                    "database_size": db_size,
                    "connection_pool": self._get_pool_status(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_type": self._get_db_type(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _measure_response_time(self) -> float:
        """Measure database response time"""
        import time
        start = time.perf_counter()
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1")).scalar()
            return round((time.perf_counter() - start) * 1000, 2)
        except:
            return -1

    def _get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status (PostgreSQL only)"""
        if "postgresql" not in self.config.database_url:
            return {"type": "sqlite", "status": "n/a"}

        try:
            pool = self.engine.pool
            return {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
        except:
            return {"error": "Could not get pool status"}

    def __enter__(self):
        self.session = self.get_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()

# Create global instance
_db_config = DatabaseConfig()
_db_service = ProductionDatabaseService(_db_config)

def get_database_service() -> ProductionDatabaseService:
    """Get the global database service instance"""
    return _db_service

def get_session() -> Session:
    """Get a database session"""
    return _db_service.get_session()

# Export key components
__all__ = [
    'DatabaseConfig',
    'ProductionDatabaseService',
    'User', 'Dispute', 'Receipt', 'Evidence', 'Settlement', 'Agent', 'SystemMetrics',
    'get_database_service',
    'get_session'
]

if __name__ == "__main__":
    # Test the database configuration
    print("Testing production database configuration...")

    config = DatabaseConfig()
    print(f"Environment: {config.environment}")
    print(f"Database URL: {config.database_url}")

    # Test database service
    db = ProductionDatabaseService(config)
    health = db.health_check()

    print("Health check:")
    print(json.dumps(health, indent=2))

    print("\nDatabase configuration test completed!")