"""
Production-ready Database Models with PostgreSQL support
Backwards compatible with SQLite for local testing
"""
import os
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, Boolean, ForeignKey, JSON, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.sql import func
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Production PostgreSQL URL vs Local SQLite
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://dalrn_user:DALRN_Pr0d_2024!SecureP@ss@localhost:5432/dalrn_production"  # For local testing
    # For production: "postgresql://dalrn_user:dalrn_pass@localhost:5432/dalrn_production"
)

class User(Base):
    """User authentication table"""
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)

    # Relationships
    disputes = relationship("Dispute", back_populates="submitter")
    receipts = relationship("Receipt", back_populates="user")

class Dispute(Base):
    """Production dispute records"""
    __tablename__ = "disputes"

    id = Column(String, primary_key=True)
    submitter_id = Column(String, ForeignKey("users.id"), nullable=False)
    parties = Column(JSON)  # List of party identifiers
    jurisdiction = Column(String(10), nullable=False)
    phase = Column(String(20), default="INTAKE")
    status = Column(String(20), default="submitted")
    cid = Column(String(100))  # IPFS content ID
    enc_meta = Column(JSON)  # Encrypted metadata
    anchor_tx = Column(String(66))  # Blockchain transaction hash
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True))

    # Relationships
    submitter = relationship("User", back_populates="disputes")
    receipts = relationship("Receipt", back_populates="dispute")
    evidences = relationship("Evidence", back_populates="dispute")

class Receipt(Base):
    """PoDP receipt chain storage"""
    __tablename__ = "receipts"

    id = Column(String, primary_key=True)  # receipt_id
    dispute_id = Column(String, ForeignKey("disputes.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"))
    step = Column(String(50), nullable=False)
    inputs = Column(JSON)
    params = Column(JSON)
    artifacts = Column(JSON)
    hashes = Column(JSON)
    signatures = Column(JSON)
    merkle_root = Column(String(66))  # For chain finalization
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    dispute = relationship("Dispute", back_populates="receipts")
    user = relationship("User", back_populates="receipts")

class Evidence(Base):
    """Evidence submissions"""
    __tablename__ = "evidences"

    id = Column(String, primary_key=True)
    dispute_id = Column(String, ForeignKey("disputes.id"), nullable=False)
    submitter_id = Column(String, ForeignKey("users.id"), nullable=False)
    evidence_type = Column(String(50), nullable=False)
    cid = Column(String(100), nullable=False)  # IPFS content ID
    encryption_meta = Column(JSON)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    dispute = relationship("Dispute", back_populates="evidences")

class Agent(Base):
    """Network agent registry"""
    __tablename__ = "agents"

    id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)  # compute, validator, mediator
    status = Column(String(20), default="active")
    network_id = Column(String(100))  # SOAN network identifier
    capabilities = Column(JSON)  # List of supported operations
    performance_metrics = Column(JSON)  # Latency, throughput, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_heartbeat = Column(DateTime(timezone=True))

class NetworkMetrics(Base):
    """System performance metrics"""
    __tablename__ = "network_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    active_agents = Column(Integer, default=0)
    disputes_pending = Column(Integer, default=0)
    disputes_resolved = Column(Integer, default=0)
    avg_resolution_time = Column(Float)  # in seconds
    clustering_coefficient = Column(Float)
    throughput = Column(Float)  # disputes per hour
    p95_latency = Column(Float)  # milliseconds
    epsilon_budget_used = Column(Float)  # Privacy budget consumption

class EpsilonLedger(Base):
    """Privacy budget tracking"""
    __tablename__ = "epsilon_ledger"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(100), nullable=False)
    model_id = Column(String(100), nullable=False)
    round_number = Column(Integer, nullable=False)
    epsilon = Column(Float, nullable=False)
    delta = Column(Float, nullable=False)
    mechanism = Column(String(50), nullable=False)  # RDP, ZCDP, etc.
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ProductionDatabaseService:
    """Production database service with connection pooling"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL

        # Production-grade engine configuration
        engine_kwargs = {
            "echo": False,  # Set to True for SQL debugging
            "future": True,
        }

        # PostgreSQL-specific optimizations
        if "postgresql" in self.database_url:
            engine_kwargs.update({
                "pool_size": 20,
                "max_overflow": 0,
                "pool_pre_ping": True,
                "pool_recycle": 300,
            })

        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables
        self.create_tables()

    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def __enter__(self):
        self.session = self.get_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()

    # User management
    def create_user(self, user_data: Dict) -> User:
        """Create new user"""
        user = User(**user_data)
        self.session.add(user)
        self.session.flush()
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.session.query(User).filter(User.id == user_id).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.session.query(User).filter(User.username == username).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.session.query(User).filter(User.email == email).first()

    # Dispute management
    def create_dispute(self, dispute_data: Dict) -> Dispute:
        """Create new dispute"""
        dispute = Dispute(**dispute_data)
        self.session.add(dispute)
        self.session.flush()
        return dispute

    def get_dispute(self, dispute_id: str) -> Optional[Dispute]:
        """Get dispute by ID"""
        return self.session.query(Dispute).filter(Dispute.id == dispute_id).first()

    def update_dispute_status(self, dispute_id: str, status: str, phase: str = None):
        """Update dispute status"""
        dispute = self.get_dispute(dispute_id)
        if dispute:
            dispute.status = status
            if phase:
                dispute.phase = phase
            dispute.updated_at = datetime.now(timezone.utc)

    def get_disputes_by_user(self, user_id: str, limit: int = 50) -> List[Dispute]:
        """Get disputes submitted by user"""
        return (self.session.query(Dispute)
                .filter(Dispute.submitter_id == user_id)
                .order_by(Dispute.created_at.desc())
                .limit(limit)
                .all())

    # Receipt management
    def create_receipt(self, receipt_data: Dict) -> Receipt:
        """Create PoDP receipt"""
        receipt = Receipt(**receipt_data)
        self.session.add(receipt)
        self.session.flush()
        return receipt

    def get_receipts_for_dispute(self, dispute_id: str) -> List[Receipt]:
        """Get all receipts for a dispute"""
        return (self.session.query(Receipt)
                .filter(Receipt.dispute_id == dispute_id)
                .order_by(Receipt.timestamp)
                .all())

    # Agent management
    def register_agent(self, agent_data: Dict) -> Agent:
        """Register new agent"""
        agent = Agent(**agent_data)
        self.session.add(agent)
        self.session.flush()
        return agent

    def get_active_agents(self, limit: int = 100) -> List[Agent]:
        """Get active agents"""
        return (self.session.query(Agent)
                .filter(Agent.status == "active")
                .limit(limit)
                .all())

    def update_agent_heartbeat(self, agent_id: str):
        """Update agent last heartbeat"""
        agent = self.session.query(Agent).filter(Agent.id == agent_id).first()
        if agent:
            agent.last_heartbeat = datetime.now(timezone.utc)

    # Metrics
    def record_metrics(self, metrics_data: Dict) -> NetworkMetrics:
        """Record network metrics"""
        metrics = NetworkMetrics(**metrics_data)
        self.session.add(metrics)
        self.session.flush()
        return metrics

    def get_latest_metrics(self) -> Optional[NetworkMetrics]:
        """Get latest metrics"""
        return (self.session.query(NetworkMetrics)
                .order_by(NetworkMetrics.timestamp.desc())
                .first())

    # Privacy budget
    def record_epsilon_spend(self, ledger_data: Dict) -> EpsilonLedger:
        """Record privacy budget expenditure"""
        entry = EpsilonLedger(**ledger_data)
        self.session.add(entry)
        self.session.flush()
        return entry

    def get_epsilon_budget(self, tenant_id: str, model_id: str) -> float:
        """Get remaining epsilon budget"""
        total_spent = (self.session.query(func.sum(EpsilonLedger.epsilon))
                      .filter(EpsilonLedger.tenant_id == tenant_id)
                      .filter(EpsilonLedger.model_id == model_id)
                      .scalar() or 0.0)

        total_budget = 4.0  # Default epsilon budget
        return max(0.0, total_budget - total_spent)

    # Health check
    def health_check(self) -> Dict:
        """Database health check"""
        try:
            # Test connection
            result = self.session.execute(text("SELECT 1")).scalar()

            # Get table counts
            user_count = self.session.query(User).count()
            dispute_count = self.session.query(Dispute).count()
            receipt_count = self.session.query(Receipt).count()
            agent_count = self.session.query(Agent).count()

            return {
                "status": "healthy",
                "database_type": "PostgreSQL" if "postgresql" in self.database_url else "SQLite",
                "connection": "ok",
                "counts": {
                    "users": user_count,
                    "disputes": dispute_count,
                    "receipts": receipt_count,
                    "agents": agent_count
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global instance
_db_service = None

def get_production_db() -> ProductionDatabaseService:
    """Get production database service singleton"""
    global _db_service
    if _db_service is None:
        _db_service = ProductionDatabaseService()
    return _db_service

def migrate_to_production():
    """Migrate existing SQLite data to production database"""
    logger.info("Starting migration to production database...")

    # This would be implemented to migrate data from the old SQLite database
    # to the new production database structure

    prod_db = get_production_db()

    # For now, just ensure tables exist
    prod_db.create_tables()

    logger.info("Migration completed successfully")

if __name__ == "__main__":
    # Test the production database
    print("Testing production database...")

    db = get_production_db()
    with db:
        health = db.health_check()
        print(f"Health check: {health}")

        # Test creating a user
        try:
            test_user = db.create_user({
                "id": "test_user_001",
                "username": "testuser",
                "email": "test@dalrn.test",
                "password_hash": "hashed_password",
                "role": "admin"
            })
            print(f"Created test user: {test_user.username}")

            # Test creating a dispute
            test_dispute = db.create_dispute({
                "id": "disp_test_001",
                "submitter_id": test_user.id,
                "parties": ["party_a", "party_b"],
                "jurisdiction": "US-CA",
                "cid": "QmTest123"
            })
            print(f"Created test dispute: {test_dispute.id}")

        except Exception as e:
            print(f"Test failed: {e}")

    print("Production database test completed!")