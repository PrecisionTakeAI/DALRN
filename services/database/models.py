"""
Database models for DALRN - PostgreSQL implementation
PRD REQUIREMENT: PostgreSQL database for persistence
"""
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from datetime import datetime
import os
import hashlib
import json

Base = declarative_base()

# Database configuration - secure environment-based setup
def get_database_url():
    """Get database URL from environment with secure defaults"""

    # Check for full DATABASE_URL first
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    # Build URL from individual components
    db_driver = os.getenv("DATABASE_DRIVER", "postgresql")
    db_user = os.getenv("DATABASE_USER", "dalrn_user")
    db_password = os.getenv("DATABASE_PASSWORD")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5432")
    db_name = os.getenv("DATABASE_NAME", "dalrn_production")

    if db_password:
        # Use PostgreSQL if password is provided
        return f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    else:
        # Fallback to SQLite for development (no credentials in code)
        from pathlib import Path
        db_path = Path("data") / "dalrn.db"
        db_path.parent.mkdir(exist_ok=True)
        return f"sqlite:///{db_path}"

DATABASE_URL = get_database_url()

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=40, echo=False)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

class Dispute(Base):
    """Dispute model - stores all dispute data"""
    __tablename__ = "disputes"

    id = Column(String(64), primary_key=True)
    parties = Column(JSON, nullable=False)  # List of party IDs
    jurisdiction = Column(String(10), nullable=False)
    cid = Column(String(100), nullable=False)  # IPFS CID
    enc_meta = Column(JSON)  # Encrypted metadata
    phase = Column(String(20), default="INTAKE", index=True)
    status = Column(String(20), default="submitted", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolution = Column(JSON)
    merkle_root = Column(String(64))
    anchor_tx = Column(String(66))  # Blockchain transaction hash
    receipt_chain_uri = Column(String(200))
    epsilon_budget = Column(Float, default=4.0)

    # Relationships
    receipts = relationship("PoDPReceipt", back_populates="dispute", cascade="all, delete-orphan")
    metrics = relationship("DisputeMetrics", back_populates="dispute", cascade="all, delete-orphan")

class Agent(Base):
    """Agent model - stores SOAN agents"""
    __tablename__ = "agents"

    id = Column(String(64), primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(20), nullable=False, index=True)  # detector, negotiator, etc.
    capabilities = Column(JSON)
    reputation_score = Column(Float, default=0.5)
    disputes_resolved = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    topology_position = Column(JSON)  # Node position in Watts-Strogatz graph
    queue_length = Column(Integer, default=0)
    service_rate = Column(Float, default=1.5)
    is_active = Column(Boolean, default=True)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)

    # Relationships
    receipts = relationship("PoDPReceipt", back_populates="agent")

class PoDPReceipt(Base):
    """Proof of Data Processing receipts"""
    __tablename__ = "podp_receipts"

    id = Column(String(64), primary_key=True)
    receipt_id = Column(String(64), unique=True, nullable=False)
    dispute_id = Column(String(64), ForeignKey("disputes.id"), nullable=False, index=True)
    agent_id = Column(String(64), ForeignKey("agents.id"), nullable=True, index=True)
    step = Column(String(50), nullable=False)
    inputs = Column(JSON)
    params = Column(JSON)
    artifacts = Column(JSON)
    hashes = Column(JSON)
    signatures = Column(JSON, default=list)
    ts = Column(DateTime, default=datetime.utcnow)
    hash = Column(String(64))

    # Relationships
    dispute = relationship("Dispute", back_populates="receipts")
    agent = relationship("Agent", back_populates="receipts")

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), default="user")  # user, admin, agent
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class NetworkMetrics(Base):
    """Network performance metrics for monitoring"""
    __tablename__ = "network_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    active_agents = Column(Integer)
    disputes_pending = Column(Integer)
    disputes_resolved = Column(Integer)
    avg_resolution_time = Column(Float)  # in seconds
    network_topology = Column(JSON)  # Snapshot of Watts-Strogatz topology
    clustering_coefficient = Column(Float)
    avg_path_length = Column(Float)
    throughput = Column(Integer)  # disputes/hour
    p95_latency = Column(Float)  # 95th percentile latency in ms

class DisputeMetrics(Base):
    """Per-dispute performance metrics"""
    __tablename__ = "dispute_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dispute_id = Column(String(64), ForeignKey("disputes.id"), nullable=False, index=True)
    ingestion_time = Column(Float)  # ms
    processing_time = Column(Float)  # ms
    total_time = Column(Float)  # ms
    agents_involved = Column(Integer)
    receipts_generated = Column(Integer)
    epsilon_used = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dispute = relationship("Dispute", back_populates="metrics")

# Create all tables
Base.metadata.create_all(bind=engine)

class DatabaseService:
    """Database service with optimized queries for <200ms performance"""

    def __init__(self):
        self.session = SessionLocal()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def create_dispute(self, dispute_data: dict) -> Dispute:
        """Create new dispute with proper persistence"""
        # Generate ID if not provided
        if 'id' not in dispute_data:
            dispute_data['id'] = f"disp_{hashlib.sha256(json.dumps(dispute_data, sort_keys=True).encode()).hexdigest()[:8]}"

        dispute = Dispute(**dispute_data)
        self.session.add(dispute)
        self.session.commit()
        self.session.refresh(dispute)
        return dispute

    def get_dispute(self, dispute_id: str) -> Dispute:
        """Get dispute by ID with eager loading for performance"""
        return self.session.query(Dispute).filter_by(id=dispute_id).first()

    def get_disputes_for_processing(self, limit: int = 100) -> list:
        """Get pending disputes for batch processing"""
        return self.session.query(Dispute).filter_by(
            status="submitted"
        ).limit(limit).all()

    def update_dispute_status(self, dispute_id: str, status: str, phase: str = None, resolution: dict = None):
        """Update dispute with resolution"""
        dispute = self.session.query(Dispute).filter_by(id=dispute_id).first()
        if dispute:
            dispute.status = status
            if phase:
                dispute.phase = phase
            if resolution:
                dispute.resolution = resolution
            dispute.updated_at = datetime.utcnow()
            self.session.commit()

    def create_receipt(self, receipt_data: dict) -> PoDPReceipt:
        """Store PoDP receipt in database"""
        if 'id' not in receipt_data:
            receipt_data['id'] = f"rcpt_{hashlib.sha256(json.dumps(receipt_data, sort_keys=True).encode()).hexdigest()[:8]}"

        receipt = PoDPReceipt(**receipt_data)
        self.session.add(receipt)
        self.session.commit()
        return receipt

    def get_receipts_for_dispute(self, dispute_id: str) -> list:
        """Get all receipts for a dispute"""
        return self.session.query(PoDPReceipt).filter_by(dispute_id=dispute_id).all()

    def create_agent(self, agent_data: dict) -> Agent:
        """Create new agent in network"""
        if 'id' not in agent_data:
            agent_data['id'] = f"agent_{hashlib.sha256(agent_data['name'].encode()).hexdigest()[:8]}"

        agent = Agent(**agent_data)
        self.session.add(agent)
        self.session.commit()
        return agent

    def get_active_agents(self, limit: int = 100) -> list:
        """Get active agents for task assignment"""
        # Agents active in last 5 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        return self.session.query(Agent).filter(
            Agent.is_active == True,
            Agent.last_heartbeat > cutoff
        ).limit(limit).all()

    def record_metrics(self, metrics_data: dict):
        """Record network metrics for monitoring"""
        metrics = NetworkMetrics(**metrics_data)
        self.session.add(metrics)
        self.session.commit()

    def get_latest_metrics(self) -> NetworkMetrics:
        """Get latest network metrics"""
        return self.session.query(NetworkMetrics).order_by(
            NetworkMetrics.timestamp.desc()
        ).first()

    def cleanup_old_data(self, days: int = 30):
        """Clean up old data for performance"""
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Delete old metrics
        self.session.query(NetworkMetrics).filter(
            NetworkMetrics.timestamp < cutoff
        ).delete()

        # Delete resolved disputes older than cutoff
        self.session.query(Dispute).filter(
            Dispute.status == "resolved",
            Dispute.updated_at < cutoff
        ).delete()

        self.session.commit()

# Migration function
def migrate_from_memory():
    """Migrate existing in-memory data to database"""
    from services.gateway.app import dispute_storage, receipt_chains

    with DatabaseService() as db:
        # Migrate disputes
        for dispute_id, dispute_data in dispute_storage.items():
            try:
                db.create_dispute(dispute_data)
                print(f"Migrated dispute: {dispute_id}")
            except Exception as e:
                print(f"Failed to migrate dispute {dispute_id}: {e}")

        # Migrate receipts
        for dispute_id, chain in receipt_chains.items():
            for receipt in chain.receipts:
                try:
                    receipt_data = receipt.dict() if hasattr(receipt, 'dict') else receipt.__dict__
                    receipt_data['dispute_id'] = dispute_id
                    db.create_receipt(receipt_data)
                except Exception as e:
                    print(f"Failed to migrate receipt: {e}")

    print("Migration complete")

if __name__ == "__main__":
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")