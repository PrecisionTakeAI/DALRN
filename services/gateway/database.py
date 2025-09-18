"""
Database abstraction layer supporting both SQLite and PostgreSQL
Allows seamless switching between database backends
"""
import os
import json
import sqlite3
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

class DatabaseInterface(ABC):
    """Abstract database interface"""

    @abstractmethod
    def connect(self):
        """Get database connection"""
        pass

    @abstractmethod
    def init_schema(self):
        """Initialize database schema"""
        pass

    @abstractmethod
    def create_dispute(self, dispute_data: dict) -> str:
        """Create dispute record"""
        pass

    @abstractmethod
    def create_receipt(self, dispute_id: str, step: str, inputs: dict, params: dict) -> str:
        """Create receipt record"""
        pass

    @abstractmethod
    def get_dispute(self, dispute_id: str) -> Optional[dict]:
        """Get dispute by ID"""
        pass

    @abstractmethod
    def get_receipts(self, dispute_id: str) -> List[dict]:
        """Get receipts for dispute"""
        pass

    @abstractmethod
    def get_dispute_count(self) -> int:
        """Get total number of disputes"""
        pass

    @abstractmethod
    def get_receipt_count(self) -> int:
        """Get total number of receipts"""
        pass

class SQLiteDatabase(DatabaseInterface):
    """SQLite database implementation"""

    def __init__(self, db_path: str = "dalrn.db"):
        self.db_path = db_path
        self.init_schema()

    def connect(self):
        """Get SQLite connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self):
        """Initialize SQLite schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create disputes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disputes (
                dispute_id TEXT PRIMARY KEY,
                parties TEXT NOT NULL,
                jurisdiction TEXT NOT NULL,
                cid TEXT NOT NULL,
                enc_meta TEXT,
                phase TEXT DEFAULT 'INTAKE',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create receipts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id TEXT PRIMARY KEY,
                dispute_id TEXT NOT NULL,
                step TEXT NOT NULL,
                inputs TEXT,
                params TEXT,
                artifacts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dispute_id) REFERENCES disputes (dispute_id)
            )
        """)

        conn.commit()
        conn.close()
        logger.info("SQLite database initialized")

    def create_dispute(self, dispute_data: dict) -> str:
        """Create dispute in SQLite"""
        conn = self.connect()
        cursor = conn.cursor()

        dispute_id = f"disp_{uuid4().hex[:12]}"

        cursor.execute("""
            INSERT INTO disputes (dispute_id, parties, jurisdiction, cid, enc_meta, phase, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            dispute_id,
            json.dumps(dispute_data['parties']),
            dispute_data['jurisdiction'],
            dispute_data['cid'],
            json.dumps(dispute_data.get('enc_meta', {})),
            'INTAKE',
            'active'
        ))

        conn.commit()
        conn.close()
        return dispute_id

    def create_receipt(self, dispute_id: str, step: str, inputs: dict, params: dict) -> str:
        """Create receipt in SQLite"""
        conn = self.connect()
        cursor = conn.cursor()

        receipt_id = f"rcpt_{uuid4().hex[:12]}"

        cursor.execute("""
            INSERT INTO receipts (receipt_id, dispute_id, step, inputs, params)
            VALUES (?, ?, ?, ?, ?)
        """, (
            receipt_id,
            dispute_id,
            step,
            json.dumps(inputs),
            json.dumps(params)
        ))

        conn.commit()
        conn.close()
        return receipt_id

    def get_dispute(self, dispute_id: str) -> Optional[dict]:
        """Get dispute from SQLite"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM disputes WHERE dispute_id = ?", (dispute_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_receipts(self, dispute_id: str) -> List[dict]:
        """Get receipts for dispute from SQLite"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM receipts WHERE dispute_id = ?
            ORDER BY created_at DESC
        """, (dispute_id,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_dispute_count(self) -> int:
        """Get total number of disputes in SQLite"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM disputes")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_receipt_count(self) -> int:
        """Get total number of receipts in SQLite"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM receipts")
        count = cursor.fetchone()[0]
        conn.close()
        return count

class PostgreSQLDatabase(DatabaseInterface):
    """PostgreSQL database implementation"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
            self.init_schema()
        except ImportError:
            logger.error("psycopg2 not available for PostgreSQL")
            raise

    def connect(self):
        """Get PostgreSQL connection"""
        return self.psycopg2.connect(
            self.database_url,
            cursor_factory=self.RealDictCursor
        )

    def init_schema(self):
        """Initialize PostgreSQL schema"""
        conn = self.connect()
        cursor = conn.cursor()

        # Create disputes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disputes (
                dispute_id VARCHAR(20) PRIMARY KEY,
                parties JSONB NOT NULL,
                jurisdiction VARCHAR(10) NOT NULL,
                cid VARCHAR(100) NOT NULL,
                enc_meta JSONB,
                phase VARCHAR(20) DEFAULT 'INTAKE',
                status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create receipts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id VARCHAR(20) PRIMARY KEY,
                dispute_id VARCHAR(20) NOT NULL,
                step VARCHAR(50) NOT NULL,
                inputs JSONB,
                params JSONB,
                artifacts JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dispute_id) REFERENCES disputes (dispute_id)
            )
        """)

        conn.commit()
        conn.close()
        logger.info("PostgreSQL database initialized")

    def create_dispute(self, dispute_data: dict) -> str:
        """Create dispute in PostgreSQL"""
        conn = self.connect()
        cursor = conn.cursor()

        dispute_id = f"disp_{uuid4().hex[:12]}"

        cursor.execute("""
            INSERT INTO disputes (dispute_id, parties, jurisdiction, cid, enc_meta, phase, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            dispute_id,
            json.dumps(dispute_data['parties']),
            dispute_data['jurisdiction'],
            dispute_data['cid'],
            json.dumps(dispute_data.get('enc_meta', {})),
            'INTAKE',
            'active'
        ))

        conn.commit()
        conn.close()
        return dispute_id

    def create_receipt(self, dispute_id: str, step: str, inputs: dict, params: dict) -> str:
        """Create receipt in PostgreSQL"""
        conn = self.connect()
        cursor = conn.cursor()

        receipt_id = f"rcpt_{uuid4().hex[:12]}"

        cursor.execute("""
            INSERT INTO receipts (receipt_id, dispute_id, step, inputs, params)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            receipt_id,
            dispute_id,
            step,
            json.dumps(inputs),
            json.dumps(params)
        ))

        conn.commit()
        conn.close()
        return receipt_id

    def get_dispute(self, dispute_id: str) -> Optional[dict]:
        """Get dispute from PostgreSQL"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM disputes WHERE dispute_id = %s", (dispute_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_receipts(self, dispute_id: str) -> List[dict]:
        """Get receipts for dispute from PostgreSQL"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM receipts WHERE dispute_id = %s
            ORDER BY created_at DESC
        """, (dispute_id,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_dispute_count(self) -> int:
        """Get total number of disputes in PostgreSQL"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM disputes")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_receipt_count(self) -> int:
        """Get total number of receipts in PostgreSQL"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM receipts")
        count = cursor.fetchone()[0]
        conn.close()
        return count

def get_database() -> DatabaseInterface:
    """Factory function to get database instance based on configuration"""
    database_url = os.getenv("DATABASE_URL")

    if database_url and database_url.startswith("postgresql://"):
        try:
            logger.info("Using PostgreSQL database")
            return PostgreSQLDatabase(database_url)
        except Exception as e:
            logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")

    # Default to SQLite
    db_path = os.getenv("DATABASE_PATH", "dalrn.db")
    logger.info(f"Using SQLite database: {db_path}")
    return SQLiteDatabase(db_path)

# Global database instance
_db_instance = None

def get_db() -> DatabaseInterface:
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = get_database()
    return _db_instance