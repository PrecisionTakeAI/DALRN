"""
Database connection manager with automatic fallback to SQLite for development.
Production-ready PostgreSQL connection with proper error handling.
"""

import os
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

# Try PostgreSQL first, fall back to SQLite if not available
try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False
    logger.warning("psycopg2 not installed, will use SQLite fallback")

import sqlite3


class DatabaseConnection:
    """Database connection manager with PostgreSQL/SQLite fallback."""

    def __init__(self):
        self.pool = None
        self.connection_type = None
        self.db_path = None
        self.init_connection()

    def init_connection(self):
        """Create database connection with retry logic and fallback."""
        # First try PostgreSQL
        if HAS_POSTGRESQL and os.getenv('DB_TYPE', 'postgres').lower() != 'sqlite':
            if self._try_postgresql():
                return

        # Fall back to SQLite
        logger.info("Using SQLite database (development mode)")
        self._setup_sqlite()

    def _try_postgresql(self) -> bool:
        """Try to establish PostgreSQL connection."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                db_params = {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': int(os.getenv('DB_PORT', 5432)),
                    'database': os.getenv('DB_NAME', 'dalrn'),
                    'user': os.getenv('DB_USER', 'dalrn_user'),
                    'password': os.getenv('DB_PASSWORD', 'dalrn_pass')
                }

                # Create connection pool
                self.pool = SimpleConnectionPool(
                    1, 20,  # min and max connections
                    **db_params
                )

                # Test the connection
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute('SELECT 1')
                        result = cur.fetchone()
                        if result[0] == 1:
                            logger.info(f"PostgreSQL connection established to {db_params['host']}:{db_params['port']}")
                            self.connection_type = 'postgresql'
                            self.ensure_tables_exist()
                            return True

            except Exception as e:
                logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return False

    def _setup_sqlite(self):
        """Setup SQLite as fallback database."""
        db_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(db_dir, exist_ok=True)

        self.db_path = os.path.join(db_dir, 'dalrn.db')
        self.connection_type = 'sqlite'

        # Test connection and create tables
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.close()

        self.ensure_tables_exist()
        logger.info(f"SQLite database initialized at {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        if self.connection_type == 'postgresql' and self.pool:
            conn = self.pool.getconn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self.pool.putconn(conn)
        else:
            # SQLite connection
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def ensure_tables_exist(self):
        """Create required tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table for authentication
            if self.connection_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        role VARCHAR(50) DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username VARCHAR(255) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        role VARCHAR(50) DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Disputes table
            if self.connection_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS disputes (
                        id SERIAL PRIMARY KEY,
                        dispute_id VARCHAR(255) UNIQUE NOT NULL,
                        user_id INTEGER REFERENCES users(id),
                        parties TEXT NOT NULL,
                        jurisdiction VARCHAR(100),
                        status VARCHAR(50) DEFAULT 'INTAKE',
                        cid VARCHAR(255),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS disputes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dispute_id VARCHAR(255) UNIQUE NOT NULL,
                        user_id INTEGER REFERENCES users(id),
                        parties TEXT NOT NULL,
                        jurisdiction VARCHAR(100),
                        status VARCHAR(50) DEFAULT 'INTAKE',
                        cid VARCHAR(255),
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Receipts table for PoDP
            if self.connection_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS receipts (
                        id SERIAL PRIMARY KEY,
                        receipt_id VARCHAR(255) UNIQUE NOT NULL,
                        dispute_id VARCHAR(255),
                        step VARCHAR(100),
                        inputs JSONB,
                        outputs JSONB,
                        hash VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS receipts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        receipt_id VARCHAR(255) UNIQUE NOT NULL,
                        dispute_id VARCHAR(255),
                        step VARCHAR(100),
                        inputs TEXT,
                        outputs TEXT,
                        hash VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Epsilon ledger for privacy budget
            if self.connection_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS epsilon_ledger (
                        id SERIAL PRIMARY KEY,
                        tenant_id VARCHAR(255) NOT NULL,
                        model_id VARCHAR(255) NOT NULL,
                        round INTEGER,
                        epsilon FLOAT NOT NULL,
                        delta FLOAT,
                        mechanism VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(tenant_id, model_id, round)
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS epsilon_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id VARCHAR(255) NOT NULL,
                        model_id VARCHAR(255) NOT NULL,
                        round INTEGER,
                        epsilon FLOAT NOT NULL,
                        delta FLOAT,
                        mechanism VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(tenant_id, model_id, round)
                    )
                """)

            logger.info(f"Database tables verified/created in {self.connection_type}")

    def execute(self, query: str, params: tuple = None) -> Any:
        """Execute a query and return results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                return cursor.lastrowid

    def execute_one(self, query: str, params: tuple = None) -> Any:
        """Execute a query and return single result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()

    def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            result = self.execute_one("SELECT 1")
            return {
                "status": "healthy",
                "type": self.connection_type,
                "connected": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "type": self.connection_type,
                "connected": False,
                "error": str(e)
            }


# Singleton instance
db = DatabaseConnection()


# Helper functions for common operations
def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username."""
    result = db.execute_one(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    )
    if result:
        if db.connection_type == 'sqlite':
            return dict(result)
        return result
    return None


def create_user(username: str, email: str, password_hash: str, role: str = 'user') -> int:
    """Create a new user."""
    return db.execute(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (username, email, password_hash, role)
    )


def create_dispute(dispute_id: str, user_id: int, parties: str,
                  jurisdiction: str, cid: str, metadata: dict) -> int:
    """Create a new dispute."""
    metadata_str = json.dumps(metadata) if db.connection_type == 'sqlite' else metadata
    return db.execute(
        """INSERT INTO disputes
           (dispute_id, user_id, parties, jurisdiction, cid, metadata)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (dispute_id, user_id, parties, jurisdiction, cid, metadata_str)
    )


def get_dispute(dispute_id: str) -> Optional[Dict]:
    """Get dispute by ID."""
    result = db.execute_one(
        "SELECT * FROM disputes WHERE dispute_id = ?",
        (dispute_id,)
    )
    if result:
        if db.connection_type == 'sqlite':
            result_dict = dict(result)
            if result_dict.get('metadata'):
                result_dict['metadata'] = json.loads(result_dict['metadata'])
            return result_dict
        return result
    return None


if __name__ == "__main__":
    # Test the connection
    print("Testing database connection...")
    health = db.health_check()
    print(f"Database health: {health}")

    # Test table creation
    print(f"Database type: {db.connection_type}")
    print("Tables created successfully")