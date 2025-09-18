"""
DALRN Database Migration Script
Migrates from SQLite to PostgreSQL production database
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.orm import sessionmaker
from services.database.production_config import (
    Base, User, Dispute, Receipt, Evidence, Settlement, Agent, SystemMetrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles migration from SQLite to PostgreSQL"""

    def __init__(self):
        # Source database (SQLite)
        self.sqlite_path = os.getenv("SQLITE_PATH", "dalrn_dev.db")
        self.sqlite_url = f"sqlite:///{self.sqlite_path}"

        # Target database (PostgreSQL)
        self.pg_host = os.getenv("POSTGRES_HOST", "localhost")
        self.pg_port = os.getenv("POSTGRES_PORT", "5432")
        self.pg_db = os.getenv("POSTGRES_DB", "dalrn_production")
        self.pg_user = os.getenv("POSTGRES_USER", "dalrn_user")
        self.pg_password = os.getenv("POSTGRES_PASSWORD", "DALRN_Pr0d_2024!SecureP@ss")
        self.postgres_url = f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"

        # Engines
        self.sqlite_engine = None
        self.postgres_engine = None

    def connect_databases(self):
        """Establish connections to both databases"""
        try:
            # Connect to SQLite
            if os.path.exists(self.sqlite_path):
                self.sqlite_engine = create_engine(self.sqlite_url)
                logger.info(f"Connected to SQLite database: {self.sqlite_path}")
            else:
                logger.warning(f"SQLite database not found: {self.sqlite_path}")
                return False

            # Connect to PostgreSQL
            self.postgres_engine = create_engine(
                self.postgres_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True
            )

            # Test connection
            with self.postgres_engine.connect() as conn:
                conn.execute("SELECT 1")

            logger.info(f"Connected to PostgreSQL database: {self.pg_db}")
            return True

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def create_schema(self):
        """Create all tables in PostgreSQL"""
        try:
            # Drop all existing tables (careful in production!)
            Base.metadata.drop_all(bind=self.postgres_engine)
            logger.info("Dropped existing tables")

            # Create all tables
            Base.metadata.create_all(bind=self.postgres_engine)
            logger.info("Created all tables in PostgreSQL")

            return True

        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            return False

    def migrate_data(self):
        """Migrate data from SQLite to PostgreSQL"""
        try:
            # Create sessions
            SqliteSession = sessionmaker(bind=self.sqlite_engine)
            PostgresSession = sessionmaker(bind=self.postgres_engine)

            sqlite_session = SqliteSession()
            postgres_session = PostgresSession()

            # Define migration order (respecting foreign key constraints)
            migration_order = [
                (User, "users"),
                (Agent, "agents"),
                (Dispute, "disputes"),
                (Receipt, "receipts"),
                (Evidence, "evidences"),
                (Settlement, "settlements"),
                (SystemMetrics, "system_metrics")
            ]

            total_records = 0

            for model, table_name in migration_order:
                try:
                    # Check if table exists in SQLite
                    inspector = inspect(self.sqlite_engine)
                    if table_name not in inspector.get_table_names():
                        logger.info(f"Table {table_name} not found in SQLite, skipping")
                        continue

                    # Get data from SQLite
                    records = sqlite_session.query(model).all()
                    count = len(records)

                    if count > 0:
                        # Migrate records
                        for record in records:
                            # Create new instance with same attributes
                            new_record = model()
                            for column in model.__table__.columns:
                                if hasattr(record, column.name):
                                    setattr(new_record, column.name, getattr(record, column.name))

                            postgres_session.add(new_record)

                        postgres_session.commit()
                        total_records += count
                        logger.info(f"Migrated {count} records from {table_name}")
                    else:
                        logger.info(f"No records found in {table_name}")

                except Exception as e:
                    logger.error(f"Failed to migrate {table_name}: {e}")
                    postgres_session.rollback()

            sqlite_session.close()
            postgres_session.close()

            logger.info(f"Total records migrated: {total_records}")
            return True

        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return False

    def verify_migration(self):
        """Verify data was migrated successfully"""
        try:
            PostgresSession = sessionmaker(bind=self.postgres_engine)
            session = PostgresSession()

            verification = {
                "users": session.query(User).count(),
                "disputes": session.query(Dispute).count(),
                "receipts": session.query(Receipt).count(),
                "evidences": session.query(Evidence).count(),
                "settlements": session.query(Settlement).count(),
                "agents": session.query(Agent).count(),
                "system_metrics": session.query(SystemMetrics).count()
            }

            session.close()

            logger.info("Migration verification:")
            for table, count in verification.items():
                logger.info(f"  {table}: {count} records")

            return verification

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return None

    def create_indexes(self):
        """Create performance indexes"""
        try:
            with self.postgres_engine.connect() as conn:
                indexes = [
                    # User indexes
                    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
                    "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
                    "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);",

                    # Dispute indexes
                    "CREATE INDEX IF NOT EXISTS idx_disputes_submitter_id ON disputes(submitter_id);",
                    "CREATE INDEX IF NOT EXISTS idx_disputes_status ON disputes(status);",
                    "CREATE INDEX IF NOT EXISTS idx_disputes_phase ON disputes(phase);",
                    "CREATE INDEX IF NOT EXISTS idx_disputes_created_at ON disputes(created_at);",

                    # Receipt indexes
                    "CREATE INDEX IF NOT EXISTS idx_receipts_dispute_id ON receipts(dispute_id);",
                    "CREATE INDEX IF NOT EXISTS idx_receipts_step ON receipts(step);",
                    "CREATE INDEX IF NOT EXISTS idx_receipts_timestamp ON receipts(timestamp);",

                    # Evidence indexes
                    "CREATE INDEX IF NOT EXISTS idx_evidences_dispute_id ON evidences(dispute_id);",
                    "CREATE INDEX IF NOT EXISTS idx_evidences_submitter_id ON evidences(submitter_id);",

                    # Agent indexes
                    "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);",
                    "CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type);",

                    # System metrics indexes
                    "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_metrics_type ON system_metrics(metric_type);"
                ]

                for index_sql in indexes:
                    conn.execute(index_sql)
                    logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")

            logger.info("All indexes created successfully")
            return True

        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False

    def update_sequences(self):
        """Update PostgreSQL sequences for auto-increment fields"""
        try:
            with self.postgres_engine.connect() as conn:
                # Update sequences for tables with auto-increment IDs
                sequences = [
                    "SELECT setval('system_metrics_id_seq', (SELECT MAX(id) FROM system_metrics));",
                ]

                for seq_sql in sequences:
                    try:
                        conn.execute(seq_sql)
                        logger.info(f"Updated sequence")
                    except:
                        pass  # Sequence might not exist for UUID-based tables

            return True

        except Exception as e:
            logger.error(f"Sequence update failed: {e}")
            return False

    def run_migration(self):
        """Execute complete migration process"""
        logger.info("=" * 60)
        logger.info("DALRN DATABASE MIGRATION TO POSTGRESQL")
        logger.info("=" * 60)

        steps = [
            ("Connecting to databases", self.connect_databases),
            ("Creating schema in PostgreSQL", self.create_schema),
            ("Migrating data", self.migrate_data),
            ("Creating indexes", self.create_indexes),
            ("Updating sequences", self.update_sequences),
            ("Verifying migration", self.verify_migration)
        ]

        for step_name, step_func in steps:
            logger.info(f"\n{step_name}...")
            result = step_func()

            if result is False:
                logger.error(f"Migration failed at step: {step_name}")
                return False

        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        # Update environment file
        self.update_env_file()

        return True

    def update_env_file(self):
        """Update .env file to use PostgreSQL"""
        env_content = f"""# DALRN Production Configuration
# Updated: {datetime.now().isoformat()}

# Database Configuration (PostgreSQL)
DATABASE_URL=postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}
DALRN_ENV=production
POSTGRES_HOST={self.pg_host}
POSTGRES_PORT={self.pg_port}
POSTGRES_DB={self.pg_db}
POSTGRES_USER={self.pg_user}
POSTGRES_PASSWORD={self.pg_password}

# Old SQLite (backup reference)
# SQLITE_PATH={self.sqlite_path}

# Performance Settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_RECYCLE=3600
DB_POOL_TIMEOUT=30

# Other Settings
SQL_DEBUG=false
"""

        env_path = Path(__file__).parent.parent / ".env.production"
        with open(env_path, "w") as f:
            f.write(env_content)

        logger.info(f"Created .env.production file with PostgreSQL configuration")

def main():
    """Run the migration"""
    migrator = DatabaseMigrator()

    # Confirm before proceeding
    print("\nWARNING: This will migrate data from SQLite to PostgreSQL.")
    print("Make sure PostgreSQL is running and the production database is created.")
    response = input("Continue? (yes/no): ")

    if response.lower() != "yes":
        print("Migration cancelled")
        return

    success = migrator.run_migration()

    if success:
        print("\n✅ Migration completed successfully!")
        print("Update your application to use DATABASE_URL from .env.production")
    else:
        print("\n❌ Migration failed! Check logs for details")

if __name__ == "__main__":
    main()