"""
Update Database URLs Throughout DALRN Codebase
Switches from SQLite to PostgreSQL configuration
"""
import os
import re
from pathlib import Path
from typing import List, Tuple

class DatabaseURLUpdater:
    """Updates database URLs from SQLite to PostgreSQL"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.pg_url = "postgresql://dalrn_user:DALRN_Pr0d_2024!SecureP@ss@localhost:5432/dalrn_production"
        self.files_updated = []

    def find_files_with_db_urls(self) -> List[Path]:
        """Find all files that might contain database URLs"""
        patterns = ["*.py", "*.env", "*.yml", "*.yaml", "*.json"]
        exclude_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}

        files = []
        for pattern in patterns:
            for file in self.project_root.rglob(pattern):
                # Skip excluded directories
                if any(excluded in str(file) for excluded in exclude_dirs):
                    continue

                # Skip this updater script
                if file.name == "update_database_urls.py":
                    continue

                files.append(file)

        return files

    def update_file(self, file_path: Path) -> bool:
        """Update database URLs in a single file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            updated = False

            # Patterns to replace
            replacements = [
                # SQLite URLs
                (r'sqlite:///[\w\-\.]+\.db', self.pg_url),
                (r'"sqlite:///[^"]+\.db"', f'"{self.pg_url}"'),
                (r"'sqlite:///[^']+\.db'", f"'{self.pg_url}'"),

                # SQLite path references
                (r'SQLITE_PATH\s*=\s*["\'][\w\-\.]+\.db["\']',
                 f'DATABASE_URL = "{self.pg_url}"'),

                # Database environment checks
                (r'if\s+self\.environment\s*==\s*"development":\s*return\s*f"sqlite:///{[^}]+}"',
                 f'if self.environment == "production":\n            return "{self.pg_url}"'),

                # Update DALRN_ENV to production
                (r'DALRN_ENV\s*=\s*["\']development["\']', 'DALRN_ENV = "production"'),
                (r'os\.getenv\("DALRN_ENV",\s*"development"\)',
                 'os.getenv("DALRN_ENV", "production")'),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            # Update specific service configurations
            if "services/database/production_config.py" in str(file_path):
                # Ensure production database is default
                content = re.sub(
                    r'self\.environment = os\.getenv\("DALRN_ENV", "development"\)',
                    'self.environment = os.getenv("DALRN_ENV", "production")',
                    content
                )

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.files_updated.append(file_path)
                return True

        except Exception as e:
            print(f"Error updating {file_path}: {e}")

        return False

    def create_env_files(self):
        """Create proper .env files for production"""
        # .env file for production
        env_content = """# DALRN Production Environment Configuration

# Database
DATABASE_URL=postgresql://dalrn_user:DALRN_Pr0d_2024!SecureP@ss@localhost:5432/dalrn_production
DALRN_ENV=production
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=dalrn_production
POSTGRES_USER=dalrn_user
POSTGRES_PASSWORD=DALRN_Pr0d_2024!SecureP@ss

# Connection Pool
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_RECYCLE=3600
DB_POOL_TIMEOUT=30
SQL_DEBUG=false

# Blockchain
DALRN_CONTRACT_ADDRESS=0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512
BLOCKCHAIN_NETWORK=local
WEB3_PROVIDER_URL=http://localhost:8545

# Service Ports
GATEWAY_PORT=8000
SEARCH_PORT=8100
FHE_PORT=8200
NEGOTIATION_PORT=8300
FL_PORT=8400
AGENTS_PORT=8500

# Performance
GATEWAY_MODE=minimal
CACHE_TTL=300
RESPONSE_CACHE_SIZE=5000
DATABASE_CACHE_SIZE=2000

# Security
JWT_SECRET=DALRN_JWT_Secret_2024_Pr0duction
API_KEY=DALRN_API_Key_2024
RATE_LIMIT_PER_MINUTE=100
"""

        env_path = self.project_root / ".env"
        with open(env_path, "w") as f:
            f.write(env_content)

        print(f"Created {env_path}")

        # .env.example for reference
        example_content = env_content.replace(
            "DALRN_Pr0d_2024!SecureP@ss", "<your-password>").replace(
            "DALRN_JWT_Secret_2024_Pr0duction", "<your-jwt-secret>").replace(
            "DALRN_API_Key_2024", "<your-api-key>")

        example_path = self.project_root / ".env.example"
        with open(example_path, "w") as f:
            f.write(example_content)

        print(f"Created {example_path}")

    def update_docker_compose(self):
        """Update docker-compose.yml for PostgreSQL"""
        compose_path = self.project_root / "infra" / "docker-compose.yml"

        if not compose_path.exists():
            compose_path = self.project_root / "docker-compose.yml"

        if compose_path.exists():
            with open(compose_path, "r") as f:
                content = f.read()

            # Add PostgreSQL service if not present
            if "postgres:" not in content:
                postgres_service = """
  postgres:
    image: postgres:14-alpine
    container_name: dalrn-postgres
    environment:
      POSTGRES_DB: dalrn_production
      POSTGRES_USER: dalrn_user
      POSTGRES_PASSWORD: DALRN_Pr0d_2024!SecureP@ss
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dalrn_user"]
      interval: 10s
      timeout: 5s
      retries: 5
"""
                # Add to services section
                content = content.replace("services:", f"services:\n{postgres_service}")

                # Add volume
                if "volumes:" not in content:
                    content += "\nvolumes:\n  postgres_data:"
                else:
                    content = content.replace("volumes:", "volumes:\n  postgres_data:")

                with open(compose_path, "w") as f:
                    f.write(content)

                print(f"Updated {compose_path} with PostgreSQL service")

    def run(self):
        """Execute the URL update process"""
        print("DALRN Database URL Update")
        print("=" * 60)
        print(f"Updating to PostgreSQL: {self.pg_url}")
        print()

        # Find files
        files = self.find_files_with_db_urls()
        print(f"Found {len(files)} files to check")

        # Update files
        for file in files:
            if self.update_file(file):
                print(f"Updated: {file.relative_to(self.project_root)}")

        # Create environment files
        self.create_env_files()

        # Update Docker Compose
        self.update_docker_compose()

        print(f"\n{'='*60}")
        print(f"Update complete!")
        print(f"Files updated: {len(self.files_updated)}")
        print(f"Database URL: {self.pg_url}")
        print()
        print("Next steps:")
        print("1. Ensure PostgreSQL is running")
        print("2. Run: python database/migrate_to_postgres.py")
        print("3. Restart all services")

def main():
    updater = DatabaseURLUpdater()
    updater.run()

if __name__ == "__main__":
    main()